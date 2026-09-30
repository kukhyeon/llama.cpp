// Execute the actual pack helpers in load/barrier/store phases on the host.
// This verifies mapping and bounds, not device scheduling or performance.
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
using uchar = unsigned char;
using ushort = unsigned short;
using uint = unsigned int;
using ulong = unsigned long;
static_assert(sizeof(uint) == 4 && sizeof(ushort) == 2, "OpenCL scalar widths");
struct char4 { char values[4]; };
static size_t work_item;
static size_t get_global_id(unsigned) { return work_item; }
static char4 vload4(size_t offset, const char * p) { char4 v; std::memcpy(v.values, p + 4 * offset, 4); return v; }
static uint as_uint(char4 v) { uint bits; std::memcpy(&bits, v.values, 4); return bits; }
#define __global
#define __local
#define __kernel static
#define GGML_HTP_PACK_HOST_TEST
#include "../ggml/src/ggml-opencl/kernels/repack_htp_q8.cl"
#undef GGML_HTP_PACK_HOST_TEST
#undef __global
#undef __local
#undef __kernel

struct source {
    int k, m;
    size_t stride, offset;
    std::vector<uchar> htp;
    std::vector<uint> q;
    std::vector<ushort> d;
    source(int nk, int nm, size_t off, int seed) : k(nk), m(nm), stride(nk + nk / 16), offset(off),
        htp(off + nm * stride + 256, 0xCD), q((size_t) nk / 4 * nm), d((size_t) nk / 32 * nm) {
        constexpr ushort bits[] = {0, 0x8000, 0x3C00, 0xBC00, 1, 0x8001, 0x7C00, 0xFC00, 0x7E01, 0xFE55, 0x7BFF};
        for (int r = 0; r < m; ++r) {
            for (int i = 0; i < k; ++i) htp[offset + r * stride + i] = (uchar) ((r * 31 + i * 17 + seed) & 255);
            for (int i = 0; i < k / 32; ++i) {
                const ushort scale = bits[(r + i + seed) % (sizeof(bits) / sizeof(bits[0]))];
                std::memcpy(htp.data() + offset + r * stride + k + i * 2, &scale, 2);
                d[(size_t) i * m + r] = scale;
            }
            for (int i = 0; i < k / 4; ++i) std::memcpy(&q[(size_t) i * m + r], htp.data() + offset + r * stride + 4 * i, 4);
        }
    }
};
struct destination {
    static constexpr size_t guard = 8;
    int k, m;
    std::vector<uint> q, expected_q;
    std::vector<ushort> d, expected_d;
    destination(int nk, int nm) : k(nk), m(nm), q((size_t) nk / 4 * nm + 2 * guard, 0xAAAAAAAA),
        expected_q(q), d((size_t) nk / 32 * nm + 2 * guard, 0xBBBB), expected_d(d) {}
    bool valid() const { return q == expected_q && d == expected_d; }
};
static void pack(const source & src, destination & dst, bool native, int ks, int kn, int os, int on, int dk, int dm) {
    if (native) {
        const size_t count = ((size_t) kn / 4 * on + 63) / 64 * 64;
        for (size_t n = count; n > 0; --n) {
            work_item = n - 1;
            kernel_repack_q8_native_region(src.q.data(), src.d.data(), src.m, ks, kn, os, on,
                dst.m, dk, dm, dst.q.data() + destination::guard, dst.d.data() + destination::guard);
        }
    } else {
        for (int qb = 0; qb < kn / 4; qb += 16) for (int rb = 0; rb < on; rb += 16) {
            std::array<uint, 16 * 17> tile_q{};
            std::array<ushort, 16 * 2> tile_d{};
            for (int tid = 63; tid >= 0; --tid) ggml_htp_q8_pack_load_tile(src.htp.data(), src.offset, src.k, src.stride,
                ks, kn, os, on, qb, rb, tid, tile_q.data(), tile_d.data());
            // The device's local-memory barrier separates these exact helper calls.
            for (int tid = 0; tid < 64; ++tid) ggml_htp_q8_pack_store_tile(kn, on, qb, rb, tid,
                dst.m, dk, dm, tile_q.data(), tile_d.data(), dst.q.data() + destination::guard, dst.d.data() + destination::guard);
        }
    }
    // Independent byte-level reference, including all untouched destination bytes.
    auto * expected = reinterpret_cast<uchar *>(dst.expected_q.data() + destination::guard);
    for (int row = 0; row < on; ++row) {
        for (int k = 0; k < kn; ++k) {
            const size_t to = ((size_t) ((dk + k) / 4) * dst.m + dm + row) * 4 + (dk + k) % 4;
            expected[to] = src.htp[src.offset + (os + row) * src.stride + ks + k];
        }
        for (int b = 0; b < kn / 32; ++b) {
            ushort value;
            std::memcpy(&value, src.htp.data() + src.offset + (os + row) * src.stride + src.k + (ks / 32 + b) * 2, 2);
            dst.expected_d[destination::guard + (size_t) (dk / 32 + b) * dst.m + dm + row] = value;
        }
    }
}
int main() {
    int count = 0;
    const std::array<std::array<int, 3>, 7> shapes = {{{32,0,32},{96,32,32},{256,32,192},{1024,256,512},
        {6656,2688,1792},{8192,960,2048},{3072,0,3072}}};
    for (auto s : shapes) for (int rows : {1,3,4,7,32,35,128}) for (int origin : {0,3,17})
        for (size_t offset : {size_t(0),size_t(256),size_t(4096)}) {
            source src(s[0], origin + rows + 3, offset, 128);
            const auto original_htp = src.htp; const auto original_q = src.q; const auto original_d = src.d;
            for (bool native : {false,true}) for (bool shifted : {false,true}) {
                destination dst(s[2] + (shifted ? 96 : 0), rows + (shifted ? 11 : 0));
                pack(src,dst,native,s[1],s[2],origin,rows,shifted ? 64 : 0,shifted ? 3 : 0);
                if (!dst.valid() || src.htp != original_htp || src.q != original_q || src.d != original_d) {
                    std::cerr << "Pack mismatch in single region " << count << '\n'; return 1;
                }
                ++count;
            }
        }
    // Whole-branch gate/up row concat and down K concat, both source orders.
    for (bool reversed : {false,true}) {
        source up0(3072, 1000, 256, 17), up1(3072, 2100, 4096, 31);
        destination up(3072,3008);
        pack(up0,up,!reversed,0,3072,40,960,0,0);
        pack(up1,up,reversed,0,3072,17,2048,0,960);
        if (!up.valid()) { std::cerr << "Whole-branch row concat mismatch\n"; return 1; }
        source down0(1024,3072,256,17), down1(3072,3072,4096,31);
        destination down(3008,3072);
        pack(down0,down,!reversed,64,960,0,3072,0,0);
        pack(down1,down,reversed,512,2048,0,3072,960,0);
        if (!down.valid()) { std::cerr << "Whole-branch K concat mismatch\n"; return 1; }
        count += 2;
    }
    std::cout << count << " exact tiled/native/whole-branch pack cases passed; source and guards preserved\n";
}
