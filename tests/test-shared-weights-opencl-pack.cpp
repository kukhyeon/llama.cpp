// Execute the actual packing kernel as a host function. This checks only its
// byte mapping and bounds, not OpenCL compilation, scheduling, or performance.
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
static char4 vload4(size_t offset, const char * pointer) {
    char4 value;
    std::memcpy(value.values, pointer + 4 * offset, sizeof(value));
    return value;
}
static uint as_uint(char4 value) {
    uint bits;
    std::memcpy(&bits, value.values, sizeof(bits));
    return bits;
}
#define __global
#define __kernel static
#include "../ggml/src/ggml-opencl/kernels/repack_htp_q8.cl"
#undef __global
#undef __kernel

static bool check_case(int parent_k, int k_start, int k_size, int rows, int out_start, size_t offset) {
    const size_t stride = parent_k + (parent_k / 32) * sizeof(ushort);
    const int parent_rows = out_start + rows + 3;
    std::vector<uchar> source(offset + parent_rows * stride + 256, 0xCD);
    // Include signed zero, negative scales, subnormals, infinities, and NaN
    // payloads. Packing must preserve their bits without floating conversion.
    constexpr ushort scales[] = {
        0x0000, 0x8000, 0x3C00, 0xBC00, 0x0001, 0x8001,
        0x7C00, 0xFC00, 0x7E01, 0xFE55, 0x7BFF,
    };
    for (int row = 0; row < parent_rows; ++row) {
        for (int k = 0; k < parent_k; ++k) {
            source[offset + row * stride + k] = (uchar) ((row * 31 + k * 17 + 128) & 255);
        }
        for (int block = 0; block < parent_k / 32; ++block) {
            const ushort bits = scales[(row + block) % (sizeof(scales) / sizeof(scales[0]))];
            std::memcpy(source.data() + offset + row * stride + parent_k + block * 2, &bits, 2);
        }
    }
    const auto original = source;
    const size_t q_count = (size_t) (k_size / 4) * rows;
    const size_t d_count = (size_t) (k_size / 32) * rows;
    constexpr size_t guard = 8;
    std::vector<uint> quants(q_count + 2 * guard, 0xAAAAAAAA);
    std::vector<ushort> scale_bits(d_count + 2 * guard, 0xBBBB);
    const size_t rounded_work_items = (q_count + 63) / 64 * 64;
    // Reverse order also checks that no work item depends on another item's writes.
    for (size_t remaining = rounded_work_items; remaining > 0; --remaining) {
        work_item = remaining - 1;
        kernel_repack_htp_q8_native(source.data(), offset, parent_k, stride,
            k_start, k_size, out_start, rows, quants.data() + guard, scale_bits.data() + guard);
    }
    for (int row = 0; row < rows; ++row) {
        for (int k = 0; k < k_size; ++k) {
            // Independent byte-level reference: a native uint word contains
            // four consecutive K bytes, and consecutive words enumerate rows.
            const size_t native_byte = ((size_t) (k / 4) * rows + row) * 4 + k % 4;
            const uchar actual = reinterpret_cast<const uchar *>(quants.data() + guard)[native_byte];
            if (actual != original[offset + (out_start + row) * stride + k_start + k]) { return false; }
        }
        for (int block = 0; block < k_size / 32; ++block) {
            ushort expected;
            std::memcpy(&expected, original.data() + offset + (out_start + row) * stride +
                parent_k + (k_start / 32 + block) * 2, 2);
            if (scale_bits[guard + (size_t) block * rows + row] != expected) { return false; }
        }
    }
    for (size_t i = 0; i < guard; ++i) {
        if (quants[i] != 0xAAAAAAAA || quants[guard + q_count + i] != 0xAAAAAAAA ||
            scale_bits[i] != 0xBBBB || scale_bits[guard + d_count + i] != 0xBBBB) { return false; }
    }
    return source == original;
}

int main() {
    const std::array<std::array<int, 3>, 7> shapes = {{
        {32, 0, 32}, {96, 32, 32}, {256, 32, 192}, {1024, 256, 512},
        {6656, 2688, 1792}, {8192, 960, 2048}, {3072, 0, 3072},
    }};
    int count = 0;
    for (const auto & shape : shapes) {
        for (int rows : {1, 3, 4, 7, 32, 35, 128}) {
            for (int out_start : {0, 3, 17}) {
                for (size_t offset : {size_t(0), size_t(256), size_t(4096)}) {
                    if (!check_case(shape[0], shape[1], shape[2], rows, out_start, offset)) {
                        std::cerr << "HTP OpenCL pack mismatch: parent_k=" << shape[0]
                            << " k_start=" << shape[1] << " k_size=" << shape[2]
                            << " rows=" << rows << " out_start=" << out_start << " offset=" << offset << '\n';
                        return 1;
                    }
                    ++count;
                }
            }
        }
    }
    std::cout << count << " exact HTP-to-OpenCL native pack cases passed; source and guards preserved\n";
    return 0;
}
