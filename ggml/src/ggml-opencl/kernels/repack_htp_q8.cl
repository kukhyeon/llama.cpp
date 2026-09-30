// A 16-row x 16-K-quad transpose: lanes read contiguous HTP K bytes, then
// write contiguous native rows. Padding separates adjacent local-memory rows.
// These helpers are also executed in two phases by the host byte-mapping test.
static inline void ggml_htp_q8_pack_load_tile(
        __global const uchar * weights, ulong weight_offset,
        int parent_k, ulong weight_stride, int k_start, int k_size,
        int out_start, int out_size, int quad_base, int row_base, int tid,
        __local uint * tile_q, __local ushort * tile_d) {
    for (int i = tid; i < 16 * 16; i += 64) {
        const int row = row_base + i / 16;
        const int quad = quad_base + i % 16;
        if (row < out_size && quad < k_size / 4) {
            __global const char * q = (__global const char *)
                (weights + weight_offset + (ulong) (out_start + row) * weight_stride);
            tile_q[(i / 16) * 17 + i % 16] = as_uint(vload4(0, q + k_start + quad * 4));
            if ((quad % 8) == 0) {
                __global const ushort * d = (__global const ushort *) (q + parent_k);
                tile_d[(i / 16) * 2 + (i % 16) / 8] = d[k_start / 32 + quad / 8];
            }
        }
    }
}

static inline void ggml_htp_q8_pack_store_tile(
        int k_size, int out_size, int quad_base, int row_base, int tid,
        int destination_m, int destination_k, int destination_row,
        __local const uint * tile_q, __local const ushort * tile_d,
        __global uint * native_q, __global ushort * native_d) {
    for (int i = tid; i < 16 * 16; i += 64) {
        const int row = row_base + i % 16;
        const int quad = quad_base + i / 16;
        if (row < out_size && quad < k_size / 4) {
            native_q[(size_t) (destination_k / 4 + quad) * destination_m + destination_row + row] =
                tile_q[(i % 16) * 17 + i / 16];
            if ((quad % 8) == 0) {
                native_d[(size_t) (destination_k / 32 + quad / 8) * destination_m + destination_row + row] =
                    tile_d[(i % 16) * 2 + (i / 16) / 8];
            }
        }
    }
}

#ifndef GGML_HTP_PACK_HOST_TEST
__kernel void kernel_repack_htp_q8_native(
        __global const uchar * weights, ulong weight_offset,
        int parent_k, ulong weight_stride,
        int k_start, int k_size, int out_start, int out_size,
        int destination_m, int destination_k, int destination_row,
        __global uint * native_q, __global ushort * native_d) {
    const int tid = get_local_id(1) * 16 + get_local_id(0);
    const int quad_base = get_group_id(0) * 16;
    const int row_base = get_group_id(1) * 16;
    __local uint tile_q[16 * 17];
    __local ushort tile_d[16 * 2];
    ggml_htp_q8_pack_load_tile(weights, weight_offset, parent_k, weight_stride,
        k_start, k_size, out_start, out_size, quad_base, row_base, tid, tile_q, tile_d);
    barrier(CLK_LOCAL_MEM_FENCE);
    ggml_htp_q8_pack_store_tile(k_size, out_size, quad_base, row_base, tid,
        destination_m, destination_k, destination_row, tile_q, tile_d, native_q, native_d);
}

#endif

// Native exclusive regions already have transposed q/d. Only their physical
// parent stride and destination origin change; all bits remain unchanged.
__kernel void kernel_repack_q8_native_region(
        __global const uint * source_q, __global const ushort * source_d,
        int parent_m, int k_start, int k_size, int out_start, int out_size,
        int destination_m, int destination_k, int destination_row,
        __global uint * native_q, __global ushort * native_d) {
    const size_t index = get_global_id(0);
    if (index >= (size_t) (k_size / 4) * out_size) { return; }
    const int row = index % out_size;
    const int quad = index / out_size;
    native_q[(size_t) (destination_k / 4 + quad) * destination_m + destination_row + row] =
        source_q[(size_t) (k_start / 4 + quad) * parent_m + out_start + row];
    if ((quad % 8) == 0) {
        native_d[(size_t) (destination_k / 32 + quad / 8) * destination_m + destination_row + row] =
            source_d[(size_t) (k_start / 32 + quad / 8) * parent_m + out_start + row];
    }
}
