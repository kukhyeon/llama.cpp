// Measurement-only kernels for GGML_SHARED_REPACK_BREAKDOWN. They are compiled
// after repack_htp_q8.cl and reuse its tile load with the same work-group
// geometry as kernel_repack_htp_q8_native, so only the stores differ.

// Reads every tile element the repack would store, then publishes one
// checksum word per work-group instead of the packed weights.
__kernel void kernel_probe_htp_q8_read(
        __global const uchar * weights, ulong weight_offset,
        int parent_k, ulong weight_stride,
        int k_start, int k_size, int out_start, int out_size,
        __global uint * checksums) {
    const int tid = get_local_id(1) * 16 + get_local_id(0);
    const int quad_base = get_group_id(0) * 16;
    const int row_base = get_group_id(1) * 16;
    __local uint tile_q[16 * 17];
    __local ushort tile_d[16 * 2];
    __local uint partial[64];
    ggml_htp_q8_pack_load_tile(weights, weight_offset, parent_k, weight_stride,
        k_start, k_size, out_start, out_size, quad_base, row_base, tid, tile_q, tile_d);
    barrier(CLK_LOCAL_MEM_FENCE);
    uint checksum = 0;
    for (int i = tid; i < 16 * 16; i += 64) {
        const int row = row_base + i % 16;
        const int quad = quad_base + i / 16;
        if (row < out_size && quad < k_size / 4) {
            checksum ^= tile_q[(i % 16) * 17 + i / 16];
            if ((quad % 8) == 0) {
                checksum += tile_d[(i % 16) * 2 + (i / 16) / 8];
            }
        }
    }
    partial[tid] = checksum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid == 0) {
        uint sum = 0;
        for (int i = 0; i < 64; ++i) {
            sum += partial[i];
        }
        checksums[get_group_id(1) * get_num_groups(0) + get_group_id(0)] = sum;
    }
}

// Stores the same q/d words row-major, without the native transposed layout.
__kernel void kernel_probe_htp_q8_copy(
        __global const uchar * weights, ulong weight_offset,
        int parent_k, ulong weight_stride,
        int k_start, int k_size, int out_start, int out_size,
        __global uint * copy_q, __global ushort * copy_d) {
    const int tid = get_local_id(1) * 16 + get_local_id(0);
    const int quad_base = get_group_id(0) * 16;
    const int row_base = get_group_id(1) * 16;
    __local uint tile_q[16 * 17];
    __local ushort tile_d[16 * 2];
    ggml_htp_q8_pack_load_tile(weights, weight_offset, parent_k, weight_stride,
        k_start, k_size, out_start, out_size, quad_base, row_base, tid, tile_q, tile_d);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = tid; i < 16 * 16; i += 64) {
        const int row = row_base + i / 16;
        const int quad = quad_base + i % 16;
        if (row < out_size && quad < k_size / 4) {
            copy_q[(size_t) row * (k_size / 4) + quad] = tile_q[(i / 16) * 17 + i % 16];
            if ((quad % 8) == 0) {
                copy_d[(size_t) row * (k_size / 32) + quad / 8] = tile_d[(i / 16) * 2 + (i % 16) / 8];
            }
        }
    }
}
