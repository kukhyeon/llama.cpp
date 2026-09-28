#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// One workgroup computes one output channel/token. HTP rows contain
// [parent_k signed quants][parent_k/32 half scales], without a native cache.
__kernel void kernel_mul_mat_htp_q8(
        __global const uchar * weights, ulong weight_offset,
        __global const uchar * input, ulong input_offset,
        __global uchar * output, ulong output_offset,
        int parent_k, ulong weight_stride,
        int k_start, int k_size, int out_start, int out_size,
        ulong input_stride, ulong output_stride, int n_tokens) {
    // Small-token fallback; the host launches exactly out_size x n_tokens.
    (void) n_tokens;
    const int row = get_group_id(0);
    const int token = get_global_id(1);
    const int lane = get_local_id(0);
    __global const char * q = (__global const char *)
        (weights + weight_offset + (ulong) (out_start + row) * weight_stride);
    __global const half * d = (__global const half *) (q + parent_k);
    __global const float * x = (__global const float *)
        (input + input_offset + (ulong) token * input_stride);
    float sum = 0.0f;
    for (int k = lane; k < k_size; k += 64) {
        const int pk = k_start + k;
        sum = fma(convert_float(q[pk]) * convert_float(d[pk / 32]), x[k], sum);
    }
    __local float partial[64];
    partial[lane] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int step = 32; step; step >>= 1) {
        if (lane < step) { partial[lane] += partial[lane + step]; }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0 && row < out_size) {
        ((__global float *) (output + output_offset + (ulong) token * output_stride))[row] = partial[0];
    }
}

// Use named vector components: the Adreno compiler used on the target device
// rejects vector subscripts, even though the host OpenCL compiler accepts them.
static inline void ggml_htp_q8_store4(
        __global uchar * output, ulong output_offset, ulong output_stride,
        int row, int token, int out_size, int n_tokens, float4 value) {
    if (row >= out_size || token >= n_tokens) { return; }
    __global float * y = (__global float *)
        (output + output_offset + (ulong)token * output_stride);
    if (row + 3 < out_size) {
        vstore4(value, 0, y + row);
    } else {
        y[row] = value.s0;
        if (row + 1 < out_size) { y[row + 1] = value.s1; }
        if (row + 2 < out_size) { y[row + 2] = value.s2; }
    }
}

// Prefill: a workgroup computes 32 output channels x 32 tokens. Each thread
// owns 4 x 4 outputs, so no workgroup reduction is needed. The shared weight
// remains in HTP format; only two 32 x 32 FP32 tiles (8 KiB) live in local memory.
// k_size/k_start are multiples of 32; channel and token tails are masked.
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void kernel_mul_mm_htp_q8(
        __global const uchar * weights, ulong weight_offset,
        __global const uchar * input, ulong input_offset,
        __global uchar * output, ulong output_offset,
        int parent_k, ulong weight_stride,
        int k_start, int k_size, int out_start, int out_size,
        ulong input_stride, ulong output_stride, int n_tokens) {
    const int tid = get_local_id(0);
    const int row_base = get_group_id(0) * 32;
    const int token_base = get_group_id(1) * 32;
    const int row_local = (tid % 8) * 4;
    const int token_local = (tid / 8) * 4;
    __local float tile_w[32 * 32];
    __local float tile_x[32 * 32];
    float4 sum0 = (float4)(0), sum1 = (float4)(0);
    float4 sum2 = (float4)(0), sum3 = (float4)(0);

    for (int kb = 0; kb < k_size; kb += 32) {
        // Consecutive threads read consecutive K quads within each HTP row.
        // The local tile is transposed for reuse by the output microtiles.
        for (int i = tid; i < 32 * 8; i += 64) {
            const int r = i / 8;
            const int k = (i % 8) * 4;
            float4 wv = (float4)(0);
            if (row_base + r < out_size) {
                __global const char * q = (__global const char *)
                    (weights + weight_offset + (ulong)(out_start + row_base + r) * weight_stride);
                __global const half * d = (__global const half *)(q + parent_k);
                wv = convert_float4(vload4(0, q + k_start + kb + k)) *
                    convert_float(d[(k_start + kb) / 32]);
            }
            tile_w[(k + 0) * 32 + r] = wv.s0;
            tile_w[(k + 1) * 32 + r] = wv.s1;
            tile_w[(k + 2) * 32 + r] = wv.s2;
            tile_w[(k + 3) * 32 + r] = wv.s3;

            float4 xv = (float4)(0);
            if (token_base + r < n_tokens) {
                __global const float * x = (__global const float *)
                    (input + input_offset + (ulong)(token_base + r) * input_stride);
                xv = vload4(0, x + kb + k);
            }
            tile_x[(k + 0) * 32 + r] = xv.s0;
            tile_x[(k + 1) * 32 + r] = xv.s1;
            tile_x[(k + 2) * 32 + r] = xv.s2;
            tile_x[(k + 3) * 32 + r] = xv.s3;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < 32; ++k) {
            const float4 wv = vload4(0, tile_w + k * 32 + row_local);
            const float4 xv = vload4(0, tile_x + k * 32 + token_local);
            sum0 = fma(wv, (float4)(xv.s0), sum0);
            sum1 = fma(wv, (float4)(xv.s1), sum1);
            sum2 = fma(wv, (float4)(xv.s2), sum2);
            sum3 = fma(wv, (float4)(xv.s3), sum3);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const int row = row_base + row_local;
    const int token = token_base + token_local;
    ggml_htp_q8_store4(output, output_offset, output_stride, row, token + 0, out_size, n_tokens, sum0);
    ggml_htp_q8_store4(output, output_offset, output_stride, row, token + 1, out_size, n_tokens, sum1);
    ggml_htp_q8_store4(output, output_offset, output_stride, row, token + 2, out_size, n_tokens, sum2);
    ggml_htp_q8_store4(output, output_offset, output_stride, row, token + 3, out_size, n_tokens, sum3);
}
