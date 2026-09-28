// Repack only the active HTP region into the existing Adreno Q8 SOA layout.
// This copies quant/scale bits exactly; it does not dequantize or requantize.
// Each work item writes one transposed four-quant word and, once per Q8 block,
// its half scale. The scratch is reused after its in-order GEMM consumer.
__kernel void kernel_repack_htp_q8_native(
        __global const uchar * weights, ulong weight_offset,
        int parent_k, ulong weight_stride,
        int k_start, int k_size, int out_start, int out_size,
        __global uint * native_q, __global ushort * native_d) {
    const size_t index = get_global_id(0);
    if (index >= (size_t) (k_size / 4) * out_size) { return; }
    const int row = index % out_size;
    const int quad = index / out_size;
    __global const char * q = (__global const char *)
        (weights + weight_offset + (ulong) (out_start + row) * weight_stride);
    native_q[index] = as_uint(vload4(0, q + k_start + quad * 4));
    if ((quad % 8) == 0) {
        __global const ushort * d = (__global const ushort *) (q + parent_k);
        native_d[(size_t) (quad / 8) * out_size + row] = d[k_start / 32 + quad / 8];
    }
}
