// Precise four-input F32 reduction used by the opt-in FFN reduce+residual
// fusion. This program is compiled without OpenCL fast-math options so the
// association matches the former ADD tree exactly:
//
//     (src0 + (src1 + src2)) + src3

kernel void kernel_add4_f32(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global char * src2,
        ulong offset2,
        global char * src3,
        ulong offset3,
        global char * dst,
        ulong offsetd,
        ulong ne) {
    const ulong i = get_global_id(0);
    if (i >= ne) {
        return;
    }

    global const float * a = (global const float *) (src0 + offset0);
    global const float * b = (global const float *) (src1 + offset1);
    global const float * c = (global const float *) (src2 + offset2);
    global const float * d = (global const float *) (src3 + offset3);
    global       float * o = (global       float *) (dst  + offsetd);

    const float bc  = b[i] + c[i];
    const float abc = a[i] + bc;
    o[i] = abc + d[i];
}
