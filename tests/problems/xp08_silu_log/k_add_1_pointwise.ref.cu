extern "C" __global__
__launch_bounds__(256) void k_add_1_pointwise(const float* silu_one, const float* add_c1, const float* x, float* add_1) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 847) {
        int a0 = _gid;
        float in2__u0 = x[a0 * 3];
        float in0__u0 = add_c1[0];
        float in1__u0 = silu_one[0];
        float v8__u0 = log((in0__u0 + fabsf(in2__u0))) + (in2__u0 * (1.0f / (in1__u0 + expf((0.0f - in2__u0)))));
        float in2__u1 = x[a0 * 3 + 1];
        float in0__u1 = add_c1[0];
        float in1__u1 = silu_one[0];
        float v8__u1 = log((in0__u1 + fabsf(in2__u1))) + (in2__u1 * (1.0f / (in1__u1 + expf((0.0f - in2__u1)))));
        float in2__u2 = x[a0 * 3 + 2];
        float in0__u2 = add_c1[0];
        float in1__u2 = silu_one[0];
        float v8__u2 = log((in0__u2 + fabsf(in2__u2))) + (in2__u2 * (1.0f / (in1__u2 + expf((0.0f - in2__u2)))));
        add_1[a0 * 3] = v8__u0;
        add_1[a0 * 3 + 1] = v8__u1;
        add_1[a0 * 3 + 2] = v8__u2;
    }
}
