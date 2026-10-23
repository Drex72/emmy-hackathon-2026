extern "C" __global__
__launch_bounds__(256) void k_add_pointwise(const float* clamp_bound0, const float* clamp_bound1, const float* x, float* add) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in0__u0 = clamp_bound0[0];
        float in2__u0 = x[a0 * 4];
        float in1__u0 = clamp_bound1[0];
        float v3__u0 = fminf(in1__u0, fmaxf(in0__u0, in2__u0)) + (in2__u0 * in2__u0);
        float in0__u1 = clamp_bound0[0];
        float in2__u1 = x[a0 * 4 + 1];
        float in1__u1 = clamp_bound1[0];
        float v3__u1 = fminf(in1__u1, fmaxf(in0__u1, in2__u1)) + (in2__u1 * in2__u1);
        float in0__u2 = clamp_bound0[0];
        float in2__u2 = x[a0 * 4 + 2];
        float in1__u2 = clamp_bound1[0];
        float v3__u2 = fminf(in1__u2, fmaxf(in0__u2, in2__u2)) + (in2__u2 * in2__u2);
        float in0__u3 = clamp_bound0[0];
        float in2__u3 = x[a0 * 4 + 3];
        float in1__u3 = clamp_bound1[0];
        float v3__u3 = fminf(in1__u3, fmaxf(in0__u3, in2__u3)) + (in2__u3 * in2__u3);
        float4 _vs_v3__u0_500171f9 = make_float4(v3__u0, v3__u1, v3__u2, v3__u3);
        *reinterpret_cast<float4*>(&add[a0 * 4]) = _vs_v3__u0_500171f9;
}
