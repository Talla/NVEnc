// K5: same-order spectral preparation and crop/add for coarse recombination.
// Included by temporal_kernels.cuh, using its EXPORT. No FFT or estimator changes.
__global__ void t2_coarse_extract(const float2 *in, const float *gain, float2 *out,
                                 int hp, int wp, int hc, int wc, float factor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = wc / 2 + 1;
    if (i >= hc * stride) return;
    int y = i / stride, x = i % stride;
    int ky = y <= (hc - 1) / 2 ? y : y - hc;
    float2 value = make_float2(0.0f, 0.0f);
    if (abs(ky) * 2 < hc && x * 2 < wc) {
        int source = ((ky + hp) % hp) * (wp / 2 + 1) + x;
        float2 v = in[source];
        float g = gain[source];
        value = make_float2((v.x * g) / factor, (v.y * g) / factor);
    }
    out[i] = value;
}

__global__ void t2_coarse_embed(const float2 *in, const float *gain, float2 *out,
                               int hp, int wp, int hc, int wc, float factor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = wp / 2 + 1;
    if (i >= hp * stride) return;
    int y = i / stride, x = i % stride;
    int ky = y <= (hp - 1) / 2 ? y : y - hp;
    float2 value = make_float2(0.0f, 0.0f);
    if (abs(ky) * 2 < hc && x * 2 < wc) {
        float2 v = in[((ky + hc) % hc) * (wc / 2 + 1) + x];
        value = make_float2(v.x * factor, v.y * factor);
    }
    float mask = gain[i] > 0.0f ? 1.0f : 0.0f;
    out[i] = make_float2(value.x * mask, value.y * mask);
}

__global__ void t2_coarse_crop_add(const float *base, const float *full, float *out,
                                  int h, int w, int wp, int py, int px) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= h * w) return;
    int y = i / w, x = i % w;
    out[i] = base[i] + full[(y + py) * wp + x + px];
}

EXPORT int dl_t2_coarse_extract(const float2 *in, const float *gain, float2 *out,
                               int hp, int wp, int hc, int wc, float factor, int device, void *stream) {
    cudaError_t status = cudaSuccess; // Native: preserve the caller-owned NVEnc CUDA context.
    if (status != cudaSuccess) return (int)status;
    int size = hc * (wc / 2 + 1);
    t2_coarse_extract<<<(size + 255) / 256, 256, 0, (cudaStream_t)stream>>>(in, gain, out, hp, wp, hc, wc, factor);
    return (int)cudaGetLastError();
}

EXPORT int dl_t2_coarse_embed(const float2 *in, const float *gain, float2 *out,
                             int hp, int wp, int hc, int wc, float factor, int device, void *stream) {
    cudaError_t status = cudaSuccess; // Native: preserve the caller-owned NVEnc CUDA context.
    if (status != cudaSuccess) return (int)status;
    int size = hp * (wp / 2 + 1);
    t2_coarse_embed<<<(size + 255) / 256, 256, 0, (cudaStream_t)stream>>>(in, gain, out, hp, wp, hc, wc, factor);
    return (int)cudaGetLastError();
}

EXPORT int dl_t2_coarse_crop_add(const float *base, const float *full, float *out,
                                int h, int w, int wp, int py, int px, int device, void *stream) {
    cudaError_t status = cudaSuccess; // Native: preserve the caller-owned NVEnc CUDA context.
    if (status != cudaSuccess) return (int)status;
    t2_coarse_crop_add<<<(h * w + 255) / 256, 256, 0, (cudaStream_t)stream>>>(base, full, out, h, w, wp, py, px);
    return (int)cudaGetLastError();
}
