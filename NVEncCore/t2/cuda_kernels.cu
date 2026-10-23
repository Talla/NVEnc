// Optional fused kernels for the F3 prototype. FP32, native CUDA streams.
// No PyTorch C++ ABI dependency: tensors and streams are passed through ctypes.
#include <cuda_runtime.h>
#include <cmath>

#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C"
#endif

#include "temporal_kernels.cuh"

__global__ void sad_costs(const float *a, const float *b, const float *init,
                         float *costs, int h, int w, int block, int radius,
                         int by, int bx) {
    extern __shared__ float values[];
    int ci = blockIdx.z, tile = blockIdx.y * bx + blockIdx.x;
    int candidates = 2 * radius + 1;
    int dx = (int)nearbyintf(init[tile]) + ci % candidates - radius;
    int dy = (int)nearbyintf(init[by * bx + tile]) + ci / candidates - radius;
    int lane = threadIdx.x;
    float v = 0.0f;
    if (lane < block * block) {
        int y = blockIdx.y * block + lane / block;
        int x = blockIdx.x * block + lane % block;
        int sy = min(max(y + dy, 0), h - 1);
        int sx = min(max(x + dx, 0), w - 1);
        v = fabsf(a[y * w + x] - b[sy * w + sx]);
    }
    values[lane] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s; s /= 2) {
        if (lane < s) values[lane] += values[lane + s];
        __syncthreads();
    }
    if (lane == 0) costs[ci * by * bx + tile] = values[0] / (block * block);
}

__device__ void spline_weights(float t, float *w) {
    float t2 = t * t, t3 = t2 * t;
    w[0] = (1 - t) * (1 - t) * (1 - t) / 6;
    w[1] = (3 * t3 - 6 * t2 + 4) / 6;
    w[2] = (-3 * t3 + 3 * t2 + 3 * t + 1) / 6;
    w[3] = t3 / 6;
}

__global__ void spline_sample(const float *coeff, const float *flow, float *out,
                              int n, int ch, int h, int w) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int pixels = h * w;
    if (index >= n * ch * pixels) return;
    int batch = index / (ch * pixels);
    int xy = index % pixels, y = xy / w, x = xy % w;
    float fx = x + flow[batch * 2 * pixels + xy];
    float fy = y + flow[batch * 2 * pixels + pixels + xy];
    int x0 = (int)floorf(fx), y0 = (int)floorf(fy);
    float wx[4], wy[4], value = 0;
    spline_weights(fx - x0, wx);
    spline_weights(fy - y0, wy);
    int base = index - xy;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        int sy = min(max(y0 + j - 1, 0), h - 1);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int sx = min(max(x0 + i - 1, 0), w - 1);
            value += wy[j] * wx[i] * coeff[base + sy * w + sx];
        }
    }
    out[index] = value;
}

EXPORT int dl_sad(const float *a, const float *b, const float *init, float *out,
                  int h, int w, int block, int radius, int device, void *stream) {
    cudaError_t status = cudaSuccess; // Native: preserve the caller-owned NVEnc CUDA context.
    if (status != cudaSuccess) return (int)status;
    int threads = 1;
    while (threads < block * block) threads *= 2;
    if (threads > 1024 || block < 1 || radius < 0 || h < block || w < block)
        return (int)cudaErrorInvalidValue;
    int by = h / block, bx = w / block, nc = (2 * radius + 1) * (2 * radius + 1);
    sad_costs<<<dim3(bx, by, nc), threads, threads * sizeof(float), (cudaStream_t)stream>>>
        (a, b, init, out, h, w, block, radius, by, bx);
    return (int)cudaGetLastError();
}

EXPORT int dl_spline(const float *coeff, const float *flow, float *out,
                     int n, int ch, int h, int w, int device, void *stream) {
    cudaError_t status = cudaSuccess; // Native: preserve the caller-owned NVEnc CUDA context.
    if (status != cudaSuccess) return (int)status;
    int count = n * ch * h * w;
    spline_sample<<<(count + 255) / 256, 256, 0, (cudaStream_t)stream>>>
        (coeff, flow, out, n, ch, h, w);
    return (int)cudaGetLastError();
}

__global__ void blur_axis(const float *in, float *out, const float *k,
                         int count, int h, int w, int radius, bool vertical) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    int pix = idx % (h * w), x = pix % w, y = pix / w, base = idx - pix;
    float value = 0;
    for (int i = -radius; i <= radius; ++i) {
        int sx = vertical ? x : min(max(x + i, 0), w - 1);
        int sy = vertical ? min(max(y + i, 0), h - 1) : y;
        value += in[base + sy * w + sx] * k[i + radius];
    }
    out[idx] = value;
}

EXPORT int dl_blur(const float *in, float *tmp, float *out, const float *k,
                   int n, int ch, int h, int w, int radius, int device, void *stream) {
    cudaError_t status = cudaSuccess; // Native: preserve the caller-owned NVEnc CUDA context.
    if (status != cudaSuccess) return (int)status;
    int count = n * ch * h * w;
    blur_axis<<<(count + 255) / 256, 256, 0, (cudaStream_t)stream>>>(in, tmp, k, count, h, w, radius, false);
    blur_axis<<<(count + 255) / 256, 256, 0, (cudaStream_t)stream>>>(tmp, out, k, count, h, w, radius, true);
    return (int)cudaGetLastError();
}

__global__ void lk_products(const float *a, const float *b, float *out, int h, int w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x, count = h * w;
    if (idx >= count) return;
    int y = idx / w, x = idx % w;
    int xm = y * w + max(x - 1, 0), xp = y * w + min(x + 1, w - 1);
    int ym = max(y - 1, 0) * w + x, yp = min(y + 1, h - 1) * w + x;
    float gx = 0.25f * ((a[xp] - a[xm]) + (b[xp] - b[xm]));
    float gy = 0.25f * ((a[yp] - a[ym]) + (b[yp] - b[ym]));
    float residual = b[idx] - a[idx];
    out[idx] = gx * gx;
    out[count + idx] = gy * gy;
    out[2 * count + idx] = gx * gy;
    out[3 * count + idx] = gx * residual;
    out[4 * count + idx] = gy * residual;
}

__global__ void lk_update(const float *stats, const float *flow, float *out,
                          int count, float lam) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float xx = stats[i] + lam, yy = stats[count + i] + lam, xy = stats[2 * count + i];
    float bx = stats[3 * count + i], by = stats[4 * count + i], det = xx * yy - xy * xy;
    float du = -(yy * bx - xy * by) / det, dv = -(xx * by - xy * bx) / det;
    out[i] = flow[i] + fminf(fmaxf(du, -1.0f), 1.0f);
    out[count + i] = flow[count + i] + fminf(fmaxf(dv, -1.0f), 1.0f);
}

EXPORT int dl_lk_products(const float *a, const float *b, float *out,
                          int h, int w, int device, void *stream) {
    cudaError_t status = cudaSuccess; // Native: preserve the caller-owned NVEnc CUDA context.
    if (status != cudaSuccess) return (int)status;
    lk_products<<<(h * w + 255) / 256, 256, 0, (cudaStream_t)stream>>>(a, b, out, h, w);
    return (int)cudaGetLastError();
}

EXPORT int dl_lk_update(const float *stats, const float *flow, float *out,
                        int h, int w, float lam, int device, void *stream) {
    cudaError_t status = cudaSuccess; // Native: preserve the caller-owned NVEnc CUDA context.
    if (status != cudaSuccess) return (int)status;
    lk_update<<<(h * w + 255) / 256, 256, 0, (cudaStream_t)stream>>>(stats, flow, out, h * w, lam);
    return (int)cudaGetLastError();
}

// ======================================================================================
// fused_motion exports (opt-in, bit-identical to the unfused chains: same expressions and
// summation order, built with --fmad=false). Additive ABI; the kernels above are unchanged.
// ======================================================================================

// SAD, one warp per (8x8 block, candidate). The original block-wide reduction of 64 values
// (values[l] += values[l + s], s = 32..1) is reproduced by one add of lane l and l + 32 followed by shuffle-downs.
#define SAD_WARPS 4
__global__ void sad_costs_warp(const float *a, const float *b, const float *init, float *costs, int h, int w,
                               int radius, int by, int bx, int nc) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int ci = blockIdx.z * SAD_WARPS + warp;
    if (ci >= nc) return;
    int tile = blockIdx.y * bx + blockIdx.x;
    int candidates = 2 * radius + 1;
    int dx = (int)nearbyintf(init[tile]) + ci % candidates - radius;
    int dy = (int)nearbyintf(init[by * bx + tile]) + ci / candidates - radius;
    float v0, v1;
    {
        int y = blockIdx.y * 8 + lane / 8, x = blockIdx.x * 8 + lane % 8;
        int sy = min(max(y + dy, 0), h - 1), sx = min(max(x + dx, 0), w - 1);
        v0 = fabsf(a[y * w + x] - b[sy * w + sx]);
    }
    {
        int y = blockIdx.y * 8 + (lane + 32) / 8, x = blockIdx.x * 8 + (lane + 32) % 8;
        int sy = min(max(y + dy, 0), h - 1), sx = min(max(x + dx, 0), w - 1);
        v1 = fabsf(a[y * w + x] - b[sy * w + sx]);
    }
    float t = v0 + v1;
    for (int s = 16; s; s >>= 1) t += __shfl_down_sync(0xffffffffu, t, s);
    if (lane == 0) costs[ci * by * bx + tile] = t / 64;
}

EXPORT int dl_sad_warp(const float *a, const float *b, const float *init, float *out,
                       int h, int w, int radius, int device, void *stream) {
    cudaError_t status = cudaSuccess; // Native: preserve the caller-owned NVEnc CUDA context.
    if (status != cudaSuccess) return (int)status;
    if (radius < 0 || h < 8 || w < 8) return (int)cudaErrorInvalidValue;
    int by = h / 8, bx = w / 8, nc = (2 * radius + 1) * (2 * radius + 1);
    sad_costs_warp<<<dim3(bx, by, (nc + SAD_WARPS - 1) / SAD_WARPS), 32 * SAD_WARPS, 0, (cudaStream_t)stream>>>
        (a, b, init, out, h, w, radius, by, bx, nc);
    return (int)cudaGetLastError();
}

// Block-matching glue: argmin over the cost volume (first index on ties, NaN smallest as torch.argmin), parabolic fit,
// init + offset. Same elementwise op order as flow.block_match. out (2, by, bx): x then y.
__device__ __forceinline__ float cl_min(float v, float lo) { return (v != v) ? v : fmaxf(v, lo); }
__device__ __forceinline__ float cl_max(float v, float hi) { return (v != v) ? v : fminf(v, hi); }
__global__ void bm_glue(const float *costs, const float *init_r, float *out, int n, int by, int bx) {
    int tile = blockIdx.x * blockDim.x + threadIdx.x, plane = by * bx;
    if (tile >= plane) return;
    float cb = costs[tile];
    int best = 0;
    for (int ci = 1; ci < n * n; ++ci) {
        float v = costs[ci * plane + tile];
        if (v < cb || (v != v && cb == cb)) { cb = v; best = ci; }
    }
    int iy = best / n, ix = best % n;
    float c0 = cb;
    float cxm = costs[(iy * n + max(ix - 1, 0)) * plane + tile], cxp = costs[(iy * n + min(ix + 1, n - 1)) * plane + tile];
    float cym = costs[(max(iy - 1, 0) * n + ix) * plane + tile], cyp = costs[(min(iy + 1, n - 1) * n + ix) * plane + tile];
    float sx = 0.0f, sy = 0.0f;
    if (ix > 0 && ix < n - 1) sx = 0.5f * (cxm - cxp) / cl_min(cxm - 2 * c0 + cxp, 1e-9f);
    if (iy > 0 && iy < n - 1) sy = 0.5f * (cym - cyp) / cl_min(cym - 2 * c0 + cyp, 1e-9f);
    sx = cl_max(cl_min(sx, -0.5f), 0.5f);
    sy = cl_max(cl_min(sy, -0.5f), 0.5f);
    out[tile] = init_r[tile] + (float)(ix - n / 2) + sx;
    out[plane + tile] = init_r[plane + tile] + (float)(iy - n / 2) + sy;
}

EXPORT int dl_bm_glue(const float *costs, const float *init_r, float *out, int n, int by, int bx, int device, void *stream) {
    cudaError_t status = cudaSuccess; // Native: preserve the caller-owned NVEnc CUDA context.
    if (status != cudaSuccess) return (int)status;
    bm_glue<<<(by * bx + 127) / 128, 128, 0, (cudaStream_t)stream>>>(costs, init_r, out, n, by, bx);
    return (int)cudaGetLastError();
}

// round(nearest-neighbour resample(fb) * 2): the next level's integer search centre (F.interpolate(mode="nearest")).
__global__ void bm_init(const float *fb, float *out, int ph, int pw, int by, int bx, float sy, float sx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= by * bx) return;
    int y = i / bx, x = i % bx;
    int iy = min((int)floorf(y * sy), ph - 1), ix = min((int)floorf(x * sx), pw - 1);
    out[i] = rintf(fb[iy * pw + ix] * 2.0f);
    out[by * bx + i] = rintf(fb[ph * pw + iy * pw + ix] * 2.0f);
}

EXPORT int dl_bm_init(const float *fb, float *out, int ph, int pw, int by, int bx, float sy, float sx, int device, void *stream) {
    cudaError_t status = cudaSuccess; // Native: preserve the caller-owned NVEnc CUDA context.
    if (status != cudaSuccess) return (int)status;
    bm_init<<<(by * bx + 127) / 128, 128, 0, (cudaStream_t)stream>>>(fb, out, ph, pw, by, bx, sy, sx);
    return (int)cudaGetLastError();
}

// ---- fused LK iteration -------------------------------------------------------------
#define LK_TX 32
#define LK_TY 8
#define LK_RMAX 8
__device__ float spline_pix(const float *coeff, const float *flow, int x, int y, int h, int w) {
    int pixels = h * w, xy = y * w + x;
    float fx = x + flow[xy];
    float fy = y + flow[pixels + xy];
    int x0 = (int)floorf(fx), y0 = (int)floorf(fy);
    float wx[4], wy[4], value = 0;
    spline_weights(fx - x0, wx);
    spline_weights(fy - y0, wy);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        int sy = min(max(y0 + j - 1, 0), h - 1);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int sx = min(max(x0 + i - 1, 0), w - 1);
            value += wy[j] * wx[i] * coeff[sy * w + sx];
        }
    }
    return value;
}

// warp the target (spline) with a 1-px halo in shared memory and form the five LK products; = spline_sample + lk_products.
__global__ void lk_sp_products(const float *coeff, const float *flow, const float *a, float *out, int h, int w) {
    __shared__ float wbs[(LK_TY + 2) * (LK_TX + 2)];
    int x0 = blockIdx.x * LK_TX, y0 = blockIdx.y * LK_TY, tid = threadIdx.y * LK_TX + threadIdx.x;
    for (int i = tid; i < (LK_TY + 2) * (LK_TX + 2); i += LK_TX * LK_TY) {
        int ly = i / (LK_TX + 2), lx = i % (LK_TX + 2);
        int gy = min(max(y0 - 1 + ly, 0), h - 1), gx = min(max(x0 - 1 + lx, 0), w - 1);
        wbs[i] = spline_pix(coeff, flow, gx, gy, h, w);
    }
    __syncthreads();
    int x = x0 + threadIdx.x, y = y0 + threadIdx.y;
    if (x >= w || y >= h) return;
    int count = h * w, idx = y * w + x;
    int xm = y * w + max(x - 1, 0), xp = y * w + min(x + 1, w - 1);
    int ym = max(y - 1, 0) * w + x, yp = min(y + 1, h - 1) * w + x;
    int lx = threadIdx.x + 1, ly = threadIdx.y + 1, ls = LK_TX + 2;
    float bxm = wbs[ly * ls + lx - 1], bxp = wbs[ly * ls + lx + 1];
    float bym = wbs[(ly - 1) * ls + lx], byp = wbs[(ly + 1) * ls + lx], bc = wbs[ly * ls + lx];
    float gx = 0.25f * ((a[xp] - a[xm]) + (bxp - bxm));
    float gy = 0.25f * ((a[yp] - a[ym]) + (byp - bym));
    float residual = bc - a[idx];
    out[idx] = gx * gx;
    out[count + idx] = gy * gy;
    out[2 * count + idx] = gx * gy;
    out[3 * count + idx] = gx * residual;
    out[4 * count + idx] = gy * residual;
}

// separable blur (horizontal then vertical, replicate borders, taps in the order of blur_axis) of 5 statistic planes
// and the 2x2 solve of lk_update; = blur_axis x2 + lk_update.
__global__ void lk_blur_solve(const float *stats, const float *flow, const float *kern, float *out, int h, int w, int r, float lam) {
    extern __shared__ float tmp[];   // 5 x (16 + 2r) x 32
    const int TY = 16, TX = 32;
    int rows = TY + 2 * r;
    int x0 = blockIdx.x * TX, y0 = blockIdx.y * TY, tid = threadIdx.y * TX + threadIdx.x, nt = TX * TY;
    int count = h * w;
    for (int i = tid; i < 5 * rows * TX; i += nt) {
        int tx = i % TX, ry = (i / TX) % rows, ch = i / (TX * rows);
        int x = x0 + tx, sy = min(max(y0 - r + ry, 0), h - 1);
        float value = 0;
        if (x < w) {
            for (int k = -r; k <= r; ++k) {
                int sx = min(max(x + k, 0), w - 1);
                value += stats[ch * count + sy * w + sx] * kern[k + r];
            }
        }
        tmp[(ch * rows + ry) * TX + tx] = value;
    }
    __syncthreads();
    int x = x0 + threadIdx.x, y = y0 + threadIdx.y;
    if (x >= w || y >= h) return;
    float s[5];
    #pragma unroll
    for (int ch = 0; ch < 5; ++ch) {
        float value = 0;
        for (int k = -r; k <= r; ++k) value += tmp[(ch * rows + threadIdx.y + r + k) * TX + threadIdx.x] * kern[k + r];
        s[ch] = value;
    }
    int i = y * w + x;
    float xx = s[0] + lam, yy = s[1] + lam, xy = s[2];
    float bx = s[3], by = s[4], det = xx * yy - xy * xy;
    float du = -(yy * bx - xy * by) / det, dv = -(xx * by - xy * bx) / det;
    out[i] = flow[i] + fminf(fmaxf(du, -1.0f), 1.0f);
    out[count + i] = flow[count + i] + fminf(fmaxf(dv, -1.0f), 1.0f);
}

// normalised convolution: blur(cat[flow * conf, conf]) -> [:2] / clamp([2], 1e-12)
__global__ void nc_fused(const float *flow, const float *conf, const float *kern, float *out, int h, int w, int r) {
    extern __shared__ float tmp[];   // 3 x (16 + 2r) x 32
    const int TY = 16, TX = 32;
    int rows = TY + 2 * r;
    int x0 = blockIdx.x * TX, y0 = blockIdx.y * TY, tid = threadIdx.y * TX + threadIdx.x, nt = TX * TY;
    int count = h * w;
    for (int i = tid; i < 3 * rows * TX; i += nt) {
        int tx = i % TX, ry = (i / TX) % rows, ch = i / (TX * rows);
        int x = x0 + tx, sy = min(max(y0 - r + ry, 0), h - 1);
        float value = 0;
        if (x < w) {
            for (int k = -r; k <= r; ++k) {
                int sx = min(max(x + k, 0), w - 1), p = sy * w + sx;
                float g = ch < 2 ? flow[ch * count + p] * conf[p] : conf[p];
                value += g * kern[k + r];
            }
        }
        tmp[(ch * rows + ry) * TX + tx] = value;
    }
    __syncthreads();
    int x = x0 + threadIdx.x, y = y0 + threadIdx.y;
    if (x >= w || y >= h) return;
    float s[3];
    #pragma unroll
    for (int ch = 0; ch < 3; ++ch) {
        float value = 0;
        for (int k = -r; k <= r; ++k) value += tmp[(ch * rows + threadIdx.y + r + k) * TX + threadIdx.x] * kern[k + r];
        s[ch] = value;
    }
    int i = y * w + x;
    float den = cl_min(s[2], 1e-12f);
    out[i] = s[0] / den;
    out[count + i] = s[1] / den;
}

EXPORT int dl_lk_sp_products(const float *coeff, const float *flow, const float *a, float *stats, int h, int w, int device, void *stream) {
    cudaError_t status = cudaSuccess; // Native: preserve the caller-owned NVEnc CUDA context.
    if (status != cudaSuccess) return (int)status;
    lk_sp_products<<<dim3((w + LK_TX - 1) / LK_TX, (h + LK_TY - 1) / LK_TY), dim3(LK_TX, LK_TY), 0, (cudaStream_t)stream>>>(coeff, flow, a, stats, h, w);
    return (int)cudaGetLastError();
}

EXPORT int dl_lk_blur_solve(const float *stats, const float *flow, const float *kern, float *out, int h, int w, int r, float lam, int device, void *stream) {
    cudaError_t status = cudaSuccess; // Native: preserve the caller-owned NVEnc CUDA context.
    if (status != cudaSuccess) return (int)status;
    if (r < 1 || r > LK_RMAX) return (int)cudaErrorInvalidValue;
    size_t smem = 5 * (16 + 2 * r) * 32 * sizeof(float);
    lk_blur_solve<<<dim3((w + 31) / 32, (h + 15) / 16), dim3(32, 16), smem, (cudaStream_t)stream>>>(stats, flow, kern, out, h, w, r, lam);
    return (int)cudaGetLastError();
}

EXPORT int dl_nc_fused(const float *flow, const float *conf, const float *kern, float *out, int h, int w, int r, int device, void *stream) {
    cudaError_t status = cudaSuccess; // Native: preserve the caller-owned NVEnc CUDA context.
    if (status != cudaSuccess) return (int)status;
    if (r < 1 || r > LK_RMAX) return (int)cudaErrorInvalidValue;
    size_t smem = 3 * (16 + 2 * r) * 32 * sizeof(float);
    nc_fused<<<dim3((w + 31) / 32, (h + 15) / 16), dim3(32, 16), smem, (cudaStream_t)stream>>>(flow, conf, kern, out, h, w, r);
    return (int)cudaGetLastError();
}
