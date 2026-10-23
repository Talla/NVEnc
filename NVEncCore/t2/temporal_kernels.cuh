// FP32 T2 tile operations. FFTs remain in cuFFT through torch.
__device__ int mirror_index(int x, int n) {
    if (n == 1) return 0;
    int period = 2 * (n - 1), r = x % period;
    if (r < 0) r += period;
    return min(r, period - r);
}

__global__ void t2_extract(const float *img, const long long *shift, const float *win,
                           float *out, int h, int w, int oy, int ox, int tiles, int nx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= tiles * 256) return;
    int tile = i / 256, bin = i % 256;
    int y = (tile / nx) * 8 - 16 - oy + bin / 16;
    int x = (tile % nx) * 8 - 16 - ox + bin % 16;
    if (shift) {
        long long dx = shift[2 * tile], dy = shift[2 * tile + 1];
        x = dx > (long long)w - 1 - x ? w - 1 : dx < -(long long)x ? 0 : x + (int)dx;
        y = dy > (long long)h - 1 - y ? h - 1 : dy < -(long long)y ? 0 : y + (int)dy;
    } else {
        x = mirror_index(x, w); y = mirror_index(y, h);
    }
    out[i] = img[y * w + x] * win[bin];
}

__global__ void t2_merge(const float2 *ref, const float2 *nb, const float *var,
                         const float *frac, float2 *acc, float *n,
                         int count, float noise_factor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float2 z = nb[i], r = ref[i];
    if (frac) {
        int bin = i % 256, tile = i / 256;
        int fx = bin % 16, fy = bin / 16;
        if (fx >= 8) fx -= 16;
        if (fy >= 8) fy -= 16;
        float angle = (2.0f * 3.14159265358979323846f / 16) *
                      (fx * frac[2 * tile] + fy * frac[2 * tile + 1]);
        // Fractional shifts are in [-.5,.5]: phase is bounded by pi.
        // The local intrinsic avoids general range reduction in this hot path.
        float s, c; __sincosf(angle, &s, &c);
        z = make_float2(z.x * c - z.y * s, z.x * s + z.y * c);
    }
    float dx = z.x - r.x, dy = z.y - r.y, d2 = dx * dx + dy * dy;
    float a = d2 / fmaxf(d2 + noise_factor * var[i], 1e-30f);
    float2 v = acc[i];
    acc[i] = make_float2(v.x + z.x + a * (r.x - z.x), v.y + z.y + a * (r.y - z.y));
    n[i] += 1.0f - a;
}

template<bool spatial>
__global__ void t2_wiener(float2 *merged, const float2 *pilot, const float *var,
                          const float *n, int count, const float *mask, int neighbours, float amount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float2 p = pilot[i], m = merged[i];
    float p2 = p.x * p.x + p.y * p.y;
    float variance = var[i];
    if constexpr (spatial) {
        float lost = neighbours > 1 ? fminf(1.0f, fmaxf(0.0f, 1.0f - (n[i] - 1.0f) / (neighbours - 1))) : 1.0f;
        variance *= 1.0f + amount * lost * mask[i % 256];
    }
    float gain = i % 256 == 0 ? 1.0f : p2 / fmaxf(p2 + variance / n[i], 1e-30f);
    merged[i] = make_float2(m.x * gain, m.y * gain);
}

__global__ void t2_fold(const float2 *tiles, float *out, int h, int w, int oy, int ox, int nx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= h * w) return;
    int py = i / w + 16 + oy, px = i % w + 16 + ox;
    int ty = py / 8, tx = px / 8;
    float value = 0;
    #pragma unroll
    for (int dy = 0; dy <= 1; ++dy) {
        #pragma unroll
        for (int dx = 0; dx <= 1; ++dx) {
            int tile = (ty - dy) * nx + tx - dx;
            int bin = (py - (ty - dy) * 8) * 16 + px - (tx - dx) * 8;
            value += tiles[tile * 256 + bin].x;
        }
    }
    out[i] = value;
}

EXPORT int dl_t2_extract(const float *img, const long long *shift, const float *win, float *out,
                         int h, int w, int oy, int ox, int tiles, int nx, int device, void *stream) {
    cudaError_t status = cudaSuccess; // Native: preserve the caller-owned NVEnc CUDA context.
    if (status != cudaSuccess) return (int)status;
    t2_extract<<<(tiles * 256 + 255) / 256, 256, 0, (cudaStream_t)stream>>>
        (img, shift, win, out, h, w, oy, ox, tiles, nx);
    return (int)cudaGetLastError();
}

EXPORT int dl_t2_merge(const float2 *ref, const float2 *nb, const float *var, const float *frac,
                       float2 *acc, float *n, int count, float noise_factor, int device, void *stream) {
    cudaError_t status = cudaSuccess; // Native: preserve the caller-owned NVEnc CUDA context.
    if (status != cudaSuccess) return (int)status;
    t2_merge<<<(count + 255) / 256, 256, 0, (cudaStream_t)stream>>>(ref, nb, var, frac, acc, n, count, noise_factor);
    return (int)cudaGetLastError();
}

EXPORT int dl_t2_wiener(float2 *merged, const float2 *pilot, const float *var, const float *n,
                        int count, int device, void *stream) {
    cudaError_t status = cudaSuccess; // Native: preserve the caller-owned NVEnc CUDA context.
    if (status != cudaSuccess) return (int)status;
    t2_wiener<false><<<(count + 255) / 256, 256, 0, (cudaStream_t)stream>>>(merged, pilot, var, n, count, nullptr, 1, 0.0f);
    return (int)cudaGetLastError();
}

// Additive ABI: existing DLLs can still use the torch variance fallback.
EXPORT int dl_t2_wiener_s2(float2 *merged, const float2 *pilot, const float *var, const float *n,
                           const float *mask, int count, int neighbours, float amount, int device, void *stream) {
    cudaError_t status = cudaSuccess; // Native: preserve the caller-owned NVEnc CUDA context.
    if (status != cudaSuccess) return (int)status;
    t2_wiener<true><<<(count + 255) / 256, 256, 0, (cudaStream_t)stream>>>
        (merged, pilot, var, n, count, mask, neighbours, amount);
    return (int)cudaGetLastError();
}

EXPORT int dl_t2_fold(const float2 *tiles, float *out, int h, int w, int oy, int ox,
                      int nx, int device, void *stream) {
    cudaError_t status = cudaSuccess; // Native: preserve the caller-owned NVEnc CUDA context.
    if (status != cudaSuccess) return (int)status;
    t2_fold<<<(h * w + 255) / 256, 256, 0, (cudaStream_t)stream>>>(tiles, out, h, w, oy, ox, nx);
    return (int)cudaGetLastError();
}

// ---- fused_reconstruction (stage 1, bit-identical class); additive ----
// Extract writing complex64 (imag 0): same values as t2_extract + torch real->complex copy.
__global__ void t2_extract_c(const float *img, const long long *shift, const float *win,
                             float2 *out, int h, int w, int oy, int ox, int tiles, int nx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= tiles * 256) return;
    int tile = i / 256, bin = i % 256;
    int y = (tile / nx) * 8 - 16 - oy + bin / 16;
    int x = (tile % nx) * 8 - 16 - ox + bin % 16;
    if (shift) {
        long long dx = shift[2 * tile], dy = shift[2 * tile + 1];
        x = dx > (long long)w - 1 - x ? w - 1 : dx < -(long long)x ? 0 : x + (int)dx;
        y = dy > (long long)h - 1 - y ? h - 1 : dy < -(long long)y ? 0 : y + (int)dy;
    } else {
        x = mirror_index(x, w); y = mirror_index(y, h);
    }
    out[i] = make_float2(img[y * w + x] * win[bin], 0.0f);
}

struct MergeArgs { const float2 *nb[8]; const float *frac[8]; int k; };

// All neighbours merged in registers: same per-bin op order as k sequential t2_merge calls, then M = acc * (1/cnt).
__global__ void t2_merge_multi(const float2 *ref, const float *var, MergeArgs a, float2 *M, float *n,
                               int count, float noise_factor, float inv_cnt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float2 r = ref[i], acc = r;
    float nn = 1.0f, v = var[i];
    int bin = i % 256, tile = i / 256, fx = bin % 16, fy = bin / 16;
    if (fx >= 8) fx -= 16;
    if (fy >= 8) fy -= 16;
    for (int k = 0; k < a.k; ++k) {
        float2 z = a.nb[k][i];
        if (a.frac[k]) {
            float angle = (2.0f * 3.14159265358979323846f / 16) *
                          (fx * a.frac[k][2 * tile] + fy * a.frac[k][2 * tile + 1]);
            float s, c; __sincosf(angle, &s, &c);
            z = make_float2(z.x * c - z.y * s, z.x * s + z.y * c);
        }
        float dx = z.x - r.x, dy = z.y - r.y, d2 = dx * dx + dy * dy;
        float al = d2 / fmaxf(d2 + noise_factor * v, 1e-30f);
        acc = make_float2(acc.x + z.x + al * (r.x - z.x), acc.y + z.y + al * (r.y - z.y));
        nn += 1.0f - al;
    }
    M[i] = make_float2(acc.x * inv_cnt, acc.y * inv_cnt);
    n[i] = nn;
}

// t2._centered_wiener (constant background) on complex tiles; zero factors kept literal like torch's real->complex promotion.
__global__ void t2_wiener_centered(const float2 *M, const float2 *P, const float *var, const float *n,
                                   const float2 *basis, float2 *out, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int bin = i % 256, t0 = i - bin;
    float m00 = M[t0].x, p00 = P[t0].x;
    float2 b = basis[bin], m = M[i], p = P[i];
    float bgx = m00 * b.x - 0.0f * b.y, bgy = m00 * b.y + 0.0f * b.x;
    float pbx = p00 * b.x - 0.0f * b.y, pby = p00 * b.y + 0.0f * b.x;
    float dx = p.x - pbx, dy = p.y - pby;
    float h = hypotf(dx, dy), power = h * h;
    float variance = var[i] / n[i];
    float g = power / fmaxf(power + variance, 1e-30f);
    float ex = m.x - bgx, ey = m.y - bgy;
    out[i] = make_float2(bgx + (ex * g - ey * 0.0f), bgy + (ex * 0.0f + ey * g));
}

// Coarse motion vectors: replicate-padded piecewise-constant dense field -> avg_pool2d(4), without the dense tensors.
__global__ void coarse_gather(const float *grid, float *out, int h, int w, int oy, int ox, int ny, int nx,
                              int hc, int wc, int left) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2 * hc * wc) return;
    int c = i / (hc * wc), cy = (i / wc) % hc, cx = i % wc;
    float sum = 0.0f;
    for (int dy = 0; dy < 4; ++dy) {
        int r = min(max(cy * 4 + dy - left, 0), h - 1);
        int jy = min((r + oy + 16 - 4) / 8, ny - 1);
        for (int dx = 0; dx < 4; ++dx) {
            int q = min(max(cx * 4 + dx - left, 0), w - 1);
            int jx = min((q + ox + 16 - 4) / 8, nx - 1);
            sum += grid[(jy * nx + jx) * 2 + c];
        }
    }
    out[i] = sum / 16.0f;
}

// S1 box filter pieces: 9x9 replicate-pad avg_pool2d sums in row-major order, divided by 81 (same as _box).
#define S1R 4
#define S1BX 32
#define S1BY 8
#define S1TW (S1BX + 2 * S1R)
#define S1TH (S1BY + 2 * S1R)
__device__ __forceinline__ void s1_load(const float *src, float *tile, int h, int w, int x0, int y0) {
    for (int j = threadIdx.y * S1BX + threadIdx.x; j < S1TW * S1TH; j += S1BX * S1BY) {
        int ty = j / S1TW, tx = j % S1TW;
        int y = min(max(y0 + ty - S1R, 0), h - 1), x = min(max(x0 + tx - S1R, 0), w - 1);
        tile[j] = src[y * w + x];
    }
}
__global__ void s1_ab(const float *g, const float *c, const float *eps, float *a_out, float *b_out, int h, int w) {
    __shared__ float tg[S1TH * S1TW], tc[S1TH * S1TW];
    int x0 = blockIdx.x * S1BX, y0 = blockIdx.y * S1BY;
    s1_load(g, tg, h, w, x0, y0); s1_load(c, tc, h, w, x0, y0);
    __syncthreads();
    int x = x0 + threadIdx.x, y = y0 + threadIdx.y;
    if (x >= w || y >= h) return;
    float sg = 0, sc = 0, sgc = 0, sgg = 0;
    for (int dy = 0; dy < 9; ++dy)
        for (int dx = 0; dx < 9; ++dx) {
            int j = (threadIdx.y + dy) * S1TW + threadIdx.x + dx;
            float gv = tg[j], cv = tc[j];
            sg += gv; sc += cv; sgc += gv * cv; sgg += gv * gv;
        }
    float mg = sg / 81.0f, mc = sc / 81.0f;
    float cov = sgc / 81.0f - mg * mc, var = sgg / 81.0f - mg * mg;
    float a = cov / fmaxf(var + eps[0], 1e-30f);
    a_out[y * w + x] = a;
    b_out[y * w + x] = mc - a * mg;
}
__global__ void s1_q(const float *a, const float *b, const float *g, const float *c, const float *thr,
                     float *out, int h, int w) {
    __shared__ float ta[S1TH * S1TW], tb[S1TH * S1TW];
    int x0 = blockIdx.x * S1BX, y0 = blockIdx.y * S1BY;
    s1_load(a, ta, h, w, x0, y0); s1_load(b, tb, h, w, x0, y0);
    __syncthreads();
    int x = x0 + threadIdx.x, y = y0 + threadIdx.y;
    if (x >= w || y >= h) return;
    float sa = 0, sb = 0;
    for (int dy = 0; dy < 9; ++dy)
        for (int dx = 0; dx < 9; ++dx) {
            int j = (threadIdx.y + dy) * S1TW + threadIdx.x + dx;
            sa += ta[j]; sb += tb[j];
        }
    float q = (sa / 81.0f) * g[y * w + x] + sb / 81.0f;
    float cv = c[y * w + x];
    out[y * w + x] = fabsf(q - cv) > thr[0] ? cv : q;
}

EXPORT int dl_t2_extract_c(const float *img, const long long *shift, const float *win, float2 *out,
                           int h, int w, int oy, int ox, int tiles, int nx, int device, void *stream) {
    cudaError_t status = cudaSuccess; // Native: preserve the caller-owned NVEnc CUDA context.
    if (status != cudaSuccess) return (int)status;
    t2_extract_c<<<(tiles * 256 + 255) / 256, 256, 0, (cudaStream_t)stream>>>
        (img, shift, win, out, h, w, oy, ox, tiles, nx);
    return (int)cudaGetLastError();
}
EXPORT int dl_t2_merge_multi(const float2 *ref, const float *var, const MergeArgs *args, float2 *M, float *n,
                             int count, float noise_factor, float inv_cnt, int device, void *stream) {
    cudaError_t status = cudaSuccess; // Native: preserve the caller-owned NVEnc CUDA context.
    if (status != cudaSuccess) return (int)status;
    t2_merge_multi<<<(count + 255) / 256, 256, 0, (cudaStream_t)stream>>>(ref, var, *args, M, n, count, noise_factor, inv_cnt);
    return (int)cudaGetLastError();
}
EXPORT int dl_t2_wiener_centered(const float2 *M, const float2 *P, const float *var, const float *n,
                                 const float2 *basis, float2 *out, int count, int device, void *stream) {
    cudaError_t status = cudaSuccess; // Native: preserve the caller-owned NVEnc CUDA context.
    if (status != cudaSuccess) return (int)status;
    t2_wiener_centered<<<(count + 255) / 256, 256, 0, (cudaStream_t)stream>>>(M, P, var, n, basis, out, count);
    return (int)cudaGetLastError();
}
EXPORT int dl_coarse_gather(const float *grid, float *out, int h, int w, int oy, int ox, int ny, int nx,
                            int hc, int wc, int left, int device, void *stream) {
    cudaError_t status = cudaSuccess; // Native: preserve the caller-owned NVEnc CUDA context.
    if (status != cudaSuccess) return (int)status;
    coarse_gather<<<(2 * hc * wc + 255) / 256, 256, 0, (cudaStream_t)stream>>>(grid, out, h, w, oy, ox, ny, nx, hc, wc, left);
    return (int)cudaGetLastError();
}
EXPORT int dl_s1_guided(const float *g, const float *c, const float *eps, const float *thr, float *a, float *b,
                        float *out, int h, int w, int device, void *stream) {
    cudaError_t status = cudaSuccess; // Native: preserve the caller-owned NVEnc CUDA context.
    if (status != cudaSuccess) return (int)status;
    dim3 block(S1BX, S1BY), grid((w + S1BX - 1) / S1BX, (h + S1BY - 1) / S1BY);
    s1_ab<<<grid, block, 0, (cudaStream_t)stream>>>(g, c, eps, a, b, h, w);
    s1_q<<<grid, block, 0, (cudaStream_t)stream>>>(a, b, g, c, thr, out, h, w);
    return (int)cudaGetLastError();
}

// ---- fused_reconstruction_fp32 (stage 2, FP32-rounding class) ----
// Coarse centered-alignment merge over up to 16 neighbours: background re-anchoring, torch phase tensors, merge, 1/cnt.
// Differs from the eager chain only where torch used FMA contraction / complex division (rounding, ~1 ulp).
struct CoarseArgs { const float2 *fz[16]; int k; };
__global__ void coarse_merge_multi(const float2 *ref, const float2 *ph, long long stride, CoarseArgs a, const float2 *ws,
                                   float w00, const float *sigD2, float c, float2 *M, float *n, int count, float inv_cnt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int bin = i % 256, t0 = i - bin;
    float2 r = ref[i], acc = r, wb = ws[bin];
    float nn = 1.0f, sd = c * sigD2[i];
    for (int k = 0; k < a.k; ++k) {
        float2 z = a.fz[k][i], z0 = a.fz[k][t0], p = ph[k * stride + i];
        float cx = z0.x / w00, cy = z0.y / w00;
        float bx = cx * wb.x - cy * wb.y, by = cx * wb.y + cy * wb.x;
        float ex = z.x - bx, ey = z.y - by;
        z = make_float2(ex * p.x - ey * p.y + bx, ex * p.y + ey * p.x + by);
        float dx = z.x - r.x, dy = z.y - r.y, d2 = dx * dx + dy * dy;
        float al = d2 / fmaxf(d2 + sd, 1e-30f);
        acc = make_float2(acc.x + z.x + al * (r.x - z.x), acc.y + z.y + al * (r.y - z.y));
        nn += 1.0f - al;
    }
    M[i] = make_float2(acc.x * inv_cnt, acc.y * inv_cnt);
    n[i] = nn;
}
EXPORT int dl_coarse_merge_multi(const float2 *ref, const float2 *ph, long long stride, const CoarseArgs *args, const float2 *ws,
                                 float w00, const float *sigD2, float c, float2 *M, float *n, int count, float inv_cnt,
                                 int device, void *stream) {
    cudaError_t status = cudaSuccess; // Native: preserve the caller-owned NVEnc CUDA context.
    if (status != cudaSuccess) return (int)status;
    coarse_merge_multi<<<(count + 255) / 256, 256, 0, (cudaStream_t)stream>>>(ref, ph, stride, *args, ws, w00, sigD2, c, M, n, count, inv_cnt);
    return (int)cudaGetLastError();
}

#include "t2_j4_kernels.cuh"

#include "t2_coarse_kernels.cuh"
