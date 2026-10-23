// J4: fused main reconstruction with an in-kernel 16x16 complex FFT (FP32-envelope class).
// Included from temporal_kernels.cuh; uses its mirror_index and EXPORT.

// Radix-2 DIT, FP32. Twiddles: stage 0 (len 2) 1 value, stage 1 (len 4) 2, stage 2 (len 8) 4, stage 3 (len 16) 8.
__constant__ float2 T2_TW[15] = {
    {1.0f, 0.0f},
    {1.0f, 0.0f},
    {6.123234262925839e-17f, -1.0f},
    {1.0f, 0.0f},
    {0.7071067690849304f, -0.7071067690849304f},
    {6.123234262925839e-17f, -1.0f},
    {-0.7071067690849304f, -0.7071067690849304f},
    {1.0f, 0.0f},
    {0.9238795042037964f, -0.3826834261417389f},
    {0.7071067690849304f, -0.7071067690849304f},
    {0.3826834261417389f, -0.9238795042037964f},
    {6.123234262925839e-17f, -1.0f},
    {-0.3826834261417389f, -0.9238795042037964f},
    {-0.7071067690849304f, -0.7071067690849304f},
    {-0.9238795042037964f, -0.3826834261417389f}};

__constant__ int T2_TW_OFF[4] = {0, 1, 3, 7};

__device__ __forceinline__ int t2_bitrev4(int x) {
    return ((x & 1) << 3) | ((x & 2) << 1) | ((x & 4) >> 1) | ((x & 8) >> 3);
}

// One 16-point FFT per 16-lane group; lane l holds the bit-reversed input on entry and X[l] on exit.
__device__ __forceinline__ float2 t2_fft16_reg(float2 v, int l, const float2 *tw, int inverse) {
    #pragma unroll
    for (int stage = 0; stage < 4; ++stage) {
        int half = 1 << stage;
        float2 other = make_float2(__shfl_xor_sync(0xffffffffu, v.x, half, 16),
                                   __shfl_xor_sync(0xffffffffu, v.y, half, 16));
        float2 w = tw[T2_TW_OFF[stage] + (l & (half - 1))];
        if (inverse) w.y = -w.y;
        bool up = (l & half) != 0;
        float2 b = up ? v : other, a = up ? other : v;
        float tr = w.x * b.x - w.y * b.y;
        float ti = w.x * b.y + w.y * b.x;
        v.x = up ? a.x - tr : a.x + tr;
        v.y = up ? a.y - ti : a.y + ti;
    }
    return v;
}

// Separable 16x16 complex FFT of one tile per thread block (256 threads, one sample each).
// Row stride 17 (one padding column) keeps the transpose and bit-reversed accesses bank-conflict free.
// Entry: buf[y*17+x]. Exit: buf[k_y*17+k_x] with torch fft2 ordering and `scale` applied.
__device__ __forceinline__ void t2_fft16_shared(float2 *buf, float2 *tmp, const float2 *tw, int inverse, float scale) {
    int tid = threadIdx.x, row = tid >> 4, l = tid & 15;
    float2 v = t2_fft16_reg(buf[row * 17 + t2_bitrev4(l)], l, tw, inverse);
    tmp[l * 17 + row] = v;                     // transpose: rows of tmp are columns of buf
    __syncthreads();
    float2 w = t2_fft16_reg(tmp[row * 17 + t2_bitrev4(l)], l, tw, inverse);
    buf[l * 17 + row] = make_float2(w.x * scale, w.y * scale);
    __syncthreads();
}

__global__ void t2_fft16_kernel(const float2 *in, float2 *out, int count, int inverse) {
    __shared__ float2 buf[272], tmp[272], tw[15];
    int tile = blockIdx.x, tid = threadIdx.x, idx = (tid >> 4) * 17 + (tid & 15);
    if (tid < 15) tw[tid] = T2_TW[tid];
    buf[idx] = in[tile * 256 + tid];
    __syncthreads();
    t2_fft16_shared(buf, tmp, tw, inverse, inverse ? 1.0f / 256.0f : 1.0f);
    out[tile * 256 + tid] = buf[idx];
}

struct MainArgs { const float *img[16]; const long long *shift[16]; const float *frac[16]; int k; };

// Same sample selection as t2_extract_c: clamp under an integer shift, mirror index otherwise.
__device__ __forceinline__ void t2_gather_window(const float *img, const long long *sh, const float *win,
                                                 int tile, int tid, int h, int w, int oy, int ox, int nx, float2 *buf) {
    int y = (tile / nx) * 8 - 16 - oy + (tid >> 4);
    int x = (tile % nx) * 8 - 16 - ox + (tid & 15);
    if (sh) {
        long long dx = sh[2 * tile], dy = sh[2 * tile + 1];
        x = dx > (long long)w - 1 - x ? w - 1 : dx < -(long long)x ? 0 : x + (int)dx;
        y = dy > (long long)h - 1 - y ? h - 1 : dy < -(long long)y ? 0 : y + (int)dy;
    } else {
        x = mirror_index(x, w); y = mirror_index(y, h);
    }
    buf[(tid >> 4) * 17 + (tid & 15)] = make_float2(img[y * w + x] * win[tid], 0.0f);
}

// Kernel A: per tile, gather + window + FFT the reference and every aligned neighbour, merge in the
// t2_merge_multi order, write M (optional), n and the inverse-transformed real windowed tile.
__global__ void t2_main_merge(const float *ref, const float *win, const float *var, MainArgs a,
                              float noise_factor, float inv_cnt, float cnt_f, int divide,
                              const float2 *ws, int centered, float w00,
                              float2 *M, float *n, float *out, int h, int w, int oy, int ox, int nx) {
    // Opt-in K4 mode retains the exact window DC without copying a scalar to the host.
    if (centered == 2) w00 = ws[0].x;
    __shared__ float2 buf[272], tmp[272], tw[15];
    int tile = blockIdx.x, tid = threadIdx.x, i = tile * 256 + tid, idx = (tid >> 4) * 17 + (tid & 15);
    if (tid < 15) tw[tid] = T2_TW[tid];
    t2_gather_window(ref, nullptr, win, tile, tid, h, w, oy, ox, nx, buf);
    __syncthreads();
    t2_fft16_shared(buf, tmp, tw, 0, 1.0f);
    float2 r = buf[idx], acc = r;
    float nn = 1.0f;
    int fx = tid & 15, fy = tid >> 4;
    if (fx >= 8) fx -= 16;
    if (fy >= 8) fy -= 16;
    for (int k = 0; k < a.k; ++k) {
        if (k > 0) __syncthreads(); // Every thread has consumed the previous DC bin.
        t2_gather_window(a.img[k], a.shift[k], win, tile, tid, h, w, oy, ox, nx, buf);
        __syncthreads();
        t2_fft16_shared(buf, tmp, tw, 0, 1.0f);
        float2 z = buf[idx], z0 = buf[0];
        float angle = (2.0f * 3.14159265358979323846f / 16) *
                      (fx * a.frac[k][2 * tile] + fy * a.frac[k][2 * tile + 1]);
        float s, c; __sincosf(angle, &s, &c);
        if (centered) {
            // t2_coarse centered_alignment: anchor the Hann-shaped brightness, ramp the detail only.
            float cx = z0.x / w00, cy = z0.y / w00;
            float2 wb = ws[tid];
            float bx = cx * wb.x - cy * wb.y, by = cx * wb.y + cy * wb.x;
            float ex = z.x - bx, ey = z.y - by;
            z = make_float2(ex * c - ey * s + bx, ex * s + ey * c + by);
        } else {
            z = make_float2(z.x * c - z.y * s, z.x * s + z.y * c);
        }
        float dx = z.x - r.x, dy = z.y - r.y, d2 = dx * dx + dy * dy;
        float al = d2 / fmaxf(d2 + noise_factor * var[i], 1e-30f);
        acc = make_float2(acc.x + z.x + al * (r.x - z.x), acc.y + z.y + al * (r.y - z.y));
        nn += 1.0f - al;
    }
    float2 m = divide ? make_float2(acc.x / cnt_f, acc.y / cnt_f)
                      : make_float2(acc.x * inv_cnt, acc.y * inv_cnt);
    if (M) M[i] = m;
    n[i] = nn;
    __syncthreads(); // Preserve the shared DC bin until all readers finish.
    buf[idx] = m;
    __syncthreads();
    t2_fft16_shared(buf, tmp, tw, 1, 1.0f / 256.0f);
    out[i] = buf[idx].x;
}

// Kernel B: re-tile the pilot, FFT, apply the centered Wiener of t2_wiener_centered, inverse FFT.
__global__ void t2_main_wiener(const float2 *M, const float *var, const float *n, const float2 *basis,
                               const float *pilot, const float *win, float *out,
                               int h, int w, int oy, int ox, int nx) {
    __shared__ float2 buf[272], tmp[272], tw[15];
    int tile = blockIdx.x, tid = threadIdx.x, i = tile * 256 + tid, idx = (tid >> 4) * 17 + (tid & 15);
    if (tid < 15) tw[tid] = T2_TW[tid];
    t2_gather_window(pilot, nullptr, win, tile, tid, h, w, oy, ox, nx, buf);
    __syncthreads();
    t2_fft16_shared(buf, tmp, tw, 0, 1.0f);
    float2 p = buf[idx], b = basis[tid];
    float m00 = M[tile * 256].x, p00 = buf[0].x;
    float bgx = m00 * b.x, bgy = m00 * b.y;
    float pbx = p00 * b.x, pby = p00 * b.y;
    float dx = p.x - pbx, dy = p.y - pby;
    float hh = hypotf(dx, dy), power = hh * hh;
    float g = power / fmaxf(power + var[i] / n[i], 1e-30f);
    float2 m = M[i];
    float ex = m.x - bgx, ey = m.y - bgy;
    __syncthreads(); // Preserve the pilot DC bin until all readers finish.
    buf[idx] = make_float2(bgx + ex * g, bgy + ey * g);
    __syncthreads();
    t2_fft16_shared(buf, tmp, tw, 1, 1.0f / 256.0f);
    out[i] = buf[idx].x;
}

// Deterministic overlap-add of real windowed tiles: same gather as t2_fold.
__global__ void t2_fold_real(const float *tiles, float *out, int h, int w, int oy, int ox, int nx) {
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
            value += tiles[tile * 256 + bin];
        }
    }
    out[i] = value;
}

EXPORT int dl_t2_fft16(const float2 *in, float2 *out, int count, int inverse, int device, void *stream) {
    cudaError_t status = cudaSuccess; // Native: preserve the caller-owned NVEnc CUDA context.
    if (status != cudaSuccess) return (int)status;
    t2_fft16_kernel<<<count, 256, 0, (cudaStream_t)stream>>>(in, out, count, inverse);
    return (int)cudaGetLastError();
}

EXPORT int dl_t2_main_merge(const float *ref, const float *win, const float *var, const MainArgs *args,
                            float noise_factor, float inv_cnt, float cnt_f, int divide,
                            const float2 *ws, int centered, float w00, float2 *M, float *n, float *out,
                            int tiles, int h, int w, int oy, int ox, int nx, int device, void *stream) {
    cudaError_t status = cudaSuccess; // Native: preserve the caller-owned NVEnc CUDA context.
    if (status != cudaSuccess) return (int)status;
    t2_main_merge<<<tiles, 256, 0, (cudaStream_t)stream>>>(ref, win, var, *args, noise_factor, inv_cnt, cnt_f,
                                                           divide, ws, centered, w00, M, n, out, h, w, oy, ox, nx);
    return (int)cudaGetLastError();
}

EXPORT int dl_t2_main_wiener(const float2 *M, const float *var, const float *n, const float2 *basis,
                             const float *pilot, const float *win, float *out, int tiles,
                             int h, int w, int oy, int ox, int nx, int device, void *stream) {
    cudaError_t status = cudaSuccess; // Native: preserve the caller-owned NVEnc CUDA context.
    if (status != cudaSuccess) return (int)status;
    t2_main_wiener<<<tiles, 256, 0, (cudaStream_t)stream>>>(M, var, n, basis, pilot, win, out, h, w, oy, ox, nx);
    return (int)cudaGetLastError();
}

EXPORT int dl_t2_fold_real(const float *tiles, float *out, int h, int w, int oy, int ox, int nx,
                           int device, void *stream) {
    cudaError_t status = cudaSuccess; // Native: preserve the caller-owned NVEnc CUDA context.
    if (status != cudaSuccess) return (int)status;
    t2_fold_real<<<(h * w + 255) / 256, 256, 0, (cudaStream_t)stream>>>(tiles, out, h, w, oy, ox, nx);
    return (int)cudaGetLastError();
}

// Additive capability marker: old DLLs must retain the scalar-host fallback.
EXPORT int dl_t2_main_device_dc() { return 1; }
