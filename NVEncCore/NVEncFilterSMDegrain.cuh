// -----------------------------------------------------------------------------------------
// NVEnc by rigaya (SMDegrain port, MIT)
// -----------------------------------------------------------------------------------------
// nvenc-smdegrain port — Phase 5 kernel header.
//
// Contains the validated standalone kernels from tools/ (ME full-search, pyramid
// downsample, pyramid refine, motion compensation, bounded temporal blend), now
// merged into the NVEncCore tree for filter integration.
//
// Clean-room reimplementation: algorithm is the classic block-matching ME + bounded
// temporal average from the SMDegrain public description. No GPL source copied.
// -----------------------------------------------------------------------------------------

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace smdegrain {

// Per-block motion-vector result.
struct MVBlock {
    int16_t mvx;   // x offset (cur -> ref), signed
    int16_t mvy;   // y offset
    int32_t sad;   // sum of absolute differences at best (mvx,mvy)
};

// ------------------------------------------------------------------------------------------
// Block-matching ME (Phase 4a — full search MVP)
// ------------------------------------------------------------------------------------------
template<int BLOCK_SIZE, typename T>
__global__ void kernel_me_fullsearch(
    const T* __restrict__ ref,
    const T* __restrict__ cur,
    const int width, const int height, const int pitch_pixels,
    const int search_radius,
    MVBlock* __restrict__ out_blocks
) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int x0 = bx * BLOCK_SIZE;
    const int y0 = by * BLOCK_SIZE;
    const int blocks_x = gridDim.x;

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int cand_side = 2 * search_radius + 1;
    const int num_candidates = cand_side * cand_side;

    int32_t best_sad = INT32_MAX;
    int16_t best_mvx = 0, best_mvy = 0;

    for (int c = tid; c < num_candidates; c += num_threads) {
        const int dx = (c % cand_side) - search_radius;
        const int dy = (c / cand_side) - search_radius;
        if (x0 + dx < 0 || x0 + dx + BLOCK_SIZE > width ||
            y0 + dy < 0 || y0 + dy + BLOCK_SIZE > height) {
            continue;
        }
        int32_t sad = 0;
        #pragma unroll
        for (int j = 0; j < BLOCK_SIZE; j++) {
            #pragma unroll
            for (int i = 0; i < BLOCK_SIZE; i++) {
                const int cur_px = (int)cur[(y0 + j) * pitch_pixels + (x0 + i)];
                const int ref_px = (int)ref[(y0 + dy + j) * pitch_pixels + (x0 + dx + i)];
                sad += abs(cur_px - ref_px);
            }
        }
        if (sad < best_sad ||
            (sad == best_sad && (dx * dx + dy * dy) < (int)(best_mvx * best_mvx + best_mvy * best_mvy))) {
            best_sad = sad;
            best_mvx = (int16_t)dx;
            best_mvy = (int16_t)dy;
        }
    }

    __shared__ int32_t s_sad[1024];
    __shared__ int16_t s_mvx[1024];
    __shared__ int16_t s_mvy[1024];
    s_sad[tid] = best_sad; s_mvx[tid] = best_mvx; s_mvy[tid] = best_mvy;
    __syncthreads();

    for (int s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            const int32_t a_sad = s_sad[tid];
            const int32_t b_sad = s_sad[tid + s];
            const int16_t a_mvx = s_mvx[tid], a_mvy = s_mvy[tid];
            const int16_t b_mvx = s_mvx[tid + s], b_mvy = s_mvy[tid + s];
            const int a_mag = (int)a_mvx * a_mvx + (int)a_mvy * a_mvy;
            const int b_mag = (int)b_mvx * b_mvx + (int)b_mvy * b_mvy;
            if (b_sad < a_sad || (b_sad == a_sad && b_mag < a_mag)) {
                s_sad[tid] = b_sad; s_mvx[tid] = b_mvx; s_mvy[tid] = b_mvy;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        MVBlock& out = out_blocks[by * blocks_x + bx];
        out.sad = s_sad[0]; out.mvx = s_mvx[0]; out.mvy = s_mvy[0];
    }
}

template<int BLOCK_SIZE, typename T>
static inline cudaError_t launch_me_fullsearch(
    const T* d_ref, const T* d_cur,
    int width, int height, int pitch_pixels,
    int search_radius,
    MVBlock* d_out_blocks,
    cudaStream_t stream = 0
) {
    const int blocks_x = width / BLOCK_SIZE;
    const int blocks_y = height / BLOCK_SIZE;
    const dim3 grid(blocks_x, blocks_y, 1);
    const dim3 block(256, 1, 1);
    kernel_me_fullsearch<BLOCK_SIZE, T><<<grid, block, 0, stream>>>(
        d_ref, d_cur, width, height, pitch_pixels, search_radius, d_out_blocks);
    return cudaGetLastError();
}

// ------------------------------------------------------------------------------------------
// Pyramid support (Phase 4b)
// ------------------------------------------------------------------------------------------
template<typename T>
__global__ void kernel_downsample_2x(
    const T* __restrict__ src, int src_w, int src_h, int src_pitch,
    T* __restrict__ dst, int dst_w, int dst_h, int dst_pitch
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;
    const int sx = x * 2;
    const int sy = y * 2;
    const int s00 = (int)src[ sy      * src_pitch + sx    ];
    const int s10 = (int)src[ sy      * src_pitch + sx + 1];
    const int s01 = (int)src[(sy + 1) * src_pitch + sx    ];
    const int s11 = (int)src[(sy + 1) * src_pitch + sx + 1];
    dst[y * dst_pitch + x] = (T)((s00 + s10 + s01 + s11 + 2) >> 2);
}

template<typename T>
static inline cudaError_t launch_downsample_2x(
    const T* d_src, int src_w, int src_h, int src_pitch,
    T* d_dst, int dst_w, int dst_h, int dst_pitch,
    cudaStream_t stream = 0
) {
    const dim3 block(16, 16, 1);
    const dim3 grid((dst_w + 15) / 16, (dst_h + 15) / 16, 1);
    kernel_downsample_2x<T><<<grid, block, 0, stream>>>(
        d_src, src_w, src_h, src_pitch, d_dst, dst_w, dst_h, dst_pitch);
    return cudaGetLastError();
}

template<int BLOCK_SIZE, typename T>
__global__ void kernel_me_refine_around_hint(
    const T* __restrict__ ref,
    const T* __restrict__ cur,
    const int width, const int height, const int pitch_pixels,
    const int search_radius,
    const MVBlock* __restrict__ hints,
    const int coarse_blocks_x,
    MVBlock* __restrict__ out_blocks
) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int x0 = bx * BLOCK_SIZE;
    const int y0 = by * BLOCK_SIZE;
    const int blocks_x = gridDim.x;

    const int coarse_bx = bx / 2;
    const int coarse_by = by / 2;
    const MVBlock hint = hints[coarse_by * coarse_blocks_x + coarse_bx];
    const int hint_mvx = (int)hint.mvx * 2;
    const int hint_mvy = (int)hint.mvy * 2;

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int cand_side = 2 * search_radius + 1;
    const int num_candidates = cand_side * cand_side;

    int32_t best_sad = INT32_MAX;
    int16_t best_mvx = (int16_t)hint_mvx;
    int16_t best_mvy = (int16_t)hint_mvy;

    for (int c = tid; c < num_candidates; c += num_threads) {
        const int ddx = (c % cand_side) - search_radius;
        const int ddy = (c / cand_side) - search_radius;
        const int dx = hint_mvx + ddx;
        const int dy = hint_mvy + ddy;
        if (x0 + dx < 0 || x0 + dx + BLOCK_SIZE > width ||
            y0 + dy < 0 || y0 + dy + BLOCK_SIZE > height) {
            continue;
        }
        int32_t sad = 0;
        #pragma unroll
        for (int j = 0; j < BLOCK_SIZE; j++) {
            #pragma unroll
            for (int i = 0; i < BLOCK_SIZE; i++) {
                const int cur_px = (int)cur[(y0 + j) * pitch_pixels + (x0 + i)];
                const int ref_px = (int)ref[(y0 + dy + j) * pitch_pixels + (x0 + dx + i)];
                sad += abs(cur_px - ref_px);
            }
        }
        if (sad < best_sad ||
            (sad == best_sad && (dx * dx + dy * dy) < (int)(best_mvx * best_mvx + best_mvy * best_mvy))) {
            best_sad = sad;
            best_mvx = (int16_t)dx;
            best_mvy = (int16_t)dy;
        }
    }

    __shared__ int32_t s_sad[1024];
    __shared__ int16_t s_mvx[1024];
    __shared__ int16_t s_mvy[1024];
    s_sad[tid] = best_sad; s_mvx[tid] = best_mvx; s_mvy[tid] = best_mvy;
    __syncthreads();

    for (int s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            const int32_t a_sad = s_sad[tid];
            const int32_t b_sad = s_sad[tid + s];
            const int16_t a_mvx = s_mvx[tid], a_mvy = s_mvy[tid];
            const int16_t b_mvx = s_mvx[tid + s], b_mvy = s_mvy[tid + s];
            const int a_mag = (int)a_mvx * a_mvx + (int)a_mvy * a_mvy;
            const int b_mag = (int)b_mvx * b_mvx + (int)b_mvy * b_mvy;
            if (b_sad < a_sad || (b_sad == a_sad && b_mag < a_mag)) {
                s_sad[tid] = b_sad; s_mvx[tid] = b_mvx; s_mvy[tid] = b_mvy;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        MVBlock& out = out_blocks[by * blocks_x + bx];
        out.sad = s_sad[0]; out.mvx = s_mvx[0]; out.mvy = s_mvy[0];
    }
}

template<int BLOCK_SIZE, typename T>
static inline cudaError_t launch_me_refine_around_hint(
    const T* d_ref, const T* d_cur,
    int width, int height, int pitch_pixels,
    int search_radius,
    const MVBlock* d_hints, int coarse_blocks_x,
    MVBlock* d_out_blocks,
    cudaStream_t stream = 0
) {
    const int blocks_x = width / BLOCK_SIZE;
    const int blocks_y = height / BLOCK_SIZE;
    const dim3 grid(blocks_x, blocks_y, 1);
    const dim3 block(64, 1, 1);
    kernel_me_refine_around_hint<BLOCK_SIZE, T><<<grid, block, 0, stream>>>(
        d_ref, d_cur, width, height, pitch_pixels, search_radius,
        d_hints, coarse_blocks_x, d_out_blocks);
    return cudaGetLastError();
}

// ------------------------------------------------------------------------------------------
// Motion compensation + bounded temporal blend (Phase 5a-c)
// ------------------------------------------------------------------------------------------
template<int BLOCK_SIZE, typename T>
__global__ void kernel_motion_compensate(
    const T* __restrict__ ref,
    const MVBlock* __restrict__ mvs,
    const int width, const int height, const int pitch_pixels,
    const int mv_blocks_x,
    T* __restrict__ out
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int bx = x / BLOCK_SIZE;
    const int by = y / BLOCK_SIZE;
    const MVBlock mv = mvs[by * mv_blocks_x + bx];

    int sx = x + (int)mv.mvx;
    int sy = y + (int)mv.mvy;
    if (sx < 0) sx = 0;
    if (sx >= width) sx = width - 1;
    if (sy < 0) sy = 0;
    if (sy >= height) sy = height - 1;
    out[y * pitch_pixels + x] = ref[sy * pitch_pixels + sx];
}

template<int BLOCK_SIZE, typename T>
static inline cudaError_t launch_motion_compensate(
    const T* d_ref, const MVBlock* d_mvs,
    int width, int height, int pitch_pixels, int mv_blocks_x,
    T* d_out, cudaStream_t stream = 0
) {
    const dim3 block(16, 16, 1);
    const dim3 grid((width + 15) / 16, (height + 15) / 16, 1);
    kernel_motion_compensate<BLOCK_SIZE, T><<<grid, block, 0, stream>>>(
        d_ref, d_mvs, width, height, pitch_pixels, mv_blocks_x, d_out);
    return cudaGetLastError();
}

// N-ref bounded blend: out = cur + clamp(avg(mc_ref[0..N-1] + cur) - cur, -limit, +limit).
// For NREFS = 1..3 causal refs. Template-specialized per N for unrolled loops.
// Unused ref pointers (e.g. r2 when NREFS=1) may be nullptr — never dereferenced.
// N-ref blend with per-block thSAD gating: refs whose block SAD exceeds thSAD are
// excluded from the average for that block (per-pixel, since SAD is looked up per block).
// This is the mechanism that differentiates media_processor's heavy_cs (thSAD=500) from
// medium_cs (thSAD=300) — higher thSAD accepts more refs → more denoising in fast-motion
// regions where ME is less reliable.
template<typename T, int NREFS, int BLOCK_SIZE>
__global__ void kernel_temporal_blend_nref(
    const T* __restrict__ cur,
    const T* __restrict__ r0,
    const T* __restrict__ r1,
    const T* __restrict__ r2,
    const MVBlock* __restrict__ mv0,
    const MVBlock* __restrict__ mv1,
    const MVBlock* __restrict__ mv2,
    const int mv_blocks_x,
    const int width, const int height, const int pitch_pixels,
    const int thSAD,
    const int limit_scaled,
    const int pix_max,
    T* __restrict__ out
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int bx = x / BLOCK_SIZE;
    const int by = y / BLOCK_SIZE;
    const int block_idx = by * mv_blocks_x + bx;

    const int c = (int)cur[y * pitch_pixels + x];
    int sum = c;
    int count = 1;
    if constexpr (NREFS >= 1) {
        if (mv0[block_idx].sad <= thSAD) { sum += (int)r0[y * pitch_pixels + x]; count++; }
    }
    if constexpr (NREFS >= 2) {
        if (mv1[block_idx].sad <= thSAD) { sum += (int)r1[y * pitch_pixels + x]; count++; }
    }
    if constexpr (NREFS >= 3) {
        if (mv2[block_idx].sad <= thSAD) { sum += (int)r2[y * pitch_pixels + x]; count++; }
    }
    const int avg = (sum + count / 2) / count;

    int delta = avg - c;
    if (delta >  limit_scaled) delta =  limit_scaled;
    if (delta < -limit_scaled) delta = -limit_scaled;
    int o = c + delta;
    if (o < 0) o = 0;
    if (o > pix_max) o = pix_max;
    out[y * pitch_pixels + x] = (T)o;
}

template<typename T, int NREFS, int BLOCK_SIZE>
static inline cudaError_t launch_temporal_blend_nref(
    const T* d_cur,
    const T* d_r0, const T* d_r1, const T* d_r2,
    const MVBlock* d_mv0, const MVBlock* d_mv1, const MVBlock* d_mv2,
    int mv_blocks_x,
    int width, int height, int pitch_pixels,
    int thSAD, int limit_scaled, int pix_max,
    T* d_out, cudaStream_t stream = 0
) {
    const dim3 block(16, 16, 1);
    const dim3 grid((width + 15) / 16, (height + 15) / 16, 1);
    kernel_temporal_blend_nref<T, NREFS, BLOCK_SIZE><<<grid, block, 0, stream>>>(
        d_cur, d_r0, d_r1, d_r2, d_mv0, d_mv1, d_mv2, mv_blocks_x,
        width, height, pitch_pixels, thSAD, limit_scaled, pix_max, d_out);
    return cudaGetLastError();
}

// 2-frame bounded blend: out = cur + clamp((mc_ref - cur) / 2, -limit, +limit).
// `pix_max` is the top end of the pixel range (255 for 8-bit, 1023 for 10-bit, etc.)
// — passed as runtime int so the same kernel serves both bit depths.
// `limit` is specified by the user in 8-bit scale per media_processor convention
// (0..255). For 10-bit content we scale it up by 4 on the caller side.
template<typename T>
__global__ void kernel_temporal_blend_2frame(
    const T* __restrict__ cur,
    const T* __restrict__ mc_ref,
    const int width, const int height, const int pitch_pixels,
    const int limit_scaled,
    const int pix_max,
    T* __restrict__ out
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int c = (int)cur[y * pitch_pixels + x];
    const int r = (int)mc_ref[y * pitch_pixels + x];
    int delta = (r - c) / 2;
    if (delta >  limit_scaled) delta =  limit_scaled;
    if (delta < -limit_scaled) delta = -limit_scaled;
    int o = c + delta;
    if (o < 0) o = 0;
    if (o > pix_max) o = pix_max;
    out[y * pitch_pixels + x] = (T)o;
}

template<typename T>
static inline cudaError_t launch_temporal_blend_2frame(
    const T* d_cur, const T* d_mc_ref,
    int width, int height, int pitch_pixels,
    int limit_scaled, int pix_max,
    T* d_out, cudaStream_t stream = 0
) {
    const dim3 block(16, 16, 1);
    const dim3 grid((width + 15) / 16, (height + 15) / 16, 1);
    kernel_temporal_blend_2frame<T><<<grid, block, 0, stream>>>(
        d_cur, d_mc_ref, width, height, pitch_pixels, limit_scaled, pix_max, d_out);
    return cudaGetLastError();
}

// ------------------------------------------------------------------------------------------
// Contrasharp post-pass (Phase 6)
//
// MVTools' SMDegrain contrasharp is a "repair-clamped unsharp mask": take the degrained
// frame, compute its own high-frequency detail (frame - blur(frame)), add it back to
// recover lost sharpness, then clamp the result between [min(degrain,source), max(degrain,source)]
// so the sharpened pixel can't exceed what was already present in the noisy source. That
// prevents amplifying noise — you only get back detail that both exists in the source and
// survived the temporal blend.
//
// Two kernels:
//   kernel_blur_3x3<T>    — separable 3x3 box (cheap; the exact blur shape barely matters
//                           for a one-tap unsharp restore since the source-clamp dominates)
//   kernel_contrasharp<T> — out = clamp(degrain + (degrain - blurred), min(d,s), max(d,s))
// ------------------------------------------------------------------------------------------

template<typename T>
__global__ void kernel_blur_3x3(
    const T* __restrict__ src,
    const int width, const int height, const int pitch_pixels,
    T* __restrict__ dst
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int xm = (x > 0)           ? x - 1 : x;
    const int xp = (x < width  - 1)  ? x + 1 : x;
    const int ym = (y > 0)           ? y - 1 : y;
    const int yp = (y < height - 1)  ? y + 1 : y;

    int sum = 0;
    sum += (int)src[ym * pitch_pixels + xm];
    sum += (int)src[ym * pitch_pixels + x ];
    sum += (int)src[ym * pitch_pixels + xp];
    sum += (int)src[y  * pitch_pixels + xm];
    sum += (int)src[y  * pitch_pixels + x ];
    sum += (int)src[y  * pitch_pixels + xp];
    sum += (int)src[yp * pitch_pixels + xm];
    sum += (int)src[yp * pitch_pixels + x ];
    sum += (int)src[yp * pitch_pixels + xp];
    dst[y * pitch_pixels + x] = (T)((sum + 4) / 9);  // rounded mean
}

template<typename T>
static inline cudaError_t launch_blur_3x3(
    const T* d_src, int width, int height, int pitch_pixels,
    T* d_dst, cudaStream_t stream = 0
) {
    const dim3 block(16, 16, 1);
    const dim3 grid((width + 15) / 16, (height + 15) / 16, 1);
    kernel_blur_3x3<T><<<grid, block, 0, stream>>>(d_src, width, height, pitch_pixels, d_dst);
    return cudaGetLastError();
}

template<typename T>
__global__ void kernel_contrasharp(
    const T* __restrict__ degrain,
    const T* __restrict__ source,
    const T* __restrict__ blurred_degrain,
    const int width, const int height, const int pitch_pixels,
    const int pix_max,
    T* __restrict__ out
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int d = (int)degrain[y * pitch_pixels + x];
    const int s = (int)source [y * pitch_pixels + x];
    const int b = (int)blurred_degrain[y * pitch_pixels + x];
    // Unsharp-mask tentative: d + (d - b) = 2d - b
    int t = 2 * d - b;
    // Repair-clamp between degrain and source — MVTools' safety net so contrasharp can't
    // drag a pixel past the original noisy value (preventing noise amplification).
    const int lo = d < s ? d : s;
    const int hi = d > s ? d : s;
    if (t < lo) t = lo;
    if (t > hi) t = hi;
    if (t < 0) t = 0;
    if (t > pix_max) t = pix_max;
    out[y * pitch_pixels + x] = (T)t;
}

template<typename T>
static inline cudaError_t launch_contrasharp(
    const T* d_degrain, const T* d_source, const T* d_blurred,
    int width, int height, int pitch_pixels, int pix_max,
    T* d_out, cudaStream_t stream = 0
) {
    const dim3 block(16, 16, 1);
    const dim3 grid((width + 15) / 16, (height + 15) / 16, 1);
    kernel_contrasharp<T><<<grid, block, 0, stream>>>(
        d_degrain, d_source, d_blurred, width, height, pitch_pixels, pix_max, d_out);
    return cudaGetLastError();
}

} // namespace smdegrain
