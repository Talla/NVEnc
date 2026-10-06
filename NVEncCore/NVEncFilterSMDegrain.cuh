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
    const int coarse_blocks_y,
    MVBlock* __restrict__ out_blocks
) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int x0 = bx * BLOCK_SIZE;
    const int y0 = by * BLOCK_SIZE;
    const int blocks_x = gridDim.x;

    // Clamp: when (height/2) is not a multiple of BLOCK_SIZE (1080 lines: 540/8 = 67.5) the last
    // full-resolution block row maps to a coarse row that does not exist.
    const int coarse_bx = min(bx / 2, coarse_blocks_x - 1);
    const int coarse_by = min(by / 2, coarse_blocks_y - 1);
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
    const MVBlock* d_hints, int coarse_blocks_x, int coarse_blocks_y,
    MVBlock* d_out_blocks,
    cudaStream_t stream = 0
) {
    const int blocks_x = width / BLOCK_SIZE;
    const int blocks_y = height / BLOCK_SIZE;
    const dim3 grid(blocks_x, blocks_y, 1);
    const dim3 block(64, 1, 1);
    kernel_me_refine_around_hint<BLOCK_SIZE, T><<<grid, block, 0, stream>>>(
        d_ref, d_cur, width, height, pitch_pixels, search_radius,
        d_hints, coarse_blocks_x, coarse_blocks_y, d_out_blocks);
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

// N-ref bounded blend with per-block thSAD gating. Refs whose block SAD exceeds thSAD
// are excluded from the average for that block.
//
// Phase 5g/5h supported NREFS=1..3 (causal-only, up to tr=3 past refs).
// Phase 5f extends to NREFS=1..6 to accommodate bidirectional mode:
//   tr=1 bidir → 2 refs (1 past + 1 future)
//   tr=2 bidir → 4 refs
//   tr=3 bidir → 6 refs
// Also covers causal-only edge cases at clip boundaries where full bidir window isn't
// available (uses whatever refs exist, 0..2*tr).
//
// Unused ref pointers may be nullptr — the if-constexpr gates ensure no dereference.
template<typename T, int NREFS, int BLOCK_SIZE>
__global__ void kernel_temporal_blend_nref(
    const T* __restrict__ cur,
    const T* __restrict__ r0, const T* __restrict__ r1, const T* __restrict__ r2,
    const T* __restrict__ r3, const T* __restrict__ r4, const T* __restrict__ r5,
    const MVBlock* __restrict__ mv0, const MVBlock* __restrict__ mv1, const MVBlock* __restrict__ mv2,
    const MVBlock* __restrict__ mv3, const MVBlock* __restrict__ mv4, const MVBlock* __restrict__ mv5,
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
    const int px_off = y * pitch_pixels + x;

    const int c = (int)cur[px_off];
    int sum = c;
    int count = 1;
    if constexpr (NREFS >= 1) {
        if (mv0[block_idx].sad <= thSAD) { sum += (int)r0[px_off]; count++; }
    }
    if constexpr (NREFS >= 2) {
        if (mv1[block_idx].sad <= thSAD) { sum += (int)r1[px_off]; count++; }
    }
    if constexpr (NREFS >= 3) {
        if (mv2[block_idx].sad <= thSAD) { sum += (int)r2[px_off]; count++; }
    }
    if constexpr (NREFS >= 4) {
        if (mv3[block_idx].sad <= thSAD) { sum += (int)r3[px_off]; count++; }
    }
    if constexpr (NREFS >= 5) {
        if (mv4[block_idx].sad <= thSAD) { sum += (int)r4[px_off]; count++; }
    }
    if constexpr (NREFS >= 6) {
        if (mv5[block_idx].sad <= thSAD) { sum += (int)r5[px_off]; count++; }
    }
    const int avg = (sum + count / 2) / count;

    int delta = avg - c;
    if (delta >  limit_scaled) delta =  limit_scaled;
    if (delta < -limit_scaled) delta = -limit_scaled;
    int o = c + delta;
    if (o < 0) o = 0;
    if (o > pix_max) o = pix_max;
    out[px_off] = (T)o;
}

template<typename T, int NREFS, int BLOCK_SIZE>
static inline cudaError_t launch_temporal_blend_nref(
    const T* d_cur,
    const T* d_r0, const T* d_r1, const T* d_r2,
    const T* d_r3, const T* d_r4, const T* d_r5,
    const MVBlock* d_mv0, const MVBlock* d_mv1, const MVBlock* d_mv2,
    const MVBlock* d_mv3, const MVBlock* d_mv4, const MVBlock* d_mv5,
    int mv_blocks_x,
    int width, int height, int pitch_pixels,
    int thSAD, int limit_scaled, int pix_max,
    T* d_out, cudaStream_t stream = 0
) {
    const dim3 block(16, 16, 1);
    const dim3 grid((width + 15) / 16, (height + 15) / 16, 1);
    kernel_temporal_blend_nref<T, NREFS, BLOCK_SIZE><<<grid, block, 0, stream>>>(
        d_cur,
        d_r0, d_r1, d_r2, d_r3, d_r4, d_r5,
        d_mv0, d_mv1, d_mv2, d_mv3, d_mv4, d_mv5,
        mv_blocks_x, width, height, pitch_pixels, thSAD, limit_scaled, pix_max, d_out);
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


// ==========================================================================================
// Phase 4 quick wins (denoise-lab project, project-plan.md Phase 4). All opt-in; with the
// defaults the filter uses the kernels above and its output is unchanged.
//
//   * sub-pel motion (half / quarter pel) with a 6-tap Lanczos-3 interpolator      (W8)
//   * chroma motion compensation with the luma vectors                             (W3)
//   * per-block thSAD scaled by the noise level at the block's brightness          (W6)
//   * SAD-weighted blend instead of binary accept/reject (MVTools weights)         (W11)
//   * scene-change detection (thSCD1 / thSCD2, MVTools semantics)
//
// Motion vectors keep the MVBlock layout. With subpel > 1 they are stored in 1/subpel luma px.
// ==========================================================================================

__device__ __forceinline__ float lanczos3(float t) {
    t = fabsf(t);
    if (t < 1e-6f) return 1.0f;
    if (t >= 3.0f) return 0.0f;
    const float pt = 3.14159265358979f * t;
    return 3.0f * __sinf(pt) * __sinf(pt * (1.0f / 3.0f)) / (pt * pt);
}

// Taps at integer offsets -2..+3 around floor(x), fx = x - floor(x) in [0,1). Normalised to sum 1.
__device__ __forceinline__ void lanczos6_weights(const float fx, float w[6]) {
    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < 6; k++) {
        w[k] = lanczos3((float)(k - 2) - fx);
        sum += w[k];
    }
    const float inv = 1.0f / sum;
    #pragma unroll
    for (int k = 0; k < 6; k++) w[k] *= inv;
}

// Sample a plane at a fractional position (edge-clamped). Exact copy at integer positions.
template<typename T>
__device__ __forceinline__ float sample_lanczos6(const T* __restrict__ src, const int width, const int height,
                                                 const int pitch_pixels, const float x, const float y) {
    const float flx = floorf(x), fly = floorf(y);
    const int ix = (int)flx, iy = (int)fly;
    const float fx = x - flx, fy = y - fly;
    if (fx == 0.0f && fy == 0.0f) {
        const int cx = min(max(ix, 0), width - 1);
        const int cy = min(max(iy, 0), height - 1);
        return (float)src[cy * pitch_pixels + cx];
    }
    float wx[6], wy[6];
    lanczos6_weights(fx, wx);
    lanczos6_weights(fy, wy);
    float acc = 0.0f;
    #pragma unroll
    for (int j = 0; j < 6; j++) {
        const int yy = min(max(iy - 2 + j, 0), height - 1);
        const T* row = src + yy * pitch_pixels;
        float r = 0.0f;
        #pragma unroll
        for (int i = 0; i < 6; i++) {
            const int xx = min(max(ix - 2 + i, 0), width - 1);
            r += wx[i] * (float)row[xx];
        }
        acc += wy[j] * r;
    }
    return acc;
}

// Block-wide float sum for blockDim.x == BLOCK_SIZE*BLOCK_SIZE (64 for 8x8: two warps).
template<int NTHREADS>
__device__ __forceinline__ float block_sum(float v, float* s_warp) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) s_warp[warp] = v;
    __syncthreads();
    float total = 0.0f;
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < (NTHREADS + 31) / 32; i++) total += s_warp[i];
        s_warp[0] = total;
    }
    __syncthreads();
    total = s_warp[0];
    __syncthreads();
    return total;
}

// Refine integer vectors (px) to 1/subpel px around the integer result: a half-pel ring, then
// (subpel == 4) a quarter-pel ring. In-place; output vectors are in 1/subpel px units.
template<int BLOCK_SIZE, typename T>
__global__ void kernel_me_subpel(
    const T* __restrict__ ref, const T* __restrict__ cur,
    const int width, const int height, const int pitch_pixels,
    const int subpel, MVBlock* __restrict__ mvs
) {
    constexpr int NT = BLOCK_SIZE * BLOCK_SIZE;
    __shared__ float s_warp[(NT + 31) / 32];
    const int bx = blockIdx.x, by = blockIdx.y;
    const int blocks_x = gridDim.x;
    const int tx = threadIdx.x % BLOCK_SIZE, ty = threadIdx.x / BLOCK_SIZE;
    const int x = bx * BLOCK_SIZE + tx, y = by * BLOCK_SIZE + ty;
    MVBlock mv = mvs[by * blocks_x + bx];
    int best_x = mv.mvx * subpel, best_y = mv.mvy * subpel;
    const float c = (float)cur[min(y, height - 1) * pitch_pixels + min(x, width - 1)];
    const float inv = 1.0f / (float)subpel;
    float best = block_sum<NT>(fabsf(c - sample_lanczos6(ref, width, height, pitch_pixels,
                                                           x + best_x * inv, y + best_y * inv)), s_warp);
    for (int step = subpel / 2; step >= 1; step /= 2) {
        const int cx = best_x, cy = best_y;
        for (int k = 0; k < 9; k++) {
            if (k == 4) continue;
            const int dx = (k % 3 - 1) * step, dy = (k / 3 - 1) * step;
            const float sad = block_sum<NT>(fabsf(c - sample_lanczos6(ref, width, height, pitch_pixels,
                                                                        x + (cx + dx) * inv, y + (cy + dy) * inv)), s_warp);
            if (sad < best) { best = sad; best_x = cx + dx; best_y = cy + dy; }
        }
    }
    if (threadIdx.x == 0) {
        mv.mvx = (int16_t)best_x;
        mv.mvy = (int16_t)best_y;
        mv.sad = (int32_t)(best + 0.5f);
        mvs[by * blocks_x + bx] = mv;
    }
}

template<int BLOCK_SIZE, typename T>
static inline cudaError_t launch_me_subpel(const T* d_ref, const T* d_cur, int width, int height, int pitch_pixels,
                                           int subpel, MVBlock* d_mvs, cudaStream_t stream = 0) {
    const dim3 grid(width / BLOCK_SIZE, height / BLOCK_SIZE, 1);
    kernel_me_subpel<BLOCK_SIZE, T><<<grid, BLOCK_SIZE * BLOCK_SIZE, 0, stream>>>(
        d_ref, d_cur, width, height, pitch_pixels, subpel, d_mvs);
    return cudaGetLastError();
}

// Luma MC at 1/subpel precision (subpel 1 still works: exact copies).
template<int BLOCK_SIZE, typename T>
__global__ void kernel_motion_compensate_subpel(
    const T* __restrict__ ref, const MVBlock* __restrict__ mvs,
    const int width, const int height, const int pitch_pixels,
    const int mv_blocks_x, const int mv_blocks_y, const int subpel, const int pix_max,
    T* __restrict__ out
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int bx = min(x / BLOCK_SIZE, mv_blocks_x - 1);
    const int by = min(y / BLOCK_SIZE, mv_blocks_y - 1);
    const MVBlock mv = mvs[by * mv_blocks_x + bx];
    const float inv = 1.0f / (float)subpel;
    const float v = sample_lanczos6(ref, width, height, pitch_pixels, x + mv.mvx * inv, y + mv.mvy * inv);
    out[y * pitch_pixels + x] = (T)min(max((int)(v + 0.5f), 0), pix_max);
}

template<int BLOCK_SIZE, typename T>
static inline cudaError_t launch_motion_compensate_subpel(const T* d_ref, const MVBlock* d_mvs, int width, int height,
    int pitch_pixels, int mv_blocks_x, int mv_blocks_y, int subpel, int pix_max, T* d_out, cudaStream_t stream = 0) {
    const dim3 block(16, 16, 1);
    const dim3 grid((width + 15) / 16, (height + 15) / 16, 1);
    kernel_motion_compensate_subpel<BLOCK_SIZE, T><<<grid, block, 0, stream>>>(
        d_ref, d_mvs, width, height, pitch_pixels, mv_blocks_x, mv_blocks_y, subpel, pix_max, d_out);
    return cudaGetLastError();
}

// Chroma MC with the luma vectors. (sx, sy) = chroma subsampling (2,2) 4:2:0, (2,1) 4:2:2, (1,1) 4:4:4.
// A luma vector of v/subpel px is v/(subpel*s) chroma px, so odd luma vectors land between chroma
// samples and are interpolated. gridDim.z selects U (0) or V (1).
template<int BLOCK_SIZE, typename T>
__global__ void kernel_motion_compensate_chroma(
    const T* __restrict__ refU, const T* __restrict__ refV, const int ref_pitch,
    T* __restrict__ outU, T* __restrict__ outV, const int out_pitch,
    const int cwidth, const int cheight, const int sx, const int sy,
    const MVBlock* __restrict__ mvs, const int mv_blocks_x, const int mv_blocks_y,
    const int subpel, const int pix_max
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cwidth || y >= cheight) return;
    const T* ref = (blockIdx.z == 0) ? refU : refV;
    T* out = (blockIdx.z == 0) ? outU : outV;
    const int bx = min((x * sx) / BLOCK_SIZE, mv_blocks_x - 1);
    const int by = min((y * sy) / BLOCK_SIZE, mv_blocks_y - 1);
    const MVBlock mv = mvs[by * mv_blocks_x + bx];
    const float v = sample_lanczos6(ref, cwidth, cheight, ref_pitch,
                                    x + (float)mv.mvx / (float)(subpel * sx), y + (float)mv.mvy / (float)(subpel * sy));
    out[y * out_pitch + x] = (T)min(max((int)(v + 0.5f), 0), pix_max);
}

template<int BLOCK_SIZE, typename T>
static inline cudaError_t launch_motion_compensate_chroma(const T* d_refU, const T* d_refV, int ref_pitch,
    T* d_outU, T* d_outV, int out_pitch, int cwidth, int cheight, int sx, int sy,
    const MVBlock* d_mvs, int mv_blocks_x, int mv_blocks_y, int subpel, int pix_max, cudaStream_t stream = 0) {
    const dim3 block(16, 16, 1);
    const dim3 grid((cwidth + 15) / 16, (cheight + 15) / 16, 2);
    kernel_motion_compensate_chroma<BLOCK_SIZE, T><<<grid, block, 0, stream>>>(
        d_refU, d_refV, ref_pitch, d_outU, d_outV, out_pitch, cwidth, cheight, sx, sy,
        d_mvs, mv_blocks_x, mv_blocks_y, subpel, pix_max);
    return cudaGetLastError();
}

// Per-block thSAD from the noise level at the block's brightness:
//   E|n1 - n2| for independent Gaussian noise of std sigma = sqrt(2) * sigma * sqrt(2/pi) = 1.128 sigma,
//   so a pure-noise block SAD is about BLOCK_SIZE^2 * 1.128 * sigma; thSAD = k * that.
// nlf_lut: SMD_NLF_BINS sigmas in 8-bit code values over the normalised level 0..1.
static constexpr int SMD_NLF_BINS = 64;
template<int BLOCK_SIZE, typename T>
__global__ void kernel_block_thsad(
    const T* __restrict__ cur, const int width, const int height, const int pitch_pixels,
    const int blocks_x, const int blocks_y, const float* __restrict__ nlf_lut,
    const float k, const int pix_max, int* __restrict__ out
) {
    const int bx = blockIdx.x * blockDim.x + threadIdx.x;
    const int by = blockIdx.y * blockDim.y + threadIdx.y;
    if (bx >= blocks_x || by >= blocks_y) return;
    float sum = 0.0f;
    for (int j = 0; j < BLOCK_SIZE; j++) {
        const T* row = cur + min(by * BLOCK_SIZE + j, height - 1) * pitch_pixels;
        for (int i = 0; i < BLOCK_SIZE; i++) sum += (float)row[min(bx * BLOCK_SIZE + i, width - 1)];
    }
    const float level = sum / (float)(BLOCK_SIZE * BLOCK_SIZE) / (float)pix_max;
    const int bin = min(max((int)(level * SMD_NLF_BINS), 0), SMD_NLF_BINS - 1);
    const float sigma = nlf_lut[bin] * ((float)pix_max / 255.0f);
    out[by * blocks_x + bx] = (int)(k * (float)(BLOCK_SIZE * BLOCK_SIZE) * 1.1284f * sigma + 0.5f);
}

template<int BLOCK_SIZE, typename T>
static inline cudaError_t launch_block_thsad(const T* d_cur, int width, int height, int pitch_pixels,
    int blocks_x, int blocks_y, const float* d_nlf, float k, int pix_max, int* d_out, cudaStream_t stream = 0) {
    const dim3 block(16, 16, 1);
    const dim3 grid((blocks_x + 15) / 16, (blocks_y + 15) / 16, 1);
    kernel_block_thsad<BLOCK_SIZE, T><<<grid, block, 0, stream>>>(
        d_cur, width, height, pitch_pixels, blocks_x, blocks_y, d_nlf, k, pix_max, d_out);
    return cudaGetLastError();
}

// Scene-change test (MVTools): count blocks whose SAD exceeds thSCD1. A reference with more than
// thSCD2/256 of its blocks over the threshold lies across a cut and is left out entirely.
__global__ void kernel_count_scd(const MVBlock* __restrict__ mvs, const int n_blocks, const int thscd1,
                                 int* __restrict__ count) {
    int local = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_blocks; i += gridDim.x * blockDim.x) {
        local += (mvs[i].sad > thscd1) ? 1 : 0;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) local += __shfl_down_sync(0xffffffff, local, o);
    if ((threadIdx.x & 31) == 0 && local) atomicAdd(count, local);
}

static inline cudaError_t launch_count_scd(const MVBlock* d_mvs, int n_blocks, int thscd1, int* d_count,
                                           cudaStream_t stream = 0) {
    kernel_count_scd<<<64, 256, 0, stream>>>(d_mvs, n_blocks, thscd1, d_count);
    return cudaGetLastError();
}

template<typename T>
struct SMDBlendRefs {
    const T* mc[6];
    const MVBlock* mv[6];
};

// Weight of one reference block (MVTools DegrainWeight): binary, or (th² - s²) / (th² + s²).
__device__ __forceinline__ float smd_ref_weight(const int sad, const int th, const int weights) {
    if (sad > th) return 0.0f;
    if (weights == 0) return 1.0f;
    const float t2 = (float)th * (float)th, s2 = (float)sad * (float)sad;
    return (t2 - s2) / fmaxf(t2 + s2, 1.0f);
}

// Luma blend with optional per-block thSAD, SAD weights and scene-change exclusion.
template<typename T, int NREFS, int BLOCK_SIZE>
__global__ void kernel_temporal_blend_v2(
    const T* __restrict__ cur, const SMDBlendRefs<T> refs,
    const int mv_blocks_x, const int mv_blocks_y,
    const int width, const int height, const int pitch_pixels,
    const int thSAD, const int* __restrict__ block_th, const int weights,
    const int* __restrict__ scd_count, const int scd_limit,
    const int limit_scaled, const int pix_max, T* __restrict__ out
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int blk = min(y / BLOCK_SIZE, mv_blocks_y - 1) * mv_blocks_x + min(x / BLOCK_SIZE, mv_blocks_x - 1);
    const int th = block_th ? block_th[blk] : thSAD;
    const int px = y * pitch_pixels + x;
    const float c = (float)cur[px];
    float sum = c, wsum = 1.0f;
    #pragma unroll
    for (int i = 0; i < NREFS; i++) {
        if (scd_count && scd_count[i] > scd_limit) continue;
        const float w = smd_ref_weight(refs.mv[i][blk].sad, th, weights);
        if (w > 0.0f) { sum += w * (float)refs.mc[i][px]; wsum += w; }
    }
    float delta = sum / wsum - c;
    delta = fminf(fmaxf(delta, -(float)limit_scaled), (float)limit_scaled);
    out[px] = (T)min(max((int)(c + delta + 0.5f), 0), pix_max);
}

template<typename T, int NREFS, int BLOCK_SIZE>
static inline cudaError_t launch_temporal_blend_v2(const T* d_cur, const SMDBlendRefs<T>& refs,
    int mv_blocks_x, int mv_blocks_y, int width, int height, int pitch_pixels,
    int thSAD, const int* d_block_th, int weights, const int* d_scd_count, int scd_limit,
    int limit_scaled, int pix_max, T* d_out, cudaStream_t stream = 0) {
    const dim3 block(16, 16, 1);
    const dim3 grid((width + 15) / 16, (height + 15) / 16, 1);
    kernel_temporal_blend_v2<T, NREFS, BLOCK_SIZE><<<grid, block, 0, stream>>>(
        d_cur, refs, mv_blocks_x, mv_blocks_y, width, height, pitch_pixels,
        thSAD, d_block_th, weights, d_scd_count, scd_limit, limit_scaled, pix_max, d_out);
    return cudaGetLastError();
}

// Chroma blend: same decisions as luma (block SAD of the co-located luma block), applied to Cb/Cr.
template<typename T>
struct SMDChromaPlanes {
    const T* cur[2];        // U, V of the frame being processed
    const T* mc[6][2];      // motion-compensated U, V per reference
    T* out[2];
    int cur_pitch, mc_pitch, out_pitch;
};

template<typename T, int NREFS, int BLOCK_SIZE>
__global__ void kernel_temporal_blend_chroma(
    const SMDChromaPlanes<T> planes, const SMDBlendRefs<T> refs,
    const int mv_blocks_x, const int mv_blocks_y,
    const int cwidth, const int cheight, const int sx, const int sy,
    const int thSAD, const int* __restrict__ block_th, const int weights,
    const int* __restrict__ scd_count, const int scd_limit,
    const int limit_scaled, const int pix_max
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int p = blockIdx.z;
    if (x >= cwidth || y >= cheight) return;
    const int blk = min((y * sy) / BLOCK_SIZE, mv_blocks_y - 1) * mv_blocks_x + min((x * sx) / BLOCK_SIZE, mv_blocks_x - 1);
    const int th = block_th ? block_th[blk] : thSAD;
    const float c = (float)planes.cur[p][y * planes.cur_pitch + x];
    float sum = c, wsum = 1.0f;
    #pragma unroll
    for (int i = 0; i < NREFS; i++) {
        if (scd_count && scd_count[i] > scd_limit) continue;
        const float w = smd_ref_weight(refs.mv[i][blk].sad, th, weights);
        if (w > 0.0f) { sum += w * (float)planes.mc[i][p][y * planes.mc_pitch + x]; wsum += w; }
    }
    float delta = sum / wsum - c;
    delta = fminf(fmaxf(delta, -(float)limit_scaled), (float)limit_scaled);
    planes.out[p][y * planes.out_pitch + x] = (T)min(max((int)(c + delta + 0.5f), 0), pix_max);
}

template<typename T, int NREFS, int BLOCK_SIZE>
static inline cudaError_t launch_temporal_blend_chroma(const SMDChromaPlanes<T>& planes, const SMDBlendRefs<T>& refs,
    int mv_blocks_x, int mv_blocks_y, int cwidth, int cheight, int sx, int sy,
    int thSAD, const int* d_block_th, int weights, const int* d_scd_count, int scd_limit,
    int limit_scaled, int pix_max, cudaStream_t stream = 0) {
    const dim3 block(16, 16, 1);
    const dim3 grid((cwidth + 15) / 16, (cheight + 15) / 16, 2);
    kernel_temporal_blend_chroma<T, NREFS, BLOCK_SIZE><<<grid, block, 0, stream>>>(
        planes, refs, mv_blocks_x, mv_blocks_y, cwidth, cheight, sx, sy,
        thSAD, d_block_th, weights, d_scd_count, scd_limit, limit_scaled, pix_max);
    return cudaGetLastError();
}

} // namespace smdegrain
