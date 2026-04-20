// -----------------------------------------------------------------------------------------
// nvenc-smdegrain port — Phase 5 motion compensation + bounded temporal blend.
// -----------------------------------------------------------------------------------------
// Given per-block MVs (from me_kernel), these kernels warp a reference frame to align
// with the current frame, then blend with limit clamping to produce a denoised output.
//
// Phase 5a scope: nearest-neighbor MC, 2-frame average (cur + one MC'd ref). Bilinear
// sub-pixel MC deferred; generalization to N refs with thSAD gate is Phase 5d.
// -----------------------------------------------------------------------------------------

#pragma once

#include "me_kernel.cuh"
#include <cuda_runtime.h>
#include <cstdint>

namespace smdegrain {

// Motion-compensation kernel: produce an output frame aligned with "cur" by sampling
// the reference frame using per-block MVs. Each output pixel uses its enclosing 8x8
// block's MV (nearest-neighbor MC — no sub-pixel, no per-pixel MV).
//
// out[x,y] = ref[clamp(x + mv.mvx), clamp(y + mv.mvy)]
//
// Clamping keeps us in-bounds for edge pixels where MV pushes the sample outside the
// ref frame. Clamp-to-edge is fine for denoising (the edge pixels are either valid or
// close enough; we're averaging not reconstructing).
template<int BLOCK_SIZE, typename T>
__global__ void kernel_motion_compensate(
    const T* __restrict__ ref,
    const MVBlock* __restrict__ mvs,   // one MV per BLOCK_SIZE×BLOCK_SIZE block
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
cudaError_t launch_motion_compensate(
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

// Bounded temporal-blend kernel.
// For each pixel:
//   blended    = (cur + mc_ref) / 2         (simple 2-frame average)
//   delta      = blended - cur              (signed)
//   limited    = clamp(delta, -limit, +limit)
//   out        = cur + limited
//
// Equivalent to: out = cur + clamp((mc_ref - cur) / 2, -limit, +limit)
//
// The `limit` is in pixel units (0..255 for 8-bit). Per media_processor presets it
// ranges 100..255. Larger limit = stronger temporal smoothing, more ghosting risk
// when MC is imperfect.
template<typename T>
__global__ void kernel_temporal_blend_2frame(
    const T* __restrict__ cur,
    const T* __restrict__ mc_ref,
    const int width, const int height, const int pitch_pixels,
    const int limit,   // max per-pixel correction
    T* __restrict__ out
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int c = (int)cur[y * pitch_pixels + x];
    const int r = (int)mc_ref[y * pitch_pixels + x];
    int delta = (r - c) / 2;   // half-strength toward ref
    if (delta > limit)  delta = limit;
    if (delta < -limit) delta = -limit;
    int o = c + delta;
    // Clamp to pixel range. Assumes 8-bit for now; template specialization for 16-bit later.
    if (o < 0) o = 0;
    if (o > 255) o = 255;
    out[y * pitch_pixels + x] = (T)o;
}

template<typename T>
cudaError_t launch_temporal_blend_2frame(
    const T* d_cur, const T* d_mc_ref,
    int width, int height, int pitch_pixels,
    int limit,
    T* d_out, cudaStream_t stream = 0
) {
    const dim3 block(16, 16, 1);
    const dim3 grid((width + 15) / 16, (height + 15) / 16, 1);
    kernel_temporal_blend_2frame<T><<<grid, block, 0, stream>>>(
        d_cur, d_mc_ref, width, height, pitch_pixels, limit, d_out);
    return cudaGetLastError();
}

} // namespace smdegrain
