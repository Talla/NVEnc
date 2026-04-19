// -----------------------------------------------------------------------------------------
// nvenc-smdegrain port — Phase 4 block-matching motion-estimation kernel.
// -----------------------------------------------------------------------------------------
// Clean-room reimplementation. Classic block-matching ME: for each block in the CURRENT
// frame, find the (dx,dy) offset into the REFERENCE frame that minimizes Sum of Absolute
// Differences (SAD). Full search over a rectangular window — O(W*H*(2R+1)^2) per pair,
// slow at large search radius but guaranteed to find the global SAD minimum within the
// window.
//
// Phase 4a scope: single-level full search. No hierarchy, no sub-pixel refinement. Once
// correctness is proven, Phase 4b adds a 3-level pyramid and diamond-refine for speed.
//
// Block size is a compile-time template parameter (8 or 16). MVTools default is 8.
// Pixel type is a template (uint8_t or uint16_t) for 8/10-bit sources.
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

// Full-search SAD kernel.
// Grid: one block per BLOCK_SIZE×BLOCK_SIZE image tile. Each CUDA thread
// evaluates one candidate (dx,dy) within the search window and contributes
// to a shared-memory SAD reduction.
//
// Layout assumption: rowMajor, pitch in PIXELS (not bytes) for template type T.
//
// search_radius: half-window in pixels. Total candidates = (2R+1)^2.
// Keep R small (8..16) for this MVP; larger R quadruples work.
template<int BLOCK_SIZE, typename T>
__global__ void kernel_me_fullsearch(
    const T* __restrict__ ref,          // reference frame (where we search)
    const T* __restrict__ cur,          // current frame (what we're matching)
    const int width,                    // pixels
    const int height,
    const int pitch_pixels,             // row stride in pixels
    const int search_radius,            // ±R pixels
    MVBlock* __restrict__ out_blocks    // [blocks_x * blocks_y], row-major
) {
    // Block coordinates in the current frame.
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int x0 = bx * BLOCK_SIZE;    // top-left of current block
    const int y0 = by * BLOCK_SIZE;

    const int blocks_x = gridDim.x;

    // Threads within a block evaluate different (dx,dy) candidates.
    // Each thread handles one (tx,ty) position in the search window.
    // For R=16 → 33×33=1089 candidates. blockDim = (33,33,1) or we iterate.
    // Simpler MVP: loop candidates sequentially within threadIdx.x 1D range.
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int num_candidates = (2 * search_radius + 1) * (2 * search_radius + 1);

    // Per-thread best SAD tracking.
    int32_t best_sad = INT32_MAX;
    int16_t best_mvx = 0;
    int16_t best_mvy = 0;

    for (int c = tid; c < num_candidates; c += num_threads) {
        const int dx = (c % (2 * search_radius + 1)) - search_radius;
        const int dy = (c / (2 * search_radius + 1)) - search_radius;

        // Check bounds: ref block at (x0+dx, y0+dy) must be fully inside frame.
        if (x0 + dx < 0 || x0 + dx + BLOCK_SIZE > width ||
            y0 + dy < 0 || y0 + dy + BLOCK_SIZE > height) {
            continue;
        }

        // Compute SAD for this candidate.
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

        // Tie-break: prefer smaller |MV| (closer to zero motion) on equal SAD.
        // This matches MVTools' zero-bias heuristic and stabilizes static regions.
        if (sad < best_sad ||
            (sad == best_sad && (dx * dx + dy * dy) < (int)(best_mvx * best_mvx + best_mvy * best_mvy))) {
            best_sad = sad;
            best_mvx = (int16_t)dx;
            best_mvy = (int16_t)dy;
        }
    }

    // Reduce across threads in the block via shared memory.
    // MVP: small block of threads (e.g. 64), simple linear reduction.
    __shared__ int32_t s_sad[1024];
    __shared__ int16_t s_mvx[1024];
    __shared__ int16_t s_mvy[1024];

    s_sad[tid] = best_sad;
    s_mvx[tid] = best_mvx;
    s_mvy[tid] = best_mvy;
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
                s_sad[tid] = b_sad;
                s_mvx[tid] = b_mvx;
                s_mvy[tid] = b_mvy;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        MVBlock& out = out_blocks[by * blocks_x + bx];
        out.sad = s_sad[0];
        out.mvx = s_mvx[0];
        out.mvy = s_mvy[0];
    }
}

// ------------------------------------------------------------------------------------------
// Pyramid support (Phase 4b)
// ------------------------------------------------------------------------------------------
// 2×2 box downsample: src[2x,2y] .. src[2x+1,2y+1] averaged -> dst[x,y].
// Used to build a 1/2-resolution pyramid level for coarse-to-fine ME.
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
cudaError_t launch_downsample_2x(
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

// Refinement variant: per-block search is centered on a hint MV (from the coarser pyramid
// level, scaled 2×). Search radius is small (typically ±2), so (2R+1)^2 ~= 25 candidates.
// The hint array is indexed by coarse-level block position; each fine block looks up the
// hint from its enclosing coarse block (fine block index / 2 per axis since we go one
// pyramid level up at 2× scale).
template<int BLOCK_SIZE, typename T>
__global__ void kernel_me_refine_around_hint(
    const T* __restrict__ ref,
    const T* __restrict__ cur,
    const int width, const int height, const int pitch_pixels,
    const int search_radius,            // small (e.g. 2) — window around the hint
    const MVBlock* __restrict__ hints,  // coarse-level MVs, indexed per coarse block
    const int coarse_blocks_x,          // stride to index hints[y*coarse_blocks_x + x]
    MVBlock* __restrict__ out_blocks    // same layout as fine-level output
) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int x0 = bx * BLOCK_SIZE;
    const int y0 = by * BLOCK_SIZE;

    const int blocks_x = gridDim.x;

    // Find enclosing coarse block (fine index / 2 per axis). Each coarse block covers a
    // 2×2 tile of fine blocks, same MV applies.
    const int coarse_bx = bx / 2;
    const int coarse_by = by / 2;
    const MVBlock hint = hints[coarse_by * coarse_blocks_x + coarse_bx];
    const int hint_mvx = (int)hint.mvx * 2;   // scale from coarse (1/2) to fine (1/1)
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
    s_sad[tid] = best_sad;
    s_mvx[tid] = best_mvx;
    s_mvy[tid] = best_mvy;
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
                s_sad[tid] = b_sad;
                s_mvx[tid] = b_mvx;
                s_mvy[tid] = b_mvy;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        MVBlock& out = out_blocks[by * blocks_x + bx];
        out.sad = s_sad[0];
        out.mvx = s_mvx[0];
        out.mvy = s_mvy[0];
    }
}

template<int BLOCK_SIZE, typename T>
cudaError_t launch_me_refine_around_hint(
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
    const dim3 block(64, 1, 1);   // 64 threads is plenty for (2*2+1)^2 = 25 candidates
    kernel_me_refine_around_hint<BLOCK_SIZE, T><<<grid, block, 0, stream>>>(
        d_ref, d_cur, width, height, pitch_pixels, search_radius,
        d_hints, coarse_blocks_x, d_out_blocks);
    return cudaGetLastError();
}

// Host launcher — computes grid / block and invokes the kernel.
template<int BLOCK_SIZE, typename T>
cudaError_t launch_me_fullsearch(
    const T* d_ref, const T* d_cur,
    int width, int height, int pitch_pixels,
    int search_radius,
    MVBlock* d_out_blocks,
    cudaStream_t stream = 0
) {
    const int blocks_x = width / BLOCK_SIZE;   // integer blocks only; partial edge blocks ignored for MVP
    const int blocks_y = height / BLOCK_SIZE;
    const dim3 grid(blocks_x, blocks_y, 1);
    // Threads per CUDA block: 256 is a safe default; each thread loops over multiple
    // candidates when 2R+1)^2 > 256.
    const dim3 block(256, 1, 1);
    kernel_me_fullsearch<BLOCK_SIZE, T><<<grid, block, 0, stream>>>(
        d_ref, d_cur, width, height, pitch_pixels, search_radius, d_out_blocks);
    return cudaGetLastError();
}

} // namespace smdegrain
