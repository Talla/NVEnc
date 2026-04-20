// -----------------------------------------------------------------------------------------
// nvenc-smdegrain port — Phase 5c end-to-end denoise test.
// -----------------------------------------------------------------------------------------
// Measures that the pipeline (pyramid ME + MC + bounded blend) actually reduces noise.
//
// Setup:
//   base     = clean textured 512x512 frame
//   cur      = base + independent_noise(σ_cur)
//   ref      = translate(base, dx, dy) + independent_noise(σ_ref)
// (so ref is "previous frame" of a camera motion dx,dy with its own noise)
//
// Pipeline:
//   ME(ref, cur) -> per-block MV ≈ (dx, dy)
//   MC(ref, MV) -> mc_ref ≈ cur's frame (denoised scene, independent noise)
//   blend(cur, mc_ref, limit) -> out
//
// Success metric:
//   std(out - base) < std(cur - base) by a measurable factor.
//   Theoretical best with 2-frame average: σ_out ≈ σ_cur / √2 ≈ 0.707 σ_cur.
//   Practically we expect somewhere in 0.75..0.9 range because MC isn't perfect at
//   block boundaries and the `limit` clamp prevents full averaging for large deltas.
//
// Build: nvcc -O2 -std=c++17 -arch=compute_75 me_denoise_test.cu -o me_denoise_test.exe
// -----------------------------------------------------------------------------------------

#include "me_kernel.cuh"
#include "mc_blend_kernel.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

#define CUDA_CHECK(expr) do { \
    cudaError_t _err = (expr); \
    if (_err != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
        std::exit(1); \
    } \
} while (0)

static void generate_base_frame(std::vector<uint8_t>& frame, int width, int height) {
    // No noise here — clean ground truth. We'll add noise separately for cur/ref.
    frame.resize((size_t)width * height);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const float cx = x - width * 0.5f;
            const float cy = y - height * 0.5f;
            const float r = std::sqrt(cx * cx + cy * cy);
            const float base = 128.0f + 60.0f * std::sin(r * 0.08f) + 40.0f * std::sin((x + y) * 0.15f);
            frame[(size_t)y * width + x] = (uint8_t)std::clamp((int)std::round(base), 0, 255);
        }
    }
}

static void add_noise(const std::vector<uint8_t>& src, std::vector<uint8_t>& dst,
                      float sigma, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> noise(0.0f, sigma);
    dst.resize(src.size());
    for (size_t i = 0; i < src.size(); i++) {
        dst[i] = (uint8_t)std::clamp((int)std::round((float)src[i] + noise(rng)), 0, 255);
    }
}

static void translate_frame(const std::vector<uint8_t>& src, std::vector<uint8_t>& dst,
                            int width, int height, int dx, int dy) {
    dst.resize(src.size());
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int sx = std::clamp(x - dx, 0, width - 1);
            int sy = std::clamp(y - dy, 0, height - 1);
            dst[(size_t)y * width + x] = src[(size_t)sy * width + sx];
        }
    }
}

// Compute std deviation of (a - b) over a rectangular interior (excludes `margin` pixels on all sides).
static double frame_stddev_vs(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b,
                               int width, int height, int margin) {
    double sum = 0.0, sum_sq = 0.0;
    long long n = 0;
    for (int y = margin; y < height - margin; y++) {
        for (int x = margin; x < width - margin; x++) {
            const int d = (int)a[(size_t)y * width + x] - (int)b[(size_t)y * width + x];
            sum += d;
            sum_sq += (double)d * d;
            n++;
        }
    }
    if (n == 0) return 0.0;
    const double mean = sum / n;
    const double var = sum_sq / n - mean * mean;
    return std::sqrt(std::max(0.0, var));
}

int main(int argc, char** argv) {
    const int width = 512;
    const int height = 512;
    const int truth_dx = 5;
    const int truth_dy = -3;
    const float sigma = 10.0f;       // noise stddev in pixel units
    const int limit = 255;            // liberal limit — let the blend run at full strength

    std::printf("== End-to-end denoise test ==\n");
    std::printf("frame %dx%d, ground-truth motion (+%d,%d), noise σ=%.1f\n",
                width, height, truth_dx, truth_dy, sigma);

    // Build base + noisy variants.
    std::vector<uint8_t> h_base, h_cur, h_ref_shifted, h_ref;
    generate_base_frame(h_base, width, height);
    add_noise(h_base, h_cur, sigma, 1001);                     // cur = base + noise_cur
    translate_frame(h_base, h_ref_shifted, width, height, truth_dx, truth_dy);
    add_noise(h_ref_shifted, h_ref, sigma, 2002);              // ref = shift(base) + noise_ref

    const double noise_before = frame_stddev_vs(h_cur, h_base, width, height, 16);
    std::printf("  std(cur - base) = %.3f  (baseline noise)\n", noise_before);

    // Upload.
    const size_t bytes = (size_t)width * height;
    uint8_t *d_cur = nullptr, *d_ref = nullptr, *d_ref_l1 = nullptr, *d_cur_l1 = nullptr;
    uint8_t *d_mc = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cur, bytes));
    CUDA_CHECK(cudaMalloc(&d_ref, bytes));
    CUDA_CHECK(cudaMalloc(&d_mc, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    const int l1_w = width / 2, l1_h = height / 2;
    CUDA_CHECK(cudaMalloc(&d_ref_l1, (size_t)l1_w * l1_h));
    CUDA_CHECK(cudaMalloc(&d_cur_l1, (size_t)l1_w * l1_h));
    CUDA_CHECK(cudaMemcpy(d_cur, h_cur.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ref, h_ref.data(), bytes, cudaMemcpyHostToDevice));

    const int blocks_x = width / 8;
    const int blocks_y = height / 8;
    const int l1_blocks_x = l1_w / 8;
    const int l1_blocks_y = l1_h / 8;
    smdegrain::MVBlock *d_coarse = nullptr, *d_fine = nullptr;
    CUDA_CHECK(cudaMalloc(&d_coarse, l1_blocks_x * l1_blocks_y * sizeof(smdegrain::MVBlock)));
    CUDA_CHECK(cudaMalloc(&d_fine, blocks_x * blocks_y * sizeof(smdegrain::MVBlock)));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    // Pipeline.
    cudaError_t e;
    e = smdegrain::launch_downsample_2x<uint8_t>(d_ref, width, height, width,
                                                  d_ref_l1, l1_w, l1_h, l1_w);
    CUDA_CHECK(e);
    e = smdegrain::launch_downsample_2x<uint8_t>(d_cur, width, height, width,
                                                  d_cur_l1, l1_w, l1_h, l1_w);
    CUDA_CHECK(e);
    e = smdegrain::launch_me_fullsearch<8, uint8_t>(d_ref_l1, d_cur_l1,
                                                     l1_w, l1_h, l1_w, 8, d_coarse);
    CUDA_CHECK(e);
    e = smdegrain::launch_me_refine_around_hint<8, uint8_t>(
            d_ref, d_cur, width, height, width, 2,
            d_coarse, l1_blocks_x, d_fine);
    CUDA_CHECK(e);
    e = smdegrain::launch_motion_compensate<8, uint8_t>(
            d_ref, d_fine, width, height, width, blocks_x, d_mc);
    CUDA_CHECK(e);
    e = smdegrain::launch_temporal_blend_2frame<uint8_t>(
            d_cur, d_mc, width, height, width, limit, d_out);
    CUDA_CHECK(e);

    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float elapsed = 0.0f; cudaEventElapsedTime(&elapsed, t0, t1);

    // Download outputs for analysis.
    std::vector<uint8_t> h_out(bytes), h_mc(bytes);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_mc.data(),  d_mc,  bytes, cudaMemcpyDeviceToHost));

    // Metrics.
    const double noise_mc_ref = frame_stddev_vs(h_mc, h_base, width, height, 16);
    const double noise_after = frame_stddev_vs(h_out, h_base, width, height, 16);

    std::printf("  std(mc_ref - base) = %.3f  (after MC, before blend)\n", noise_mc_ref);
    std::printf("  std(out - base)    = %.3f  (after denoise)\n", noise_after);

    const double reduction = (noise_before > 0.0) ? (noise_after / noise_before) : 1.0;
    std::printf("  noise reduction ratio = %.3f  (ideal for 2-frame avg: 0.707)\n", reduction);
    std::printf("  pipeline time: %.3f ms\n", elapsed);

    // Sanity: top MVs should cluster around (5,-3).
    std::vector<smdegrain::MVBlock> h_fine(blocks_x * blocks_y);
    CUDA_CHECK(cudaMemcpy(h_fine.data(), d_fine,
                          blocks_x * blocks_y * sizeof(smdegrain::MVBlock),
                          cudaMemcpyDeviceToHost));
    int gt_match = 0, valid = 0;
    for (int by = 4; by < blocks_y - 4; by++) {
        for (int bx = 4; bx < blocks_x - 4; bx++) {
            const auto& mv = h_fine[by * blocks_x + bx];
            valid++;
            if (mv.mvx == truth_dx && mv.mvy == truth_dy) gt_match++;
        }
    }
    std::printf("  interior ME accuracy: %d/%d = %.1f%% recovered (+%d,%d)\n",
                gt_match, valid, 100.0 * gt_match / valid, truth_dx, truth_dy);

    // Cleanup.
    cudaFree(d_cur); cudaFree(d_ref); cudaFree(d_ref_l1); cudaFree(d_cur_l1);
    cudaFree(d_mc); cudaFree(d_out); cudaFree(d_coarse); cudaFree(d_fine);
    cudaEventDestroy(t0); cudaEventDestroy(t1);

    // Accept any reduction below 0.90 as "the denoising pipeline works." (Conservative
    // vs the theoretical 0.707 because MC isn't perfect and our frames are small.)
    if (reduction < 0.90) {
        std::printf("\nPASS (noise reduced by %.1f%%)\n", (1.0 - reduction) * 100.0);
        return 0;
    } else {
        std::printf("\nFAIL (expected ratio < 0.90, got %.3f)\n", reduction);
        return 1;
    }
}
