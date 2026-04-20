// -----------------------------------------------------------------------------------------
// nvenc-smdegrain port — Phase 4b standalone ME kernel test harness.
// -----------------------------------------------------------------------------------------
// Synthetic-translation validation: generate a textured reference frame, produce a translated
// copy (known ground-truth MV), run the ME kernel, verify recovered MVs match.
//
// Build: nvcc -O2 -arch=compute_75 me_standalone_test.cu -o me_standalone_test.exe
// Run:   me_standalone_test.exe
// -----------------------------------------------------------------------------------------

#include "me_kernel.cuh"
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

// Generate a deterministic textured 8-bit grayscale frame with noise + gradients.
// Enough spatial detail that block matching can disambiguate positions.
static void generate_texture_frame(std::vector<uint8_t>& frame, int width, int height, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> noise(-20, 20);
    frame.resize((size_t)width * height);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Base: radial gradient + diagonal stripes + noise. Gives each block a unique signature.
            const float cx = x - width * 0.5f;
            const float cy = y - height * 0.5f;
            const float r = std::sqrt(cx * cx + cy * cy);
            const int base = (int)(128.0f + 60.0f * std::sin(r * 0.08f) + 40.0f * std::sin((x + y) * 0.15f));
            const int val = std::clamp(base + noise(rng), 0, 255);
            frame[(size_t)y * width + x] = (uint8_t)val;
        }
    }
}

// Translate a frame by (dx,dy). Out-of-bounds source pixels become the edge clamp value.
// Produces the "reference" frame that the current frame (with zero motion) should match
// by searching at offset (-dx,-dy) — i.e. cur[i,j] = ref[i-dx, j-dy] → ME should report MV = (-dx,-dy).
// But we conventionally define MV as "offset INTO ref FROM cur" — so cur[i,j] ≈ ref[i+mvx, j+mvy] and
// a synthetic shift where ref is cur shifted by (+dx,+dy) means mv = (+dx,+dy).
static void translate_frame(const std::vector<uint8_t>& src, std::vector<uint8_t>& dst,
                            int width, int height, int dx, int dy) {
    dst.resize(src.size());
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // dst[x,y] = src[x-dx, y-dy] with edge clamp
            int sx = std::clamp(x - dx, 0, width - 1);
            int sy = std::clamp(y - dy, 0, height - 1);
            dst[(size_t)y * width + x] = src[(size_t)sy * width + sx];
        }
    }
}

int main(int argc, char** argv) {
    const int width = 512;
    const int height = 512;
    const int block_size = 8;
    const int search_radius = 16;
    const int truth_dx = 7;
    const int truth_dy = 3;

    std::printf("== ME kernel synthetic-translation test ==\n");
    std::printf("frame: %dx%d, block %d, search ±%d, ground-truth MV = (%d,%d)\n",
                width, height, block_size, search_radius, truth_dx, truth_dy);

    // Generate frames.
    // Convention here: CURRENT frame is the textured base; REFERENCE frame is CURRENT shifted by
    // (truth_dx, truth_dy). The kernel searches for the (dx,dy) such that ref[y+dy, x+dx] ≈ cur[y, x],
    // which is (dx,dy) = (truth_dx, truth_dy).
    std::vector<uint8_t> h_cur, h_ref;
    generate_texture_frame(h_cur, width, height, 42);
    translate_frame(h_cur, h_ref, width, height, truth_dx, truth_dy);

    // Upload to device.
    uint8_t *d_cur = nullptr, *d_ref = nullptr;
    const size_t frame_bytes = (size_t)width * height;
    CUDA_CHECK(cudaMalloc(&d_cur, frame_bytes));
    CUDA_CHECK(cudaMalloc(&d_ref, frame_bytes));
    CUDA_CHECK(cudaMemcpy(d_cur, h_cur.data(), frame_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ref, h_ref.data(), frame_bytes, cudaMemcpyHostToDevice));

    // Allocate output.
    const int blocks_x = width / block_size;
    const int blocks_y = height / block_size;
    const int num_blocks = blocks_x * blocks_y;
    smdegrain::MVBlock* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, num_blocks * sizeof(smdegrain::MVBlock)));

    // Run kernel.
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));

    cudaError_t err = smdegrain::launch_me_fullsearch<8, uint8_t>(
        d_ref, d_cur, width, height, width, search_radius, d_out);
    CUDA_CHECK(err);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, t0, t1));

    // Download results.
    std::vector<smdegrain::MVBlock> h_out(num_blocks);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, num_blocks * sizeof(smdegrain::MVBlock),
                          cudaMemcpyDeviceToHost));

    // Analyse.
    int correct = 0;
    int edge_blocks = 0;          // blocks too close to the edge for the search to find ground truth
    int total_interior = 0;
    int sad_zero = 0;
    int64_t sad_total = 0;
    int max_sad_correct = 0;
    int worst_mvx_diff = 0, worst_mvy_diff = 0;

    for (int by = 0; by < blocks_y; by++) {
        for (int bx = 0; bx < blocks_x; bx++) {
            const auto& mv = h_out[by * blocks_x + bx];
            const int x0 = bx * block_size;
            const int y0 = by * block_size;
            // An interior block's ground-truth shifted position must fit in the frame.
            const bool gt_in_bounds =
                (x0 + truth_dx >= 0 && x0 + truth_dx + block_size <= width) &&
                (y0 + truth_dy >= 0 && y0 + truth_dy + block_size <= height);
            // Also the search radius must encompass the ground truth.
            const bool gt_in_search =
                (std::abs(truth_dx) <= search_radius && std::abs(truth_dy) <= search_radius);
            if (!gt_in_bounds || !gt_in_search) {
                edge_blocks++;
                continue;
            }
            total_interior++;
            sad_total += mv.sad;
            if (mv.sad == 0) sad_zero++;
            const int dx_diff = std::abs((int)mv.mvx - truth_dx);
            const int dy_diff = std::abs((int)mv.mvy - truth_dy);
            if (dx_diff == 0 && dy_diff == 0) {
                correct++;
                if (mv.sad > max_sad_correct) max_sad_correct = mv.sad;
            } else {
                if (dx_diff > worst_mvx_diff) worst_mvx_diff = dx_diff;
                if (dy_diff > worst_mvy_diff) worst_mvy_diff = dy_diff;
            }
        }
    }

    std::printf("\n== Results ==\n");
    std::printf("  total blocks:        %d (%dx%d)\n", num_blocks, blocks_x, blocks_y);
    std::printf("  edge blocks skipped: %d\n", edge_blocks);
    std::printf("  interior blocks:     %d\n", total_interior);
    std::printf("  correct MV recovery: %d  (%.1f%%)\n",
                correct, total_interior ? 100.0 * correct / total_interior : 0.0);
    std::printf("  SAD=0 blocks:        %d\n", sad_zero);
    std::printf("  mean SAD (interior): %.2f\n",
                total_interior ? (double)sad_total / total_interior : 0.0);
    std::printf("  max SAD among correct: %d\n", max_sad_correct);
    if (correct < total_interior) {
        std::printf("  worst mvx diff: %d, worst mvy diff: %d\n", worst_mvx_diff, worst_mvy_diff);
    }
    std::printf("  kernel time: %.3f ms\n", elapsed_ms);

    // Print a couple of sample blocks for sanity.
    std::printf("\n== Sample blocks ==\n");
    for (int by : {8, 32, 48}) {
        for (int bx : {8, 32, 48}) {
            const auto& mv = h_out[by * blocks_x + bx];
            std::printf("  block (%d,%d): mv=(%d,%d) sad=%d\n",
                        bx, by, (int)mv.mvx, (int)mv.mvy, (int)mv.sad);
        }
    }

    // Cleanup.
    cudaFree(d_cur);
    cudaFree(d_ref);
    cudaFree(d_out);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);

    // Exit code reflects test pass.
    // Target: ≥95% of interior blocks recover the ground-truth MV with SAD small (tolerates the
    // ±20 noise injection — SAD proportional to noise magnitude times block area = 20*64 = 1280 max).
    const double pass_threshold = 0.95;
    const double correct_rate = total_interior ? (double)correct / total_interior : 0.0;
    if (correct_rate >= pass_threshold) {
        std::printf("\nPASS\n");
        return 0;
    } else {
        std::printf("\nFAIL (correct rate %.1f%% < %.1f%%)\n",
                    100.0 * correct_rate, 100.0 * pass_threshold);
        return 1;
    }
}
