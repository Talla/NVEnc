// -----------------------------------------------------------------------------------------
// nvenc-smdegrain port — Phase 4d ME sanity check on real frame pairs.
// -----------------------------------------------------------------------------------------
// Loads two raw Y8 planes from files and runs the ME kernel. No ground truth; this is
// a sanity check that the MV distribution looks like real camera motion (clustered near
// zero for static-ish footage, spread for action) rather than noise.
//
// Build: nvcc -O2 -std=c++17 -arch=compute_75 me_real_frames_test.cu -o me_real_frames_test.exe
// Run:   me_real_frames_test.exe <ref.y8> <cur.y8> <width> <height> [search_radius=16]
// -----------------------------------------------------------------------------------------

#include "me_kernel.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <map>
#include <algorithm>

#define CUDA_CHECK(expr) do { \
    cudaError_t _err = (expr); \
    if (_err != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
        std::exit(1); \
    } \
} while (0)

static bool load_raw_y8(const char* path, std::vector<uint8_t>& out, size_t expected) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "failed to open %s\n", path);
        return false;
    }
    out.resize(expected);
    size_t got = std::fread(out.data(), 1, expected, f);
    std::fclose(f);
    if (got != expected) {
        std::fprintf(stderr, "%s: expected %zu bytes, got %zu\n", path, expected, got);
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: %s <ref.y8> <cur.y8> <width> <height> [search_radius=16]\n", argv[0]);
        return 2;
    }
    const char* ref_path = argv[1];
    const char* cur_path = argv[2];
    const int width = std::atoi(argv[3]);
    const int height = std::atoi(argv[4]);
    const int search_radius = (argc >= 6) ? std::atoi(argv[5]) : 16;
    const int block_size = 8;

    const size_t frame_bytes = (size_t)width * height;
    std::vector<uint8_t> h_ref, h_cur;
    if (!load_raw_y8(ref_path, h_ref, frame_bytes)) return 1;
    if (!load_raw_y8(cur_path, h_cur, frame_bytes)) return 1;

    std::printf("== ME real-frame sanity check ==\n");
    std::printf("ref: %s\ncur: %s\nframe %dx%d, block %d, search ±%d\n",
                ref_path, cur_path, width, height, block_size, search_radius);

    uint8_t *d_ref = nullptr, *d_cur = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ref, frame_bytes));
    CUDA_CHECK(cudaMalloc(&d_cur, frame_bytes));
    CUDA_CHECK(cudaMemcpy(d_ref, h_ref.data(), frame_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cur, h_cur.data(), frame_bytes, cudaMemcpyHostToDevice));

    const int blocks_x = width / block_size;
    const int blocks_y = height / block_size;
    const int num_blocks = blocks_x * blocks_y;
    smdegrain::MVBlock* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, num_blocks * sizeof(smdegrain::MVBlock)));

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));

    cudaError_t launch_err = smdegrain::launch_me_fullsearch<8, uint8_t>(
        d_ref, d_cur, width, height, width, search_radius, d_out);
    CUDA_CHECK(launch_err);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, t0, t1));

    std::vector<smdegrain::MVBlock> h_out(num_blocks);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, num_blocks * sizeof(smdegrain::MVBlock),
                          cudaMemcpyDeviceToHost));

    // Statistics.
    std::map<std::pair<int,int>, int> mv_hist;
    int zero_mv = 0;
    int64_t sad_total = 0;
    int sad_max = 0, sad_min = INT32_MAX;
    int valid = 0;
    for (const auto& mv : h_out) {
        if (mv.sad == INT32_MAX) continue;   // all edge blocks were filtered (no candidates)
        mv_hist[{mv.mvx, mv.mvy}]++;
        if (mv.mvx == 0 && mv.mvy == 0) zero_mv++;
        sad_total += mv.sad;
        if (mv.sad > sad_max) sad_max = mv.sad;
        if (mv.sad < sad_min) sad_min = mv.sad;
        valid++;
    }

    std::printf("\n== MV distribution ==\n");
    std::printf("  total blocks: %d (%dx%d), valid: %d\n", num_blocks, blocks_x, blocks_y, valid);
    std::printf("  zero-MV blocks: %d  (%.1f%%)\n", zero_mv, valid ? 100.0 * zero_mv / valid : 0.0);
    std::printf("  mean SAD: %.1f, min %d, max %d\n",
                valid ? (double)sad_total / valid : 0.0, sad_min, sad_max);

    // Top 10 most common MVs — should show a tight cluster near (0,0) for static footage.
    std::vector<std::pair<std::pair<int,int>, int>> sorted_hist(mv_hist.begin(), mv_hist.end());
    std::sort(sorted_hist.begin(), sorted_hist.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    std::printf("\n  top-10 MVs:\n");
    const int topn = std::min<int>(10, sorted_hist.size());
    for (int i = 0; i < topn; i++) {
        const auto& mv = sorted_hist[i].first;
        const int count = sorted_hist[i].second;
        std::printf("    (%+3d,%+3d): %5d blocks (%.1f%%)\n",
                    mv.first, mv.second, count, valid ? 100.0 * count / valid : 0.0);
    }

    // MV magnitude histogram.
    std::map<int, int> mag_hist;
    for (const auto& mv : h_out) {
        if (mv.sad == INT32_MAX) continue;
        const int mag = std::max(std::abs((int)mv.mvx), std::abs((int)mv.mvy));
        mag_hist[mag]++;
    }
    std::printf("\n  |MV|_∞ histogram:\n");
    for (const auto& p : mag_hist) {
        std::printf("    %2d: %5d blocks (%.1f%%)\n",
                    p.first, p.second, valid ? 100.0 * p.second / valid : 0.0);
    }

    std::printf("\n  kernel time: %.3f ms  (%dx%d, R=%d)\n",
                elapsed_ms, width, height, search_radius);

    cudaFree(d_ref);
    cudaFree(d_cur);
    cudaFree(d_out);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return 0;
}
