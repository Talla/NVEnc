// -----------------------------------------------------------------------------------------
// nvenc-smdegrain port — Phase 4b pyramid ME validation + speedup measurement.
// -----------------------------------------------------------------------------------------
// Compares:
//   (A) flat full-search at L0 with R=16 (MVP)
//   (B) pyramid: coarse full-search at L1 (half-res, R=8) + refine at L0 (R=2) around hint
//
// Both modes run on synthetic translation and real Fuji frames. Verifies that (B) gets
// essentially the same MVs as (A) but much faster.
//
// Build: nvcc -O2 -std=c++17 -arch=compute_75 me_pyramid_test.cu -o me_pyramid_test.exe
// -----------------------------------------------------------------------------------------

#include "me_kernel.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <map>

#define CUDA_CHECK(expr) do { \
    cudaError_t _err = (expr); \
    if (_err != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
        std::exit(1); \
    } \
} while (0)

static void generate_texture_frame(std::vector<uint8_t>& frame, int width, int height, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> noise(-20, 20);
    frame.resize((size_t)width * height);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const float cx = x - width * 0.5f;
            const float cy = y - height * 0.5f;
            const float r = std::sqrt(cx * cx + cy * cy);
            const int base = (int)(128.0f + 60.0f * std::sin(r * 0.08f) + 40.0f * std::sin((x + y) * 0.15f));
            const int val = std::clamp(base + noise(rng), 0, 255);
            frame[(size_t)y * width + x] = (uint8_t)val;
        }
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

static bool load_raw_y8(const char* path, std::vector<uint8_t>& out, size_t expected) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    out.resize(expected);
    size_t got = std::fread(out.data(), 1, expected, f);
    std::fclose(f);
    return got == expected;
}

struct ModeResult {
    std::vector<smdegrain::MVBlock> blocks;
    float elapsed_ms;
};

// Mode A: flat full-search at L0.
static ModeResult run_flat(const std::vector<uint8_t>& h_ref, const std::vector<uint8_t>& h_cur,
                           int width, int height, int search_radius) {
    const size_t frame_bytes = (size_t)width * height;
    const int blocks_x = width / 8;
    const int blocks_y = height / 8;
    const int num_blocks = blocks_x * blocks_y;

    uint8_t *d_ref = nullptr, *d_cur = nullptr;
    smdegrain::MVBlock* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ref, frame_bytes));
    CUDA_CHECK(cudaMalloc(&d_cur, frame_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, num_blocks * sizeof(smdegrain::MVBlock)));
    CUDA_CHECK(cudaMemcpy(d_ref, h_ref.data(), frame_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cur, h_cur.data(), frame_bytes, cudaMemcpyHostToDevice));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    cudaError_t err = smdegrain::launch_me_fullsearch<8, uint8_t>(
        d_ref, d_cur, width, height, width, search_radius, d_out);
    CUDA_CHECK(err);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float elapsed = 0.0f;
    cudaEventElapsedTime(&elapsed, t0, t1);

    ModeResult r;
    r.blocks.resize(num_blocks);
    CUDA_CHECK(cudaMemcpy(r.blocks.data(), d_out, num_blocks * sizeof(smdegrain::MVBlock),
                          cudaMemcpyDeviceToHost));
    r.elapsed_ms = elapsed;

    cudaFree(d_ref); cudaFree(d_cur); cudaFree(d_out);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return r;
}

// Mode B: pyramid — downsample 2x, coarse ME at L1, refine at L0.
static ModeResult run_pyramid(const std::vector<uint8_t>& h_ref, const std::vector<uint8_t>& h_cur,
                              int width, int height,
                              int coarse_radius,      // search radius at L1 (effective 2x at L0)
                              int refine_radius) {    // search radius at L0 around propagated hint
    const size_t frame_bytes = (size_t)width * height;
    const int l1_w = width / 2;
    const int l1_h = height / 2;
    const size_t l1_bytes = (size_t)l1_w * l1_h;

    const int l0_blocks_x = width / 8;
    const int l0_blocks_y = height / 8;
    const int l0_num_blocks = l0_blocks_x * l0_blocks_y;
    const int l1_blocks_x = l1_w / 8;
    const int l1_blocks_y = l1_h / 8;
    const int l1_num_blocks = l1_blocks_x * l1_blocks_y;

    uint8_t *d_ref_l0 = nullptr, *d_cur_l0 = nullptr;
    uint8_t *d_ref_l1 = nullptr, *d_cur_l1 = nullptr;
    smdegrain::MVBlock *d_coarse = nullptr, *d_fine = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ref_l0, frame_bytes));
    CUDA_CHECK(cudaMalloc(&d_cur_l0, frame_bytes));
    CUDA_CHECK(cudaMalloc(&d_ref_l1, l1_bytes));
    CUDA_CHECK(cudaMalloc(&d_cur_l1, l1_bytes));
    CUDA_CHECK(cudaMalloc(&d_coarse, l1_num_blocks * sizeof(smdegrain::MVBlock)));
    CUDA_CHECK(cudaMalloc(&d_fine, l0_num_blocks * sizeof(smdegrain::MVBlock)));
    CUDA_CHECK(cudaMemcpy(d_ref_l0, h_ref.data(), frame_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cur_l0, h_cur.data(), frame_bytes, cudaMemcpyHostToDevice));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    // 1. Downsample L0 -> L1 for both ref and cur.
    cudaError_t err1a = smdegrain::launch_downsample_2x<uint8_t>(
        d_ref_l0, width, height, width, d_ref_l1, l1_w, l1_h, l1_w);
    CUDA_CHECK(err1a);
    cudaError_t err1b = smdegrain::launch_downsample_2x<uint8_t>(
        d_cur_l0, width, height, width, d_cur_l1, l1_w, l1_h, l1_w);
    CUDA_CHECK(err1b);

    // 2. Coarse ME at L1.
    cudaError_t err2 = smdegrain::launch_me_fullsearch<8, uint8_t>(
        d_ref_l1, d_cur_l1, l1_w, l1_h, l1_w, coarse_radius, d_coarse);
    CUDA_CHECK(err2);

    // 3. Refine at L0 around coarse hint (propagated 2x).
    cudaError_t err3 = smdegrain::launch_me_refine_around_hint<8, uint8_t>(
        d_ref_l0, d_cur_l0, width, height, width, refine_radius,
        d_coarse, l1_blocks_x, d_fine);
    CUDA_CHECK(err3);

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float elapsed = 0.0f;
    cudaEventElapsedTime(&elapsed, t0, t1);

    ModeResult r;
    r.blocks.resize(l0_num_blocks);
    CUDA_CHECK(cudaMemcpy(r.blocks.data(), d_fine, l0_num_blocks * sizeof(smdegrain::MVBlock),
                          cudaMemcpyDeviceToHost));
    r.elapsed_ms = elapsed;

    cudaFree(d_ref_l0); cudaFree(d_cur_l0);
    cudaFree(d_ref_l1); cudaFree(d_cur_l1);
    cudaFree(d_coarse); cudaFree(d_fine);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return r;
}

static void compare_modes(const char* label,
                          const ModeResult& A, const ModeResult& B, int num_blocks) {
    int same_mv = 0, close_mv = 0, different_mv = 0;
    int64_t sad_diff_total = 0;
    int worst_dx = 0, worst_dy = 0;
    for (int i = 0; i < num_blocks; i++) {
        const auto& a = A.blocks[i];
        const auto& b = B.blocks[i];
        const int dx = std::abs((int)a.mvx - (int)b.mvx);
        const int dy = std::abs((int)a.mvy - (int)b.mvy);
        if (dx == 0 && dy == 0) same_mv++;
        else if (dx <= 1 && dy <= 1) close_mv++;
        else different_mv++;
        sad_diff_total += std::abs((int)a.sad - (int)b.sad);
        if (dx > worst_dx) worst_dx = dx;
        if (dy > worst_dy) worst_dy = dy;
    }
    const double total = num_blocks ? (double)num_blocks : 1.0;
    std::printf("  [%s] flat=%.2fms  pyr=%.2fms  speedup=%.1fx\n",
                label, A.elapsed_ms, B.elapsed_ms, A.elapsed_ms / B.elapsed_ms);
    std::printf("       agreement: same=%d(%.1f%%)  within±1=%d(%.1f%%)  different=%d(%.1f%%)\n",
                same_mv, 100.0 * same_mv / total,
                close_mv, 100.0 * close_mv / total,
                different_mv, 100.0 * different_mv / total);
    std::printf("       worst MV delta: (%d,%d),  mean |SAD diff|: %.1f\n",
                worst_dx, worst_dy, sad_diff_total / total);
}

int main(int argc, char** argv) {
    const int width = 512;
    const int height = 512;

    // ---- Test 1: synthetic translation (7,3) — pyramid should match flat 100%.
    std::printf("== Test 1: synthetic translation (+7,+3) ==\n");
    std::vector<uint8_t> h_cur_syn, h_ref_syn;
    generate_texture_frame(h_cur_syn, width, height, 42);
    translate_frame(h_cur_syn, h_ref_syn, width, height, 7, 3);

    auto A_syn = run_flat(h_ref_syn, h_cur_syn, width, height, 16);
    auto B_syn = run_pyramid(h_ref_syn, h_cur_syn, width, height, 8, 2);
    compare_modes("synthetic", A_syn, B_syn, (width / 8) * (height / 8));

    // How many blocks recovered the ground truth in each mode?
    int gt_count_A = 0, gt_count_B = 0;
    int interior = 0;
    for (int by = 0; by < height / 8; by++) {
        for (int bx = 0; bx < width / 8; bx++) {
            const int x0 = bx * 8, y0 = by * 8;
            if (x0 + 7 + 8 > width || y0 + 3 + 8 > height) continue;
            interior++;
            const int i = by * (width / 8) + bx;
            if (A_syn.blocks[i].mvx == 7 && A_syn.blocks[i].mvy == 3) gt_count_A++;
            if (B_syn.blocks[i].mvx == 7 && B_syn.blocks[i].mvy == 3) gt_count_B++;
        }
    }
    std::printf("       ground-truth (+7,+3): flat %d/%d (%.1f%%)  pyramid %d/%d (%.1f%%)\n",
                gt_count_A, interior, 100.0 * gt_count_A / interior,
                gt_count_B, interior, 100.0 * gt_count_B / interior);

    // ---- Test 2: real Fuji frames. No ground truth; we just want agreement.
    std::printf("\n== Test 2: real Fuji XH2S ProRes frames ==\n");
    const size_t frame_bytes = (size_t)width * height;
    std::vector<uint8_t> h_ref_real, h_cur_real;
    if (!load_raw_y8("ref.y8", h_ref_real, frame_bytes) ||
        !load_raw_y8("cur.y8", h_cur_real, frame_bytes)) {
        std::fprintf(stderr, "  (ref.y8 / cur.y8 not found — skipping real-frame test)\n");
    } else {
        auto A_real = run_flat(h_ref_real, h_cur_real, width, height, 16);
        auto B_real = run_pyramid(h_ref_real, h_cur_real, width, height, 8, 2);
        compare_modes("real", A_real, B_real, (width / 8) * (height / 8));

        // Show top MVs in each mode.
        for (const auto& tag_res : std::vector<std::pair<const char*, const ModeResult*>>{
                {"flat", &A_real}, {"pyramid", &B_real}}) {
            std::map<std::pair<int,int>, int> hist;
            for (const auto& mv : tag_res.second->blocks) hist[{mv.mvx, mv.mvy}]++;
            std::vector<std::pair<std::pair<int,int>, int>> s(hist.begin(), hist.end());
            std::sort(s.begin(), s.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });
            std::printf("  top-5 MVs [%s]:", tag_res.first);
            for (int i = 0; i < std::min<int>(5, s.size()); i++) {
                std::printf("  (%+d,%+d)×%d", s[i].first.first, s[i].first.second, s[i].second);
            }
            std::printf("\n");
        }
    }

    return 0;
}
