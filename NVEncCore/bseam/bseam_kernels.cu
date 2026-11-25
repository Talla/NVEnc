// AX700 Round 2 joint seam detector and repair from denoise-lab.
// used_by: NVEncFilterBSeam.cpp. Caller owns the workspace, frame lifetimes and stream.
#include "bseam_kernels.h"
#include <math_constants.h>
#include <cstdint>
#include <cmath>
#include <type_traits>

namespace {
constexpr int TW = 160, TH = 104, NC = 35, PAD = 3, FIR_RADIUS = 12;
constexpr int PW = TW + 2 * PAD;
constexpr int AREA = TW * TH, SCRATCH = 3 * AREA, WORKERS = 144;
constexpr int THREADS = 256;
enum { G1, G2, G3, G12, G15, G16, G375, D16, D375, J4, J6, J8, BUMP, TABLES };
constexpr int TABLE_WIDTH = 97;
// FP64 keeps the original row-band locator precision. Repair uses FP32.
__constant__ double locator_gaussian[9] = {
        1.33830624614741749e-04, 4.43186162003126553e-03, 5.39911274207044087e-02, 2.41971445656600725e-01, 3.98943469356097757e-01,
        2.41971445656600725e-01, 5.39911274207044087e-02, 4.43186162003126553e-03, 1.33830624614741749e-04,
    };

__constant__ float repair_filters[TABLES][TABLE_WIDTH] = {
    {
        1.33830624614741749e-04f, 4.43186162003126553e-03f, 5.39911274207044087e-02f, 2.41971445656600725e-01f, 3.98943469356097757e-01f,
        2.41971445656600725e-01f, 5.39911274207044087e-02f, 4.43186162003126553e-03f, 1.33830624614741749e-04f,
    },
    {
        6.69162895726355312e-05f, 4.36349020506788317e-04f, 2.21596317259655515e-03f, 8.76430436278586959e-03f, 2.69959579672988462e-02f,
        6.47599366047274388e-02f, 1.20987489765349038e-01f, 1.76035758884790366e-01f, 1.99474647864745003e-01f, 1.76035758884790366e-01f,
        1.20987489765349038e-01f, 6.47599366047274388e-02f, 2.69959579672988462e-02f, 8.76430436278586959e-03f, 2.21596317259655515e-03f,
        4.36349020506788317e-04f, 6.69162895726355312e-05f,
    },
    {
        4.46113427726488033e-05f, 1.60094765891977581e-04f, 5.14107605821972750e-04f, 1.47732477840040698e-03f, 3.79876994030548237e-03f,
        8.74087804769129242e-03f, 1.79975001908605561e-02f, 3.31599884208674187e-02f, 5.46715782456540650e-02f, 8.06591998987869901e-02f,
        1.06485694027244698e-01f, 1.25797983460310653e-01f, 1.32984538550783710e-01f, 1.25797983460310653e-01f, 1.06485694027244698e-01f,
        8.06591998987869901e-02f, 5.46715782456540650e-02f, 3.31599884208674187e-02f, 1.79975001908605561e-02f, 8.74087804769129242e-03f,
        3.79876994030548237e-03f, 1.47732477840040698e-03f, 5.14107605821972750e-04f, 1.60094765891977581e-04f, 4.46113427726488033e-05f,
    },
    {
        1.11531077939528136e-05f, 1.55114629552385144e-05f, 2.14236609424589000e-05f, 2.93845252868002954e-05f, 4.00246679491474307e-05f,
        5.41403254857469906e-05f, 7.27273985539103032e-05f, 9.70195621120554028e-05f, 1.28530037184599520e-04f, 1.69096267848579018e-04f,
        2.20926329683392435e-04f, 2.86645414613496236e-04f, 3.69340205340761170e-04f, 4.72598392221928564e-04f, 6.00540027406517803e-04f,
        7.57836895451934403e-04f, 9.49715655154648262e-04f, 1.18194022950021053e-03f, 1.46076884898195102e-03f, 1.79288134765121887e-03f,
        2.18527282571433007e-03f, 2.64511067094540557e-03f, 3.17955320001280699e-03f, 3.79552984276218487e-03f, 4.49948482100880412e-03f,
        5.29708860834395782e-03f, 6.19292400300927061e-03f, 7.19015627109400885e-03f, 8.29019936003603901e-03f, 9.49239245681267677e-03f,
        1.07937029726979904e-02f, 1.21884731797224713e-02f, 1.36682280232360061e-02f, 1.52215609453537435e-02f, 1.68341127824079458e-02f,
        1.84886559191846760e-02f, 2.01652919444671799e-02f, 2.18417661900188777e-02f, 2.34938969644290889e-02f, 2.50961113038630675e-02f,
        2.66220730017542317e-02f, 2.80453829333352118e-02f, 2.93403266524693250e-02f, 3.04826402806156020e-02f, 3.14502631526917403e-02f,
        3.22240447741847166e-02f, 3.27883745172495994e-02f, 3.31317051611000266e-02f, 3.32469457587209888e-02f, 3.31317051611000266e-02f,
        3.27883745172495994e-02f, 3.22240447741847166e-02f, 3.14502631526917403e-02f, 3.04826402806156020e-02f, 2.93403266524693250e-02f,
        2.80453829333352118e-02f, 2.66220730017542317e-02f, 2.50961113038630675e-02f, 2.34938969644290889e-02f, 2.18417661900188777e-02f,
        2.01652919444671799e-02f, 1.84886559191846760e-02f, 1.68341127824079458e-02f, 1.52215609453537435e-02f, 1.36682280232360061e-02f,
        1.21884731797224713e-02f, 1.07937029726979904e-02f, 9.49239245681267677e-03f, 8.29019936003603901e-03f, 7.19015627109400885e-03f,
        6.19292400300927061e-03f, 5.29708860834395782e-03f, 4.49948482100880412e-03f, 3.79552984276218487e-03f, 3.17955320001280699e-03f,
        2.64511067094540557e-03f, 2.18527282571433007e-03f, 1.79288134765121887e-03f, 1.46076884898195102e-03f, 1.18194022950021053e-03f,
        9.49715655154648262e-04f, 7.57836895451934403e-04f, 6.00540027406517803e-04f, 4.72598392221928564e-04f, 3.69340205340761170e-04f,
        2.86645414613496236e-04f, 2.20926329683392435e-04f, 1.69096267848579018e-04f, 1.28530037184599520e-04f, 9.70195621120554028e-05f,
        7.27273985539103032e-05f, 5.41403254857469906e-05f, 4.00246679491474307e-05f, 2.93845252868002954e-05f, 2.14236609424589000e-05f,
        1.55114629552385144e-05f, 1.11531077939528136e-05f,
    },
    {
        8.92210686353923102e-05f, 1.02819657813890903e-03f, 7.59740219659692934e-03f, 3.59943480737088331e-02f, 1.09341174954530496e-01f,
        2.12967528547854001e-01f, 2.65964257161070916e-01f, 2.12967528547854001e-01f, 1.09341174954530496e-01f, 3.59943480737088331e-02f,
        7.59740219659692934e-03f, 1.02819657813890903e-03f, 8.92210686353923102e-05f,
    },
    {
        1.10020044943487938e-02f, 4.31751450217583363e-02f, 1.14643519437383018e-01f, 2.05977096991565828e-01f, 2.50404468109888034e-01f,
        2.05977096991565828e-01f, 1.14643519437383018e-01f, 4.31751450217583363e-02f, 1.10020044943487938e-02f,
    },
    {
        6.03865075738404047e-03f, 1.10521507485053898e-02f, 1.88395499381827994e-02f, 2.99096330009212714e-02f, 4.42250651664203506e-02f,
        6.09035651202444739e-02f, 7.81148830298857844e-02f, 9.33129017048303783e-02f, 1.03816513419469861e-01f, 1.07574174228311392e-01f,
        1.03816513419469861e-01f, 9.33129017048303783e-02f, 7.81148830298857844e-02f, 6.09035651202444739e-02f, 4.42250651664203506e-02f,
        2.99096330009212714e-02f, 1.88395499381827994e-02f, 1.10521507485053898e-02f, 6.03865075738404047e-03f,
    },
    {
        -1.71906320224199868e-02f, -5.05958730723730377e-02f, -8.95652495604554733e-02f, -8.04598035123303967e-02f, 0.00000000000000000e+00f,
        8.04598035123303967e-02f, 8.95652495604554733e-02f, 5.05958730723730377e-02f, 1.71906320224199868e-02f,
    },
    {
        -3.86473648472578611e-03f, -6.28744575914973267e-03f, -9.37790930256210409e-03f, -1.27614434137264098e-02f, -1.57244676147272368e-02f,
        -1.73236807453139852e-02f, -1.66645083797089671e-02f, -1.32711682424647651e-02f, -7.38250762094007883e-03f, 0.00000000000000000e+00f,
        7.38250762094007883e-03f, 1.32711682424647651e-02f, 1.66645083797089671e-02f, 1.73236807453139852e-02f, 1.57244676147272368e-02f,
        1.27614434137264098e-02f, 9.37790930256210409e-03f, 6.28744575914973267e-03f, 3.86473648472578611e-03f,
    },
    {
        8.49999999999999978e-01f, 4.50000000000000011e-01f, 4.99999999999999889e-02f, -3.49999999999999978e-01f,
    },
    {
        5.95238095238095233e-01f, 4.23809523809523747e-01f, 2.52380952380952372e-01f, 8.09523809523809423e-02f, -9.04761904761904601e-02f,
        -2.61904761904761862e-01f,
    },
    {
        4.58333333333333315e-01f, 3.63095238095238082e-01f, 2.67857142857142849e-01f, 1.72619047619047616e-01f, 7.73809523809523836e-02f,
        -1.78571428571428492e-02f, -1.13095238095238082e-01f, -2.08333333333333315e-01f,
    },
    {
        -4.80367989919239236e-03f, -4.21325969243636633e-02f, -1.11107441745099511e-01f, -2.01227419495967952e-01f, -2.98772580504032104e-01f,
        -3.88892558254900544e-01f, -4.57867403075636337e-01f, -4.95196320100807608e-01f, 4.95196320100807608e-01f, 4.57867403075636337e-01f,
        3.88892558254900544e-01f, 2.98772580504032104e-01f, 2.01227419495967952e-01f, 1.11107441745099511e-01f, 4.21325969243636633e-02f,
        4.80367989919239236e-03f,
    },
};

__constant__ float float_filters[9][25] = {
    {1.3383062461e-04f, 4.4318616200e-03f, 5.3991127421e-02f, 2.4197144566e-01f, 3.9894346936e-01f, 2.4197144566e-01f, 5.3991127421e-02f, 4.4318616200e-03f, 1.3383062461e-04f},
    {6.6916289573e-05f, 4.3634902051e-04f, 2.2159631726e-03f, 8.7643043628e-03f, 2.6995957967e-02f, 6.4759936605e-02f, 1.2098748977e-01f, 1.7603575888e-01f, 1.9947464786e-01f, 1.7603575888e-01f, 1.2098748977e-01f, 6.4759936605e-02f, 2.6995957967e-02f, 8.7643043628e-03f, 2.2159631726e-03f, 4.3634902051e-04f, 6.6916289573e-05f},
    {4.4611342773e-05f, 1.6009476589e-04f, 5.1410760582e-04f, 1.4773247784e-03f, 3.7987699403e-03f, 8.7408780477e-03f, 1.7997500191e-02f, 3.3159988421e-02f, 5.4671578246e-02f, 8.0659199899e-02f, 1.0648569403e-01f, 1.2579798346e-01f, 1.3298453855e-01f, 1.2579798346e-01f, 1.0648569403e-01f, 8.0659199899e-02f, 5.4671578246e-02f, 3.3159988421e-02f, 1.7997500191e-02f, 8.7408780477e-03f, 3.7987699403e-03f, 1.4773247784e-03f, 5.1410760582e-04f, 1.6009476589e-04f, 4.4611342773e-05f},
    {},
    {},
    {1.1002004494e-02f, 4.3175145022e-02f, 1.1464351944e-01f, 2.0597709699e-01f, 2.5040446811e-01f, 2.0597709699e-01f, 1.1464351944e-01f, 4.3175145022e-02f, 1.1002004494e-02f},
    {6.0386507574e-03f, 1.1052150749e-02f, 1.8839549938e-02f, 2.9909633001e-02f, 4.4225065166e-02f, 6.0903565120e-02f, 7.8114883030e-02f, 9.3312901705e-02f, 1.0381651342e-01f, 1.0757417423e-01f, 1.0381651342e-01f, 9.3312901705e-02f, 7.8114883030e-02f, 6.0903565120e-02f, 4.4225065166e-02f, 2.9909633001e-02f, 1.8839549938e-02f, 1.1052150749e-02f, 6.0386507574e-03f},
    {-1.7190632022e-02f, -5.0595873072e-02f, -8.9565249560e-02f, -8.0459803512e-02f, 0.0000000000e+00f, 8.0459803512e-02f, 8.9565249560e-02f, 5.0595873072e-02f, 1.7190632022e-02f},
    {-3.8647364847e-03f, -6.2874457591e-03f, -9.3779093026e-03f, -1.2761443414e-02f, -1.5724467615e-02f, -1.7323680745e-02f, -1.6664508380e-02f, -1.3271168242e-02f, -7.3825076209e-03f, 0.0000000000e+00f, 7.3825076209e-03f, 1.3271168242e-02f, 1.6664508380e-02f, 1.7323680745e-02f, 1.5724467615e-02f, 1.2761443414e-02f, 9.3779093026e-03f, 6.2874457591e-03f, 3.8647364847e-03f},
};

struct Plane {
    const void* pixels;
    size_t pitch;
    int is_float;
    __device__ float get(int y, int x) const {
        const auto* row = static_cast<const unsigned char*>(pixels) + y * pitch;
        return is_float ? reinterpret_cast<const float*>(row)[x] : row[x];
    }
};
struct Inputs { Plane plane[4]; };

__host__ __device__ int min_i(int a, int b) { return a < b ? a : b; }
__host__ __device__ int max_i(int a, int b) { return a > b ? a : b; }
__host__ __device__ int tiles(int n, int tile) {
    const int last = max_i(0, n - tile);
    return 1 + (last + tile / 2 - 1) / (tile / 2);
}
__host__ __device__ int start(int i, int n, int tile) {
    return min_i(i * (tile / 2), max_i(0, n - tile));
}
__device__ int clamp_i(int x, int n) { return max_i(0, min_i(n - 1, x)); }
__device__ int reflect(int x, int n) {
    if (x >= 0 && x < n) return x;
    x %= 2 * n;
    if (x < 0) x += 2 * n;
    return x < n ? x : 2 * n - x - 1;
}
__device__ float clip(float x, float low, float high) { return fminf(high, fmaxf(low, x)); }

struct Tile {
    int first, top, bottom;
    unsigned short runs[NC];
};
struct AX700JointWorkspace {
    int h, w, q, nx, ny, nt, workers;
    float *sampled, *dy, *dx, *sum, *weight;
    double* chunks;
    unsigned char *contour, *band, *near, *seam;
    unsigned* votes;
    int *active, *count;
    Tile *proposal, *stable;
    float* scratch;
};

template<class T>
T* reserve(uintptr_t base, size_t& bytes, size_t n) {
    bytes = (bytes + 255) & ~size_t(255);
    auto* p = reinterpret_cast<T*>(base + bytes);
    bytes += n * sizeof(T);
    return p;
}

AX700JointWorkspace layout(void* raw, int h, int w, size_t& bytes) {
    AX700JointWorkspace s{};
    s.h = h; s.w = w; s.q = (w + 3) / 4;
    s.nx = tiles(w, TW); s.ny = tiles(h, TH); s.nt = s.nx * s.ny;
    s.workers = min_i(WORKERS, s.nt);
    const size_t nq = size_t(h) * s.q, n = size_t(h) * w;
    const auto base = reinterpret_cast<uintptr_t>(raw);
    bytes = 0;
    s.active = reserve<int>(base, bytes, s.nt);
    s.count = reserve<int>(base, bytes, 1);
    s.stable = reserve<Tile>(base, bytes, s.nt);
    s.sum = reserve<float>(base, bytes, n);
    s.weight = reserve<float>(base, bytes, n);
    size_t repair_bytes = bytes;
    s.sampled = reserve<float>(base, bytes, nq);
    s.dy = reserve<float>(base, bytes, nq);
    s.dx = reserve<float>(base, bytes, nq);
    s.chunks = reserve<double>(base, bytes, size_t(h) * (s.nx + 2));
    s.contour = reserve<unsigned char>(base, bytes, nq);
    s.band = reserve<unsigned char>(base, bytes, nq);
    s.near = reserve<unsigned char>(base, bytes, nq);
    s.seam = reserve<unsigned char>(base, bytes, nq);
    s.votes = reserve<unsigned>(base, bytes, (nq + 3) / 4);
    s.proposal = reserve<Tile>(base, bytes, s.nt);
    // Only stable tile metadata survives into repair.
    s.scratch = reserve<float>(base, repair_bytes, size_t(s.workers) * SCRATCH);
    if (repair_bytes > bytes) bytes = repair_bytes;
    return s;
}

__global__ void sample(Plane in, AX700JointWorkspace s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= s.h * s.q) return;
    int y = i / s.q, x = 4 * (i % s.q);
    float value = float_filters[G1][4] * in.get(y, x);
    #pragma unroll
    for (int j = 1; j <= 4; ++j)
        value += float_filters[G1][j + 4] * (in.get(y, reflect(x + j, s.w)) + in.get(y, reflect(x - j, s.w)));
    s.sampled[i] = value;
}

__global__ void gradient_y(AX700JointWorkspace s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= s.h * s.q) return;
    int y = i / s.q, x = i % s.q;
    float gy = 0, gx = float_filters[G16][4] * s.sampled[i];
    #pragma unroll
    for (int j = 1; j <= 4; ++j) {
        float lo = s.sampled[reflect(y - j, s.h) * s.q + x];
        float hi = s.sampled[reflect(y + j, s.h) * s.q + x];
        gy += float_filters[D16][j + 4] * (hi - lo);
        gx += float_filters[G16][j + 4] * (hi + lo);
    }
    s.dy[i] = gy;
    s.dx[i] = gx;
}

__global__ void gradient_x(AX700JointWorkspace s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= s.h * s.q) return;
    int row = i / s.q * s.q, x = i % s.q;
    float gy = float_filters[G375][9] * s.dy[i], gx = 0;
    #pragma unroll
    for (int j = 1; j <= 9; ++j) {
        int lo = row + reflect(x - j, s.q), hi = row + reflect(x + j, s.q);
        gy += float_filters[G375][j + 9] * (s.dy[lo] + s.dy[hi]);
        gx += float_filters[D375][j + 9] * (s.dx[hi] - s.dx[lo]);
    }
    const float a = fabsf(gy), b = fabsf(gx) / 4;
    s.contour[i] = a >= 1.75f && a > 2 * b;
}

__global__ void dilate(AX700JointWorkspace s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= s.h * s.q) return;
    int y = i / s.q, x = i % s.q;
    bool band = false, near = false;
    for (int j = -13; j <= 13; ++j) {
        bool on = s.contour[clamp_i(y + j, s.h) * s.q + x];
        near |= on;
        band |= on && abs(j) <= 8;
    }
    s.band[i] = band;
    s.near[i] = near;
}

template<int N, int FILTER>
__device__ float jump(const float* row, int x) {
    float sum = 0;
    #pragma unroll
    for (int j = 0; j < N; ++j)
        sum += repair_filters[FILTER][j] * (float(row[x + j]) - row[x - 1 - j]);
    return sum;
}

template<int N>
__device__ int integer_jump(const unsigned short* row, int column) {
    constexpr int w4[]{17, 9, 1, -7};
    constexpr int w6[]{125, 89, 53, 17, -19, -55};
    constexpr int w8[]{77, 61, 45, 29, 13, -3, -19, -35};
    int value = 0;
    #pragma unroll
    for (int d = 0; d < N; ++d)
        value += (N == 4 ? w4[d] : N == 6 ? w6[d] : w8[d]) *
            (int(row[column + d]) - row[column - 1 - d]);
    return value;
}

// Rolling sums share all overlapping detector windows, including the reference
// flanks. Keep the four planes local to the block.
template<bool U8>
__global__ void is_seam(Inputs input, AX700JointWorkspace s) {
    constexpr int COLS = 32, ROWS = 8, PITCH = 4 * COLS + 16;
    using Window = typename std::conditional<U8, unsigned short, float>::type;
    using Accumulator = typename std::conditional<U8, int, float>::type;
    __shared__ Window sums[4][ROWS][PITCH];
    int x0 = blockIdx.x * COLS * 4, y0 = blockIdx.y * ROWS;
    int lane = threadIdx.x;
    for (int i = lane; i < 4 * PITCH; i += THREADS) {
        const int plane = i / PITCH, local_x = i % PITCH;
        const int x = x0 - 8 + local_x;
        if (x < 0 || x >= s.w) continue;
        Plane src = input.plane[plane];
        if constexpr (U8) src.is_float = 0;
        Accumulator value = 0;
        #pragma unroll
        for (int r = 0; r < 8; ++r) value += Accumulator(src.get(y0 + r, x));
        sums[plane][0][local_x] = U8 ? Window(value) : Window(value / 8.0f);
        for (int r = 1; r < ROWS && y0 + r < s.h - 7; ++r) {
            value += Accumulator(src.get(y0 + r + 7, x)) - Accumulator(src.get(y0 + r - 1, x));
            sums[plane][r][local_x] = U8 ? Window(value) : Window(value / 8.0f);
        }
    }
    __syncthreads();
    int x = x0 + (lane % COLS) * 4, y = y0 + lane / COLS;
    if (x >= s.w || y >= s.h - 7) return;
    bool selected = false;
    if (x >= 10 && x < s.w - 10) {
        const Window* a = sums[0][lane / COLS];
        int column = 8 + (lane % COLS) * 4;
        if constexpr (U8) {
            int central = integer_jump<6>(a, column), magnitude = abs(central);
            if (magnitude >= 6384) {
                int sign = central >= 0 ? 1 : -1;
                if (sign * integer_jump<4>(a, column) > 0 && sign * integer_jump<8>(a, column) > 0) {
                    int virtual_max = max_i(max_i(abs(integer_jump<6>(a, column - 2)), abs(integer_jump<6>(a, column - 1))),
                                            max_i(abs(integer_jump<6>(a, column + 1)), abs(integer_jump<6>(a, column + 2))));
                    // Convert the 210-denominator weights and 8-row mean
                    // into exact integer cross products before comparison.
                    selected = 1250 * magnitude >= 1475 * virtual_max + 24780;
                    for (int p = 1; p < 4 && selected; ++p)
                        selected = 5 * sign * integer_jump<6>(sums[p][lane / COLS], column) < 4 * magnitude;
                }
            }
        } else {
            float central = jump<6, J6>(a, column);
            float magnitude = fabsf(central);
            if (magnitude >= 3.8f && central * jump<4, J4>(a, column) > 0 &&
                central * jump<8, J8>(a, column) > 0) {
                float virtual_max = fmaxf(fmaxf(fabsf(jump<6, J6>(a, column - 2)), fabsf(jump<6, J6>(a, column - 1))),
                                          fmaxf(fabsf(jump<6, J6>(a, column + 1)), fabsf(jump<6, J6>(a, column + 2))));
                selected = magnitude >= 1.18f * (virtual_max + 0.01f);
                for (int p = 1; p < 4 && selected; ++p)
                    selected = copysignf(1.0f, central) * jump<6, J6>(sums[p][lane / COLS], column) < 0.8f * magnitude;
            }
        }
    }
    s.seam[y * s.q + x / 4] = selected;
}

// Sum 80-column chunks. The edge-aligned final tile gets two
// independent chunks when its start differs from the regular grid.
__global__ void horizontal_chunks(Plane plane, AX700JointWorkspace s) {
    const int y = blockIdx.x, lane = threadIdx.x % 32;
    const int last = start(s.nx - 1, s.w, TW);
    const bool tail = last % (TW / 2) != 0;
    const int count = s.nx + 1 + int(tail);
    for (int c = threadIdx.x / 32; c < count; c += THREADS / 32) {
        const int begin = tail && c >= s.nx - 1
            ? (c == s.nx - 1 ? c * (TW / 2) : last + (c - s.nx) * (TW / 2))
            : c * (TW / 2);
        const int end = min_i(s.w, begin + TW / 2);
        double value = 0;
        for (int x = begin + lane; x < end; x += 32) value += plane.get(y, x);
        for (int d = 16; d; d /= 2) value += __shfl_down_sync(0xffffffff, value, d);
        if (!lane) s.chunks[size_t(y) * (s.nx + 2) + c] = value;
    }
}

// Convert selected 8-row windows to disjoint span runs, exactly as in the CPU scan.
__device__ bool next_run(unsigned& mask, int first, int& a, int& b, unsigned& group) {
    if (!mask) return false;
    int begin = __ffs(mask) - 1, end = begin;
    group = 0;
    do {
        end = __ffs(mask) - 1;
        group |= 1u << end;
        mask &= mask - 1;
    } while (mask && __ffs(mask) - 1 <= end + 2);
    a = first + 4 * begin; b = first + 4 * end + 8;
    return true;
}

__global__ void locate(AX700JointWorkspace s) {
    int t = blockIdx.x, lane = threadIdx.x;
    int ox = start(t % s.nx, s.w, TW), oy = start(t / s.nx, s.h, TH);
    int w = min_i(s.w, TW), h = min_i(s.h, TH), q0 = ox / 4, q1 = (ox + w + 3) / 4;
    __shared__ int has_guide;
    __shared__ double mean[TH], smooth[TH];
    Tile& proposal = s.proposal[t];
    if (lane < NC) proposal.runs[lane] = 0;
    if (!lane) { has_guide = 0; proposal.first = 0; proposal.top = 0; proposal.bottom = 0; }
    if (w < 20 || h < 20) return;
    __syncthreads();
    for (int i = lane; i < h * (q1 - q0); i += blockDim.x) {
        if (s.band[(oy + i / (q1 - q0)) * s.q + q0 + i % (q1 - q0)]) {
            atomicOr(&has_guide, 1);
            break;
        }
    }
    __syncthreads();
    if (!has_guide) return;
    if (lane < h) {
        auto* row = s.chunks + size_t(oy + lane) * (s.nx + 2);
        const int c = t % s.nx + int(t % s.nx == s.nx - 1 && ox % (TW / 2) != 0);
        mean[lane] = (row[c] + row[c + 1]) / w;
    }
    __syncthreads();
    if (lane < h) {
        double v = 0;
        for (int d = -4; d <= 4; ++d) v += locator_gaussian[d + 4] * mean[reflect(lane + d, h)];
        smooth[lane] = v;
    }
    __syncthreads();
    if (!lane) {
        double peak = 0;
        int center = 0;
        for (int y = 0; y < h; ++y) {
            double g = fabs(smooth[min_i(h - 1, y + 1)] - smooth[max_i(0, y - 1)]);
            if (y && y < h - 1) g *= 0.5;
            if (g > peak) { peak = g; center = y; }
        }
        if (peak >= 0.5) {
            proposal.top = max_i(0, center - 19);
            proposal.bottom = min_i(h, center + 19);
            proposal.first = max_i(0, proposal.top - 5);
        }
    }
    __syncthreads();
    if (lane >= NC || !proposal.bottom) return;
    int x = ((ox + 13) & ~3) + 4 * lane;
    if (x >= ox + w - 10) return;
    unsigned bits = 0;
    for (int y = proposal.first, j = 0; y < min_i(h - 7, proposal.bottom + 2); y += 4, ++j)
        bits |= unsigned(s.seam[(oy + y) * s.q + x / 4]) << j;
    proposal.runs[lane] = bits;
    int a, b;
    unsigned group;
    while (next_run(bits, proposal.first, a, b, group))
        for (int y = a; y < b; ++y) {
            const int i = (oy + y) * s.q + x / 4;
            // Half-overlap tiles contribute at most nine votes per sample.
            atomicAdd(s.votes + i / 4, 1u << (8 * (i % 4)));
        }
}

__device__ int coverage(int p, int n, int tile, int margin) {
    int last = max_i(0, n - tile), stride = tile / 2, length = min_i(n, tile);
    int hi = min_i(last / stride, (p - margin) / stride);
    int lo = max_i(0, (p - (length - margin) + stride) / stride);
    int scans = p >= margin ? max_i(0, hi - lo + 1) : 0;
    if (last % stride && p >= last + margin && p < last + length - margin) ++scans;
    return scans;
}

__global__ void corroborate(AX700JointWorkspace s) {
    int t = blockIdx.x, lane = threadIdx.x;
    int ox = start(t % s.nx, s.w, TW), oy = start(t / s.nx, s.h, TH);
    const Tile& proposal = s.proposal[t];
    Tile& stable = s.stable[t];
    __shared__ int any;
    if (!lane) {
        any = 0;
        stable.first = proposal.first;
        stable.top = proposal.top;
        stable.bottom = proposal.bottom;
    }
    __syncthreads();
    if (lane < NC) {
        unsigned accepted = 0, bits = proposal.runs[lane], group;
        int x = ((ox + 13) & ~3) + 4 * lane, a, b;
        while (next_run(bits, proposal.first, a, b, group)) {
            int corroborated = 0, xv = coverage(x, s.w, TW, 10);
            for (int y = oy + a; y < oy + b; ++y) {
                int i = y * s.q + x / 4;
                corroborated += s.near[i] && int((s.votes[i / 4] >> (8 * (i % 4))) & 255u) >= min_i(2, xv * coverage(y, s.h, TH, 0));
            }
            if (corroborated >= 4) accepted |= group;
        }
        stable.runs[lane] = accepted;
        if (accepted) atomicOr(&any, 1);
    }
    __syncthreads();
    if (!lane && any) s.active[atomicAdd(s.count, 1)] = t;
}

__device__ float response(int distance) {
    switch (abs(distance)) {
        case 0: return 1.1276756959619711f;
        case 4: return -0.48513370169476444f;
        case 8: return 0.063837847980985657f;
        case 12: return 0.011469346423450747f;
        default: return 0;
    }
}

template<int B>
__device__ void factor(float (*band)[B + 1], int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = max_i(0, i - B); j <= i; ++j) {
            float v = band[i][i - j];
            for (int k = max_i(0, i - B); k < j; ++k)
                v -= band[i][i - k] * band[j][j - k];
            band[i][i - j] = i == j ? sqrtf(v) : v / band[j][0];
        }
    }
}

template<int B>
__device__ void solve(const float (*band)[B + 1], float* rhs, int n) {
    for (int i = 0; i < n; ++i) {
        for (int k = max_i(0, i - B); k < i; ++k) rhs[i] -= band[i][i - k] * rhs[k];
        rhs[i] /= band[i][0];
    }
    for (int i = n - 1; i >= 0; --i) {
        for (int k = i + 1; k < min_i(i + B + 1, n); ++k) rhs[i] -= band[k][k - i] * rhs[k];
        rhs[i] /= band[i][0];
    }
}

struct RepairShared {
    int columns[NC], lookup[TW], envelope_index[NC], n, direction, good;
    float center;
    unsigned char envelopes[NC][TH];
    union { float observed[TH][NC]; float projection[TH][NC]; };
    float coefficients[TH][NC];
    float normal[NC][7];
    float position[TW], confidence[TW], weights[TW], curve[TW], residual[TW];
    float smooth[TW], dx[TW], dy[TW], sorted[THREADS], median, mad;
    float gradient[TH];
    unsigned long long row_mask[TH];
};

__device__ float correction(int x, const float* coeff, const RepairShared& z) {
    float v = 0;
    for (int c = z.lookup[x]; c < z.n && z.columns[c] <= x + 8; ++c)
        v += coeff[c] * float(repair_filters[BUMP][x - z.columns[c] + 8]);
    return v;
}

__device__ float nearest_filter(const float* v, int x, int w, int f, int radius) {
    float result = 0;
    for (int d = -radius; d <= radius; ++d)
        result += repair_filters[f][d + radius] * v[clamp_i(x + d, w)];
    return result;
}

__device__ void median_sort(float value, float* sorted, int lane) {
    for (int span = 2; span <= THREADS; span *= 2) {
        for (int step = span / 2; step; step /= 2) {
            float other;
            if (step < 32) {
                other = __shfl_xor_sync(0xffffffff, value, step);
            } else {
                sorted[lane] = value;
                __syncthreads();
                other = sorted[lane ^ step];
                __syncthreads();
            }
            bool upper = bool(lane & span) != bool(lane & step);
            value = upper ? fmaxf(value, other) : fminf(value, other);
        }
    }
    sorted[lane] = value;
    __syncthreads();
}

__device__ float profile_gradient(const float* guide, int y, int x, int h) {
    return (guide[min_i(y + 1, h - 1) * TW + x] - guide[max_i(y - 1, 0) * TW + x])
        * (y == 0 || y == h - 1 ? 1.f : 0.5f);
}

// Inverse cubic B-spline filter. Apply the symmetric taps to
// differences from the center to preserve constant input exactly.
__constant__ float spline_taps[FIR_RADIUS] = {-4.6410161514e-01f, 1.2435565298e-01f, -3.3320996791e-02f, 8.9283341811e-03f, -2.3923399336e-03f, 6.4102555322e-04f, -1.7176227931e-04f, 4.6023564032e-05f, -1.2331976815e-05f, 3.3043432287e-06f, -8.8539609965e-07f, 2.3724116988e-07f};

__device__ void cubic_weights(float f, float* out) {
    float g = 1 - f;
    out[0] = g*g*g / 6;
    out[1] = (4 - 6*f*f + 3*f*f*f) / 6;
    out[2] = (4 - 6*g*g + 3*g*g*g) / 6;
    out[3] = f*f*f / 6;
}

__device__ float spline_offset(const float* s, int x, int y, float dx, float dy, float baseline) {
    int sx = int(floorf(dx)), sy = int(floorf(dy));
    int ix = x + sx, iy = y + sy;
    float wx[4], wy[4];
    cubic_weights(dx - sx, wx);
    cubic_weights(dy - sy, wy);
    float out = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j)
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            out += wx[i] * wy[j] * (s[(iy - 1 + j) * PW + ix - 1 + i] - baseline);
    return out;
}

__device__ float horizontal_low(const float* vertical, int y, int x, int w) {
    float sum = repair_filters[G1][4] * vertical[y * TW + x];
    #pragma unroll
    for (int d = 1; d <= 4; ++d)
        sum += repair_filters[G1][d + 4] *
            (float(vertical[y * TW + reflect(x - d, w)]) + vertical[y * TW + reflect(x + d, w)]);
    return sum;
}

template<bool FULL>
__global__ void repair(Plane plane, AX700JointWorkspace s) {
    __shared__ RepairShared z;
    const int lane = threadIdx.x, w = FULL ? TW : min_i(s.w, TW), h = FULL ? TH : min_i(s.h, TH);
    float* corrected = s.scratch + size_t(blockIdx.x) * SCRATCH;
    float* vertical = corrected + AREA;
    float* low = vertical + AREA;
    for (int work = blockIdx.x; work < *s.count; work += gridDim.x) {
        const int t = s.active[work];
        const Tile& tile = s.stable[t];
        int ox = start(t % s.nx, s.w, TW), oy = start(t / s.nx, s.h, TH);
        if (!lane) {
            z.n = 0;
            z.center = plane.get(oy, ox);
            for (int i = 0; i < NC; ++i) if (tile.runs[i]) {
                z.envelope_index[z.n] = i;
                z.columns[z.n++] = ((ox + 13) & ~3) - ox + 4 * i;
            }
            z.good = 0;
        }
        __syncthreads();
        int n = z.n;
        if (lane < w) {
            int c = 0;
            while (c < n && z.columns[c] + 8 <= lane) ++c;
            z.lookup[lane] = c;
        }
        for (int i = lane; i < n * h; i += THREADS) {
            const int c = i / h, y = i % h;
            unsigned bits = tile.runs[z.envelope_index[c]], group;
            int a, b, e = 0;
            while (next_run(bits, tile.first, a, b, group))
                e = max_i(e, max_i(0, min_i(8, min_i(y - a + 8, b + 8 - y))));
            z.envelopes[c][y] = e;
            const int x = ox + z.columns[c];
            float j = 0;
            for (int d = 0; d < 6; ++d)
                j += repair_filters[J6][d] * (float(plane.get(oy + y, x + d)) - plane.get(oy + y, x - 1 - d));
            z.observed[y][c] = float(j);
        }
        for (int i = lane; i < n * 7; i += THREADS) {
            int c = i / 7, d = i % 7, b = c - d;
            if (b < 0) continue;
            float val = d == 0 ? 0.05f : 0;
            for (int k = max_i(0, c - 3); k < min_i(n, b + 4); ++k)
                val += response(z.columns[c] - z.columns[k]) * response(z.columns[b] - z.columns[k]);
            z.normal[c][d] = val;
        }
        __syncthreads();
        if (!lane) factor<6>(z.normal, n);
        // Hold filtered targets separately until all readers have finished.
        for (int i = lane; i < n * h; i += THREADS) {
            int y = i / n, c = i % n;
            float v = repair_filters[G3][12] * z.observed[y][c];
            #pragma unroll
            for (int d = 1; d <= 12; ++d)
                v += repair_filters[G3][d + 12] *
                    (float(z.observed[clamp_i(y - d, h)][c]) + z.observed[clamp_i(y + d, h)][c]);
            z.coefficients[y][c] = copysignf(clip(fabsf(v) - 0.5f, 0, 8), v) * (z.envelopes[c][y] / 8.f);
        }
        __syncthreads();
        if (lane < h) {
            float rhs[NC];
            for (int i = 0; i < n; ++i) {
                rhs[i] = 0;
                for (int k = max_i(0, i - 3); k < min_i(n, i + 4); ++k)
                    rhs[i] -= response(z.columns[i] - z.columns[k]) * z.coefficients[lane][k];
            }
            solve<6>(z.normal, rhs, n);
            for (int i = 0; i < n; ++i) z.coefficients[lane][i] = z.envelopes[i][lane] ? rhs[i] : 0;
        }
        __syncthreads();
        for (int i = lane; i < h * w; i += THREADS) {
            int y = i / w, x = i % w;
            corrected[y * TW + x] = (plane.get(oy + y, ox + x) - z.center) + correction(x, z.coefficients[y], z);
        }
        __syncthreads();
        for (int i = lane; i < h * w; i += THREADS) {
            int y = i / w, x = i % w;
            float value = repair_filters[G1][4] * corrected[y * TW + x];
            #pragma unroll
            for (int d = 1; d <= 4; ++d)
                value += repair_filters[G1][d + 4] *
                    (float(corrected[reflect(y - d, h) * TW + x]) + corrected[reflect(y + d, h) * TW + x]);
            vertical[y * TW + x] = float(value);
        }
        __syncthreads();
        const int top = tile.top, bottom = tile.bottom;
        int begin = max_i(0, top - 1), end = min_i(h, bottom + 1);
        for (int i = lane; i < (end - begin) * w; i += THREADS) {
            int y = begin + i / w, x = i % w;
            float value = repair_filters[G2][8] * vertical[y * TW + x];
            #pragma unroll
            for (int d = 1; d <= 8; ++d)
                value += repair_filters[G2][d + 8] *
                    (float(vertical[y * TW + reflect(x - d, w)]) + vertical[y * TW + reflect(x + d, w)]);
            low[y * TW + x] = float(value);
        }
        __syncthreads();
        for (int y = top + lane / 32; y < bottom; y += THREADS / 32) {
            float value = 0;
            for (int x = lane % 32; x < w; x += 32) value += profile_gradient(low, y, x, h);
            for (int d = 16; d; d /= 2) value += __shfl_down_sync(0xffffffff, value, d);
            if (lane % 32 == 0) z.gradient[y] = value;
        }
        __syncthreads();
        if (!lane) {
            float strongest = 0, weakest = 0;
            for (int y = top; y < bottom; ++y) {
                strongest = fmaxf(strongest, z.gradient[y]);
                weakest = fminf(weakest, z.gradient[y]);
            }
            z.direction = strongest >= -weakest ? 1 : -1;
        }
        __syncthreads();
        if (lane < w) {
            float strength = -CUDART_INF_F;
            int peak = top;
            for (int y = top; y < bottom; ++y) {
                float v = z.direction * profile_gradient(low, y, lane, h);
                if (v > strength) { strength = v; peak = y; }
            }
            int j = max_i(top + 1, min_i(bottom - 2, peak));
            float left = z.direction * profile_gradient(low, j - 1, lane, h);
            float mid = z.direction * profile_gradient(low, j, lane, h);
            float right = z.direction * profile_gradient(low, j + 1, lane, h);
            float curvature = left - 2 * mid + right;
            float sub = fabsf(curvature) > 1e-8f ? 0.5f * (left - right) / curvature : 0;
            z.position[lane] = (j - top) + fminf(0.5f, fmaxf(-0.5f, sub));
            float confidence = j <= top + 1 || j >= bottom - 2 ? 0 : clip((strength - 1.5f) / 2, 0, 1);
            z.confidence[lane] = z.weights[lane] = confidence;
            if (confidence > 0) atomicAdd(&z.good, 1);
        }
        __syncthreads();
        const bool geometry = z.good >= max_i(12, w / 2);
        if (geometry) {
            for (int iter = 0; iter < 4; ++iter) {
                if (lane < w) {
                    float num = 0, den = 0;
                    for (int d = -48; d <= 48; ++d) {
                        int c = clamp_i(lane + d, w);
                        float weight = repair_filters[G12][d + 48] * z.weights[c];
                        num += weight * z.position[c];
                        den += weight;
                    }
                    z.curve[lane] = num / fmaxf(den, 1e-12f);
                    z.residual[lane] = z.position[lane] - z.curve[lane];
                }
                __syncthreads();
                bool good = lane < w && z.confidence[lane] > 0;
                median_sort(good ? z.residual[lane] : CUDART_INF_F, z.sorted, lane);
                if (!lane) z.median = 0.5f * (z.sorted[(z.good - 1) / 2] + z.sorted[z.good / 2]);
                __syncthreads();
                median_sort(good ? fabsf(z.residual[lane] - z.median) : CUDART_INF_F, z.sorted, lane);
                if (!lane) z.mad = fmaxf(0.35f, 1.4826f * 0.5f *
                    (z.sorted[(z.good - 1) / 2] + z.sorted[z.good / 2]));
                __syncthreads();
                if (lane < w) z.weights[lane] = z.confidence[lane] *
                    fminf(1.0f, 1.5f * z.mad / fmaxf(fabsf(z.residual[lane]), 1e-8f));
                __syncthreads();
            }
            if (lane < w) {
                z.residual[lane] = (z.curve[lane] - z.position[lane]) * z.confidence[lane];
                z.smooth[lane] = nearest_filter(z.position, lane, w, G2, 8);
            }
            __syncthreads();
            if (lane < w) {
                float motion = nearest_filter(z.residual, lane, w, G15, 6);
                motion = fmaxf(-1.25f, fminf(1.25f, motion)) * fminf(1.0f, min_i(lane, w - 1 - lane) / 8.f);
                float slope = z.smooth[min_i(w - 1, lane + 1)] - z.smooth[max_i(0, lane - 1)];
                if (lane && lane < w - 1) slope *= 0.5f;
                z.dy[lane] = motion / (1 + slope * slope);
                z.dx[lane] = -slope * z.dy[lane];
            }
        }
        const int band_begin = max_i(0, top - 10);
        const int band_end = min_i(h, bottom + 10);
        const int spline_begin = band_begin - PAD;
        const int spline_rows = band_end - band_begin + 2 * PAD;
        if (geometry) {
            const int pw = w + 2 * PAD;
            const int low_begin = max_i(0, spline_begin - FIR_RADIUS);
            const int low_end = min_i(h, spline_begin + spline_rows + FIR_RADIUS);
            for (int i = lane; i < (low_end - low_begin) * w; i += THREADS) {
                const int y = low_begin + i / w, x = i % w;
                low[y * TW + x] = horizontal_low(vertical, y, x, w);
            }
            __syncthreads();
            for (int i = lane; i < (low_end - low_begin) * pw; i += THREADS) {
                const int y = i / pw, x = i % pw - PAD;
                const float* row = low + (y + low_begin) * TW;
                const float center = row[clamp_i(x, w)];
                float delta = 0;
                #pragma unroll
                for (int d = 1; d <= FIR_RADIUS; ++d)
                    delta += spline_taps[d - 1] * ((row[clamp_i(x - d, w)] - center)
                                                  + (row[clamp_i(x + d, w)] - center));
                corrected[y * PW + x + PAD] = center + delta;
            }
            __syncthreads();
            for (int i = lane; i < spline_rows * pw; i += THREADS) {
                const int y = spline_begin + i / pw, x = i % pw;
                const float center = corrected[(clamp_i(y, h) - low_begin) * PW + x];
                float delta = 0;
                #pragma unroll
                for (int d = 1; d <= FIR_RADIUS; ++d)
                    delta += spline_taps[d - 1] *
                        ((corrected[(clamp_i(y - d, h) - low_begin) * PW + x] - center)
                         + (corrected[(clamp_i(y + d, h) - low_begin) * PW + x] - center));
                vertical[(y - spline_begin) * PW + x] = center + delta;
            }
            __syncthreads();
        }
        for (int i = lane; i < h * w; i += THREADS) {
            int y = i / w, x = i % w;
            float g = 0;
            if (geometry && y >= top - 10 && y < bottom + 10 && z.dy[x] != 0) {
                int support = 0;
                for (int c = z.lookup[x]; c < n && z.columns[c] <= x + 8; ++c) {
                    int edge = (x - z.columns[c]) * 2 + 1;
                    support = max_i(support, min_i(2, z.envelopes[c][y]) * min_i(4, 16 - abs(edge)));
                }
                float taper = clip((9 - fabsf((y - top) - z.position[x])) / 4, 0, 1);
                if (support && taper) {
                    float delta = spline_offset(vertical, x + PAD, y - spline_begin,
                                            -z.dx[x], -z.dy[x], low[y * TW + x]);
                    // Subtract the zero-motion FIR reconstruction term so
                    // truncation and FP32 DC error cannot introduce motion.
                    delta -= spline_offset(vertical, x + PAD, y - spline_begin,
                                       0, 0, low[y * TW + x]);
                    g = delta * taper * (support / 8.f);
                }
            }
            corrected[y * TW + x] = g;
        }
        __syncthreads();
        // Consecutive rows with the same positive envelope share one factor.
        // The low-pass and spline planes are free after confinement.
        auto* projection_factor = reinterpret_cast<float (*)[4]>(vertical);
        auto* active = reinterpret_cast<int*>(low);
        unsigned long long mask = 0;
        if (lane < h && geometry) {
            for (int i = 0; i < n; ++i)
                if (z.envelopes[i][lane]) mask |= 1ULL << i;
            z.row_mask[lane] = mask;
        }
        __syncthreads();
        if (lane < h && mask && (!lane || mask != z.row_mask[lane - 1])) {
            int* columns = active + lane * NC;
            auto* band = projection_factor + lane * NC;
            int k = 0;
            for (auto bits = mask; bits; bits &= bits - 1) columns[k++] = __ffsll(bits) - 1;
            for (int i = 0; i < k; ++i)
                for (int d = 0; d <= min_i(3, i); ++d)
                    band[i][d] = response(z.columns[columns[i]] - z.columns[columns[i - d]]);
            factor<3>(band, k);
        }
        __syncthreads();
        if (lane < h) {
            float* row = z.projection[lane];
            for (int i = 0; i < n; ++i) row[i] = 0;
            if (mask) {
                int leader = lane;
                while (leader && z.row_mask[leader - 1] == mask) --leader;
                const int* columns = active + leader * NC;
                const int k = __popcll(mask);
                bool nonzero = false;
                for (int i = 0; i < k; ++i) {
                    row[i] = -jump<6, J6>(corrected + lane * TW, z.columns[columns[i]]);
                    nonzero |= row[i] != 0;
                }
                if (nonzero) solve<3>(projection_factor + leader * NC, row, k);
                for (int i = k - 1; i >= 0; --i) {
                    float coefficient = row[i];
                    row[i] = 0;
                    row[columns[i]] = coefficient;
                }
            }
        }
        __syncthreads();
        for (int i = lane; i < h * w; i += THREADS) {
            int y = i / w, x = i % w;
            float delta = correction(x, z.coefficients[y], z) + corrected[y * TW + x]
                + correction(x, z.projection[y], z);
            // Suppress FP32 zero-motion noise before adding blend weight.
            if (fabsf(delta) > 1e-7f) {
                float wx = fminf(1.f, min_i(x + 1, w - x) / 24.f);
                float wy = fminf(1.f, min_i(y + 1, h - y) / 20.f);
                int index = (oy + y) * s.w + ox + x;
                atomicAdd(s.sum + index, delta * (wx * wy));
                atomicAdd(s.weight + index, wx * wy);
            }
        }
        __syncthreads();

    }
}

template<bool Wide>
__global__ void writePlane(Plane source, const float* sum, const float* weight,
                           uint8_t* output, size_t pitch, int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int i = y * width + x;
    const float raw = source.get(y, x);
    const float value = weight && weight[i] > 0
        ? raw + clip(sum[i] / weight[i], -8, 8) : raw;
    auto* row = output + y * pitch;
    if constexpr (Wide) {
        reinterpret_cast<uint16_t*>(row)[x] = min_i(1023, max_i(0, lrintf(value * 4.f))) << 6;
    } else {
        row[x] = min_i(255, max_i(0, lrintf(value)));
    }
}

int blocks(int count, int threads = THREADS) { return (count + threads - 1) / threads; }
} // namespace

namespace bseam {
size_t workspaceBytes(int height, int width) {
    size_t bytes;
    layout(nullptr, height, width, bytes);
    return bytes;
}

cudaError_t write(const uint8_t* source, size_t sourcePitch, uint8_t* output,
                  size_t outputPitch, int width, int height, cudaStream_t stream) {
    const dim3 block(32, 8), grid(blocks(width, 32), blocks(height, 8));
    writePlane<true><<<grid, block, 0, stream>>>(
        {source, sourcePitch, 0}, nullptr, nullptr, output, outputPitch, width, height);
    return cudaGetLastError();
}

cudaError_t repairFrame(void* memory, int h, int w, const uint8_t* const* planes,
                        const size_t* pitches, uint8_t* output, size_t outputPitch,
                        bool wide, cudaStream_t stream) {
    size_t bytes;
    auto s = layout(memory, h, w, bytes);
    Inputs input{};
    for (int i = 0; i < 4; ++i) input.plane[i] = {planes[i], pitches[i], 0};
    const int qgrid = blocks(h * s.q);
    sample<<<qgrid, THREADS, 0, stream>>>(input.plane[0], s);
    gradient_y<<<qgrid, THREADS, 0, stream>>>(s);
    gradient_x<<<qgrid, THREADS, 0, stream>>>(s);
    dilate<<<qgrid, THREADS, 0, stream>>>(s);
    if (h >= 20 && w >= 20) {
        dim3 grid_seam(blocks(s.q, 32), blocks(h - 7, 8));
        is_seam<true><<<grid_seam, THREADS, 0, stream>>>(input, s);
    }
    auto error = cudaMemsetAsync(s.count, 0, sizeof(int), stream);
    if (error != cudaSuccess) return error;
    error = cudaMemsetAsync(s.votes, 0, ((size_t(h) * s.q + 3) / 4) * sizeof(unsigned), stream);
    if (error != cudaSuccess) return error;
    horizontal_chunks<<<h, THREADS, 0, stream>>>(input.plane[0], s);
    locate<<<s.nt, 128, 0, stream>>>(s);
    corroborate<<<s.nt, 64, 0, stream>>>(s);
    error = cudaMemsetAsync(s.sum, 0, size_t(h) * w * sizeof(float), stream);
    if (error != cudaSuccess) return error;
    error = cudaMemsetAsync(s.weight, 0, size_t(h) * w * sizeof(float), stream);
    if (error != cudaSuccess) return error;
    if (h >= TH && w >= TW) repair<true><<<s.workers, THREADS, 0, stream>>>(input.plane[0], s);
    else repair<false><<<s.workers, THREADS, 0, stream>>>(input.plane[0], s);
    const dim3 block(32, 8), grid(blocks(w, 32), blocks(h, 8));
    if (wide) {
        writePlane<true><<<grid, block, 0, stream>>>(input.plane[0], s.sum, s.weight, output, outputPitch, w, h);
    } else {
        writePlane<false><<<grid, block, 0, stream>>>(input.plane[0], s.sum, s.weight, output, outputPitch, w, h);
    }
    return cudaGetLastError();
}
} // namespace bseam
