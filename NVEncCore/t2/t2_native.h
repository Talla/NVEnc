// Native T2 engine. Image buffers are FP32 normalized code values; fit boundaries are FP64.
// used_by: NVEncFilterT2.cpp, tests/t2_native_test.cu
// rules: one caller-owned CUDA stream; immutable owned history; no global mutable state.
#pragma once
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>
namespace native_t2 {
struct PlaneModel {
    std::vector<double> levels, sigma, psd;
    int psdSize = 0;
    double rho = 0;
};
struct Model { PlaneModel planes[3]; };
struct PlaneView { void *ptr; size_t pitch; int width, height, bytes, shift, bits, step; };
struct FrameView { PlaneView plane[3]; };
class Engine {
public:
    Engine(int width, int height, int sx, int sy, const std::string& mode, Model model, double strength = 1.0);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    void ingest(int64_t index, const FrameView& frame, cudaStream_t stream);
    void render(int64_t index, int64_t received, const FrameView& output, cudaStream_t stream);
    std::string statistics() const;
    uint64_t fp64ContextCount() const;
private:
    struct Impl;
    std::unique_ptr<Impl> p;
};
}
