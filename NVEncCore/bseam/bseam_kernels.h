// CUDA dispatch for NVEncFilterBSeam.
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace bseam {
size_t workspaceBytes(int height, int width);
cudaError_t write(const uint8_t* source, size_t sourcePitch, uint8_t* output,
                  size_t outputPitch, int width, int height, cudaStream_t stream);
cudaError_t repairFrame(void* memory, int height, int width, const uint8_t* const* planes,
                        const size_t* pitches, uint8_t* output, size_t outputPitch,
                        bool wide, cudaStream_t stream);
}
