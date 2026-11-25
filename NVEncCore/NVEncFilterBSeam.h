// AX700 B-picture seam repair VPP. used_by: NVEncCore.cpp.
#pragma once

#include <array>
#include "NVEncFilter.h"

class NVEncFilterBSeam : public NVEncFilter {
public:
    NVEncFilterBSeam();
    ~NVEncFilterBSeam();
    RGY_ERR init(shared_ptr<NVEncFilterParam> param, shared_ptr<RGYLog> log) override;
    void close() override;

protected:
    RGY_ERR run_filter(const RGYFrameInfo* input, RGYFrameInfo** output,
                       int* count, cudaStream_t stream) override;

private:
    const RGYFrameInfo& picture(int64_t index) const;
    int64_t groupStart() const;

    std::array<std::unique_ptr<CUFrameBuf>, 5> m_ring;
    void* m_workspace = nullptr;
    int64_t m_received = 0;
    int64_t m_emitted = 0;
    bool m_draining = false;
};
