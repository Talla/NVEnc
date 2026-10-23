// Native T2 VPP integration. used_by: NVEncCore.cpp, NVEncFilterT2.cpp
// rules: retained source ownership, originating metadata, repeated EOS, per-instance state.
#pragma once
#include "NVEncFilter.h"
#include "rgy_prm.h"
#include "t2/t2_native.h"
#include <array>
class NVEncFilterParamT2 : public NVEncFilterParam {
public:
    VppT2 t2;
    tstring print() const override { return t2.print(); }
};
class NVEncFilterT2 : public NVEncFilter {
public:
    NVEncFilterT2();
    ~NVEncFilterT2();
    RGY_ERR init(shared_ptr<NVEncFilterParam> param, shared_ptr<RGYLog> log) override;
protected:
    RGY_ERR run_filter(const RGYFrameInfo *input, RGYFrameInfo **output, int *count, cudaStream_t stream) override;
    void close() override;
private:
    std::unique_ptr<native_t2::Engine> m_engine;
    std::array<RGYFrameInfo,17> m_metadata;
    int64_t m_received, m_emitted;
    bool m_draining;
    uint64_t m_errors;
    bool m_fp64Logged;
};
