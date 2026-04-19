// -----------------------------------------------------------------------------------------
// NVEnc by rigaya (SMDegrain port, MIT)
// -----------------------------------------------------------------------------------------
// nvenc-smdegrain port — Phase 3 filter skeleton with ring buffer + identity passthrough.
// Clean-room reimplementation of the SMDegrain function (havsfunc / MVTools). Name retained
// for discoverability; algorithm and implementation are new code, no GPL source copied.
// -----------------------------------------------------------------------------------------

#pragma once

#include "NVEncFilter.h"
#include "rgy_prm.h"
#include <array>

#if (defined(WIN32) || defined(WIN64)) && defined(_M_IX86)
#define ENABLE_VPP_SMDEGRAIN 0
#else
#define ENABLE_VPP_SMDEGRAIN 1
#endif

class NVEncFilterParamSMDegrain : public NVEncFilterParam {
public:
    VppSMDegrain smdegrain;
    std::pair<int, int> compute_capability;

    NVEncFilterParamSMDegrain() : smdegrain(), compute_capability() {};
    virtual ~NVEncFilterParamSMDegrain() {};
    virtual tstring print() const override;
};

class NVEncFilterSMDegrain : public NVEncFilter {
public:
    NVEncFilterSMDegrain();
    virtual ~NVEncFilterSMDegrain();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) override;
    virtual void close() override;
    RGY_ERR checkParam(const NVEncFilterParamSMDegrain *prm);

    // Motion-estimation reference-frame ring: 2*tr+1 frames. Phase 3 allocates but
    // does not yet index into it; Phase 4+ will store refs here for block-matching ME.
    std::vector<std::unique_ptr<CUFrameBuf>> m_ringBuf;
    int m_ringSize;
};
