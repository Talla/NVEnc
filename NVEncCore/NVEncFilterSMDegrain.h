// -----------------------------------------------------------------------------------------
// NVEnc by rigaya (SMDegrain port, MIT)
// -----------------------------------------------------------------------------------------
// nvenc-smdegrain port — Phase 5 filter class.
// Clean-room reimplementation of the SMDegrain function (havsfunc / MVTools). Name retained
// for discoverability; algorithm and implementation are new code, no GPL source copied.
// -----------------------------------------------------------------------------------------

#pragma once

#include "NVEncFilter.h"
#include "rgy_prm.h"
#include <array>
#include <vector>

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
    RGY_ERR allocateWorkspaces(const NVEncFilterParamSMDegrain *prm);

    // Templated pipeline body. Instantiated for uint8_t (8-bit) and uint16_t (10/12/16-bit).
    // Handles ring store, L1 downsample, UV passthrough, first-frame identity, and the
    // full ME+MC+blend pipeline for frame N>=1. Called from run_filter() after a bit-depth
    // dispatch. `pix_max` and `limit_scaled` are computed by the caller from bit_depth.
    template<typename T>
    RGY_ERR runDenoiseImpl(
        const RGYFrameInfo *pInputFrame,
        RGYFrameInfo **ppOutputFrames,
        int *pOutputFrameNum,
        cudaStream_t stream,
        int pix_max,
        int limit_scaled,
        int thSAD_scaled);

    // Frame ring: holds 2*tr+1 recent input frames. Phase 5f uses this as a
    // bidirectional window — at steady state we process the frame at the center
    // of the ring so it has tr past refs AND tr future refs available.
    std::vector<std::unique_ptr<CUFrameBuf>> m_ringBuf;
    int  m_ringSize;     // cached = 2*tr+1
    int  m_ringIdx;      // number of input frames received so far
    int  m_outputIdx;    // number of output frames emitted so far (== frame index of next emit)
    bool m_flushed;      // true once end-of-stream flush has been started (null input seen)

    // L1 (half-resolution) Y-plane buffers for pyramid ME — one per ring slot.
    std::vector<std::unique_ptr<CUMemBuf>> m_l1Buf;
    int  m_l1Width;
    int  m_l1Height;

    // MV arrays.
    //   m_coarseMVs: scratch at L1 block grid, reused across refs (only the fine result matters).
    //   m_fineMVs: one per ref (2*tr slots — bidirectional window) so the blend kernel can
    //     read each ref's SAD for per-block thSAD gating.
    std::unique_ptr<CUMemBuf> m_coarseMVs;
    std::vector<std::unique_ptr<CUMemBuf>> m_fineMVs;

    // Scratch frames — one per ref (2*tr slots in Phase 5f bidirectional mode).
    // Full CUFrameBuf so Y-plane pitch matches the ring-buffer frames; the blend kernel
    // assumes identical pitches for cur / mc_ref(s) / out, which encoder-allocated frames
    // satisfy but a contiguous cudaMalloc does not (GPU pitch alignment > width).
    std::vector<std::unique_ptr<CUFrameBuf>> m_mcScratch;

    // Contrasharp scratch (Phase 6): post-blend pass that restores detail lost to
    // temporal averaging. m_degrainScratch holds the blend output before sharpening;
    // m_blurScratch holds the 3x3-blurred version used as the unsharp reference.
    std::unique_ptr<CUFrameBuf> m_degrainScratch;
    std::unique_ptr<CUFrameBuf> m_blurScratch;

    // Cached param snapshot for reallocation detection.
    int m_cachedWidth;
    int m_cachedHeight;
    int m_cachedTr;
    RGY_CSP m_cachedCsp;
};
