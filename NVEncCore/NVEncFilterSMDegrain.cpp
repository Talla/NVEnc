// -----------------------------------------------------------------------------------------
// NVEnc by rigaya (SMDegrain port, MIT)
// -----------------------------------------------------------------------------------------
// nvenc-smdegrain port — Phase 3: filter skeleton with ring buffer + identity passthrough.
// See NVEncFilterSMDegrain.h for the clean-room rationale.
// -----------------------------------------------------------------------------------------

#include "convert_csp.h"
#include "NVEncFilterSMDegrain.h"
#include "rgy_prm.h"

NVEncFilterSMDegrain::NVEncFilterSMDegrain() :
    m_ringBuf(),
    m_ringSize(0) {
    m_name = _T("smdegrain");
}

NVEncFilterSMDegrain::~NVEncFilterSMDegrain() {
    close();
}

tstring NVEncFilterParamSMDegrain::print() const {
    return smdegrain.print();
}

RGY_ERR NVEncFilterSMDegrain::checkParam(const NVEncFilterParamSMDegrain *prm) {
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter (non-positive frame size).\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->smdegrain.tr < 1 || prm->smdegrain.tr > 3) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, tr must be 1..3 (got %d).\n"), prm->smdegrain.tr);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->smdegrain.thSAD < 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, thSAD must be >= 0 (got %d).\n"), prm->smdegrain.thSAD);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->smdegrain.limit < 0 || prm->smdegrain.limit > 255) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, limit must be 0..255 (got %d).\n"), prm->smdegrain.limit);
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterSMDegrain::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamSMDegrain>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if ((sts = checkParam(prm.get())) != RGY_ERR_NONE) {
        return sts;
    }

    // Motion-estimation reference-frame ring: 2*tr+1 frames.
    // Phase 3 only exercises the allocation; actual indexing lands in Phase 4+.
    const int newRingSize = 2 * prm->smdegrain.tr + 1;
    const bool resolutionChanged = !m_param ||
        cmpFrameInfoCspResolution(&m_param->frameOut, &prm->frameOut);
    const bool needRealloc = (m_ringSize != newRingSize) || resolutionChanged;
    if (needRealloc) {
        m_ringBuf.clear();
        m_ringBuf.resize(newRingSize);
        for (auto& buf : m_ringBuf) {
            buf = std::unique_ptr<CUFrameBuf>(new CUFrameBuf());
            sts = buf->alloc(prm->frameOut.width, prm->frameOut.height, prm->frameOut.csp);
            if (sts != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate SMDegrain ring buffer frame: %s.\n"), get_err_mes(sts));
                return sts;
            }
        }
        m_ringSize = newRingSize;
        AddMessage(RGY_LOG_DEBUG, _T("SMDegrain: allocated %d-frame ring buffer (tr=%d, %dx%d csp=%s).\n"),
            newRingSize, prm->smdegrain.tr, prm->frameOut.width, prm->frameOut.height,
            RGY_CSP_NAMES[prm->frameOut.csp]);
    }

    // Output frame buffer — single slot for identity passthrough.
    sts = AllocFrameBuf(prm->frameOut, 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate SMDegrain output buffer: %s.\n"), get_err_mes(sts));
        return sts;
    }
    for (int i = 0; i < RGY_CSP_PLANES[pParam->frameOut.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    setFilterInfo(pParam->print());
    m_param = pParam;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterSMDegrain::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    RGY_ERR sts = RGY_ERR_NONE;

    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        auto pOutFrame = m_frameBuf[0].get();
        ppOutputFrames[0] = &pOutFrame->frame;
    }
    ppOutputFrames[0]->picstruct = pInputFrame->picstruct;

    // End-of-stream flush: upstream signals EOF via null pointer.
    if (pInputFrame->ptr[0] == nullptr) {
        *pOutputFrameNum = 0;
        ppOutputFrames[0] = nullptr;
        return sts;
    }

    // Phase 3 identity passthrough: device-to-device copy, no kernel work yet.
    const auto memcpyKind = getCudaMemcpyKind(pInputFrame->mem_type, ppOutputFrames[0]->mem_type);
    if (memcpyKind != cudaMemcpyDeviceToDevice) {
        AddMessage(RGY_LOG_ERROR, _T("SMDegrain: only supported on device memory.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    sts = copyFrameAsync(ppOutputFrames[0], pInputFrame, stream);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("SMDegrain: failed to copy frame: %s.\n"), get_err_mes(sts));
        return sts;
    }
    copyFramePropWithoutRes(ppOutputFrames[0], pInputFrame);

    m_nFrameIdx++;
    return RGY_ERR_NONE;
}

void NVEncFilterSMDegrain::close() {
    m_frameBuf.clear();
    m_ringBuf.clear();
    m_ringSize = 0;
}
