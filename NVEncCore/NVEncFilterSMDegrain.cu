// -----------------------------------------------------------------------------------------
// NVEnc by rigaya (SMDegrain port, MIT)
// -----------------------------------------------------------------------------------------
// nvenc-smdegrain port — Phase 5e filter implementation.
//
// Wires the validated standalone kernels (ME pyramid + MC + bounded blend) into the
// NVEncC filter chain. Phase 5e MVP scope:
//   - tr=1 causal mode only (current + previous frame — no future lookahead delay).
//   - Y plane is denoised; U and V planes pass through unchanged.
//   - 8-bit only (YV12). 10-bit rejected with a clear error in checkParam.
//   - contrasharp is a no-op for now (Phase 6).
//   - thSAD gating is not yet applied (Phase 5f); all refs contribute.
//
// Later phases extend to bidirectional tr (3,5,7 frames), N-ref weighted averaging,
// thSAD rejection, 10-bit, sub-pixel MC, contrasharp.
// -----------------------------------------------------------------------------------------

#include <algorithm>
#include "NVEncFilterSMDegrain.h"
#include "NVEncFilterSMDegrain.cuh"
#include "convert_csp.h"
#include "rgy_prm.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)

static constexpr int SMD_BLOCK_SIZE = 8;
static constexpr int SMD_COARSE_RADIUS = 8;   // at L1 (half-res) = effective ±16 at L0
static constexpr int SMD_REFINE_RADIUS = 2;   // at L0 around propagated hint

NVEncFilterSMDegrain::NVEncFilterSMDegrain() :
    m_ringBuf(),
    m_ringSize(0),
    m_ringIdx(0),
    m_l1Buf(),
    m_l1Width(0),
    m_l1Height(0),
    m_coarseMVs(),
    m_fineMVs(),
    m_mcScratch(),
    m_cachedWidth(0),
    m_cachedHeight(0),
    m_cachedTr(0),
    m_cachedCsp(RGY_CSP_NA) {
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
    // Phase 5e MVP: only 8-bit YV12 (NV12→YV12 is performed by an earlier cspconv filter).
    if (RGY_CSP_BIT_DEPTH[prm->frameOut.csp] != 8) {
        AddMessage(RGY_LOG_ERROR, _T("SMDegrain MVP supports 8-bit sources only (phase 5e). got %d-bit.\n"),
            RGY_CSP_BIT_DEPTH[prm->frameOut.csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    if (RGY_CSP_CHROMA_FORMAT[prm->frameOut.csp] != RGY_CHROMAFMT_YUV420) {
        AddMessage(RGY_LOG_ERROR, _T("SMDegrain MVP supports YV12/YUV420 only (phase 5e).\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if ((prm->frameOut.width & 1) || (prm->frameOut.height & 1)) {
        AddMessage(RGY_LOG_ERROR, _T("SMDegrain requires even frame dimensions (got %dx%d).\n"),
            prm->frameOut.width, prm->frameOut.height);
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterSMDegrain::allocateWorkspaces(const NVEncFilterParamSMDegrain *prm) {
    RGY_ERR sts = RGY_ERR_NONE;
    const int width = prm->frameOut.width;
    const int height = prm->frameOut.height;
    const int tr = prm->smdegrain.tr;
    const int newRingSize = 2 * tr + 1;
    const bool paramsChanged = (width != m_cachedWidth) || (height != m_cachedHeight) ||
        (tr != m_cachedTr) || (prm->frameOut.csp != m_cachedCsp) || (m_ringSize != newRingSize);

    if (!paramsChanged) {
        return RGY_ERR_NONE;
    }

    m_ringBuf.clear();
    m_l1Buf.clear();
    m_coarseMVs.reset();
    m_fineMVs.clear();
    m_mcScratch.clear();

    m_ringBuf.resize(newRingSize);
    for (auto& buf : m_ringBuf) {
        buf = std::unique_ptr<CUFrameBuf>(new CUFrameBuf());
        sts = buf->alloc(width, height, prm->frameOut.csp);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate SMDegrain ring buffer: %s.\n"), get_err_mes(sts));
            return sts;
        }
    }

    m_l1Width = width / 2;
    m_l1Height = height / 2;
    const size_t l1_bytes = (size_t)m_l1Width * m_l1Height;
    m_l1Buf.resize(newRingSize);
    for (auto& buf : m_l1Buf) {
        buf = std::unique_ptr<CUMemBuf>(new CUMemBuf(l1_bytes));
        sts = buf->alloc();
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate L1 buffer: %s.\n"), get_err_mes(sts));
            return sts;
        }
    }

    const int l0_blocks_x = width / SMD_BLOCK_SIZE;
    const int l0_blocks_y = height / SMD_BLOCK_SIZE;
    const int l1_blocks_x = m_l1Width / SMD_BLOCK_SIZE;
    const int l1_blocks_y = m_l1Height / SMD_BLOCK_SIZE;
    const size_t coarse_bytes = (size_t)l1_blocks_x * l1_blocks_y * sizeof(smdegrain::MVBlock);
    const size_t fine_bytes = (size_t)l0_blocks_x * l0_blocks_y * sizeof(smdegrain::MVBlock);
    m_coarseMVs = std::unique_ptr<CUMemBuf>(new CUMemBuf(coarse_bytes));
    if ((sts = m_coarseMVs->alloc()) != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate coarse MV buffer: %s.\n"), get_err_mes(sts));
        return sts;
    }
    // One fine-MV buffer per past ref so thSAD gating can read each ref's per-block SAD.
    m_fineMVs.clear();
    m_fineMVs.resize(tr);
    for (auto& buf : m_fineMVs) {
        buf = std::unique_ptr<CUMemBuf>(new CUMemBuf(fine_bytes));
        if ((sts = buf->alloc()) != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate fine MV buffer: %s.\n"), get_err_mes(sts));
            return sts;
        }
    }

    // One MC scratch frame per past ref (tr of them).
    m_mcScratch.clear();
    m_mcScratch.resize(tr);
    for (auto& buf : m_mcScratch) {
        buf = std::unique_ptr<CUFrameBuf>(new CUFrameBuf());
        sts = buf->alloc(width, height, prm->frameOut.csp);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate MC scratch frame: %s.\n"), get_err_mes(sts));
            return sts;
        }
    }

    m_ringSize = newRingSize;
    m_ringIdx = 0;
    m_cachedWidth = width;
    m_cachedHeight = height;
    m_cachedTr = tr;
    m_cachedCsp = prm->frameOut.csp;

    AddMessage(RGY_LOG_DEBUG,
        _T("SMDegrain: allocated ring %d, L1 %dx%d, L0 blocks %dx%d, L1 blocks %dx%d.\n"),
        newRingSize, m_l1Width, m_l1Height, l0_blocks_x, l0_blocks_y, l1_blocks_x, l1_blocks_y);
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
    if ((sts = allocateWorkspaces(prm.get())) != RGY_ERR_NONE) {
        return sts;
    }

    sts = AllocFrameBuf(prm->frameOut, 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate output buffer: %s.\n"), get_err_mes(sts));
        return sts;
    }
    for (int i = 0; i < RGY_CSP_PLANES[pParam->frameOut.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    setFilterInfo(pParam->print());
    m_param = pParam;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterSMDegrain::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames,
                                         int *pOutputFrameNum, cudaStream_t stream) {
    RGY_ERR sts = RGY_ERR_NONE;

    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        auto pOutFrame = m_frameBuf[0].get();
        ppOutputFrames[0] = &pOutFrame->frame;
    }
    ppOutputFrames[0]->picstruct = pInputFrame->picstruct;

    if (pInputFrame->ptr[0] == nullptr) {
        *pOutputFrameNum = 0;
        ppOutputFrames[0] = nullptr;
        return RGY_ERR_NONE;
    }

    const auto memcpyKind = getCudaMemcpyKind(pInputFrame->mem_type, ppOutputFrames[0]->mem_type);
    if (memcpyKind != cudaMemcpyDeviceToDevice) {
        AddMessage(RGY_LOG_ERROR, _T("SMDegrain: only supported on device memory.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    auto prm = std::dynamic_pointer_cast<NVEncFilterParamSMDegrain>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    // Stash the full current frame into the ring.
    const int cur_slot = m_ringIdx % m_ringSize;
    CUFrameBuf* curRing = m_ringBuf[cur_slot].get();
    sts = copyFrameAsync(&curRing->frame, pInputFrame, stream);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("SMDegrain: failed to copy input into ring: %s.\n"), get_err_mes(sts));
        return sts;
    }

    // Build L1 (half-res Y) for this frame at arrival time; subsequent frames that reference
    // this one as "prev" can reuse the cached L1 instead of re-downsampling per run_filter.
    auto curY  = getPlane(&curRing->frame, RGY_PLANE_Y);
    const int pitch_y0 = curY.pitch[0];
    const int pitch_l1 = m_l1Width;
    const int width = prm->frameOut.width;
    const int height = prm->frameOut.height;
    uint8_t* d_cur_l1  = (uint8_t*)m_l1Buf[cur_slot]->ptr;
    cudaError_t cerr = smdegrain::launch_downsample_2x<uint8_t>(
        (const uint8_t*)curY.ptr[0], width, height, pitch_y0,
        d_cur_l1, m_l1Width, m_l1Height, pitch_l1, stream);
    if (cerr != cudaSuccess) { AddMessage(RGY_LOG_ERROR, _T("SMDegrain downsample(cur) failed: %d.\n"), (int)cerr); return RGY_ERR_CUDA; }

    // UV passthrough (Phase 5g still denoises Y only).
    auto curInputU = getPlane(pInputFrame, RGY_PLANE_U);
    auto curInputV = getPlane(pInputFrame, RGY_PLANE_V);
    auto outU = getPlane(ppOutputFrames[0], RGY_PLANE_U);
    auto outV = getPlane(ppOutputFrames[0], RGY_PLANE_V);
    sts = copyPlaneAsync(&outU, &curInputU, stream);
    if (sts != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("SMDegrain: U passthrough failed: %s.\n"), get_err_mes(sts)); return sts; }
    sts = copyPlaneAsync(&outV, &curInputV, stream);
    if (sts != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("SMDegrain: V passthrough failed: %s.\n"), get_err_mes(sts)); return sts; }

    auto outY = getPlane(ppOutputFrames[0], RGY_PLANE_Y);

    // First frame: no prev yet, identity Y.
    if (m_ringIdx == 0) {
        auto curInputY = getPlane(pInputFrame, RGY_PLANE_Y);
        sts = copyPlaneAsync(&outY, &curInputY, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("SMDegrain: first-frame Y copy failed: %s.\n"), get_err_mes(sts));
            return sts;
        }
        copyFramePropWithoutRes(ppOutputFrames[0], pInputFrame);
        m_ringIdx++;
        m_nFrameIdx++;
        return RGY_ERR_NONE;
    }

    // Frame N>=1: pipeline using up to `tr` causal past refs.
    const int tr = prm->smdegrain.tr;
    const int have_refs = std::min(m_ringIdx, tr);  // ramps during warmup

    const uint8_t* d_cur_l0 = (const uint8_t*)curY.ptr[0];
    uint8_t* d_out = (uint8_t*)outY.ptr[0];
    const int out_pitch = outY.pitch[0];
    smdegrain::MVBlock* d_coarse = (smdegrain::MVBlock*)m_coarseMVs->ptr;
    const int l1_blocks_x = m_l1Width / SMD_BLOCK_SIZE;
    const int l0_blocks_x = width / SMD_BLOCK_SIZE;

    // For each past ref k in 1..have_refs: ME(ref, cur) -> fine_MV[k-1], MC -> mc_scratch[k-1].
    for (int k = 1; k <= have_refs; k++) {
        const int ref_slot = (m_ringIdx - k) % m_ringSize;
        CUFrameBuf* refRing = m_ringBuf[ref_slot].get();
        auto refY = getPlane(&refRing->frame, RGY_PLANE_Y);
        const uint8_t* d_ref_l0 = (const uint8_t*)refY.ptr[0];
        uint8_t* d_ref_l1 = (uint8_t*)m_l1Buf[ref_slot]->ptr;

        smdegrain::MVBlock* d_fine_k = (smdegrain::MVBlock*)m_fineMVs[k - 1]->ptr;

        cerr = smdegrain::launch_me_fullsearch<SMD_BLOCK_SIZE, uint8_t>(
            d_ref_l1, d_cur_l1, m_l1Width, m_l1Height, pitch_l1,
            SMD_COARSE_RADIUS, d_coarse, stream);
        if (cerr != cudaSuccess) { AddMessage(RGY_LOG_ERROR, _T("SMDegrain coarse ME k=%d failed: %d.\n"), k, (int)cerr); return RGY_ERR_CUDA; }

        cerr = smdegrain::launch_me_refine_around_hint<SMD_BLOCK_SIZE, uint8_t>(
            d_ref_l0, d_cur_l0, width, height, pitch_y0,
            SMD_REFINE_RADIUS, d_coarse, l1_blocks_x, d_fine_k, stream);
        if (cerr != cudaSuccess) { AddMessage(RGY_LOG_ERROR, _T("SMDegrain refine ME k=%d failed: %d.\n"), k, (int)cerr); return RGY_ERR_CUDA; }

        auto mcY = getPlane(&m_mcScratch[k - 1]->frame, RGY_PLANE_Y);
        uint8_t* d_mc_k = (uint8_t*)mcY.ptr[0];
        const int mc_k_pitch = mcY.pitch[0];
        if (mc_k_pitch != pitch_y0) {
            AddMessage(RGY_LOG_ERROR, _T("SMDegrain pitch mismatch at k=%d (mc=%d, y0=%d).\n"), k, mc_k_pitch, pitch_y0);
            return RGY_ERR_UNSUPPORTED;
        }
        cerr = smdegrain::launch_motion_compensate<SMD_BLOCK_SIZE, uint8_t>(
            d_ref_l0, d_fine_k, width, height, pitch_y0, l0_blocks_x,
            d_mc_k, stream);
        if (cerr != cudaSuccess) { AddMessage(RGY_LOG_ERROR, _T("SMDegrain MC k=%d failed: %d.\n"), k, (int)cerr); return RGY_ERR_CUDA; }
    }

    if (out_pitch != pitch_y0) {
        AddMessage(RGY_LOG_ERROR, _T("SMDegrain output pitch mismatch (y0=%d, out=%d).\n"), pitch_y0, out_pitch);
        return RGY_ERR_UNSUPPORTED;
    }

    const int limit_scaled = prm->smdegrain.limit;
    const int thSAD       = prm->smdegrain.thSAD;
    const int pix_max = 255;
    const uint8_t* mc0 = (have_refs >= 1) ? (const uint8_t*)getPlane(&m_mcScratch[0]->frame, RGY_PLANE_Y).ptr[0] : nullptr;
    const uint8_t* mc1 = (have_refs >= 2) ? (const uint8_t*)getPlane(&m_mcScratch[1]->frame, RGY_PLANE_Y).ptr[0] : nullptr;
    const uint8_t* mc2 = (have_refs >= 3) ? (const uint8_t*)getPlane(&m_mcScratch[2]->frame, RGY_PLANE_Y).ptr[0] : nullptr;
    const smdegrain::MVBlock* mv0 = (have_refs >= 1) ? (const smdegrain::MVBlock*)m_fineMVs[0]->ptr : nullptr;
    const smdegrain::MVBlock* mv1 = (have_refs >= 2) ? (const smdegrain::MVBlock*)m_fineMVs[1]->ptr : nullptr;
    const smdegrain::MVBlock* mv2 = (have_refs >= 3) ? (const smdegrain::MVBlock*)m_fineMVs[2]->ptr : nullptr;

    switch (have_refs) {
        case 1:
            cerr = smdegrain::launch_temporal_blend_nref<uint8_t, 1, SMD_BLOCK_SIZE>(
                d_cur_l0, mc0, nullptr, nullptr, mv0, nullptr, nullptr,
                l0_blocks_x, width, height, pitch_y0,
                thSAD, limit_scaled, pix_max, d_out, stream);
            break;
        case 2:
            cerr = smdegrain::launch_temporal_blend_nref<uint8_t, 2, SMD_BLOCK_SIZE>(
                d_cur_l0, mc0, mc1, nullptr, mv0, mv1, nullptr,
                l0_blocks_x, width, height, pitch_y0,
                thSAD, limit_scaled, pix_max, d_out, stream);
            break;
        case 3:
            cerr = smdegrain::launch_temporal_blend_nref<uint8_t, 3, SMD_BLOCK_SIZE>(
                d_cur_l0, mc0, mc1, mc2, mv0, mv1, mv2,
                l0_blocks_x, width, height, pitch_y0,
                thSAD, limit_scaled, pix_max, d_out, stream);
            break;
        default:
            AddMessage(RGY_LOG_ERROR, _T("SMDegrain: unexpected have_refs=%d.\n"), have_refs);
            return RGY_ERR_UNSUPPORTED;
    }
    if (cerr != cudaSuccess) { AddMessage(RGY_LOG_ERROR, _T("SMDegrain blend failed: %d.\n"), (int)cerr); return RGY_ERR_CUDA; }

    copyFramePropWithoutRes(ppOutputFrames[0], pInputFrame);
    m_ringIdx++;
    m_nFrameIdx++;
    return RGY_ERR_NONE;
}

void NVEncFilterSMDegrain::close() {
    m_frameBuf.clear();
    m_ringBuf.clear();
    m_l1Buf.clear();
    m_coarseMVs.reset();
    m_fineMVs.clear();
    m_mcScratch.clear();
    m_ringSize = 0;
    m_ringIdx = 0;
    m_l1Width = 0;
    m_l1Height = 0;
    m_cachedWidth = 0;
    m_cachedHeight = 0;
    m_cachedTr = 0;
    m_cachedCsp = RGY_CSP_NA;
}
