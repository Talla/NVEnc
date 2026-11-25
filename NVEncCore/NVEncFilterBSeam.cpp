// AX700 B-picture seam repair VPP. used_by: NVEncCore.cpp.
// The stream orders ring ingestion, repair, and all downstream reads.
#include "NVEncFilterBSeam.h"
#include "bseam/bseam_kernels.h"

NVEncFilterBSeam::NVEncFilterBSeam() {
    m_name = _T("bseam");
}

NVEncFilterBSeam::~NVEncFilterBSeam() {
    close();
}

void NVEncFilterBSeam::close() {
    if (m_workspace) cudaFree(m_workspace);
    m_workspace = nullptr;
    for (auto& slot : m_ring) slot.reset();
    m_frameBuf.clear();
    m_received = m_emitted = 0;
    m_draining = false;
}

RGY_ERR NVEncFilterBSeam::init(shared_ptr<NVEncFilterParam> param, shared_ptr<RGYLog> log) {
    m_pLog = log;
    close();
    const auto& input = param->frameIn;
    if ((input.csp != RGY_CSP_NV12 && input.csp != RGY_CSP_YV12) ||
        interlaced(input) || input.width <= 0 || input.height <= 0 ||
        (input.width & 1) || (input.height & 1)) {
        AddMessage(RGY_LOG_ERROR, _T("BSeam requires progressive 8-bit NV12 or YV12 with even dimensions.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    for (auto& slot : m_ring) {
        slot = std::make_unique<CUFrameBuf>(input);
        const auto status = slot->alloc();
        if (status != RGY_ERR_NONE) {
            close();
            return status;
        }
    }
    auto status = AllocFrameBuf(param->frameOut, 2);
    if (status == RGY_ERR_NONE) {
        status = err_to_rgy(cudaMalloc(&m_workspace, bseam::workspaceBytes(input.height, input.width)));
    }
    if (status != RGY_ERR_NONE) {
        close();
        return status;
    }
    for (int p = 0; p < RGY_CSP_PLANES[param->frameOut.csp]; ++p) {
        param->frameOut.pitch[p] = m_frameBuf[0]->frame.pitch[p];
    }
    m_pathThrough = FILTER_PATHTHROUGH_NONE;
    m_param = param;
    setFilterInfo(_T("bseam: AX700 H.264 B-picture seam repair"));
    return RGY_ERR_NONE;
}

const RGYFrameInfo& NVEncFilterBSeam::picture(int64_t index) const {
    return m_ring[index % m_ring.size()]->frame;
}

int64_t NVEncFilterBSeam::groupStart() const {
    const auto& target = picture(m_emitted);
    if (target.width < 20 || target.height < 20 || target.sourceFrameType != RGY_FRAMETYPE_B) return -1;
    const auto anchor = [](RGY_FRAMETYPE type) {
        return type == (RGY_FRAMETYPE_I | RGY_FRAMETYPE_REF) ||
               type == (RGY_FRAMETYPE_P | RGY_FRAMETYPE_REF);
    };
    for (auto first : {m_emitted - 1, m_emitted - 2}) {
        if (first < 0 || first + 3 >= m_received) continue;
        const auto& before = picture(first);
        if (anchor(before.sourceFrameType) && anchor(picture(first + 3).sourceFrameType) &&
            picture(first + 1).sourceFrameType == RGY_FRAMETYPE_B &&
            picture(first + 2).sourceFrameType == RGY_FRAMETYPE_B &&
            picture(first + 1).inputFrameId == before.inputFrameId + 1 &&
            picture(first + 2).inputFrameId == before.inputFrameId + 2 &&
            picture(first + 3).inputFrameId == before.inputFrameId + 3) {
            return first;
        }
    }
    return -1;
}

RGY_ERR NVEncFilterBSeam::run_filter(const RGYFrameInfo* input, RGYFrameInfo** output,
                                    int* count, cudaStream_t stream) {
    *count = 0;
    output[0] = nullptr;
    if (!m_workspace) return RGY_ERR_INVALID_CALL;
    if (input && input->ptr[0]) {
        if (m_draining) {
            if (m_received != m_emitted) return RGY_ERR_INVALID_CALL;
            m_received = m_emitted = 0;
            m_draining = false;
        }
        if (input->csp != m_param->frameIn.csp || input->width != m_param->frameIn.width ||
            input->height != m_param->frameIn.height || interlaced(*input)) {
            AddMessage(RGY_LOG_ERROR, _T("BSeam input format changed or became interlaced.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
        auto& slot = m_ring[m_received % m_ring.size()]->frame;
        const auto status = copyFrameAsync(&slot, input, stream);
        if (status != RGY_ERR_NONE) return status;
        copyFramePropWithoutRes(&slot, input);
        ++m_received;
    } else {
        m_draining = true;
    }
    if (m_emitted >= m_received || (!m_draining && m_received - m_emitted <= 2)) return RGY_ERR_NONE;

    const auto& target = picture(m_emitted);
    const auto first = groupStart();
    AddMessage(RGY_LOG_DEBUG, _T("BSeam frame %d pts %lld source type %u eligible %d\n"),
               target.inputFrameId, (long long)target.timestamp, (unsigned)target.sourceFrameType, first >= 0);
    auto& dst = m_frameBuf[m_emitted % m_frameBuf.size()]->frame;
    const bool wide = dst.csp != target.csp;
    if (first >= 0) {
        const auto& sibling = picture(m_emitted == first + 1 ? first + 2 : first + 1);
        const auto& before = picture(first);
        const auto& after = picture(first + 3);
        const uint8_t* planes[] = {target.ptr[0], before.ptr[0], after.ptr[0], sibling.ptr[0]};
        const size_t pitches[] = {size_t(target.pitch[0]), size_t(before.pitch[0]),
                                  size_t(after.pitch[0]), size_t(sibling.pitch[0])};
        const auto status = bseam::repairFrame(m_workspace, target.height, target.width, planes,
                                              pitches, dst.ptr[0], dst.pitch[0], wide, stream);
        if (status != cudaSuccess) return err_to_rgy(status);
    }
    for (int p = first >= 0 ? 1 : 0; p < RGY_CSP_PLANES[dst.csp]; ++p) {
        const auto srcPlane = getPlane(&target, (RGY_PLANE)p);
        auto dstPlane = getPlane(&dst, (RGY_PLANE)p);
        if (wide) {
            const int width = p == 0 || target.csp == RGY_CSP_NV12 ? target.width : target.width / 2;
            const auto status = bseam::write(srcPlane.ptr[0], srcPlane.pitch[0], dstPlane.ptr[0],
                                            dstPlane.pitch[0], width, srcPlane.height, stream);
            if (status != cudaSuccess) return err_to_rgy(status);
        } else {
            const auto status = copyPlaneAsync(&dstPlane, &srcPlane, stream);
            if (status != RGY_ERR_NONE) return status;
        }
    }
    copyFramePropWithoutRes(&dst, &target);
    ++m_emitted;
    output[0] = &dst;
    *count = 1;
    return RGY_ERR_NONE;
}
