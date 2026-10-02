// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2014-2016 rigaya
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// ------------------------------------------------------------------------------------------

#include <array>
#include <map>
#define _USE_MATH_DEFINES
#include <cmath>
#include "convert_csp.h"
#include "NVEncFilterDenoiseFFT3D.h"
#include "rgy_prm.h"

std::unique_ptr<DenoiseFFT3DBase> getDenoiseFFT3DFunc8FP32(const int block_size);
std::unique_ptr<DenoiseFFT3DBase> getDenoiseFFT3DFunc8FP16(const int block_size);
std::unique_ptr<DenoiseFFT3DBase> getDenoiseFFT3DFunc16FP16(const int block_size);
std::unique_ptr<DenoiseFFT3DBase> getDenoiseFFT3DFunc16FP32(const int block_size);

std::unique_ptr<DenoiseFFT3DBase> getDenoiseFunc(const RGY_CSP csp, const int block_size, VppFpPrecision prec) {
    switch (RGY_CSP_DATA_TYPE[csp]) {
    case RGY_DATA_TYPE_U8:
        if (prec == VppFpPrecision::VPP_FP_PRECISION_FP32) {
            return getDenoiseFFT3DFunc8FP32(block_size);
        } else {
            return getDenoiseFFT3DFunc8FP16(block_size);
        }
    case RGY_DATA_TYPE_U16:
        if (prec == VppFpPrecision::VPP_FP_PRECISION_FP32) {
            return getDenoiseFFT3DFunc16FP32(block_size);
        } else {
            return getDenoiseFFT3DFunc16FP16(block_size);
        }
    default:
        return nullptr;
    }
}

RGY_ERR NVEncFilterDenoiseFFT3DBuffer::alloc(int width, int height, RGY_CSP csp, int frames) {
    m_bufFFT.resize(frames);
    for (auto& buf : m_bufFFT) {
        if (!buf || buf->frame.width != width || buf->frame.height != height || buf->frame.csp != csp) {
            buf = std::unique_ptr<CUFrameBuf>(new CUFrameBuf());
            auto sts = buf->alloc(width, height, csp);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
    }
    return RGY_ERR_NONE;
}

NVEncFilterDenoiseFFT3D::NVEncFilterDenoiseFFT3D() :
    m_bufIdx(0),
    m_ov1(0),
    m_ov2(0),
    m_bufFFT(),
    m_filteredBlocks(),
    m_windowBuf(),
    m_windowBufInverse(),
    m_sigmaTable(),
    m_sigmaTableCurve(),
    m_sigmaTableBlockSize(0),
    m_sigmaTableTemporalCount(0),
    m_sigmaTableBitDepth(0) {
    m_name = _T("denoise-fft");
}

NVEncFilterDenoiseFFT3D::~NVEncFilterDenoiseFFT3D() {
    close();
}

RGY_ERR NVEncFilterDenoiseFFT3D::checkParam(const NVEncFilterParamDenoiseFFT3D *prm) {
    //パラメータチェック
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->fft3d.sigma < 0.0f || 100.0f < prm->fft3d.sigma) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, sigma must be 0 - 100.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->fft3d.amount < 0.0f || 1.0f < prm->fft3d.amount) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, strength must be 0 - 1.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (get_cx_index(list_vpp_fft3d_block_size, prm->fft3d.block_size) < 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid block_size.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->fft3d.overlap < 0.0f || 0.8f < prm->fft3d.overlap) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, overlap must be 0 - 0.8.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->fft3d.overlap2 < 0.0f || 0.8f < prm->fft3d.overlap2) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, overlap2 must be 0 - 0.8.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (0.8f < prm->fft3d.overlap + prm->fft3d.overlap2) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, sum of overlap and overlap2 must be below 0.8.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->fft3d.method != 0 && prm->fft3d.method != 1) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, method must be 0 or 1.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->fft3d.temporal != 0 && prm->fft3d.temporal != 1) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, temporal must be 0 or 1.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->fft3d.tbsize != 0 && prm->fft3d.tbsize != 1 && prm->fft3d.tbsize != 3 && prm->fft3d.tbsize != 5) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, tbsize must be 1, 3, or 5.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (get_cx_index(list_vpp_fp_prec, prm->fft3d.precision) < 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid precision.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterDenoiseFFT3D::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDenoiseFFT3D>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if ((sts = checkParam(prm.get())) != RGY_ERR_NONE) {
        return sts;
    }
    if (prm->fft3d.precision != VppFpPrecision::VPP_FP_PRECISION_FP32 && prm->compute_capability.first < 7) {
        prm->fft3d.precision = VppFpPrecision::VPP_FP_PRECISION_FP32;
    }
    if (!m_param
        || prm->fft3d.block_size != std::dynamic_pointer_cast<NVEncFilterParamDenoiseFFT3D>(m_param)->fft3d.block_size
        || prm->fft3d.overlap != std::dynamic_pointer_cast<NVEncFilterParamDenoiseFFT3D>(m_param)->fft3d.overlap
        || prm->fft3d.overlap2 != std::dynamic_pointer_cast<NVEncFilterParamDenoiseFFT3D>(m_param)->fft3d.overlap2
        || prm->fft3d.temporal != std::dynamic_pointer_cast<NVEncFilterParamDenoiseFFT3D>(m_param)->fft3d.temporal
        || prm->fft3d.tbsize != std::dynamic_pointer_cast<NVEncFilterParamDenoiseFFT3D>(m_param)->fft3d.tbsize
        || prm->fft3d.precision != std::dynamic_pointer_cast<NVEncFilterParamDenoiseFFT3D>(m_param)->fft3d.precision
        || cmpFrameInfoCspResolution(&m_param->frameOut, &prm->frameOut)) {
        m_ov1 = (int)(prm->fft3d.block_size * 0.5 * prm->fft3d.overlap + 0.5);
        m_ov2 = (int)(prm->fft3d.block_size * 0.5 * (prm->fft3d.overlap + prm->fft3d.overlap2) + 0.5) - m_ov1;

        //より小さいUVに合わせてブロック数を計算し、そこから確保するメモリを決める
        auto planeUV = getPlane(&prm->frameOut, RGY_PLANE_U);
        const auto blocksUV = getBlockCount(planeUV.width, planeUV.height, prm->fft3d.block_size, m_ov1, m_ov2);
        const int complexSize = (prm->fft3d.precision == VppFpPrecision::VPP_FP_PRECISION_FP32) ? 8 : 4;

        RGY_CSP fft_csp = RGY_CSP_NA;
        int blockGlobalWidth = 0, blockGlobalHeight = 0;
        if (RGY_CSP_CHROMA_FORMAT[prm->frameOut.csp] == RGY_CHROMAFMT_YUV420) {
            fft_csp = RGY_CSP_YV12;
            blockGlobalWidth = blocksUV.first * prm->fft3d.block_size * 2;
            blockGlobalHeight = blocksUV.second * prm->fft3d.block_size * 2;
        } else if (RGY_CSP_CHROMA_FORMAT[prm->frameOut.csp] == RGY_CHROMAFMT_YUV444) {
            fft_csp = RGY_CSP_YUV444;
            blockGlobalWidth = blocksUV.first * prm->fft3d.block_size;
            blockGlobalHeight = blocksUV.second * prm->fft3d.block_size;
        } else {
            AddMessage(RGY_LOG_ERROR, _T("Invalid colorformat: %s.\n"), RGY_CSP_NAMES[prm->frameOut.csp]);
            return RGY_ERR_UNSUPPORTED;
        }

        if ((sts = m_bufFFT.alloc(blockGlobalWidth * complexSize, blockGlobalHeight * complexSize, fft_csp, prm->fft3d.effectiveTbsize())) != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory for FFT: %s.\n"), get_err_mes(sts));
            return sts;
        }

        m_filteredBlocks = std::unique_ptr<CUFrameBuf>(new CUFrameBuf());
        if ((sts = m_filteredBlocks->alloc(blockGlobalWidth, blockGlobalHeight, prm->frameOut.csp)) != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory for filtered blocks: %s.\n"), get_err_mes(sts));
            return sts;
        }

        sts = AllocFrameBuf(prm->frameOut, 1);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
            return sts;
        }
        for (int i = 0; i < RGY_CSP_PLANES[pParam->frameOut.csp]; i++) {
            prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
        }

        if (!m_param || !m_windowBuf || prm->fft3d.block_size != std::dynamic_pointer_cast<NVEncFilterParamDenoiseFFT3D>(m_param)->fft3d.block_size) {
            std::vector<float> blockWindow(prm->fft3d.block_size);
            std::vector<float> blockWindowInv(prm->fft3d.block_size);
            auto winFunc = [block_size = prm->fft3d.block_size](const int x) { return 0.50f - 0.50f * std::cos(2.0f * (float)M_PI * x / (float)block_size); };
            for (int i = 0; i < prm->fft3d.block_size; i++) {
                blockWindow[i] = winFunc(i);
                blockWindowInv[i] = 1.0f / blockWindow[i];
            }

            m_windowBuf = std::unique_ptr<CUMemBuf>(new CUMemBuf(blockWindow.size() * sizeof(blockWindow[0])));
            m_windowBufInverse = std::unique_ptr<CUMemBuf>(new CUMemBuf(blockWindowInv.size() * sizeof(blockWindowInv[0])));

            if ((sts = m_windowBuf->alloc()) != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory for FFT window: %s.\n"), get_err_mes(sts));
                return sts;
            }
            if ((sts = m_windowBufInverse->alloc()) != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory for FFT window (inverse): %s.\n"), get_err_mes(sts));
                return sts;
            }
            if ((sts = err_to_rgy(cudaMemcpy(m_windowBuf->ptr, blockWindow.data(), blockWindow.size() * sizeof(blockWindow[0]), cudaMemcpyHostToDevice))) != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to copy memory for FFT window: %s.\n"), get_err_mes(sts));
                return sts;
            }
            if ((sts = err_to_rgy(cudaMemcpy(m_windowBufInverse->ptr, blockWindowInv.data(), blockWindowInv.size() * sizeof(blockWindowInv[0]), cudaMemcpyHostToDevice))) != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to copy memory for FFT window (inverse): %s.\n"), get_err_mes(sts));
                return sts;
            }
        }
    }

    // Build per-bin sigma² LUT from sigma_curve (dfttest slocation semantics).
    // Only matters when curve is non-empty; otherwise the scalar-sigma path is used unchanged.
    {
        const int block_size = prm->fft3d.block_size;
        const int temporalCount = prm->fft3d.effectiveTbsize();
        const int bit_depth = RGY_CSP_BIT_DEPTH[prm->frameOut.csp];
        const auto &curve = prm->fft3d.sigma_curve;

        const bool paramsChanged =
            block_size != m_sigmaTableBlockSize ||
            temporalCount != m_sigmaTableTemporalCount ||
            bit_depth != m_sigmaTableBitDepth ||
            curve != m_sigmaTableCurve;

        if (paramsChanged) {
            if (curve.empty()) {
                m_sigmaTable.reset();
            } else {
                const size_t lutCount = (size_t)temporalCount * block_size * block_size;
                std::vector<float> lut(lutCount);
                // Match the scalar path's host-side sigma normalization at .cuh:499 — divide by (1<<8)-1 = 255.
                // Upstream uses 8 here regardless of bit_depth; mirror it so a flat curve produces scalar-equivalent output.
                const float scaleInv = 1.0f / 255.0f;
                for (int bz = 0; bz < temporalCount; bz++) {
                    const float fz = (temporalCount > 1)
                        ? (float)std::min(bz, temporalCount - bz) / ((float)temporalCount * 0.5f)
                        : 0.0f;
                    for (int by = 0; by < block_size; by++) {
                        const float fy = (float)std::min(by, block_size - by) / ((float)block_size * 0.5f);
                        for (int bx = 0; bx < block_size; bx++) {
                            const float fx = (float)std::min(bx, block_size - bx) / ((float)block_size * 0.5f);
                            float radial = std::sqrt((fx * fx + fy * fy) * 0.5f + fz * fz);
                            if (radial > 1.0f) radial = 1.0f;
                            // Piecewise-linear interpolation of (freq, sigma) pairs sorted ascending on freq.
                            float sigma_bin;
                            if (radial <= curve.front().first) {
                                sigma_bin = curve.front().second;
                            } else if (radial >= curve.back().first) {
                                sigma_bin = curve.back().second;
                            } else {
                                sigma_bin = curve.back().second;
                                for (size_t i = 1; i < curve.size(); i++) {
                                    if (radial <= curve[i].first) {
                                        const float f0 = curve[i - 1].first;
                                        const float f1 = curve[i].first;
                                        const float s0 = curve[i - 1].second;
                                        const float s1 = curve[i].second;
                                        const float t = (radial - f0) / (f1 - f0);
                                        sigma_bin = s0 + t * (s1 - s0);
                                        break;
                                    }
                                }
                            }
                            // Store sigma² so a flat curve sigma=S matches scalar sigma=S² after the shared /255 scaling.
                            lut[(size_t)bz * block_size * block_size + (size_t)by * block_size + bx] =
                                sigma_bin * sigma_bin * scaleInv;
                        }
                    }
                }

                m_sigmaTable = std::unique_ptr<CUMemBuf>(new CUMemBuf(lutCount * sizeof(float)));
                if ((sts = m_sigmaTable->alloc()) != RGY_ERR_NONE) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory for sigma_curve LUT: %s.\n"), get_err_mes(sts));
                    return sts;
                }
                if ((sts = err_to_rgy(cudaMemcpy(m_sigmaTable->ptr, lut.data(), lutCount * sizeof(float), cudaMemcpyHostToDevice))) != RGY_ERR_NONE) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to copy memory for sigma_curve LUT: %s.\n"), get_err_mes(sts));
                    return sts;
                }
                AddMessage(RGY_LOG_DEBUG, _T("Built sigma_curve LUT: %d x %d x %d (%zu floats)\n"),
                    temporalCount, block_size, block_size, lutCount);
            }
            m_sigmaTableCurve = curve;
            m_sigmaTableBlockSize = block_size;
            m_sigmaTableTemporalCount = temporalCount;
            m_sigmaTableBitDepth = bit_depth;
        }
    }

    setFilterInfo(pParam->print());
    m_pathThrough = FILTER_PATHTHROUGH_ALL;
    if (prm->fft3d.effectiveTbsize() > 1) {
        m_pathThrough &= (~(FILTER_PATHTHROUGH_TIMESTAMP | FILTER_PATHTHROUGH_FLAGS | FILTER_PATHTHROUGH_DATA));
    }
    m_param = pParam;
    return sts;
}

tstring NVEncFilterParamDenoiseFFT3D::print() const {
    return fft3d.print();
}

RGY_ERR NVEncFilterDenoiseFFT3D::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    RGY_ERR sts = RGY_ERR_NONE;

    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        auto pOutFrame = m_frameBuf[0].get();
        ppOutputFrames[0] = &pOutFrame->frame;
    }
    ppOutputFrames[0]->picstruct = pInputFrame->picstruct;

    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDenoiseFFT3D>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto denosieFunc = getDenoiseFunc(prm->frameIn.csp, prm->fft3d.block_size, prm->fft3d.precision);
    if (!denosieFunc) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp or block_size.\n"));
        return RGY_ERR_UNSUPPORTED;
    }

    const int tbsize = prm->fft3d.effectiveTbsize();
    const int lag = tbsize / 2; // 0 for tbsize=1, 1 for tbsize=3, 2 for tbsize=5

    const bool finalOutput = pInputFrame->ptr[0] == nullptr;
    if (finalOutput) {
        if (tbsize == 1 || m_nFrameIdx >= m_bufIdx) {
            //終了
            *pOutputFrameNum = 0;
            ppOutputFrames[0] = nullptr;
            return sts;
        }
    } else {
        //if (interlaced(*pInputFrame)) {
        //    return filter_as_interlaced_pair(pInputFrame, ppOutputFrames[0], stream);
        //}
        const auto memcpyKind = getCudaMemcpyKind(pInputFrame->mem_type, ppOutputFrames[0]->mem_type);
        if (memcpyKind != cudaMemcpyDeviceToDevice) {
            AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
            return RGY_ERR_INVALID_PARAM;
        }
        if (m_param->frameOut.csp != m_param->frameIn.csp) {
            AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
            return RGY_ERR_INVALID_PARAM;
        }
        auto fftBuf = m_bufFFT.get(m_bufIdx++);
        if (!fftBuf || !fftBuf->frame.ptr[0]) {
            AddMessage(RGY_LOG_ERROR, _T("failed to get fft buffer.\n"));
            return RGY_ERR_NULL_PTR;
        }
        sts = denosieFunc->fft()(&fftBuf->frame, pInputFrame, m_ov1, m_ov2, (const float *)m_windowBuf->ptr, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to run fft: %s.\n"), get_err_mes(sts));
            return RGY_ERR_NONE;
        }
        copyFramePropWithoutRes(&fftBuf->frame, pInputFrame);
    }

    auto planeUV = getPlane(&prm->frameOut, RGY_PLANE_U);

    if (tbsize > 1) {
        // Need `lag` frames buffered before the first center emits on the streaming path.
        if (!finalOutput && m_bufIdx <= lag) {
            *pOutputFrameNum = 0;
            ppOutputFrames[0] = nullptr;
            return sts;
        }
        // Center frame index (0-based stream position). Non-final: most-recent-fft-minus-lag.
        // Final: drain in emission order (m_nFrameIdx is the next index to emit).
        const int centerIdx = finalOutput ? m_nFrameIdx : (m_bufIdx - 1 - lag);
        const int lastIdx = m_bufIdx - 1;
        // Edge frames duplicate via clamp to [0, lastIdx] — same pattern as the original tbsize=3 path.
        const int idxPP   = std::max(centerIdx - 2, 0);
        const int idxP    = std::max(centerIdx - 1, 0);
        const int idxC    = centerIdx;
        const int idxN    = std::min(centerIdx + 1, lastIdx);
        const int idxNN   = std::min(centerIdx + 2, lastIdx);

        const float *sigmaTablePtr = m_sigmaTable ? (const float *)m_sigmaTable->ptr : nullptr;
        if (tbsize == 3) {
            auto fftPrev = m_bufFFT.get(idxP);
            auto fftCur  = m_bufFFT.get(idxC);
            auto fftNext = m_bufFFT.get(idxN);
            sts = denosieFunc->tfft_filter_ifft(1, 3)(&m_filteredBlocks->frame,
                &fftPrev->frame, &fftCur->frame, &fftNext->frame, nullptr, nullptr,
                (const float *)m_windowBufInverse->ptr,
                prm->frameOut.width, prm->frameOut.height, planeUV.width, planeUV.height, m_ov1, m_ov2,
                prm->fft3d.sigma, 1.0f - prm->fft3d.amount, prm->fft3d.method,
                sigmaTablePtr, stream);
            if (sts != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to run tfft_filter_ifft(1, 3): %s.\n"), get_err_mes(sts));
                return RGY_ERR_NONE;
            }
            copyFramePropWithoutRes(ppOutputFrames[0], &fftCur->frame);
        } else { // tbsize == 5
            auto fftPP   = m_bufFFT.get(idxPP);
            auto fftPrev = m_bufFFT.get(idxP);
            auto fftCur  = m_bufFFT.get(idxC);
            auto fftNext = m_bufFFT.get(idxN);
            auto fftNN   = m_bufFFT.get(idxNN);
            sts = denosieFunc->tfft_filter_ifft(2, 5)(&m_filteredBlocks->frame,
                &fftPP->frame, &fftPrev->frame, &fftCur->frame, &fftNext->frame, &fftNN->frame,
                (const float *)m_windowBufInverse->ptr,
                prm->frameOut.width, prm->frameOut.height, planeUV.width, planeUV.height, m_ov1, m_ov2,
                prm->fft3d.sigma, 1.0f - prm->fft3d.amount, prm->fft3d.method,
                sigmaTablePtr, stream);
            if (sts != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to run tfft_filter_ifft(2, 5): %s.\n"), get_err_mes(sts));
                return RGY_ERR_NONE;
            }
            copyFramePropWithoutRes(ppOutputFrames[0], &fftCur->frame);
        }
    } else {
        auto fftCur = m_bufFFT.get(m_bufIdx - 1);
        const float *sigmaTablePtr = m_sigmaTable ? (const float *)m_sigmaTable->ptr : nullptr;
        sts = denosieFunc->tfft_filter_ifft(0, 1)(&m_filteredBlocks->frame, &fftCur->frame, nullptr, nullptr, nullptr, nullptr, (const float *)m_windowBufInverse->ptr,
            prm->frameOut.width, prm->frameOut.height, planeUV.width, planeUV.height, m_ov1, m_ov2,
            prm->fft3d.sigma, 1.0f - prm->fft3d.amount, prm->fft3d.method,
            sigmaTablePtr, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to run tfft_filter_ifft(0, 1): %s.\n"), get_err_mes(sts));
            return RGY_ERR_NONE;
        }
    }
    sts = denosieFunc->merge()(ppOutputFrames[0], &m_filteredBlocks->frame, m_ov1, m_ov2, stream);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to run merge: %s.\n"), get_err_mes(sts));
        return RGY_ERR_NONE;
    }

    m_nFrameIdx++;
    return sts;
}

void NVEncFilterDenoiseFFT3D::close() {
    m_frameBuf.clear();
    m_bufFFT.clear();
    m_windowBuf.reset();
    m_windowBufInverse.reset();
    m_sigmaTable.reset();
    m_sigmaTableCurve.clear();
    m_sigmaTableBlockSize = 0;
    m_sigmaTableTemporalCount = 0;
    m_sigmaTableBitDepth = 0;
}
