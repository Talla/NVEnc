// Native T2 VPP. used_by: NVEncCore.cpp; rules: explicit extended model, native geometry,
// owned FP32 ring, >=10-bit output, delayed originating metadata and complete EOS drain.
#include "NVEncFilterT2.h"
#include "rgy_util.h"
#include "convert_csp.h"
#include "../json/json.hpp"
#include <fstream>
#include <filesystem>
#include <cmath>
#include <set>
#include <stdexcept>

namespace {
native_t2::Model loadModel(const VppT2& prm) {
    std::ifstream f(std::filesystem::path(prm.profile));
    if(!f)throw std::runtime_error("cannot open T2 profile");
    nlohmann::json j;f>>j;
    const auto& e=j.at("picture_profiles").at(tchar_to_string(prm.pp)).at("noise_profiles").at(tchar_to_string(prm.iso)).at("extended");
    const auto levels=e.at("nlf").at("levels").get<std::vector<double>>();
    if(levels.empty())throw std::runtime_error("T2 NLF levels empty");
    for(size_t i=0;i<levels.size();++i)if(!std::isfinite(levels[i])||levels[i]<0||levels[i]>1||(i&&levels[i]<=levels[i-1]))throw std::runtime_error("T2 NLF levels must be increasing finite 0..1");
    const auto& psds=e.at("psd");if(!psds.is_array()||psds.empty())throw std::runtime_error("T2 PSD missing");
    native_t2::Model model;const char* names[]={"Y","Cb","Cr"};
    for(int p=0;p<3;++p){auto& m=model.planes[p];const auto& sig=e.at("nlf").at("sigma").at(names[p]);if(sig.size()!=levels.size())throw std::runtime_error("T2 NLF length mismatch");for(size_t k=0;k<levels.size();++k){if(sig[k].is_null())continue;double v=sig[k].get<double>();if(!std::isfinite(v)||v<0)throw std::runtime_error("T2 invalid sigma");m.levels.push_back(levels[k]);m.sigma.push_back(v/255.);}if(m.levels.empty())throw std::runtime_error("T2 plane has no finite measured NLF bins");
        const auto& a=psds.at(psds.size()/2).at(names[p]);int n=(int)a.size();if(n<2)throw std::runtime_error("T2 PSD too small");double sum=0;for(const auto& row:a){if(row.size()!=n)throw std::runtime_error("T2 PSD must be square");for(const auto& v:row){double x=v.get<double>();if(!std::isfinite(x)||x<0)throw std::runtime_error("T2 invalid PSD energy");m.psd.push_back(x);sum+=x;}}if(sum<=0)throw std::runtime_error("T2 zero PSD energy");m.psdSize=n; // The lab preserves profile PSD normalization for coarse propagation.
        for(const char* band:{"all","LF","MF","HF"}){const auto& corr=e.at("temporal_corr").at(names[p]).at(band);if(!corr.is_array()||corr.empty())throw std::runtime_error("T2 temporal correlation missing");double x=corr.at(0).get<double>();if(!std::isfinite(x)||x< -1||x>1)throw std::runtime_error("T2 invalid temporal correlation");if(std::string(band)=="HF")m.rho=std::max(0.,std::min(.95,x));}
    }return model;
}
std::pair<int,int> sampling(RGY_CSP csp){auto c=RGY_CSP_CHROMA_FORMAT[csp];if(c==RGY_CHROMAFMT_YUV420)return {2,2};if(c==RGY_CHROMAFMT_YUV422)return {2,1};if(c==RGY_CHROMAFMT_YUV444)return {1,1};throw std::runtime_error("T2 supports YUV420/422/444 only");}
bool semi(RGY_CSP c){return c==RGY_CSP_NV12||c==RGY_CSP_P010||c==RGY_CSP_NV16||c==RGY_CSP_P210;}
native_t2::FrameView view(const RGYFrameInfo& frame){native_t2::FrameView out{};auto ss=sampling(frame.csp);const bool sp=semi(frame.csp);int bytes=RGY_CSP_BIT_DEPTH[frame.csp]>8?2:1;int bits=RGY_CSP_BIT_DEPTH[frame.csp];
    // Semi-planar decoder buffers contain nominal-bit codes in the high bits.
    // Keep exact nominal / (2^bits-1) scaling when reader provenance retains bitdepth.
    if(sp&&bytes==2&&frame.bitdepth>=9&&frame.bitdepth<16)bits=frame.bitdepth;
    for(int p=0;p<3;++p){auto plane=getPlane(&frame,(RGY_PLANE)p);void* ptr=plane.ptr[0];size_t pitch=plane.pitch[0];if(sp&&p){auto uv=getPlane(&frame,RGY_PLANE_U);ptr=(void*)((char*)uv.ptr[0]+(p==2?bytes:0));pitch=uv.pitch[0];}out.plane[p]={ptr,pitch,frame.width/(p?ss.first:1),frame.height/(p?ss.second:1),bytes,sp&&bytes==2?16-bits:0,bits,sp&&p?2:1};}return out;
}
}
NVEncFilterT2::NVEncFilterT2():m_received(0),m_emitted(0),m_draining(false),m_errors(0),m_fp64Logged(false){m_name=_T("mcwiener");}
NVEncFilterT2::~NVEncFilterT2(){close();}
void NVEncFilterT2::close(){if(m_engine)AddMessage(RGY_LOG_DEBUG,_T("MCWiener fit statistics: %s.\n"),char_to_tstring(m_engine->statistics()).c_str());m_engine.reset();m_fp64Logged=false;m_received=m_emitted=0;m_draining=false;for(auto& f:m_metadata)f=RGYFrameInfo();}
RGY_ERR NVEncFilterT2::init(shared_ptr<NVEncFilterParam> param,shared_ptr<RGYLog> log){m_pLog=log;close();auto prm=std::dynamic_pointer_cast<NVEncFilterParamT2>(param);if(!prm)return RGY_ERR_INVALID_PARAM;
    try{if(prm->t2.profile.empty()||prm->t2.pp.empty()||prm->t2.iso.empty())throw std::runtime_error("T2 requires profile,pp,iso (iso is an entry selector)");auto ss=sampling(prm->frameIn.csp);if(rgy_csp_has_alpha(prm->frameIn.csp)||prm->frameIn.mem_type!=RGY_MEM_TYPE_GPU)throw std::runtime_error("T2 requires device YUV without alpha");if(interlaced(prm->frameIn))throw std::runtime_error("T2 requires progressive input");if(prm->frameIn.csp!=prm->frameOut.csp)throw std::runtime_error("T2 surface format must be preserved at this seam");if(RGY_CSP_BIT_DEPTH[prm->frameOut.csp]<10)throw std::runtime_error("T2 requires >=10-bit surfaces; request a 10-bit decoder/reader format");m_engine.reset(new native_t2::Engine(prm->frameIn.width,prm->frameIn.height,ss.first,ss.second,tchar_to_string(prm->t2.mode),loadModel(prm->t2),prm->t2.strength));}
    catch(const std::exception& e){++m_errors;AddMessage(RGY_LOG_ERROR,_T("T2: %s.\n"),char_to_tstring(e.what()).c_str());return RGY_ERR_INVALID_PARAM;}
    auto sts=AllocFrameBuf(prm->frameOut,2);if(sts!=RGY_ERR_NONE)return sts;for(int p=0;p<RGY_CSP_PLANES[prm->frameOut.csp];++p)prm->frameOut.pitch[p]=m_frameBuf[0]->frame.pitch[p];m_pathThrough=FILTER_PATHTHROUGH_NONE;m_param=param;setFilterInfo(prm->print());return RGY_ERR_NONE;
}
RGY_ERR NVEncFilterT2::run_filter(const RGYFrameInfo* input,RGYFrameInfo** output,int* count,cudaStream_t stream){*count=0;output[0]=nullptr;auto prm=std::dynamic_pointer_cast<NVEncFilterParamT2>(m_param);if(!prm||!m_engine)return RGY_ERR_INVALID_CALL;const bool eos=!input||!input->ptr[0];
    try{if(!eos){if(m_draining){if(m_emitted!=m_received)throw std::runtime_error("new input before T2 drain completed");AddMessage(RGY_LOG_DEBUG,_T("MCWiener completed clip fit statistics: %s.\n"),char_to_tstring(m_engine->statistics()).c_str());m_received=m_emitted=0;m_draining=false;m_fp64Logged=false;auto ss=sampling(input->csp);m_engine.reset(new native_t2::Engine(input->width,input->height,ss.first,ss.second,tchar_to_string(prm->t2.mode),loadModel(prm->t2),prm->t2.strength));}if(cmpFrameInfoCspResolution(input,&prm->frameIn)||interlaced(*input))throw std::runtime_error("T2 input format changed or interlaced");m_engine->ingest(m_received,view(*input),stream);auto& metadata=m_metadata[m_received%17];metadata=RGYFrameInfo();copyFramePropWithoutRes(&metadata,input);++m_received;}else m_draining=true;
        if(m_emitted>=m_received||(!m_draining&&m_received<=m_emitted+8))return RGY_ERR_NONE;
        auto& dst=m_frameBuf[m_emitted%2]->frame;m_engine->render(m_emitted,m_received,view(dst),stream);if(!m_fp64Logged&&m_engine->fp64ContextCount()){AddMessage(RGY_LOG_WARN,_T("MCWiener: native FP64 definition fit selected outside unchanged FP32 envelope; %s.\n"),char_to_tstring(m_engine->statistics()).c_str());m_fp64Logged=true;}copyFramePropWithoutRes(&dst,&m_metadata[m_emitted%17]);output[0]=&dst;*count=1;++m_emitted;return RGY_ERR_NONE;
    }catch(const std::exception& e){++m_errors;AddMessage(RGY_LOG_ERROR,_T("T2: %s (%s, host_errors=%llu).\n"),char_to_tstring(e.what()).c_str(),char_to_tstring(m_engine->statistics()).c_str(),(unsigned long long)m_errors);close();return RGY_ERR_CUDA;}
}
