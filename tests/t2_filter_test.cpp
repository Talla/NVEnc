// Actual NVEncFilterT2 lifecycle/metadata conformance. used_by: native port receipt.
// rules: production filter API, owned reused input, original IDs/PTS/duration/flags/data pointers.
#include "../NVEncCore/NVEncFilterT2.h"
#include "../NVEncCore/rgy_frame.h"
#include <iostream>
#include <stdexcept>
#include <vector>
static void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
struct Result {std::vector<unsigned short> pixels;};
int main(int argc,char** argv){try{
    require(argc==2,"profile argument required");cudaStream_t stream;require(cudaStreamCreate(&stream)==cudaSuccess,"stream");
    auto log=std::make_shared<RGYLog>(nullptr,RGY_LOG_ERROR);
    int cases=0;
    for(const auto& mode:{_T("plain"),_T("box"),_T("full")}){
        NVEncFilterT2 filter;auto prm=std::make_shared<NVEncFilterParamT2>();prm->t2.enable=true;prm->t2.mode=mode;prm->t2.profile=char_to_tstring(argv[1]);prm->t2.pp=_T("SIM");prm->t2.iso=_T("fixture");prm->frameIn=RGYFrameInfo(128,96,RGY_CSP_YUV422_16,16,RGY_PICSTRUCT_FRAME,RGY_MEM_TYPE_GPU);prm->frameOut=prm->frameIn;require(filter.init(prm,log)==RGY_ERR_NONE,"init");CUFrameBuf input(prm->frameIn);require(input.alloc()==RGY_ERR_NONE,"input allocation");
        for(int length:{0,1,2,8,9,16,17,18}){
            std::vector<RGYFrameInfo> props;std::vector<Result> results;for(int i=0;i<length;++i){auto meta=prm->frameIn;meta.timestamp=12345+i*13+i%3;meta.duration=11+i%3;meta.inputFrameId=900+i;meta.flags=(i%2)?RGY_FRAME_FLAG_RFF:RGY_FRAME_FLAG_NONE;meta.dataList.push_back(std::make_shared<RGYFrameData>());props.push_back(meta);}
            auto consume=[&](RGYFrameInfo* out,int n){require(n==0||n==1,"output count per call");require((n==0)==(out==nullptr),"null output convention");if(!n)return;int j=int(results.size());require(j<length,"extra output");auto& expected=props[j];require(out->timestamp==expected.timestamp&&out->duration==expected.duration&&out->inputFrameId==expected.inputFrameId,"origin PTS/duration/id");require(out->flags==expected.flags&&out->picstruct==expected.picstruct&&out->dataList==expected.dataList,"origin flags/picstruct/dataList");require(cudaStreamSynchronize(stream)==cudaSuccess,"output sync");auto y=getPlane(out,RGY_PLANE_Y);Result r;r.pixels.resize(128*96);require(cudaMemcpy2D(r.pixels.data(),256,y.ptr[0],y.pitch[0],256,96,cudaMemcpyDeviceToHost)==cudaSuccess,"output download");results.push_back(std::move(r));};
            auto execute=[&](NVEncFilterT2& target){
            for(int i=0;i<length;++i){for(int p=0;p<3;++p){auto v=getPlane(&input.frame,(RGY_PLANE)p);std::vector<unsigned short> host(v.width*v.height);for(size_t j=0;j<host.size();++j)host[j]=(unsigned short)(20000+(i*313+j*17+p*41)%1000);require(cudaMemcpy2DAsync(v.ptr[0],v.pitch[0],host.data(),v.width*2,v.width*2,v.height,cudaMemcpyHostToDevice,stream)==cudaSuccess,"input upload");require(cudaStreamSynchronize(stream)==cudaSuccess,"upload lifetime");}copyFramePropWithoutRes(&input.frame,&props[i]);RGYFrameInfo* out=nullptr;int n=0;require(target.filter(&input.frame,&out,&n,stream)==RGY_ERR_NONE,"feed");consume(out,n);}
            for(int call=0;call<length+3;++call){RGYFrameInfo* out=nullptr;int n=0;require(target.filter(nullptr,&out,&n,stream)==RGY_ERR_NONE,"drain");consume(out,n);}require(int(results.size())==length,"exact count after repeated EOS");
            };execute(filter);
            if(length==18){auto reused=results;results.clear();NVEncFilterT2 isolated;require(isolated.init(prm,log)==RGY_ERR_NONE,"independent instance init");execute(isolated);require(reused.size()==results.size(),"second instance count");for(size_t j=0;j<results.size();++j)require(reused[j].pixels==results[j].pixels,"second clip has no stale history");}
            ++cases;
        }
    }
    cudaStreamDestroy(stream);std::cout<<"{\"cases\":"<<cases<<",\"result\":\"PASS\",\"scope\":\"3 modes x lengths0,1,2,8,9,16,17,18; reused filter second clips + exact comparison to isolated instance; VFR nonzero PTS, IDs, flags, side-data identity; repeated EOS\"}\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
