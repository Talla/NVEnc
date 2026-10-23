// Bounded raw adapter for native T2 numerical conformance. used_by: test_t2_native_port.py
// rules: no surrogate math, production Engine and typed plane adapter; writes fresh caller paths.
#include "../NVEncCore/t2/t2_native.h"
#include "../json/json.hpp"
#include <fstream>
#include <iostream>
#include <chrono>
#include <stdexcept>
#include <cstdlib>
static void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
int main(int argc,char** argv){try{
    if(argc!=10&&argc!=11&&argc!=13)throw std::runtime_error("profile mode width height chroma count input_u16 output_u16 bits");
    int w=std::stoi(argv[3]),h=std::stoi(argv[4]),count=std::stoi(argv[6]),bits=std::stoi(argv[9]);int sx=std::string(argv[5])=="444"?1:2,sy=std::string(argv[5])=="420"?2:1;
    nlohmann::json doc;std::ifstream pf(argv[1]);pf>>doc;const auto& e=doc["picture_profiles"][argc==13?argv[11]:"SIM"]["noise_profiles"][argc==13?argv[12]:"fixture"]["extended"];
    native_t2::Model m;const char* names[]={"Y","Cb","Cr"};for(int p=0;p<3;++p){auto& a=m.planes[p];auto levels=e["nlf"]["levels"].get<std::vector<double>>();const auto& sigmas=e["nlf"]["sigma"][names[p]];for(size_t i=0;i<levels.size();++i)if(!sigmas[i].is_null()){a.levels.push_back(levels[i]);a.sigma.push_back(sigmas[i].get<double>()/255);}auto ps=e["psd"][e["psd"].size()/2][names[p]];a.psdSize=int(ps.size());for(auto& row:ps)for(auto& v:row)a.psd.push_back(v.get<double>());a.rho=std::max(0.,std::min(.95,e["temporal_corr"][names[p]]["HF"][0].get<double>()));}
    native_t2::Engine engine(w,h,sx,sy,argv[2],std::move(m),argc>=11?std::strtod(argv[10],nullptr):1.0);cudaStream_t stream;check(cudaStreamCreate(&stream));native_t2::FrameView in{},out{};std::vector<unsigned short> host[3];size_t total=0;for(int p=0;p<3;++p){int W=w/(p?sx:1),H=h/(p?sy:1);host[p].resize(size_t(W)*H);void *a,*b;check(cudaMalloc(&a,W*H*2));check(cudaMalloc(&b,W*H*2));in.plane[p]={a,size_t(W)*2,W,H,2,0,bits,1};out.plane[p]={b,size_t(W)*2,W,H,2,0,bits,1};total+=W*H*2;}
    std::ifstream input(argv[7],std::ios::binary);std::ofstream output(argv[8],std::ios::binary);if(!input||!output)throw std::runtime_error("test file open failed");int emitted=0;std::vector<double> ms;
    auto emit=[&](){auto begin=std::chrono::steady_clock::now();engine.render(emitted,count<emitted+9?count:emitted+9,out,stream);check(cudaStreamSynchronize(stream));ms.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count());for(int p=0;p<3;++p){check(cudaMemcpy(host[p].data(),out.plane[p].ptr,host[p].size()*2,cudaMemcpyDeviceToHost));output.write((const char*)host[p].data(),host[p].size()*2);}++emitted;};
    auto begin=std::chrono::steady_clock::now();for(int i=0;i<count;++i){for(int p=0;p<3;++p){input.read((char*)host[p].data(),host[p].size()*2);if(!input)throw std::runtime_error("truncated input");check(cudaMemcpyAsync(in.plane[p].ptr,host[p].data(),host[p].size()*2,cudaMemcpyHostToDevice,stream));}engine.ingest(i,in,stream);if(i>=8)emit();}while(emitted<count)emit();check(cudaStreamSynchronize(stream));double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();double sum=0,max=0;for(double x:ms){sum+=x;max=std::max(max,x);}std::cout<<"{\"frames\":"<<emitted<<",\"total_ms\":"<<elapsed<<",\"render_mean_ms\":"<<(ms.empty()?0:sum/ms.size())<<",\"render_max_ms\":"<<max<<",\"statistics\":\""<<engine.statistics()<<"\"}\n";for(int p=0;p<3;++p){cudaFree(in.plane[p].ptr);cudaFree(out.plane[p].ptr);}cudaStreamDestroy(stream);return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
