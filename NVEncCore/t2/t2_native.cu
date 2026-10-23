// Native translation of the frozen T2 plain/box/full orchestration (2026-10-20).
// used_by: NVEncFilterT2.cpp; rules: source samples never prefiltered for reconstruction,
// FP32 image accumulation, FP64 fit ABI, owned 17-slot history, caller's persistent stream.
#include "t2_native.h"
#include <cufft.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <fstream>
#include <cstdlib>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_reduce.cuh>
// These sources are verbatim copies of the validated lab launchers. See PROVENANCE.md.
#include "cuda_kernels.cu"
// Compile-only SM75 integer reduction compatibility. Production Engine still
// rejects devices below8.6; >=SM80 uses the unchanged validated intrinsics.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
__device__ int native_t2_warp_add_legacy(unsigned mask,int v){for(int d=16;d;d>>=1)v+=__shfl_xor_sync(mask,v,d);return v;}
__device__ unsigned native_t2_warp_min_legacy(unsigned mask,unsigned v){for(int d=16;d;d>>=1)v=min(v,__shfl_xor_sync(mask,v,d));return v;}
__device__ unsigned native_t2_warp_max_legacy(unsigned mask,unsigned v){for(int d=16;d;d>>=1)v=max(v,__shfl_xor_sync(mask,v,d));return v;}
#define __reduce_add_sync native_t2_warp_add_legacy
#define __reduce_min_sync native_t2_warp_min_legacy
#define __reduce_max_sync native_t2_warp_max_legacy
#endif
#include "t2_field_kernels.cu"
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#undef __reduce_add_sync
#undef __reduce_min_sync
#undef __reduce_max_sync
#endif
#include "t2_strength_kernels.cuh"
#include "t2_fp64_kernels.cuh"

namespace native_t2 {
static void ck(cudaError_t e) { if(e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
static void fk(cufftResult e) { if(e != CUFFT_SUCCESS) throw std::runtime_error("T2 cuFFT error " + std::to_string(e)); }
template<class T> struct Buf {
    T *p=nullptr; size_t n=0;
    Buf()=default; explicit Buf(size_t count) { size(count); }
    Buf(const Buf&)=delete; Buf& operator=(const Buf&)=delete;
    Buf(Buf&& b) noexcept : p(b.p),n(b.n) { b.p=nullptr;b.n=0; }
    Buf& operator=(Buf&& b) noexcept { if(this!=&b){if(p)cudaFree(p);p=b.p;n=b.n;b.p=nullptr;b.n=0;}return *this; }
    ~Buf(){if(p)cudaFree(p);}
    void size(size_t count){if(count>n){if(p)ck(cudaFree(p));p=nullptr;ck(cudaMalloc((void**)&p,count*sizeof(T)));n=count;}}
    void put(const std::vector<T>& v,cudaStream_t s){size(v.size());ck(cudaMemcpyAsync(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice,s));ck(cudaStreamSynchronize(s));}
    std::vector<T> get(size_t count,cudaStream_t s) const {std::vector<T> v(count);ck(cudaMemcpyAsync(v.data(),p,count*sizeof(T),cudaMemcpyDeviceToHost,s));ck(cudaStreamSynchronize(s));return v;}
};
struct Image { int h=0,w=0; Buf<float> b; Image()=default; Image(int hh,int ww):h(hh),w(ww),b(size_t(hh)*ww){} };
static int blocks(size_t n){return int((n+255)/256);}
template<class T> static void trace_buffer(const Buf<T>& b,size_t n,const std::string& name,cudaStream_t stream){const char* dir=std::getenv("T2_NATIVE_TRACE");if(!dir)return;auto v=b.get(n,stream);std::ofstream f(std::string(dir)+"/"+name,std::ios::binary);if(!f)throw std::runtime_error("T2 diagnostic trace path unavailable");f.write((const char*)v.data(),v.size()*sizeof(T));}
constexpr double PI=3.14159265358979323846;
static double low(double f){double lo=std::pow(2.,-.25)/16.,hi=std::pow(2.,.25)/16.;return f<=lo?1.:f>=hi?0.:.5+.5*std::cos(PI*(std::log2(f)-std::log2(lo))/.5);}
static double freq(int i,int n){return double(i<(n+1)/2?i:i-n)/n;}
static double median(std::vector<double> v){v.erase(std::remove_if(v.begin(),v.end(),[](double x){return !std::isfinite(x);}),v.end());if(v.empty())return NAN;size_t k=(v.size()-1)/2;std::nth_element(v.begin(),v.begin()+k,v.end());return v[k];}
static double sigma(const PlaneModel& m,double x){if(x<=m.levels.front())return m.sigma.front();if(x>=m.levels.back())return m.sigma.back();auto j=std::upper_bound(m.levels.begin(),m.levels.end(),x)-m.levels.begin();double t=(x-m.levels[j-1])/(m.levels[j]-m.levels[j-1]);return m.sigma[j-1]+t*(m.sigma[j]-m.sigma[j-1]);}

__global__ void read_plane(PlaneView v,float *out){int i=blockIdx.x*256+threadIdx.x;if(i>=v.width*v.height)return;const char* row=(const char*)v.ptr+(i/v.width)*v.pitch;int x=(i%v.width)*v.step;unsigned code=v.bytes==1?((const unsigned char*)row)[x]:((const unsigned short*)row)[x];out[i]=float(code>>v.shift)/float((1u<<v.bits)-1);}
__global__ void write_plane(PlaneView v,const float *in){int i=blockIdx.x*256+threadIdx.x;if(i>=v.width*v.height)return;char* row=(char*)v.ptr+(i/v.width)*v.pitch;int x=(i%v.width)*v.step;float val=fminf(1.f,fmaxf(0.f,in[i]));unsigned code=(unsigned)nearbyintf(val*float((1u<<v.bits)-1));if(v.bytes==1)((unsigned char*)row)[x]=(unsigned char)code;else ((unsigned short*)row)[x]=(unsigned short)(code<<v.shift);}
__global__ void pool(const float* in,float* out,int h,int w,int oh,int ow){int i=blockIdx.x*256+threadIdx.x;if(i>=oh*ow)return;int x=i%ow,y=i/ow,x0=x*w/ow,x1=((x+1)*w+ow-1)/ow,y0=y*h/oh,y1=((y+1)*h+oh-1)/oh;float z=0;for(int yy=y0;yy<y1;++yy)for(int xx=x0;xx<x1;++xx)z+=in[yy*w+xx];out[i]=z/((x1-x0)*(y1-y0));}
__global__ void resize_flow(const float* in,float* out,int h,int w,int oh,int ow,int corners,float mx,float my){int i=blockIdx.x*256+threadIdx.x;if(i>=2*oh*ow)return;int a=i/(oh*ow),j=i%(oh*ow),y=j/ow,x=j%ow;float fy=corners?(oh>1?float(y)*(h-1)/(oh-1):0):fmaxf(0.f,(y+.5f)*h/oh-.5f);float fx=corners?(ow>1?float(x)*(w-1)/(ow-1):0):fmaxf(0.f,(x+.5f)*w/ow-.5f);int y0=(int)fy,x0=(int)fx,y1=min(h-1,y0+1),x1=min(w-1,x0+1);fy-=y0;fx-=x0;const float* p=in+a*h*w;float top=p[y0*w+x0]+fx*(p[y0*w+x1]-p[y0*w+x0]);float bot=p[y1*w+x0]+fx*(p[y1*w+x1]-p[y1*w+x0]);out[i]=(top+fy*(bot-top))*(a?my:mx);}
__global__ void pad_complex(const float* in,float2* out,int h,int w,int hp,int wp,int py,int px){int i=blockIdx.x*256+threadIdx.x;if(i>=hp*wp)return;out[i]=make_float2(in[mirror_index(i/wp-py,h)*w+mirror_index(i%wp-px,w)],0);}
__global__ void project(const float2* in,float* out,int hp,int wp,int h,int w,int py,int px,float scale){int i=blockIdx.x*256+threadIdx.x;if(i<h*w)out[i]=in[(i/w+py)*wp+i%w+px].x*scale;}
__global__ void to_complex(const float* in,float2* out,int n){int i=blockIdx.x*256+threadIdx.x;if(i<n)out[i]=make_float2(in[i],0);}
__global__ void spectral_gain(float2* p,int h,int w,int sx,int sy,int spline){int i=blockIdx.x*256+threadIdx.x;if(i>=h*w)return;int y=i/w,x=i%w;float fy=float(y<(h+1)/2?y:y-h)/h,fx=float(x<(w+1)/2?x:x-w)/w;float gain;if(spline)gain=36.f/((4+2*cosf(2*float(PI)*fy))*(4+2*cosf(2*float(PI)*fx)));else {float f=hypotf(fx/sx,fy/sy),lo=powf(2.f,-.25f)/16,hi=powf(2.f,.25f)/16;gain=f<=lo?1:f>=hi?0:.5f+.5f*cosf(float(PI)*(log2f(f)-log2f(lo))/.5f);}p[i].x*=gain;p[i].y*=gain;}
__global__ void coarse_extract_full(const float2* in,float2* out,int hp,int wp,int hc,int wc,int area){int i=blockIdx.x*256+threadIdx.x;if(i>=hc*wc)return;int y=i/wc,x=i%wc,ky=y<(hc+1)/2?y:y-hc,kx=x<(wc+1)/2?x:x-wc;out[i]=abs(ky)*2>=hc||abs(kx)*2>=wc?make_float2(0,0):make_float2(in[((ky+hp)%hp)*wp+(kx+wp)%wp].x/area,in[((ky+hp)%hp)*wp+(kx+wp)%wp].y/area);}
__global__ void coarse_embed_full(const float2* in,float2* out,int hp,int wp,int hc,int wc,int sx,int sy,int area){int i=blockIdx.x*256+threadIdx.x;if(i>=hp*wp)return;int y=i/wp,x=i%wp,ky=y<(hp+1)/2?y:y-hp,kx=x<(wp+1)/2?x:x-wp;float2 z=make_float2(0,0);float f=hypotf(float(kx)/wp/sx,float(ky)/hp/sy);if(abs(ky)*2<hc&&abs(kx)*2<wc&&f<powf(2.f,.25f)/16){z=in[((ky+hc)%hc)*wc+(kx+wc)%wc];z.x*=area;z.y*=area;}out[i]=z;}
__global__ void subtract(float* a,const float* b,int n){int i=blockIdx.x*256+threadIdx.x;if(i<n)a[i]-=b[i];}
__global__ void crop_add(float* a,const float2* b,int h,int w,int hp,int wp,int py,int px){int i=blockIdx.x*256+threadIdx.x;if(i<h*w)a[i]+=b[(i/w+py)*wp+i%w+px].x/(hp*wp);}
__global__ void gradient_products(const float* in,float* out,int h,int w){int i=blockIdx.x*256+threadIdx.x;if(i>=h*w)return;int x=i%w,y=i/w;float gx=.5f*(in[y*w+min(w-1,x+1)]-in[y*w+max(0,x-1)]),gy=.5f*(in[min(h-1,y+1)*w+x]-in[max(0,y-1)*w+x]);out[i]=gx*gx;out[h*w+i]=gy*gy;out[2*h*w+i]=gx*gy;}
__global__ void confidence(const float* in,float* out,int n){int i=blockIdx.x*256+threadIdx.x;if(i>=n)return;float xx=in[i],yy=in[n+i],xy=in[2*n+i];out[i]=.5f*(xx+yy-sqrtf(fmaxf(0,(xx-yy)*(xx-yy)+4*xy*xy)));}
__global__ void divide(float* p,int n,float den){int i=blockIdx.x*256+threadIdx.x;if(i<n)p[i]/=den;}
__global__ void coarse_variance(const float* var,float* out,int n,float rho){int i=blockIdx.x*256+threadIdx.x;if(i<n)out[i]=(2.f*var[i])*(1.f-rho);}
__global__ void tile_vectors(const float* flow,float* out,int h,int w,int H,int W,int oy,int ox,int nx,int count){int i=blockIdx.x*256+threadIdx.x;if(i>=2*count)return;int tile=i/2,a=i%2;float sum=0;for(int yy=0;yy<16;++yy)for(int xx=0;xx<16;++xx){int y=mirror_index((tile/nx)*8-16-oy+yy,H),x=mirror_index((tile%nx)*8-16-ox+xx,W);float fy=fmaxf(0.f,float(h)/H*(y+.5f)-.5f),fx=fmaxf(0.f,float(w)/W*(x+.5f)-.5f);int y0=int(fy),x0=int(fx),y1=min(h-1,y0+1),x1=min(w-1,x0+1);fy-=y0;fx-=x0;const float* p=flow+a*h*w;float top=p[y0*w+x0]+fx*(p[y0*w+x1]-p[y0*w+x0]),bot=p[y1*w+x0]+fx*(p[y1*w+x1]-p[y1*w+x0]);sum+=top+fy*(bot-top);}out[i]=sum/256.f*(a?float(H)/h:float(W)/w);}
__global__ void alignment(const float* v,long long* sh,float* fr,int n){int i=blockIdx.x*256+threadIdx.x;if(i<n){float r=nearbyintf(v[i]);sh[i]=(long long)r;fr[i]=v[i]-r;}}
__global__ void plane_vectors(const float* v,float* out,int h,int w,int H,int W,int sx,int sy,int oy,int ox,int nx,int lnx,int lny,int count){int i=blockIdx.x*256+threadIdx.x;if(i>=2*count)return;int tile=i/2,a=i%2;float cx=((tile%nx)*8-8-ox)*sx,cy=((tile/nx)*8-8-oy)*sy;int jx=max(0,min(lnx-1,(int)nearbyintf((cx+8+ox)/8))),jy=max(0,min(lny-1,(int)nearbyintf((cy+8+oy)/8)));out[i]=v[2*(jy*lnx+jx)+a]/(a?sy:sx);}
__global__ void means(const float* in,float* out,int h,int w,int oy,int ox,int nx,int count){int tile=blockIdx.x*256+threadIdx.x;if(tile>=count)return;float v=0;for(int y=0;y<16;++y)for(int x=0;x<16;++x)v+=in[mirror_index(tile/nx*8-16-oy+y,h)*w+mirror_index(tile%nx*8-16-ox+x,w)];out[tile]=v/256;}
__global__ void variance(const float* level,const float* xp,const float* fp,int len,const float* shape,float* out,int count){int i=blockIdx.x*256+threadIdx.x;if(i>=count*256)return;float x=level[i/256],v=fp[0];if(x>=xp[len-1])v=fp[len-1];else if(x>xp[0]){int j=1;while(j<len&&xp[j]<x)++j;float t=fminf(1.f,fmaxf(0.f,(x-xp[j-1])/(xp[j]-xp[j-1])));v=fp[j-1]+t*(fp[j]-fp[j-1]);}out[i]=(v*v*8.f)*shape[i%256]*36.f;}
__global__ void center_tiles(const float* in,const float* win,float2* out,float* level,int h,int w,int oy,int ox,int nx){__shared__ float v[256];int tile=blockIdx.x,tid=threadIdx.x;int y=mirror_index(tile/nx*8-16-oy+(tid>>4),h),x=mirror_index(tile%nx*8-16-ox+(tid&15),w);v[tid]=in[y*w+x];__syncthreads();float mean=0,raw=0;for(int j=0;j<256;++j){mean+=v[j]*win[j];raw+=v[j];}out[tile*256+tid]=make_float2((v[tid]-mean/64.f)*win[tid],0);if(!tid)level[tile]=raw/256.f;}
__global__ void texture_power(const float2* p,const float* var,double* out,int count){int tile=blockIdx.x*256+threadIdx.x;if(tile>=count)return;float z=0,n=0;for(int j=1;j<256;++j){float2 a=p[tile*256+j];z+=a.x*a.x+a.y*a.y;n+=var[tile*256+j]/8.f;}out[tile]=double(z-n);}
// Score the original centred, reflect-padded noisy tiles. This uses the copied 16x16 FFT.
__global__ void score(const float* ref,const float* nb,const float* v,const float* win,const float* var,float* out,int h,int w,int oy,int ox,int nx){__shared__ float2 b[272],tmp[272],tw[15];__shared__ float vals[256];int tile=blockIdx.x,tid=threadIdx.x,idx=(tid>>4)*17+(tid&15);if(tid<15)tw[tid]=T2_TW[tid];float2 spectra[2];for(int side=0;side<2;++side){int dx=side?(int)nearbyintf(v[2*tile]):0,dy=side?(int)nearbyintf(v[2*tile+1]):0;int y=max(-96,min(h+95,tile/nx*8-16-oy+(tid>>4)+dy)),x=max(-96,min(w+95,tile%nx*8-16-ox+(tid&15)+dx));vals[tid]=(side?nb:ref)[mirror_index(y,h)*w+mirror_index(x,w)];__syncthreads();float mean=0;for(int j=0;j<256;++j)mean+=vals[j]*win[j];b[idx]=make_float2((vals[tid]-mean/64.f)*win[tid],0);__syncthreads();t2_fft16_shared(b,tmp,tw,0,1);spectra[side]=b[idx];__syncthreads();}int fx=tid&15,fy=tid>>4;if(fx>=8)fx-=16;if(fy>=8)fy-=16;float angle=2*float(PI)/16*(fx*(v[2*tile]-nearbyintf(v[2*tile]))+fy*(v[2*tile+1]-nearbyintf(v[2*tile+1]))),sn,cs;__sincosf(angle,&sn,&cs);float2 z=spectra[1];float dx=spectra[0].x-(z.x*cs-z.y*sn),dy=spectra[0].y-(z.x*sn+z.y*cs);vals[tid]=tid?(dx*dx+dy*dy)/(2*fmaxf(var[tile*256+tid]/8.f,1e-12f)):0;__syncthreads();if(!tid){float sum=0;for(int j=0;j<256;++j)sum+=vals[j];out[tile]=sum/255.f;}}

// Host-only exact lower median seeds for the existing FP64 direct local fit.
extern "C" DL_EXPORT int dl_t2_native_median_seed64_host(const double* Y,const double* W,double* beta,unsigned char* finite,int B,int ny,int nx,int r){
 if(!Y||!W||!beta||!finite||B<1||B>16||ny<1||nx<1||r<0||r>24)return int(cudaErrorInvalidValue);
 const int N=ny*nx;std::vector<double> values[2];for(auto& a:values)a.reserve((2*r+1)*(2*r+1));
 for(int b=0;b<B;++b)for(int y=0;y<ny;++y)for(int x=0;x<nx;++x){for(auto& a:values)a.clear();for(int yy=std::max(0,y-r);yy<=std::min(ny-1,y+r);++yy)for(int xx=std::max(0,x-r);xx<=std::min(nx-1,x+r);++xx){int j=yy*nx+xx;if(W[b*N+j]>0){values[0].push_back(Y[(b*2)*N+j]);values[1].push_back(Y[(b*2+1)*N+j]);}}int i=b*N+y*nx+x;finite[i]=!values[0].empty();for(int a=0;a<2;++a)beta[i*2+a]=finite[i]?median(values[a]):0.;}
 return 0;
}

struct Engine::Impl {
    struct Source { int64_t index=-1; std::array<Image,3> image,coarse; std::vector<Image> pyr,coeff,conf; };
    int h,w,sx,sy,device; std::string mode; Model model; double strength; cudaStream_t stream=nullptr; bool bound=false;
    std::array<Source,17> ring;
    Buf<float> win; Buf<float2> ws,basis; std::array<Buf<float>,3> xp,fp,shapes,cshapes;
    std::map<std::pair<int,int>,cufftHandle> plans;
    std::map<int,Buf<float>> gaussian;
    uint64_t frames=0,pairs=0,refined=0,domainErrors=0,fp32DomainExits=0,fp64BoxFits=0,fp64LocalFits=0; std::string domainNote;
    Impl(int W,int H,int X,int Y,std::string M,Model mm,double C):h(H),w(W),sx(X),sy(Y),mode(std::move(M)),model(std::move(mm)),strength(C){ck(cudaGetDevice(&device));cudaDeviceProp props{};ck(cudaGetDeviceProperties(&props,device));if(props.major*10+props.minor<86)throw std::runtime_error("MCWiener requires compute capability 8.6 or newer");}
    ~Impl(){for(auto& x:plans)cufftDestroy(x.second);}
    void bind(cudaStream_t s){if(bound&&stream!=s)throw std::runtime_error("T2 stream changed within one filter instance");stream=s;bound=true;}
    void fft(Buf<float2>& b,int H,int W,bool inverse){auto key=std::make_pair(H,W);auto it=plans.find(key);if(it==plans.end()){cufftHandle plan;fk(cufftPlan2d(&plan,H,W,CUFFT_C2C));it=plans.emplace(key,plan).first;}fk(cufftSetStream(it->second,stream));fk(cufftExecC2C(it->second,b.p,b.p,inverse?CUFFT_INVERSE:CUFFT_FORWARD));}
    Buf<float>& kernel(float sig){int k=int(sig*10+.5f);auto it=gaussian.find(k);if(it==gaussian.end()){int r=int(std::ceil(3*sig));std::vector<float> v(2*r+1);float sum=0;for(int j=-r;j<=r;++j){v[j+r]=std::exp(-.5f*(j/sig)*(j/sig));sum+=v[j+r];}for(auto& x:v)x/=sum;Buf<float> b;b.put(v,stream);it=gaussian.emplace(k,std::move(b)).first;}return it->second;}
    Image blur(const Image& a,float sig,int channels=1){Image b(a.h,a.w);b.b.size(size_t(a.h)*a.w*channels);Buf<float> tmp(size_t(a.h)*a.w*channels);auto& k=kernel(sig);ck((cudaError_t)dl_blur(a.b.p,tmp.p,b.b.p,k.p,1,channels,a.h,a.w,int(k.n/2),device,stream));return b;}
    Image down(const Image& a,int factor){Image b(a.h/factor,a.w/factor);pool<<<blocks(b.h*b.w),256,0,stream>>>(a.b.p,b.b.p,a.h,a.w,b.h,b.w);return b;}
    Image spline(const Image& a){int hp=a.h+32,wp=a.w+32;Buf<float2> z(size_t(hp)*wp);pad_complex<<<blocks(hp*wp),256,0,stream>>>(a.b.p,z.p,a.h,a.w,hp,wp,16,16);fft(z,hp,wp,false);spectral_gain<<<blocks(hp*wp),256,0,stream>>>(z.p,hp,wp,1,1,1);fft(z,hp,wp,true);Image out(a.h,a.w);project<<<blocks(a.h*a.w),256,0,stream>>>(z.p,out.b.p,hp,wp,a.h,a.w,16,16,1.f/(hp*wp));return out;}
    Image conf(const Image& a){Image p(a.h,a.w);p.b.size(size_t(3)*a.h*a.w);gradient_products<<<blocks(a.h*a.w),256,0,stream>>>(a.b.p,p.b.p,a.h,a.w);Image b=blur(p,2.f,3);Image c(a.h,a.w);confidence<<<blocks(a.h*a.w),256,0,stream>>>(b.b.p,c.b.p,a.h*a.w);auto v=c.b.get(a.h*a.w,stream);float maxval=std::max(1e-12f,*std::max_element(v.begin(),v.end()));divide<<<blocks(a.h*a.w),256,0,stream>>>(c.b.p,a.h*a.w,maxval);return c;}
    std::pair<int,int> padded(int H,int W,int X,int Y){int dx=4/X,dy=4/Y;return {H+64/Y+(dy-H%dy)%dy,W+64/X+(dx-W%dx)%dx};}
    Image decimate(const Image& a,int X,int Y){auto hw=padded(a.h,a.w,X,Y);int hp=hw.first,wp=hw.second,hc=hp/(4/Y),wc=wp/(4/X),area=16/(X*Y);Buf<float2> z(size_t(hp)*wp),c(size_t(hc)*wc);pad_complex<<<blocks(hp*wp),256,0,stream>>>(a.b.p,z.p,a.h,a.w,hp,wp,32/Y,32/X);fft(z,hp,wp,false);spectral_gain<<<blocks(hp*wp),256,0,stream>>>(z.p,hp,wp,X,Y,0);coarse_extract_full<<<blocks(hc*wc),256,0,stream>>>(z.p,c.p,hp,wp,hc,wc,area);fft(c,hc,wc,true);Image out(hc,wc);project<<<blocks(hc*wc),256,0,stream>>>(c.p,out.b.p,hc,wc,hc,wc,0,0,1.f/(hc*wc));return out;}
    void recombine(Image& a,Image& c,int X,int Y){Image d=decimate(a,X,Y);subtract<<<blocks(c.h*c.w),256,0,stream>>>(c.b.p,d.b.p,c.h*c.w);auto hw=padded(a.h,a.w,X,Y);int hp=hw.first,wp=hw.second;Buf<float2> z(size_t(c.h)*c.w),full(size_t(hp)*wp);to_complex<<<blocks(c.h*c.w),256,0,stream>>>(c.b.p,z.p,c.h*c.w);fft(z,c.h,c.w,false);coarse_embed_full<<<blocks(hp*wp),256,0,stream>>>(z.p,full.p,hp,wp,c.h,c.w,X,Y,16/(X*Y));fft(full,hp,wp,true);crop_add<<<blocks(a.h*a.w),256,0,stream>>>(a.b.p,full.p,a.h,a.w,hp,wp,32/Y,32/X);}
    void constants();
    void ingest(int64_t index,const FrameView& frame);
    Buf<float> motion(Source& a,Source& b,int oy,int ox,Buf<float>* dense);
    Buf<float> vars(const Image& levelImage,int p,int oy,int ox,bool coarse);
    Image reconstruction(const Image& ref,const std::vector<const Image*>& neighbours,const std::vector<Buf<float>>& v,const std::vector<int>& use,const Image& levelImage,int p,int oy,int ox,bool coarse);
    void s1(std::array<Image,3>& res);
    void fit(std::vector<Buf<float>>& v,const std::vector<int>& ks,Source& ref,int oy,int ox);
    void render(int64_t index,int64_t received,const FrameView& out);
};

void Engine::Impl::constants(){
    if(win.p)return;
    std::vector<float> window(256);std::vector<float2> spectrum(256),b(256);
    for(int y=0;y<16;++y)for(int x=0;x<16;++x)window[y*16+x]=float((.5-.5*std::cos(2*PI*y/16))*(.5-.5*std::cos(2*PI*x/16)));
    for(int y=0;y<16;++y)for(int x=0;x<16;++x){std::complex<double> z=0.;for(int j=0;j<16;++j)for(int i=0;i<16;++i)z+=double(window[j*16+i])*std::polar(1.,-2*PI*(double(y*j+x*i)/16));spectrum[y*16+x]=make_float2(float(z.real()),float(z.imag()));b[y*16+x]=make_float2(float(z.real()/64),float(z.imag()/64));}
    win.put(window,stream);ws.put(spectrum,stream);basis.put(b,stream);
    for(int p=0;p<3;++p){auto& m=model.planes[p];std::vector<float> a(m.levels.begin(),m.levels.end()),s(m.sigma.begin(),m.sigma.end()),shape(256),coarse(256);xp[p].put(a,stream);fp[p].put(s,stream);int n=m.psdSize;
        auto ps=[&](int y,int x){return m.psd[((y+n/2)%n)*n+(x+n/2)%n];};
        double sum=0;for(int y=0;y<16;++y)for(int x=0;x<16;++x){int iy=0,ix=0;for(int j=1;j<n;++j){if(std::abs(freq(j,n)-freq(y,16))<std::abs(freq(iy,n)-freq(y,16)))iy=j;if(std::abs(freq(j,n)-freq(x,16))<std::abs(freq(ix,n)-freq(x,16)))ix=j;}sum+=shape[y*16+x]=float(ps(iy,ix));}for(auto& x:shape)x=float(x/(sum/256));shapes[p].put(shape,stream);
        // Exact 128-grid propagation of the PSD through coarse lowpass and periodic Hann.
        int X=p?sx:1,Y=p?sy:1;std::vector<double> A(16*128),S(128*128),tmp(16*128);
        for(int k=0;k<16;++k)for(int j=0;j<128;++j){std::complex<double> z=0.;for(int t=0;t<16;++t)z+=(.5-.5*std::cos(2*PI*t/16))*std::polar(1.,-2*PI*(freq(k,16)-freq(j,128))*t);A[k*128+j]=std::norm(z);}
        double dc=.25*(ps(1,0)+ps(n-1,0)+ps(0,1)+ps(0,n-1));
        auto src=[&](int y,int x){y=(y%n+n)%n;x=(x%n+n)%n;return (x||y)?ps(y,x):dc;};
        for(int y=0;y<128;++y)for(int x=0;x<128;++x){double fy=freq(y,128)/4,fx=freq(x,128)/4,py=fy*Y*n,px=fx*X*n;int iy=int(std::floor(py)),ix=int(std::floor(px));double ty=py-iy,tx=px-ix;double val=(src(iy,ix)*(1-tx)+src(iy,ix+1)*tx)*(1-ty)+(src(iy+1,ix)*(1-tx)+src(iy+1,ix+1)*tx)*ty;double H=low(std::hypot(fy,fx));S[y*128+x]=val*H*H/(16/(X*Y));}
        for(int k=0;k<16;++k)for(int x=0;x<128;++x){double z=0;for(int y=0;y<128;++y)z+=A[k*128+y]*S[y*128+x];tmp[k*128+x]=z;}
        for(int y=0;y<16;++y)for(int x=0;x<16;++x){double z=0;for(int j=0;j<128;++j)z+=tmp[y*128+j]*A[x*128+j];coarse[y*16+x]=float(z/(128*128*36.));}cshapes[p].put(coarse,stream);
    }
}

void Engine::Impl::ingest(int64_t index,const FrameView& frame){
    constants();Source& s=ring[index%17];s.index=index;
    for(int p=0;p<3;++p){auto& v=frame.plane[p];s.image[p]=Image(v.height,v.width);read_plane<<<blocks(v.height*v.width),256,0,stream>>>(v,s.image[p].b.p);s.coarse[p]=decimate(s.image[p],p?sx:1,p?sy:1);}
    s.pyr.clear();s.coeff.clear();s.conf.clear();Image g=blur(s.image[0],1.f);s.pyr.push_back(down(g,4));for(int j=1;j<3;++j){Image b=blur(s.pyr.back(),.8f);s.pyr.push_back(down(b,2));}for(auto& a:s.pyr){s.coeff.push_back(spline(a));s.conf.push_back(conf(a));}ck(cudaGetLastError());
}

Buf<float> Engine::Impl::motion(Source& a,Source& b,int oy,int ox,Buf<float>* dense){
    int levels=1;while(levels<3&&std::min(a.pyr[levels-1].h,a.pyr[levels-1].w)>=16)++levels;
    Buf<float> fb;int ph=0,pw=0;int block=std::min(8,std::min(a.pyr[0].h,a.pyr[0].w));
    for(int lvl=levels-1;lvl>=0;--lvl){auto& A=a.pyr[lvl];auto& B=b.pyr[lvl];int by=A.h/block,bx=A.w/block,n=by*bx,r=lvl==levels-1?8:4;Buf<float> init(2*n),cost(size_t((2*r+1)*(2*r+1))*n),out(2*n);if(!fb.p)ck(cudaMemsetAsync(init.p,0,2*n*sizeof(float),stream));else ck((cudaError_t)dl_bm_init(fb.p,init.p,ph,pw,by,bx,float(ph)/by,float(pw)/bx,device,stream));if(block==8)ck((cudaError_t)dl_sad_warp(A.b.p,B.b.p,init.p,cost.p,A.h,A.w,r,device,stream));else ck((cudaError_t)dl_sad(A.b.p,B.b.p,init.p,cost.p,A.h,A.w,block,r,device,stream));ck((cudaError_t)dl_bm_glue(cost.p,init.p,out.p,2*r+1,by,bx,device,stream));fb=std::move(out);ph=by;pw=bx;}
    Buf<float> flow(size_t(2)*a.pyr[0].h*a.pyr[0].w);resize_flow<<<blocks(2*a.pyr[0].h*a.pyr[0].w),256,0,stream>>>(fb.p,flow.p,ph,pw,a.pyr[0].h,a.pyr[0].w,0,1,1);ph=a.pyr[0].h;pw=a.pyr[0].w;
    for(int lvl=2;lvl>=0;--lvl){auto& A=a.pyr[lvl];int n=A.h*A.w;Buf<float> f(2*n);resize_flow<<<blocks(2*n),256,0,stream>>>(flow.p,f.p,ph,pw,A.h,A.w,1,float(A.w-1)/std::max(pw-1,1),float(A.h-1)/std::max(ph-1,1));flow=std::move(f);Buf<float> stats(5*n),upd(2*n),sm(2*n);auto& k=kernel(2.f);for(int j=0;j<4;++j){ck((cudaError_t)dl_lk_sp_products(b.coeff[lvl].b.p,flow.p,A.b.p,stats.p,A.h,A.w,device,stream));ck((cudaError_t)dl_lk_blur_solve(stats.p,flow.p,k.p,upd.p,A.h,A.w,6,1e-4f,device,stream));ck((cudaError_t)dl_nc_fused(upd.p,a.conf[lvl].b.p,k.p,sm.p,A.h,A.w,6,device,stream));std::swap(flow,sm);}ph=A.h;pw=A.w;}
    int nx=(w+16+ox)/8+1,ny=(h+16+oy)/8+1;Buf<float> out(2*nx*ny);tile_vectors<<<blocks(2*nx*ny),256,0,stream>>>(flow.p,out.p,ph,pw,h,w,oy,ox,nx,nx*ny);if(dense)*dense=std::move(flow);++pairs;return out;
}

Buf<float> Engine::Impl::vars(const Image& image,int p,int oy,int ox,bool coarse){int nx=(image.w+16+ox)/8+1,ny=(image.h+16+oy)/8+1,N=nx*ny;Buf<float> lvl(N),v(size_t(N)*256);means<<<blocks(N),256,0,stream>>>(image.b.p,lvl.p,image.h,image.w,oy,ox,nx,N);variance<<<blocks(size_t(N)*256),256,0,stream>>>(lvl.p,xp[p].p,fp[p].p,int(xp[p].n),(coarse?cshapes[p]:shapes[p]).p,v.p,N);return v;}
Image Engine::Impl::reconstruction(const Image& ref,const std::vector<const Image*>& nb,const std::vector<Buf<float>>& v,const std::vector<int>& use,const Image& level,int p,int oy,int ox,bool coarse){
    int nx=(ref.w+16+ox)/8+1,ny=(ref.h+16+oy)/8+1,N=nx*ny;Buf<float> var=vars(level,p,oy,ox,coarse),count(size_t(N)*256),tiles(size_t(N)*256);Buf<float2> M(size_t(N)*256);MainArgs args{};std::vector<Buf<long long>> sh;std::vector<Buf<float>> fr;sh.reserve(use.size());fr.reserve(use.size());for(int i:use){sh.emplace_back(2*N);fr.emplace_back(2*N);alignment<<<blocks(2*N),256,0,stream>>>(v[i].p,sh.back().p,fr.back().p,2*N);int j=args.k++;args.img[j]=nb[i]->b.p;args.shift[j]=sh.back().p;args.frac[j]=fr.back().p;}float cnt=float(1+use.size());Buf<float> mergeVar;const float* mergeVariance=var.p;float noiseFactor=float(2*(1-model.planes[p].rho));if(coarse){mergeVar.size(size_t(N)*256);coarse_variance<<<blocks(size_t(N)*256),256,0,stream>>>(var.p,mergeVar.p,N*256,float(model.planes[p].rho));mergeVariance=mergeVar.p;noiseFactor=1.f;}if(strength==1.0){t2_main_merge<<<N,256,0,stream>>>(ref.b.p,win.p,mergeVariance,args,noiseFactor,1/cnt,cnt,coarse?0:!(use.size()>=1&&use.size()<=8),ws.p,coarse?2:0,64,M.p,count.p,tiles.p,ref.h,ref.w,oy,ox,nx);}else{native_t2_main_merge_scaled<<<N,256,0,stream>>>(ref.b.p,win.p,mergeVariance,args,noiseFactor,strength,1/cnt,cnt,coarse?0:!(use.size()>=1&&use.size()<=8),ws.p,coarse?2:0,64,M.p,count.p,tiles.p,ref.h,ref.w,oy,ox,nx);}Image pilot(ref.h,ref.w),out(ref.h,ref.w);t2_fold_real<<<blocks(ref.h*ref.w),256,0,stream>>>(tiles.p,pilot.b.p,ref.h,ref.w,oy,ox,nx);t2_main_wiener<<<N,256,0,stream>>>(M.p,var.p,count.p,basis.p,pilot.b.p,win.p,tiles.p,ref.h,ref.w,oy,ox,nx);t2_fold_real<<<blocks(ref.h*ref.w),256,0,stream>>>(tiles.p,out.b.p,ref.h,ref.w,oy,ox,nx);ck(cudaGetLastError());return out;
}

void Engine::Impl::s1(std::array<Image,3>& res){for(int p=1;p<3;++p){int H=res[p].h,W=res[p].w;Image g(H,W);pool<<<blocks(H*W),256,0,stream>>>(res[0].b.p,g.b.p,res[0].h,res[0].w,H,W);auto guide=g.b.get(H*W,stream);double mean=std::accumulate(guide.begin(),guide.end(),0.)/guide.size();float sig=float(sigma(model.planes[p],mean)*.5),eps=sig*sig,thr=3*sig;Buf<float> e,t;e.put({eps},stream);t.put({thr},stream);Image a(H,W),b(H,W),out(H,W);ck((cudaError_t)dl_s1_guided(g.b.p,res[p].b.p,e.p,t.p,a.b.p,b.b.p,out.b.p,H,W,device,stream));res[p]=std::move(out);}}

void Engine::Impl::fit(std::vector<Buf<float>>& v,const std::vector<int>& ks,Source& ref,int oy,int ox){
    // The frozen boundary estimator returns plain vectors until all sixteen direct pairs exist.
    if(mode=="plain"||ks.size()!=16)return;
    int nx=(w+16+ox)/8+1,ny=(h+16+oy)/8+1,N=nx*ny;std::vector<int> inner;for(int j=0;j<N;++j){int x=j%nx*8-8-ox,y=j/nx*8-8-oy;if(x>=32&&w-x>=32&&y>=32&&h-y>=32)inner.push_back(j);}if(inner.empty()){++domainErrors;throw std::runtime_error("T2 fit has no interior tiles");}
    Buf<float> var=vars(ref.image[0],0,oy,ox,false),lvl(N);Buf<float2> centered(size_t(N)*256),spectra(size_t(N)*256);Buf<double> texture(N);center_tiles<<<N,256,0,stream>>>(ref.image[0].b.p,win.p,centered.p,lvl.p,h,w,oy,ox,nx);t2_fft16_kernel<<<N,256,0,stream>>>(centered.p,spectra.p,N,0);texture_power<<<blocks(N),256,0,stream>>>(spectra.p,var.p,texture.p,N);auto tex=texture.get(N,stream);std::vector<double> vals;for(int j:inner)vals.push_back(tex[j]);double tm=median(vals);std::vector<int> sel;for(int j:inner)if(tex[j]>=tm)sel.push_back(j);
    std::vector<std::vector<float>> V;for(auto& a:v)V.push_back(a.get(2*N,stream));std::array<std::array<double,2>,16> g{};
    for(int k=0;k<16;++k)for(int ax=0;ax<2;++ax){std::vector<double> y;for(int j:sel)y.push_back(V[k][2*j+ax]);double beta=median(y);std::vector<double> res(y.size()),dev(y.size()),wt(y.size());auto scale=[&](){for(size_t j=0;j<y.size();++j)res[j]=y[j]-beta;double med=median(res);for(size_t j=0;j<y.size();++j)dev[j]=std::abs(res[j]-med);double s=std::max(1e-4,1.4826*median(dev));for(size_t j=0;j<y.size();++j)wt[j]=std::min(1.,1.345*s/std::max(std::abs(res[j]),1e-12));};scale();for(int it=0;it<21;++it){double num=0,den=1e-9;for(size_t j=0;j<y.size();++j){num+=wt[j]*y[j];den+=wt[j];}beta=num/den;scale();}g[k][ax]=beta;}
    // Symmetric full16 quadratic design has diagonal odd part and a 2x2 even block.
    double K=16,sum2=0,sum4=0;for(int k:ks){sum2+=k*k;sum4+=double(k)*k*k*k;}double det=K*sum4-sum2*sum2;std::vector<double> s2(N),s2c(N),weight(N),sig(N);for(int j=0;j<N;++j){double rss=0;for(int ax=0;ax<2;++ax){double b0=0,b1=0,b2=0;for(int i=0;i<16;++i){double z=double(V[i][2*j+ax])-g[i][ax];b0+=z;b1+=ks[i]*z;b2+=ks[i]*ks[i]*z;}double c=(sum4*b0-sum2*b2)/det,u=b1/sum2,a=(K*b2-sum2*b0)/det;for(int i=0;i<16;++i){double z=double(V[i][2*j+ax])-g[i][ax]-c-u*ks[i]-a*ks[i]*ks[i];rss+=z*z;}}s2[j]=rss/26;}
    vals.clear();for(int j:sel)if(tex[j]>0)vals.push_back(s2[j]*tex[j]);double ac;if(!vals.empty())ac=median(vals);else {std::vector<double> a,b;for(int j:inner){a.push_back(s2[j]);b.push_back(std::max(1.,tex[j]));}ac=median(a)*median(b);}vals.clear();for(int j:sel)vals.push_back(tex[j]);double floor=.05*std::max(1e-9,median(vals));for(int j=0;j<N;++j){s2c[j]=std::max(1e-6,(26*std::max(1e-6,s2[j])+8*ac/std::max(floor,tex[j]))/34);sig[j]=std::sqrt(s2c[j]);}for(int j:inner)weight[j]=1/s2c[j];
    std::vector<int> targets;for(int i=0;i<16;++i)if(mode=="full"||std::abs(ks[i])<=2)targets.push_back(i);int B=int(targets.size());std::vector<double> Y(size_t(B)*2*N),W(size_t(B)*N),S(size_t(B)*N),m(B),inv(size_t(B)*N);for(int b=0;b<B;++b){int k=targets[b];for(int j=0;j<N;++j){for(int ax=0;ax<2;++ax)Y[(b*2+ax)*N+j]=double(V[k][2*j+ax])-g[k][ax];W[b*N+j]=weight[j];S[b*N+j]=sig[j];}}
    Buf<double> dy,dw,ds,dz(size_t(B)*N);dy.put(Y,stream);dw.put(W,stream);ds.put(S,stream);ck((cudaError_t)dl_t2_scale_z2(dy.p,dw.p,ds.p,dz.p,B,ny,nx,stream));auto z=dz.get(size_t(B)*N,stream);for(int b=0;b<B;++b){std::vector<double> row(z.begin()+b*N,z.begin()+(b+1)*N);double med=median(row);m[b]=std::max(.05,(std::isfinite(med)?med:1.)/(2*std::log(2.)));for(int j=0;j<N;++j){double scale=(4.685*sig[j])*std::sqrt(m[b]);inv[b*N+j]=1/(scale*scale);}}
    bool domain=true;for(double x:Y)domain&=std::isfinite(x)&&std::abs(x)<=1e6;for(double x:weight)domain&=std::isfinite(x)&&(x==0||(x>=1e-12&&x<=1e6));for(double x:sig)domain&=std::isfinite(x)&&x>=1e-3&&x<=1e6;for(double x:m)domain&=std::isfinite(x)&&x>=.05&&x<=1e20;if(!domain){trace_buffer(dy,Y.size(),"domain-Y.f64",stream);std::ostringstream why;auto range=[&](const char* name,const std::vector<double>& a){auto mm=std::minmax_element(a.begin(),a.end());why<<" "<<name<<"=["<<*mm.first<<","<<*mm.second<<"]";};range("Y",Y);range("W",weight);range("sig",sig);range("m",m);why<<" tex_median="<<tm<<" confidence_a="<<ac;domainNote=why.str();bool safe=true;for(double x:Y)safe&=std::isfinite(x)&&std::isfinite(x*x);for(double x:weight)safe&=std::isfinite(x)&&x>=0;for(double x:sig)safe&=std::isfinite(x)&&x>0;for(double x:m)safe&=std::isfinite(x)&&x>0;for(double x:inv)safe&=std::isfinite(x)&&x>0;if(!safe){++domainErrors;throw std::runtime_error("T2 input outside finite native FP64 fit domain:"+domainNote);}++fp32DomainExits;}
    Buf<double> dinv,dm,sharedW;dinv.put(inv,stream);dm.put(m,stream);sharedW.put(weight,stream);std::vector<std::vector<Buf<float>>> candidates(6);int radii[]={24,12,6,3,2,1};for(int c=0;c<6;++c){int r=radii[c];Buf<double> out(size_t(B)*2*N);if(!domain){if(mode=="box"){Buf<double> scratch(size_t(8)*B*N);Buf<unsigned char> good(size_t(B)*N);ck((cudaError_t)dl_t2_native_boxfit_fp64(dy.p,sharedW.p,dinv.p,out.p,scratch.p,good.p,B,ny,nx,r,6,stream));++fp64BoxFits;}else {std::vector<double> G((2*r+1)*(2*r+1)),seed(size_t(B)*N*2);std::vector<unsigned char> valid(size_t(B)*N);double ss=std::max(.5*r,.6);for(int y=-r;y<=r;++y)for(int x=-r;x<=r;++x)G[(y+r)*(2*r+1)+x+r]=std::exp(-(x*x+y*y)/(2*ss*ss));int error=dl_t2_native_median_seed64_host(Y.data(),W.data(),seed.data(),valid.data(),B,ny,nx,r);if(error)throw std::runtime_error("T2 invalid FP64 median-seed descriptor");Buf<double> dg,beta;dg.put(G,stream);beta.put(seed,stream);Buf<bool> finite(size_t(B)*N);std::unique_ptr<bool[]> hostFinite(new bool[size_t(B)*N]);for(size_t i=0;i<size_t(B)*N;++i)hostFinite[i]=valid[i]!=0;ck(cudaMemcpyAsync(finite.p,hostFinite.get(),sizeof(bool)*B*N,cudaMemcpyHostToDevice,stream));ck((cudaError_t)dl_t2_field_constant(dy.p,dw.p,ds.p,dm.p,dg.p,beta.p,finite.p,out.p,dinv.p,B,2,ny,nx,r,6,1,0.,4.685,stream));ck(cudaStreamSynchronize(stream));++fp64LocalFits;}}else if(mode=="box"){Buf<float> scratch(size_t(8)*B*N);Buf<unsigned char> good(size_t(B)*N);ck((cudaError_t)dl_t2_boxfit_f32(dy.p,sharedW.p,dinv.p,out.p,scratch.p,good.p,B,ny,nx,r,6,stream));}else {std::vector<float> G((2*r+1)*(2*r+1));double ss=std::max(.5*r,.6);for(int y=-r;y<=r;++y)for(int x=-r;x<=r;++x)G[(y+r)*(2*r+1)+x+r]=float(std::exp(-(x*x+y*y)/(2*ss*ss)));Buf<float> dg;dg.put(G,stream);Buf<float4> scratch(size_t(B)*N);ck((cudaError_t)dl_t2_field_tiled_f32_rs(dy.p,dw.p,ds.p,dm.p,dg.p,dinv.p,out.p,scratch.p,B,ny,nx,r,6,0,4.685,stream));}auto result=out.get(size_t(B)*2*N,stream);for(int b=0;b<B;++b){std::vector<float> cv(2*N);for(int j=0;j<N;++j)for(int ax=0;ax<2;++ax)cv[2*j+ax]=float(result[(b*2+ax)*N+j]+g[targets[b]][ax]);Buf<float> d;d.put(cv,stream);candidates[c].push_back(std::move(d));}}
    for(int b=0;b<B;++b){int k=targets[b];std::array<std::vector<float>,7> energies;Source& nb=ring[(ref.index+ks[k])%17];for(int c=0;c<7;++c){Buf<float> e(N);score<<<N,256,0,stream>>>(ref.image[0].b.p,nb.image[0].b.p,c==0?v[k].p:candidates[c-1][b].p,win.p,var.p,e.p,h,w,oy,ox,nx);energies[c]=e.get(N,stream);}std::array<std::vector<float>,6> hostCandidates;for(int c=0;c<6;++c)hostCandidates[c]=candidates[c][b].get(2*N,stream);auto chosen=hostCandidates[0];for(int j:inner){float best=energies[0][j];for(int c=1;c<7;++c)best=std::min(best,energies[c][j]);int choice=0;for(int c=1;c<7;++c)if((double(energies[c][j])-double(best))*255<=-2*std::log(.001)){choice=c;break;}if(choice==0){chosen[2*j]=V[k][2*j];chosen[2*j+1]=V[k][2*j+1];}else {const auto& cv=hostCandidates[choice-1];chosen[2*j]=cv[2*j];chosen[2*j+1]=cv[2*j+1];}}v[k].put(chosen,stream);++refined;}
}

void Engine::Impl::render(int64_t index,int64_t received,const FrameView& out){Source& ref=ring[index%17];if(ref.index!=index)throw std::runtime_error("T2 stale source slot");int oy=1+int(index*3%7),ox=1+int((index*5+2)%7);std::vector<int> ks;std::vector<Buf<float>> v,dense;for(int k=-8;k<=8;++k)if(k&&index+k>=0&&index+k<received){Source& nb=ring[(index+k)%17];if(nb.index!=index+k)throw std::runtime_error("T2 stale neighbour slot");ks.push_back(k);dense.emplace_back();v.push_back(motion(ref,nb,oy,ox,&dense.back()));}if(index==4){for(int i=0;i<int(ks.size());++i){trace_buffer(v[i],v[i].n,"vectors-"+std::to_string(ks[i])+".f32",stream);trace_buffer(dense[i],dense[i].n,"flow-"+std::to_string(ks[i])+".f32",stream);}}fit(v,ks,ref,oy,ox);std::vector<int> nearby,all;for(int i=0;i<int(ks.size());++i){all.push_back(i);if(std::abs(ks[i])<=2)nearby.push_back(i);}std::array<Image,3> res,longRes;int lnx=(w+16+ox)/8+1,lny=(h+16+oy)/8+1;
    for(int p=0;p<3;++p){int X=p?sx:1,Y=p?sy:1,H=ref.image[p].h,W=ref.image[p].w,N=((W+16+ox)/8+1)*((H+16+oy)/8+1);std::vector<Buf<float>> pv;std::vector<const Image*> nb;for(int i=0;i<int(ks.size());++i){nb.push_back(&ring[(index+ks[i])%17].image[p]);Buf<float> d(2*N);if(!p)ck(cudaMemcpyAsync(d.p,v[i].p,2*N*sizeof(float),cudaMemcpyDeviceToDevice,stream));else if(mode!="plain"&&ks.size()==16)plane_vectors<<<blocks(2*N),256,0,stream>>>(v[i].p,d.p,H,W,h,w,X,Y,oy,ox,(W+16+ox)/8+1,lnx,lny,N);else {
            Buf<float> full(size_t(2)*h*w),native(size_t(2)*H*W);
            resize_flow<<<blocks(2*h*w),256,0,stream>>>(dense[i].p,full.p,h/4,w/4,h,w,0,float(w)/(w/4),float(h)/(h/4));
            resize_flow<<<blocks(2*H*W),256,0,stream>>>(full.p,native.p,h,w,H,W,0,1.f/X,1.f/Y);
            tile_vectors<<<blocks(2*N),256,0,stream>>>(native.p,d.p,H,W,H,W,oy,ox,(W+16+ox)/8+1,N);
        }pv.push_back(std::move(d));}Image level(H,W);pool<<<blocks(H*W),256,0,stream>>>(ref.image[0].b.p,level.b.p,h,w,H,W);res[p]=reconstruction(ref.image[p],nb,pv,nearby,level,p,oy,ox,false);if(mode=="full")longRes[p]=reconstruction(ref.image[p],nb,pv,all,level,p,oy,ox,false);}
    s1(res);if(mode=="full"){s1(longRes);for(int p=0;p<3;++p){ // Preserve long-main coarse-band selection before coarse replacement.
        int H=res[p].h,W=res[p].w,hp=H+64,wp=W+64;Image delta(H,W);ck(cudaMemcpyAsync(delta.b.p,longRes[p].b.p,H*W*sizeof(float),cudaMemcpyDeviceToDevice,stream));subtract<<<blocks(H*W),256,0,stream>>>(delta.b.p,res[p].b.p,H*W);Buf<float2> spec(size_t(hp)*wp);pad_complex<<<blocks(hp*wp),256,0,stream>>>(delta.b.p,spec.p,H,W,hp,wp,32,32);fft(spec,hp,wp,false);spectral_gain<<<blocks(hp*wp),256,0,stream>>>(spec.p,hp,wp,p?sx:1,p?sy:1,0);fft(spec,hp,wp,true);crop_add<<<blocks(H*W),256,0,stream>>>(res[p].b.p,spec.p,H,W,hp,wp,32,32);
    }}
    int hc=ref.coarse[0].h,wc=ref.coarse[0].w,ncx=(wc+16+ox)/8+1,ncy=(hc+16+oy)/8+1;std::vector<Buf<float>> cv;for(auto& a:v){Image cg(hc,wc);cg.b.size(size_t(2)*hc*wc);ck((cudaError_t)dl_coarse_gather(a.p,cg.b.p,h,w,oy,ox,lny,lnx,hc,wc,34,device,stream));Buf<float> vec(2*ncx*ncy);tile_vectors<<<blocks(2*ncx*ncy),256,0,stream>>>(cg.b.p,vec.p,hc,wc,hc,wc,oy,ox,ncx,ncx*ncy);divide<<<blocks(2*ncx*ncy),256,0,stream>>>(vec.p,2*ncx*ncy,4);cv.push_back(std::move(vec));}for(int p=0;p<3;++p){std::vector<const Image*> nb;for(int k:ks)nb.push_back(&ring[(index+k)%17].coarse[p]);Image c=reconstruction(ref.coarse[p],nb,cv,all,ref.coarse[0],p,oy,ox,true);recombine(res[p],c,p?sx:1,p?sy:1);write_plane<<<blocks(res[p].h*res[p].w),256,0,stream>>>(out.plane[p],res[p].b.p);}ck(cudaGetLastError());++frames;
}
Engine::Engine(int w,int h,int sx,int sy,const std::string& mode,Model model,double strength):p(new Impl(w,h,sx,sy,mode,std::move(model),strength)){if(!std::isfinite(strength)||strength<0)throw std::runtime_error("MCWiener strength must be finite and nonnegative (lab c)");if(w<68||h<68||w%4||h%4)throw std::runtime_error("T2 requires dimensions >=68 and divisible by four");if(mode!="plain"&&mode!="box"&&mode!="full")throw std::runtime_error("T2 unknown mode");}
Engine::~Engine()=default;
uint64_t Engine::fp64ContextCount()const{return p->fp32DomainExits;}
static void validate_frame(const FrameView& v,int w,int h,int sx,int sy){for(int plane=0;plane<3;++plane){const auto& a=v.plane[plane];const int width=w/(plane?sx:1),height=h/(plane?sy:1);if(!a.ptr||a.width!=width||a.height!=height||(a.bytes!=1&&a.bytes!=2)||(a.step!=1&&a.step!=2)||a.bits<1||a.bits>16||a.shift<0||a.bits+a.shift>a.bytes*8||a.pitch<size_t(width)*a.bytes*a.step)throw std::runtime_error("T2 invalid typed plane descriptor (geometry/pitch/storage/code units)");}}
void Engine::ingest(int64_t i,const FrameView& v,cudaStream_t s){validate_frame(v,p->w,p->h,p->sx,p->sy);if(i<0)throw std::runtime_error("T2 invalid source index");p->bind(s);p->ingest(i,v);}
void Engine::render(int64_t i,int64_t received,const FrameView& v,cudaStream_t s){validate_frame(v,p->w,p->h,p->sx,p->sy);if(i<0||i>=received)throw std::runtime_error("T2 invalid output index");p->bind(s);p->render(i,received,v);}
std::string Engine::statistics()const{std::ostringstream s;s<<"frames="<<p->frames<<", pairs="<<p->pairs<<", refined="<<p->refined<<", domain_errors="<<p->domainErrors<<", fp32_domain_exits="<<p->fp32DomainExits<<", box_fp64_calls="<<p->fp64BoxFits<<", local_fp64_calls="<<p->fp64LocalFits;if(p->fp32DomainExits)s<<", last_domain="<<p->domainNote;return s.str();}
}
