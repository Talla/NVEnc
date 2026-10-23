// Experimental T2 deg0/P2 FP64 IRLS: direct neighbors, original Torch median.
// No fast math, preserve sqrt/divide/square Tukey expression. Reduction order
// differs from Torch einsum and requires explicit candidate/output validation.
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_reduce.cuh>
#ifdef _WIN32
#define DL_EXPORT __declspec(dllexport)
#else
#define DL_EXPORT __attribute__((visibility("default")))
#endif

template<bool Algebra>
__global__ void field_constant(const double* Y,const double* W,const double* S,
 const double* M,const double* G,const double* init,const bool* finite,double* out,const double* inverse,
 int B,int P,int ny,int nx,int r,int iters,double prior,double c) {
 const int center=blockIdx.x, N=ny*nx,b=center/N,pos=center%N,y=pos/nx,x=pos%nx;
 const int side=2*r+1,K=side*side;
 __shared__ double vals[4][128];
 __shared__ double bx,by; __shared__ bool good;
 if(threadIdx.x==0){bx=init[center*2];by=init[center*2+1];good=false;}
 __syncthreads();
 const double rootm=sqrt(M[b]);
 for(int it=0;it<iters;++it){
  double sw=0,sx=0,sy=0,sbase=0;
  for(int j=threadIdx.x;j<K;j+=blockDim.x){
   int yy=y+j/side-r,xx=x+j%side-r;
   bool inside=yy>=0&&yy<ny&&xx>=0&&xx<nx;
   int idx=b*N+yy*nx+xx;
   double v0=inside?Y[(b*P)*N+yy*nx+xx]:0;
   double v1=inside?Y[(b*P+1)*N+yy*nx+xx]:0;
   double base=(inside?W[idx]:0)*G[j];
   double sigma=inside?S[idx]:1;
   double dx=v0-bx,dy=v1-by;
   double norm=dx*dx+dy*dy;
   double squared;
   if constexpr(Algebra){
    double inv=inside?inverse[idx]:1.0/((c*rootm)*(c*rootm));
    // Exceptional reciprocal underflow/overflow or nonfinite residuals retain
    // the original sqrt/divide/square behavior. No epsilon/floor is introduced.
    if(isfinite(inv)&&inv>0.0&&isfinite(norm))squared=(norm<0.0?0.0:norm)*inv;
    else {double z=sqrt((norm<0.0?0.0:norm))/(c*sigma*rootm);squared=z*z;}
   }else{double z=sqrt((norm<0.0?0.0:norm))/(c*sigma*rootm);squared=z*z;}
   double q=1.0-squared; q=q<0.0?0.0:q; double wt=base*(q*q);
   sw+=wt;sx+=wt*v0;sy+=wt*v1;sbase+=base;
  }
  vals[0][threadIdx.x]=sw;vals[1][threadIdx.x]=sx;vals[2][threadIdx.x]=sy;vals[3][threadIdx.x]=sbase;
  __syncthreads();
  for(int d=64;d;d/=2){if(threadIdx.x<d){for(int q=0;q<4;++q)vals[q][threadIdx.x]+=vals[q][threadIdx.x+d];}__syncthreads();}
  if(threadIdx.x==0){double a=vals[0][0]+vals[3][0]*prior;good=a>1e-12;
   if(good&&finite[center]){bx=vals[1][0]/a;by=vals[2][0]/a;}}
  __syncthreads();
 }
 if(threadIdx.x==0){out[(b*P)*N+pos]=good&&finite[center]?bx:Y[(b*P)*N+pos];out[(b*P+1)*N+pos]=good&&finite[center]?by:Y[(b*P+1)*N+pos];}
}
extern "C" DL_EXPORT int dl_t2_field_constant(const double* Y,const double* W,const double* S,
 const double* M,const double* G,const double* init,const bool* finite,double* out,const double* inverse,
 int B,int P,int ny,int nx,int r,int iters,int algebra,double prior,double c,void* stream){
 if(algebra)field_constant<true><<<B*ny*nx,128,0,(cudaStream_t)stream>>>(Y,W,S,M,G,init,finite,out,inverse,B,P,ny,nx,r,iters,prior,c);
 else field_constant<false><<<B*ny*nx,128,0,(cudaStream_t)stream>>>(Y,W,S,M,G,init,finite,out,inverse,B,P,ny,nx,r,iters,prior,c);
 return (int)cudaGetLastError();
}

extern "C" DL_EXPORT int dl_t2_field_abi_version(){return 2;}

// Experimental FP32 iterations; FP64 source arrays, median and scale retained.
template<bool Algebra>
__global__ void field_constant_float(const double* Y,const double* W,const double* S,
 const double* M,const double* G,const double* init,const bool* finite,double* out,const double* inverse,
 int B,int P,int ny,int nx,int r,int iters,double prior,double c) {
 const int center=blockIdx.x, N=ny*nx,b=center/N,pos=center%N,y=pos/nx,x=pos%nx;
 const int side=2*r+1,K=side*side;
 __shared__ float vals[4][128];
 __shared__ float bx,by; __shared__ bool good;
 if(threadIdx.x==0){bx=init[center*2];by=init[center*2+1];good=false;}
 __syncthreads();
 const double rootm=sqrt(M[b]);
 for(int it=0;it<iters;++it){
  float sw=0,sx=0,sy=0,sbase=0;
  for(int j=threadIdx.x;j<K;j+=blockDim.x){
   int yy=y+j/side-r,xx=x+j%side-r;
   bool inside=yy>=0&&yy<ny&&xx>=0&&xx<nx;
   int idx=b*N+yy*nx+xx;
   float v0=inside?(float)Y[(b*P)*N+yy*nx+xx]:0;
   float v1=inside?(float)Y[(b*P+1)*N+yy*nx+xx]:0;
   float base=(float)((inside?W[idx]:0)*G[j]);
   double sigma=inside?S[idx]:1;
   float dx=v0-bx,dy=v1-by;
   float norm=dx*dx+dy*dy;
   float squared;
   if constexpr(Algebra){
    float inv=(float)(inside?inverse[idx]:1.0/((c*rootm)*(c*rootm)));
    // Exceptional reciprocal underflow/overflow or nonfinite residuals retain
    // the original sqrt/divide/square behavior. No epsilon/floor is introduced.
    if(isfinite(inv)&&inv>0.0&&isfinite(norm))squared=(norm<0.0f?0.0f:norm)*inv;
    else {double z=sqrt((norm<0.0?0.0:norm))/(c*sigma*rootm);squared=z*z;}
   }else{double z=sqrt((norm<0.0?0.0:norm))/(c*sigma*rootm);squared=z*z;}
   float q=1.0f-squared; q=q<0.0f?0.0f:q; float wt=base*(q*q);
   sw+=wt;sx+=wt*v0;sy+=wt*v1;sbase+=base;
  }
  vals[0][threadIdx.x]=sw;vals[1][threadIdx.x]=sx;vals[2][threadIdx.x]=sy;vals[3][threadIdx.x]=sbase;
  __syncthreads();
  for(int d=64;d;d/=2){if(threadIdx.x<d){for(int q=0;q<4;++q)vals[q][threadIdx.x]+=vals[q][threadIdx.x+d];}__syncthreads();}
  if(threadIdx.x==0){float a=vals[0][0]+vals[3][0]*(float)prior;good=a>1e-12;
   if(good&&finite[center]){bx=vals[1][0]/a;by=vals[2][0]/a;}}
  __syncthreads();
 }
 if(threadIdx.x==0){out[(b*P)*N+pos]=good&&finite[center]?bx:Y[(b*P)*N+pos];out[(b*P+1)*N+pos]=good&&finite[center]?by:Y[(b*P+1)*N+pos];}
}

extern "C" DL_EXPORT int dl_t2_field_float(const double* Y,const double* W,const double* S,
 const double* M,const double* G,const double* init,const bool* finite,double* out,const double* inverse,
 int B,int P,int ny,int nx,int r,int iters,int algebra,double prior,double c,void* stream){
 field_constant_float<true><<<B*ny*nx,128,0,(cudaStream_t)stream>>>(Y,W,S,M,G,init,finite,out,inverse,B,P,ny,nx,r,iters,prior,c);
 return (int)cudaGetLastError();
}

// Optional ABI2 initializer only. CUB is supplied by the installed CUDA toolkit;
// its redistribution notices remain in the installed header/distribution terms.
// All existing fitting kernels and exports above retain their arithmetic.
template<int Radius,int Threads,int Items>
__global__ void field_median_float(const float* Y,const bool* valid,double* beta,bool* finite,int ny,int nx){
 using Sort=cub::BlockRadixSort<float,Threads,Items>;
 using Reduce=cub::BlockReduce<int,Threads>;
 __shared__ union {typename Sort::TempStorage sort;typename Reduce::TempStorage reduce;} storage;
 __shared__ int count;
 const int center=blockIdx.x,N=ny*nx,b=center/N,pos=center%N,y=pos/nx,x=pos%nx;
 constexpr int side=2*Radius+1,K=side*side;
 float keys[Items];
 int local_count=0;
 #pragma unroll
 for(int i=0;i<Items;++i){
  const int j=threadIdx.x+i*Threads,yy=y+j/side-Radius,xx=x+j%side-Radius;
  if(j<K&&yy>=0&&yy<ny&&xx>=0&&xx<nx&&valid[b*N+yy*nx+xx])++local_count;
 }
 const int total=Reduce(storage.reduce).Sum(local_count);
 if(threadIdx.x==0){count=total;finite[center]=total>0;}
 __syncthreads();
 if(count==0){if(threadIdx.x==0){beta[center*2]=0.;beta[center*2+1]=0.;}return;}
 const int median=(count-1)/2;
 for(int axis=0;axis<2;++axis){
  #pragma unroll
  for(int i=0;i<Items;++i){
   const int j=threadIdx.x+i*Threads,yy=y+j/side-Radius,xx=x+j%side-Radius;
   const bool keep=j<K&&yy>=0&&yy<ny&&xx>=0&&xx<nx&&valid[b*N+yy*nx+xx];
   keys[i]=keep?Y[(b*2+axis)*N+yy*nx+xx]:CUDART_INF_F;
  }
  Sort(storage.sort).Sort(keys);
  #pragma unroll
  for(int i=0;i<Items;++i)if(int(threadIdx.x)*Items+i==median)beta[center*2+axis]=double(keys[i]);
  __syncthreads();
 }
}

extern "C" DL_EXPORT int dl_t2_field_median_float(const float* Y,const bool* valid,double* beta,bool* finite,
 int B,int ny,int nx,int r,void* stream){
 if(B<1||ny<1||nx<1)return int(cudaErrorInvalidValue);
 const int blocks=B*ny*nx;
 switch(r){
  case 24:field_median_float<24,128,20><<<blocks,128,0,(cudaStream_t)stream>>>(Y,valid,beta,finite,ny,nx);break;
  case 12:field_median_float<12,128,5><<<blocks,128,0,(cudaStream_t)stream>>>(Y,valid,beta,finite,ny,nx);break;
  case 6:field_median_float<6,128,2><<<blocks,128,0,(cudaStream_t)stream>>>(Y,valid,beta,finite,ny,nx);break;
  case 3:field_median_float<3,64,1><<<blocks,64,0,(cudaStream_t)stream>>>(Y,valid,beta,finite,ny,nx);break;
  case 2:field_median_float<2,32,1><<<blocks,32,0,(cudaStream_t)stream>>>(Y,valid,beta,finite,ny,nx);break;
  case 1:field_median_float<1,32,1><<<blocks,32,0,(cudaStream_t)stream>>>(Y,valid,beta,finite,ny,nx);break;
  default:return int(cudaErrorInvalidValue);
 }
 return int(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// tiled_fp32_exact backend (new, additive; every kernel above is unchanged).
// One launch per radius covers all pairs (grid z) and all IRLS iterations. A block
// loads its (B+2r)^2 neighbour tile (FP32 values, exactly the (float)Y, W, (float)inverse
// the streamed kernel reads) into shared memory once; one warp owns one centre at a time.
// The warp reproduces the 128-thread partial-sum structure of field_constant_float
// (partial t = lane+32*s accumulates j = t+128*i in order; then the tree d=64..1), so the
// FP32 sums are bit-identical to dl_t2_field_float. The median initializer is the exact
// lower median by bisection on order-preserving float keys (same ordering as CUB radix
// sort, so -0 < +0 as in dl_t2_field_median_float). Build uses --fmad=false.
// -0 and +0 share one key: CUB's radix sort treats them as equal and orders them by stable (thread,item) position,
// which the zero-class branch below reproduces so the median's zero sign matches dl_t2_field_median_float.
__device__ __forceinline__ unsigned dl_f2key(float f){if(f==0.f)return 0x80000000u;unsigned u=__float_as_uint(f);return u^((u>>31)?0xFFFFFFFFu:0x80000000u);}
__device__ __forceinline__ float dl_key2f(unsigned k){unsigned u=k^((k>>31)?0x80000000u:0xFFFFFFFFu);return __uint_as_float(u);}

// (float)(w*g) with the FP64 multiply and the double->float rounding (RNE) done as 1 DMUL + integer ops: the FP64 pipe is
// 1/64 rate on this GPU and the generic conversion is a second FP64-pipe op per neighbour. Out-of-range results take the generic path.
__device__ __forceinline__ float dl_base_f(double w,double g){
 const double d=__dmul_rn(w,g);
 const unsigned long long u=(unsigned long long)__double_as_longlong(d);
 const unsigned hi=(unsigned)(u>>32),lo=(unsigned)u,e=(hi>>20)&0x7FFu;
 if(e==0u&&(((hi&0xFFFFFu)|lo)==0u))return __uint_as_float(hi&0x80000000u);
 if(e-897u>253u)return (float)d;
 const unsigned m=((hi&0xFFFFFu)<<3)|(lo>>29),rem=lo&0x1FFFFFFFu;
 unsigned f=(hi&0x80000000u)|((e-896u)<<23)|m;
 if(rem>0x10000000u||(rem==0x10000000u&&(m&1u)))++f;
 return __uint_as_float(f);
}

// Phase 0: median + IRLS fused. Phase 1: median only -> medbuf (bx,by,fin). Phase 2: IRLS from medbuf (radius 24, where
// the per-lane key array and the hoisted base array do not fit one register allocation).
// F32 (field_backend "tiled_fp32", FP32-rounding class): the neighbour weight base is float(W)*float(G) as one FP32 multiply
// (W held as float in shared memory, G32 = float Gaussian table); no FP64 arithmetic in the centre loop. F32=false is the
// unchanged tiled_fp32_exact kernel ((float)(W*G) rounded from the FP64 product).
template<int Phase,bool Hoist,int R,int BX,int BY,int NW,bool F32=false,bool RS=false>
__global__ void __launch_bounds__(NW*32,1) field_tiled(const double* Y,const double* W,const double* S,
 const double* M,const double* G,const double* inverse,double* out,float4* medbuf,int ny,int nx,int iters,double prior,double c,const float* G32=nullptr){
 constexpr int TW=BX+2*R,TH=BY+2*R,T=TW*TH,side=2*R+1,K=side*side,KPL=(K+31)/32,BPL=(K+127)/128;
 // CUB BlockRadixSort layout of the current median kernel (threads x items per thread)
 constexpr int CT=R>=6?128:(R==3?64:32),CI=R==24?20:(R==12?5:(R==6?2:1));
 extern __shared__ __align__(16) unsigned char smem_raw[];
 // F32: one float4 (v0,v1,inverse scale,float(W)) per neighbour tile position, so the IRLS reads a neighbour with one LDS.128.
 double* sW=(double*)smem_raw; float4* sp=(float4*)smem_raw;
 float* sv0=(float*)(smem_raw+size_t(T)*sizeof(double)); float* sv1=sv0+T; float* sinv=sv1+T;
 unsigned char* sok=F32?(unsigned char*)(sp+T):(unsigned char*)(sinv+T);
 // RS (median_select "radix"): 256-bin histogram per warp after the tile arrays (16-byte aligned).
 unsigned* hist=(unsigned*)(smem_raw+((size_t(T)*(F32?sizeof(float4)+1:sizeof(double)+3*sizeof(float)+1)+15)&~size_t(15)))+(threadIdx.x>>5)*256;
 auto ld0=[&](int e)->float{if constexpr(F32)return sp[e].x;else return sv0[e];};
 auto ld1=[&](int e)->float{if constexpr(F32)return sp[e].y;else return sv1[e];};
 const int b=blockIdx.z,N=ny*nx,x0=blockIdx.x*BX,y0=blockIdx.y*BY;
 const double rootm=sqrt(M[b]);
 const float outside_inv=(float)(1.0/((c*rootm)*(c*rootm)));
 for(int e=threadIdx.x;e<T;e+=NW*32){
  const int yy=y0+e/TW-R,xx=x0+e%TW-R;
  if(yy>=0&&yy<ny&&xx>=0&&xx<nx){
   const int p=yy*nx+xx;
   const double w=W[b*N+p];
   if constexpr(F32)sp[e]=make_float4((float)Y[(b*2)*N+p],(float)Y[(b*2+1)*N+p],(float)inverse[b*N+p],(float)w);
   else{sv0[e]=(float)Y[(b*2)*N+p];sv1[e]=(float)Y[(b*2+1)*N+p];sW[e]=w;sinv[e]=(float)inverse[b*N+p];}
   sok[e]=w>0.0;
  }else{
   if constexpr(F32)sp[e]=make_float4(0.f,0.f,outside_inv,0.f);
   else{sv0[e]=0.f;sv1[e]=0.f;sW[e]=0.0;sinv[e]=outside_inv;}
   sok[e]=0;
  }
 }
 __syncthreads();
 const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
 for(int ci=warp;ci<BX*BY;ci+=NW){
  const int cy=ci/BX,cx=ci%BX,y=y0+cy,x=x0+cx;
  if(y>=ny||x>=nx)continue;
  const int pos=y*nx+x,c0=cy*TW+cx; // smem index of neighbour j: c0 + j + (j/side)*(TW-side)
  float bx=0.f,by=0.f;bool fin=true;
  if constexpr(Phase==2){const float4 mv=medbuf[b*N+pos];bx=mv.x;by=mv.y;fin=mv.z>0.5f;}
  if constexpr(Phase!=2){
  // ---- exact lower median of valid (W>0) neighbours, per component
  unsigned key[KPL];int cnt=0;
  #pragma unroll
  for(int i=0;i<KPL;++i){
   const int j=lane+32*i;unsigned k=0xFFFFFFFFu;
   if(j<K){const int e=c0+j+(j/side)*(TW-side);if(sok[e]){k=dl_f2key(ld0(e));++cnt;}}
   key[i]=k;
  }
  cnt=__reduce_add_sync(0xFFFFFFFFu,cnt);
  fin=cnt>0;
  if(fin){
   const int kth=(cnt-1)/2;
   #pragma unroll
   for(int axis=0;axis<2;++axis){
    if(axis==1){
     #pragma unroll
     for(int i=0;i<KPL;++i){
      const int j=lane+32*i;
      if(key[i]!=0xFFFFFFFFu){const int e=c0+j+(j/side)*(TW-side);key[i]=dl_f2key(ld1(e));}
     }
    }
    unsigned lo=0xFFFFFFFFu,hi=0u;
    if constexpr(RS){ // kth smallest valid key by four 8-bit radix passes (same key the bisection returns)
     lo=0u;int rank=kth;
     #pragma unroll
     for(int pass=0;pass<4;++pass){
      const int shift=24-8*pass;
      #pragma unroll
      for(int q=0;q<8;++q)hist[lane+32*q]=0u;
      __syncwarp();
      #pragma unroll
      for(int i=0;i<KPL;++i){
       const unsigned k=key[i];
       bool take=k!=0xFFFFFFFFu;
       if(pass>0)take=take&&((k>>(shift+8))==(lo>>(shift+8)));
       if(take)atomicAdd(hist+((k>>shift)&255u),1u);
      }
      __syncwarp();
      unsigned b8[8],sum=0u;
      #pragma unroll
      for(int q=0;q<8;++q){b8[q]=hist[lane*8+q];sum+=b8[q];}
      unsigned incl=sum;
      #pragma unroll
      for(int d=1;d<32;d<<=1){const unsigned v=__shfl_up_sync(0xFFFFFFFFu,incl,d);if(lane>=d)incl+=v;}
      const unsigned owner=__ffs(__ballot_sync(0xFFFFFFFFu,incl>(unsigned)rank))-1u;
      unsigned digit=0u,before=0u;
      if(lane==owner){
       unsigned run=incl-sum;
       #pragma unroll
       for(int q=0;q<8;++q){if(run+b8[q]<=(unsigned)rank)run+=b8[q];else{digit=owner*8u+q;before=run;break;}}
      }
      digit=__shfl_sync(0xFFFFFFFFu,digit,owner);before=__shfl_sync(0xFFFFFFFFu,before,owner);
      lo|=digit<<shift;rank-=(int)before;
      __syncwarp();
     }
    }else{
    #pragma unroll
    for(int i=0;i<KPL;++i)if(key[i]!=0xFFFFFFFFu){lo=min(lo,key[i]);hi=max(hi,key[i]);}
    lo=__reduce_min_sync(0xFFFFFFFFu,lo);hi=__reduce_max_sync(0xFFFFFFFFu,hi);
    while(lo<hi){
     const unsigned mid=lo+((hi-lo)>>1);int cc=0;
     #pragma unroll
     for(int i=0;i<KPL;++i)cc+=key[i]<=mid;
     cc=__reduce_add_sync(0xFFFFFFFFu,cc);
     if(cc>kth)hi=mid;else lo=mid+1;
    }
    }
    float med=dl_key2f(lo);
    if(lo==0x80000000u){ // zero class: reproduce the sign CUB's stable (thread,item)-ordered sort leaves at this rank
     int below=0;
     #pragma unroll
     for(int i=0;i<KPL;++i)below+=key[i]<0x80000000u;
     below=__reduce_add_sync(0xFFFFFFFFu,below);
     const int g=kth-below;
     int plo=0,phi=CT*CI-1;
     while(plo<phi){
      const int mid=(plo+phi)>>1;int cc=0;
      #pragma unroll
      for(int i=0;i<KPL;++i){const int j=lane+32*i;cc+=(key[i]==0x80000000u)&&((j%CT)*CI+j/CT<=mid);}
      cc=__reduce_add_sync(0xFFFFFFFFu,cc);
      if(cc>g)phi=mid;else plo=mid+1;
     }
     const int j2=plo/CI+(plo%CI)*CT;
     const int e2=c0+j2+(j2/side)*(TW-side);
     med=axis==0?ld0(e2):ld1(e2);
    }
    if(axis==0)bx=med;else by=med;
   }
  }
  }
  if constexpr(Phase==1){if(lane==0)medbuf[b*N+pos]=make_float4(bx,by,fin?1.f:0.f,0.f);continue;}
  // ---- IRLS, identical FP32 expressions and summation structure to field_constant_float.
  // base=(float)(W*G) and its sum do not change between iterations, so they are computed once per centre
  // (same values, same summation order as the streamed kernel recomputes every iteration).
  bool good=false;
  if(fin){
   float base[Hoist?4:1][Hoist?BPL:1];float tb=0.f;
   if constexpr(Hoist){
    float sb[4]={0.f,0.f,0.f,0.f};
    #pragma unroll
    for(int i=0;i<BPL;++i){
     #pragma unroll
     for(int s=0;s<4;++s){
      const int j=lane+32*s+128*i;float bv=0.f;
      if(j<K){const int e=c0+j+(j/side)*(TW-side);if constexpr(F32)bv=sp[e].w*__ldg(G32+j);else bv=dl_base_f(sW[e],__ldg(G+j));sb[s]+=bv;}
      base[s][i]=bv;
     }
    }
    float a=(sb[0]+sb[2])+(sb[1]+sb[3]);
    #pragma unroll
    for(int d=16;d;d>>=1)a+=__shfl_down_sync(0xFFFFFFFFu,a,d);
    tb=__shfl_sync(0xFFFFFFFFu,a,0);
   }
   for(int it=0;it<iters;++it){ // iters<0 (parity tests only): median initializer alone
    float sw[4]={0.f,0.f,0.f,0.f},sx[4]={0.f,0.f,0.f,0.f},sy[4]={0.f,0.f,0.f,0.f},sb[4]={0.f,0.f,0.f,0.f};
    auto eval=[&](const int s,const int i){
      const int j=lane+32*s+128*i;
      if(j<K){
       const int jy=j/side,e=c0+j+jy*(TW-side);
       if constexpr(F32){
        // FP32 class: inverse scale finite and > 0 inside the numerical-domain gate, so no FP64 fallback; contracted FMAs.
        const float4 p=sp[e];
        const float dx=p.x-bx,dy=p.y-by;
        const float squared=__fmaf_rn(dy,dy,dx*dx)*p.z;
        float bs;
        if constexpr(Hoist)bs=base[s][i];else{bs=p.w*__ldg(G32+j);sb[s]+=bs;}
        const float q=fmaxf(1.0f-squared,0.0f);const float wt=bs*(q*q);
        sw[s]+=wt;sx[s]=__fmaf_rn(wt,p.x,sx[s]);sy[s]=__fmaf_rn(wt,p.y,sy[s]);
       }else{
       const float v0=sv0[e],v1=sv1[e];
       const float inv=sinv[e];
       const float dx=v0-bx,dy=v1-by;
       const float norm=dx*dx+dy*dy;
       float squared;
       if(isfinite(inv)&&inv>0.0f&&isfinite(norm))squared=(norm<0.0f?0.0f:norm)*inv;
       else{
        const int yy=y+jy-R,xx=x+(j-jy*side)-R;
        const double sigma=(yy>=0&&yy<ny&&xx>=0&&xx<nx)?S[b*N+yy*nx+xx]:1.0;
        const double z=sqrt((norm<0.0?0.0:norm))/(c*sigma*rootm);squared=z*z;
       }
       float bs;
       if constexpr(Hoist)bs=base[s][i];else{bs=dl_base_f(sW[e],__ldg(G+j));sb[s]+=bs;}
       float q=1.0f-squared;q=q<0.0f?0.0f:q;const float wt=bs*(q*q);
       sw[s]+=wt;sx[s]+=wt*v0;sy[s]+=wt*v1;
       }
      }
    };
    if constexpr(Hoist){ // base array in registers: fully unrolled, s-major
     #pragma unroll
     for(int s=0;s<4;++s){
      #pragma unroll
      for(int i=0;i<BPL;++i)eval(s,i);
     }
    }else{ // rolled i loop, four independent partial sums per lane (register pressure at radius 24)
     #pragma unroll 1
     for(int i=0;i<BPL;++i){
      #pragma unroll
      for(int s=0;s<4;++s)eval(s,i);
     }
    }
    float tot[4];
    #pragma unroll
    for(int q=0;q<(Hoist?3:4);++q){
     const float* part=q==0?sw:(q==1?sx:(q==2?sy:sb));
     float a=(part[0]+part[2])+(part[1]+part[3]);
     #pragma unroll
     for(int d=16;d;d>>=1)a+=__shfl_down_sync(0xFFFFFFFFu,a,d);
     tot[q]=__shfl_sync(0xFFFFFFFFu,a,0);
    }
    const float a=tot[0]+(Hoist?tb:tot[3])*(float)prior;
    good=a>1e-12;
    if(good){bx=tot[1]/a;by=tot[2]/a;}
   }
  }
  if(iters<0)good=true;
  if(lane==0){
   const bool ok=good&&fin;
   out[(b*2)*N+pos]=ok?(double)bx:Y[(b*2)*N+pos];
   out[(b*2+1)*N+pos]=ok?(double)by:Y[(b*2+1)*N+pos];
  }
 }
}

template<int Phase,bool Hoist,int R,int BX,int BY,int NW,bool F32=false,bool RS=false>
static int launch_one(const double* Y,const double* W,const double* S,const double* M,const double* G,const double* inverse,
 double* out,float4* medbuf,int B,int ny,int nx,int iters,double prior,double c,cudaStream_t stream,const float* G32=nullptr){
 constexpr int T=(BX+2*R)*(BY+2*R);
 size_t smem=F32?size_t(T)*(sizeof(float4)+1):size_t(T)*(sizeof(double)+3*sizeof(float)+1);
 if(RS)smem=((smem+15)&~size_t(15))+size_t(NW)*256*sizeof(unsigned);
 static bool configured=false;
 if(!configured){
  cudaError_t e=cudaFuncSetAttribute(field_tiled<Phase,Hoist,R,BX,BY,NW,F32,RS>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem);
  if(e!=cudaSuccess)return (int)e;
  configured=true;
 }
 dim3 grid((nx+BX-1)/BX,(ny+BY-1)/BY,B); field_tiled<Phase,Hoist,R,BX,BY,NW,F32,RS><<<grid,NW*32,smem,stream>>>(Y,W,S,M,G,inverse,out,medbuf,ny,nx,iters,prior,c,G32);
 return (int)cudaGetLastError();
}

template<bool Hoist,int R,int BX,int BY,int NW,bool F32=false>
static int launch_tiled(const double* Y,const double* W,const double* S,const double* M,const double* G,const double* inverse,
 double* out,float4* medbuf,int B,int ny,int nx,int iters,double prior,double c,cudaStream_t stream,const float* G32=nullptr){
 return launch_one<0,Hoist,R,BX,BY,NW,F32>(Y,W,S,M,G,inverse,out,medbuf,B,ny,nx,iters,prior,c,stream,G32);
}

extern "C" DL_EXPORT int dl_t2_field_tiled(const double* Y,const double* W,const double* S,const double* M,const double* G,
 const double* inverse,double* out,void* medbuf_,int B,int ny,int nx,int r,int iters,double prior,double c,void* stream){
 if(B<1||ny<1||nx<1||iters==0||B>65535)return int(cudaErrorInvalidValue);
 cudaStream_t s=(cudaStream_t)stream;float4* medbuf=(float4*)medbuf_;
 switch(r){
  case 24:{ // two launches (median, then IRLS) sharing a float4 scratch of B*ny*nx; no host sync
   if(!medbuf)return int(cudaErrorInvalidValue);
   int e=launch_one<1,false,24,8,8,16>(Y,W,S,M,G,inverse,out,medbuf,B,ny,nx,iters,prior,c,s);
   if(e)return e;
   return launch_one<2,false,24,8,12,24>(Y,W,S,M,G,inverse,out,medbuf,B,ny,nx,iters,prior,c,s);
  }
  case 12:return launch_tiled<true,12,16,16,8>(Y,W,S,M,G,inverse,out,medbuf,B,ny,nx,iters,prior,c,s);
  case 6:return launch_tiled<true,6,16,16,8>(Y,W,S,M,G,inverse,out,medbuf,B,ny,nx,iters,prior,c,s);
  case 3:return launch_tiled<true,3,16,16,8>(Y,W,S,M,G,inverse,out,medbuf,B,ny,nx,iters,prior,c,s);
  case 2:return launch_tiled<true,2,16,16,8>(Y,W,S,M,G,inverse,out,medbuf,B,ny,nx,iters,prior,c,s);
  case 1:return launch_tiled<true,1,16,16,8>(Y,W,S,M,G,inverse,out,medbuf,B,ny,nx,iters,prior,c,s);
  default:return int(cudaErrorInvalidValue);
 }
}

// field_backend "tiled_fp32": same kernels with the FP32 weight base float(W)*float(G32) (G32 = float Gaussian table). Additive export.
extern "C" DL_EXPORT int dl_t2_field_tiled_f32(const double* Y,const double* W,const double* S,const double* M,const float* G32,
 const double* inverse,double* out,void* medbuf_,int B,int ny,int nx,int r,int iters,double prior,double c,void* stream){
 if(B<1||ny<1||nx<1||iters==0||B>65535)return int(cudaErrorInvalidValue);
 cudaStream_t s=(cudaStream_t)stream;float4* medbuf=(float4*)medbuf_;
 switch(r){
  case 24:{
   if(!medbuf)return int(cudaErrorInvalidValue);
   int e=launch_one<1,false,24,8,8,16,true>(Y,W,S,M,nullptr,inverse,out,medbuf,B,ny,nx,iters,prior,c,s,G32);
   if(e)return e;
   return launch_one<2,false,24,8,12,24,true>(Y,W,S,M,nullptr,inverse,out,medbuf,B,ny,nx,iters,prior,c,s,G32);
  }
  case 12:return launch_tiled<true,12,16,16,8,true>(Y,W,S,M,nullptr,inverse,out,medbuf,B,ny,nx,iters,prior,c,s,G32);
  case 6:return launch_tiled<true,6,16,16,8,true>(Y,W,S,M,nullptr,inverse,out,medbuf,B,ny,nx,iters,prior,c,s,G32);
  case 3:return launch_tiled<true,3,16,16,8,true>(Y,W,S,M,nullptr,inverse,out,medbuf,B,ny,nx,iters,prior,c,s,G32);
  case 2:return launch_tiled<true,2,16,16,8,true>(Y,W,S,M,nullptr,inverse,out,medbuf,B,ny,nx,iters,prior,c,s,G32);
  case 1:return launch_tiled<true,1,16,16,8,true>(Y,W,S,M,nullptr,inverse,out,medbuf,B,ny,nx,iters,prior,c,s,G32);
  default:return int(cudaErrorInvalidValue);
 }
}

// field_backend "tiled_fp32" with median_select "radix": radius 24 median initializer by shared-memory radix select
// (bit-identical to dl_t2_field_tiled_f32); every other radius forwards unchanged. Additive export.
extern "C" DL_EXPORT int dl_t2_field_tiled_f32_rs(const double* Y,const double* W,const double* S,const double* M,const float* G32,
 const double* inverse,double* out,void* medbuf_,int B,int ny,int nx,int r,int iters,double prior,double c,void* stream){
 if(r!=24)return dl_t2_field_tiled_f32(Y,W,S,M,G32,inverse,out,medbuf_,B,ny,nx,r,iters,prior,c,stream);
 if(B<1||ny<1||nx<1||iters==0||B>65535||!medbuf_)return int(cudaErrorInvalidValue);
 cudaStream_t s=(cudaStream_t)stream;float4* medbuf=(float4*)medbuf_;
 int e=launch_one<1,false,24,8,8,16,true,true>(Y,W,S,M,nullptr,inverse,out,medbuf,B,ny,nx,iters,prior,c,s,G32);
 if(e)return e;
 return launch_one<2,false,24,8,12,24,true>(Y,W,S,M,nullptr,inverse,out,medbuf,B,ny,nx,iters,prior,c,s,G32);
}

// Per-tile squared residual to the 5x5 lower median (field.scale_ratio, M=None, P=2), FP64, no contraction.
// NaN where the tile is invalid or z2 is not finite; the caller takes the nanmedian per pair exactly as Torch does.
__global__ void scale_z2_kernel(const double* Y,const double* W,const double* S,double* z2,int ny,int nx){
 const int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y,b=blockIdx.z;
 if(x>=nx||y>=ny)return;
 const int N=ny*nx,pos=y*nx+x;
 double v0[25],v1[25];int n=0;
 for(int dy=-2;dy<=2;++dy)for(int dx=-2;dx<=2;++dx){
  const int yy=y+dy,xx=x+dx;
  if(yy>=0&&yy<ny&&xx>=0&&xx<nx&&W[b*N+yy*nx+xx]>0.0){v0[n]=Y[(b*2)*N+yy*nx+xx];v1[n]=Y[(b*2+1)*N+yy*nx+xx];++n;}
 }
 double res=CUDART_NAN;
 if(W[b*N+pos]>0.0&&n>0){
  const int k=(n-1)/2;double med[2];
  for(int a=0;a<2;++a){
   const double* v=a?v1:v0;med[a]=0.0;
   for(int i=0;i<n;++i){int l=0,le=0;for(int j=0;j<n;++j){l+=v[j]<v[i];le+=v[j]<=v[i];}if(l<=k&&k<le){med[a]=v[i];break;}}
  }
  const double r0=__dsub_rn(Y[(b*2)*N+pos],med[0]),r1=__dsub_rn(Y[(b*2+1)*N+pos],med[1]);
  const double sg=S[b*N+pos];
  res=__ddiv_rn(__dadd_rn(__dmul_rn(r0,r0),__dmul_rn(r1,r1)),__dmul_rn(sg,sg));
  if(!isfinite(res))res=CUDART_NAN;
 }
 z2[b*N+pos]=res;
}

extern "C" DL_EXPORT int dl_t2_scale_z2(const double* Y,const double* W,const double* S,double* z2,int B,int ny,int nx,void* stream){
 if(B<1||ny<1||nx<1||B>65535)return int(cudaErrorInvalidValue);
 dim3 blk(16,16),grid((nx+15)/16,(ny+15)/16,B);
 scale_z2_kernel<<<grid,blk,0,(cudaStream_t)stream>>>(Y,W,S,z2,ny,nx);
 return int(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Exact FP64 row lower-median (robust stage 2). Matches torch.median(dim) (nanmode 0: any NaN -> NaN) and
// torch.nanmedian(dim) (nanmode 1: NaN ignored, all-NaN -> NaN) on CUDA: element k=(n_valid-1)/2 of the ascending order,
// ordering by the monotone 64-bit key in which -0 < +0 (as torch's radixSelect). Radix select, 8 passes of 8 bits, one block per row.
__device__ __forceinline__ unsigned long long dl_d2key(double d){const unsigned long long u=(unsigned long long)__double_as_longlong(d);return (u>>63)?~u:(u|0x8000000000000000ULL);}
__device__ __forceinline__ double dl_key2d(unsigned long long k){const unsigned long long u=(k>>63)?(k&0x7FFFFFFFFFFFFFFFULL):~k;return __longlong_as_double((long long)u);}

__global__ void row_median_kernel(const double* x,double* out,int n,long long rs,long long cs,int nanmode){
 __shared__ unsigned hist[256];__shared__ unsigned long long prefix;__shared__ int nnan,kk;
 const double* row=x+blockIdx.x*rs;
 if(threadIdx.x==0){nnan=0;prefix=0;}
 __syncthreads();
 int ln=0;
 for(int i=threadIdx.x;i<n;i+=blockDim.x)ln+=isnan(row[i*cs]);
 if(ln)atomicAdd(&nnan,ln);
 __syncthreads();
 if(nnan>0&&nanmode==0||nnan==n){if(threadIdx.x==0)out[blockIdx.x]=CUDART_NAN;return;}
 if(threadIdx.x==0)kk=(n-nnan-1)/2;
 for(int pass=0;pass<8;++pass){
  const int shift=56-8*pass;
  for(int b=threadIdx.x;b<256;b+=blockDim.x)hist[b]=0;
  __syncthreads();
  const unsigned long long pre=prefix;
  for(int i=threadIdx.x;i<n;i+=blockDim.x){
   const double v=row[i*cs];
   if(isnan(v))continue;
   const unsigned long long k=dl_d2key(v);
   if(pass==0||(k>>(shift+8))==(pre>>(shift+8)))atomicAdd(&hist[(unsigned)((k>>shift)&255ULL)],1u);
  }
  __syncthreads();
  if(threadIdx.x==0){
   int k=kk;unsigned digit=255;
   for(unsigned b=0;b<256;++b){if(k<(int)hist[b]){digit=b;break;}k-=(int)hist[b];}
   kk=k;prefix|=((unsigned long long)digit)<<shift;
  }
  __syncthreads();
 }
 if(threadIdx.x==0)out[blockIdx.x]=dl_key2d(prefix);
}

extern "C" DL_EXPORT int dl_t2_row_median(const double* x,double* out,int rows,int n,long long rs,long long cs,int nanmode,void* stream){
 if(rows<1||n<1||rows>65535)return int(cudaErrorInvalidValue);
 row_median_kernel<<<rows,512,0,(cudaStream_t)stream>>>(x,out,n,rs,cs,nanmode);
 return int(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Multi-block exact FP64 row lower-median (fused_motion "median"): same selection (8 radix passes of 8 bits over the monotone
// 64-bit key) and the same result bits as row_median_kernel / torch.median, but R blocks cooperate on each row. The blocks of a row
// meet at a software barrier after each pass (counters in ws, zeroed per launch; all blocks must be co-resident, the host side caps
// rows*R at 4 blocks per SM). ws layout per row: 8*256 histogram bins, NaN count, barrier counter (2056 words).
#define MB_ROW_WORDS 2056
#define MB_UNROLL 8
__global__ void row_median_mb_kernel(const double* x,double* out,int n,long long rs,long long cs,int nanmode,unsigned* ws,int R){
 __shared__ unsigned sh[260];__shared__ unsigned wsum[8];__shared__ int sdig,skk;
 const int row=blockIdx.y,part=blockIdx.x,tid=threadIdx.x,lane=tid&31,warp=tid>>5;
 unsigned* gh=ws+(size_t)row*MB_ROW_WORDS;unsigned* gnan=gh+2048;unsigned* gbar=gh+2049;
 const double* rowp=x+(long long)row*rs;
 const int chunk=(n+R-1)/R,i0=part*chunk,i1=min(n,i0+chunk);
 unsigned long long prefix=0;int kk=0;unsigned phase=0;
 for(int pass=0;pass<8;++pass){
  const int shift=56-8*pass;
  for(int b=tid;b<260;b+=256)sh[b]=0;
  if(tid==0){sdig=255;skk=kk;}
  __syncthreads();
  for(int base=i0;base<i1;base+=256*MB_UNROLL){
   double vv[MB_UNROLL];
   #pragma unroll
   for(int u=0;u<MB_UNROLL;++u){const int i=base+u*256+tid;vv[u]=i<i1?rowp[(long long)i*cs]:0.0;}
   #pragma unroll
   for(int u=0;u<MB_UNROLL;++u){
    const int i=base+u*256+tid;unsigned digit=256;bool nan_=false;
    if(i<i1){
     const double v=vv[u];nan_=isnan(v);
     if(!nan_){const unsigned long long k=dl_d2key(v);if(pass==0||(k>>(shift+8))==(prefix>>(shift+8)))digit=(unsigned)((k>>shift)&255ULL);}
    }
    if(pass==0){const unsigned nm=__ballot_sync(0xffffffffu,nan_);if(lane==0&&nm)atomicAdd(&sh[256],(unsigned)__popc(nm));}
    if(digit<256u)atomicAdd(&sh[digit],1u);
   }
  }
  __syncthreads();
  if(sh[tid])atomicAdd(&gh[pass*256+tid],sh[tid]);
  if(pass==0&&tid==0&&sh[256])atomicAdd(gnan,sh[256]);
  __threadfence();
  __syncthreads();
  if(tid==0){
   __threadfence();atomicAdd(gbar,1u);++phase;
   while(*((volatile unsigned*)gbar)<(unsigned)R*phase){}
   __threadfence();
  }
  __syncthreads();
  if(pass==0){
   const int nn=(int)__ldcg(gnan);
   if((nn>0&&nanmode==0)||nn==n){if(part==0&&tid==0)out[row]=CUDART_NAN;return;}
   kk=(n-nn-1)/2;if(tid==0)skk=kk;
  }
  const unsigned c=__ldcg(&gh[pass*256+tid]);
  unsigned v=c;
  for(int o=1;o<32;o<<=1){const unsigned u=__shfl_up_sync(0xffffffffu,v,o);if(lane>=o)v+=u;}
  if(lane==31)wsum[warp]=v;
  __syncthreads();
  unsigned off=0;for(int w=0;w<warp;++w)off+=wsum[w];
  const unsigned incl=v+off,excl=incl-c;
  if((unsigned)kk>=excl&&(unsigned)kk<incl){sdig=tid;skk=kk-(int)excl;}
  __syncthreads();
  kk=skk;prefix|=((unsigned long long)sdig)<<shift;
  __syncthreads();
 }
 if(part==0&&tid==0)out[row]=dl_key2d(prefix);
}

extern "C" DL_EXPORT int dl_t2_row_median_mb(const double* x,double* out,int rows,int n,long long rs,long long cs,int nanmode,unsigned* ws,int R,void* stream){
 if(rows<1||n<1||R<1||R>64||rows>65535)return int(cudaErrorInvalidValue);
 cudaMemsetAsync(ws,0,(size_t)rows*MB_ROW_WORDS*sizeof(unsigned),(cudaStream_t)stream);
 row_median_mb_kernel<<<dim3(R,rows),256,0,(cudaStream_t)stream>>>(x,out,n,rs,cs,nanmode,ws,R);
 return int(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Exact FP64 row selection (J5a, global translation): the same element as row_median_kernel / torch.median(dim), i.e. the
// element at rank (n_valid-1)/2 of the monotone 64-bit double key (-0 < +0; ties are equal values), but it stops the radix
// passes as soon as the selected bucket fits a gather list and finishes the selection over that list. For the tile-vector
// rows (n ~ 6e4) the top-byte bucket is still large, so the usual cost is two 8-bit passes plus one gather instead of eight
// passes. Same multi-block software barrier and the same shared 256-bin histogram as row_median_mb, so each pass keeps the
// ~1 TB/s effective bandwidth of that kernel. ws layout per row: 8x256 histogram bins, NaN count, barrier counter, gather
// counter, then RSEL_CAP 64-bit keys (2 words each).
#define RSEL_CAP 8192
#define RSEL_HIST 2048
#define RSEL_HDR 4
#define RSEL_G_NAN RSEL_HIST
#define RSEL_G_BAR (RSEL_HIST+1)
#define RSEL_G_CNT (RSEL_HIST+2)
#define RSEL_ROW_HDR (RSEL_HIST + RSEL_HDR)                 // zeroed per launch; the key list after the last row is not
#define RSEL_ROW_WORDS (RSEL_ROW_HDR + RSEL_CAP*2)
#define RSEL_UNROLL 8
__global__ void row_select_mb_kernel(const double* x,double* out,int n,long long rs,long long cs,int nanmode,
                                     unsigned* ws,int R){
 __shared__ unsigned sh[260];__shared__ unsigned wsum[8];__shared__ int sdig,skk;
 const int row=blockIdx.y,part=blockIdx.x,tid=threadIdx.x,lane=tid&31,warp=tid>>5;
 unsigned* gh=ws+(size_t)row*RSEL_ROW_HDR;
 unsigned* gnan=gh+RSEL_G_NAN;unsigned* gbar=gh+RSEL_G_BAR;unsigned* gcnt=gh+RSEL_G_CNT;
 unsigned long long* keys=(unsigned long long*)(ws+(size_t)gridDim.y*RSEL_ROW_HDR)+row*RSEL_CAP;
 const double* rowp=x+(long long)row*rs;
 const int chunk=(n+R-1)/R,i0=part*chunk,i1=min(n,i0+chunk);
 unsigned long long prefix=0;int kk=0,ncur=n+1;unsigned phase=0;int consumed=0;
 for(int pass=0;pass<8;++pass){
  if(pass>0&&ncur<=RSEL_CAP)break;
  const int shift=56-8*pass;
  for(int b=tid;b<260;b+=256)sh[b]=0;
  if(tid==0){sdig=255;skk=kk;}
  __syncthreads();
  for(int base=i0;base<i1;base+=256*RSEL_UNROLL){
   double vv[RSEL_UNROLL];
   #pragma unroll
   for(int u=0;u<RSEL_UNROLL;++u){const int i=base+u*256+tid;vv[u]=i<i1?rowp[(long long)i*cs]:0.0;}
   #pragma unroll
   for(int u=0;u<RSEL_UNROLL;++u){
    const int i=base+u*256+tid;unsigned digit=256;bool nan_=false;
    if(i<i1){
     const double v=vv[u];nan_=isnan(v);
     if(!nan_){const unsigned long long k=dl_d2key(v);if(pass==0||(k>>(shift+8))==(prefix>>(shift+8)))digit=(unsigned)((k>>shift)&255ULL);}
    }
    if(pass==0){const unsigned nm=__ballot_sync(0xffffffffu,nan_);if(lane==0&&nm)atomicAdd(&sh[256],(unsigned)__popc(nm));}
    if(digit<256u)atomicAdd(&sh[digit],1u);
   }
  }
  __syncthreads();
  if(sh[tid])atomicAdd(&gh[pass*256+tid],sh[tid]);
  if(pass==0&&tid==0&&sh[256])atomicAdd(gnan,sh[256]);
  __threadfence();
  __syncthreads();
  if(tid==0){
   __threadfence();atomicAdd(gbar,1u);++phase;
   while(*((volatile unsigned*)gbar)<(unsigned)R*phase){}
   __threadfence();
  }
  __syncthreads();
  if(pass==0){
   const int nn=(int)__ldcg(gnan);
   if((nn>0&&nanmode==0)||nn==n){if(part==0&&tid==0)out[row]=CUDART_NAN;return;}
   kk=(n-nn-1)/2;if(tid==0)skk=kk;
  }
  const unsigned c=__ldcg(&gh[pass*256+tid]);
  unsigned v=c;
  for(int o=1;o<32;o<<=1){const unsigned u=__shfl_up_sync(0xffffffffu,v,o);if(lane>=o)v+=u;}
  if(lane==31)wsum[warp]=v;
  __syncthreads();
  unsigned off=0;for(int w=0;w<warp;++w)off+=wsum[w];
  const unsigned incl=v+off,excl=incl-c;
  if((unsigned)kk>=excl&&(unsigned)kk<incl){sdig=tid;skk=kk-(int)excl;}
  __syncthreads();
  const int digit=sdig;
  kk=skk;prefix|=((unsigned long long)digit)<<shift;
  __syncthreads();
  ncur=(int)__ldcg(&gh[pass*256+digit]);
  consumed=pass+1;
 }
 if(consumed>=8){
  if(part==0&&tid==0)out[row]=dl_key2d(prefix);
  return;
 }
 // gather the selected bucket once, then finish exactly over the list (all keys share the fixed prefix)
 const int shf=64-8*consumed;
 for(int i=i0+tid;i<i1;i+=blockDim.x){
  const double v=rowp[(long long)i*cs];
  if(isnan(v))continue;
  const unsigned long long k=dl_d2key(v);
  if((k>>shf)==(prefix>>shf)){const int p=(int)atomicAdd(gcnt,1u);if(p<RSEL_CAP)keys[p]=k;}
 }
 __threadfence();__syncthreads();
 if(tid==0){__threadfence();atomicAdd(gbar,1u);++phase;
  while(*((volatile unsigned*)gbar)<(unsigned)R*phase){}__threadfence();}
 __syncthreads();
 if(part!=0)return;
 const int total=(int)__ldcg(gcnt);
 for(int pass=0;pass<8;++pass){
  const int shift=56-8*pass;
  for(int b=tid;b<260;b+=256)sh[b]=0;
  if(tid==0){sdig=255;skk=kk;}
  __syncthreads();
  for(int i=tid;i<total;i+=256){
   const unsigned long long k=keys[i];
   if(pass==0||(k>>(shift+8))==(prefix>>(shift+8)))atomicAdd(&sh[(unsigned)((k>>shift)&255ULL)],1u);
  }
  __syncthreads();
  const unsigned c=sh[tid];
  unsigned v=c;
  for(int o=1;o<32;o<<=1){const unsigned u=__shfl_up_sync(0xffffffffu,v,o);if(lane>=o)v+=u;}
  if(lane==31)wsum[warp]=v;
  __syncthreads();
  unsigned off=0;for(int w=0;w<warp;++w)off+=wsum[w];
  const unsigned incl=v+off,excl=incl-c;
  if((unsigned)kk>=excl&&(unsigned)kk<incl){sdig=tid;skk=kk-(int)excl;}
  __syncthreads();
  kk=skk;prefix|=((unsigned long long)sdig)<<shift;
  __syncthreads();
 }
 if(tid==0)out[row]=dl_key2d(prefix);
}

extern "C" DL_EXPORT int dl_t2_row_select(const double* x,double* out,int rows,int n,long long rs,long long cs,int nanmode,
                                          unsigned* ws,int R,void* stream){
 if(rows<1||n<1||rows>65535||R<1||R>64)return int(cudaErrorInvalidValue);
 cudaMemsetAsync(ws,0,(size_t)rows*RSEL_ROW_HDR*sizeof(unsigned),(cudaStream_t)stream);
 row_select_mb_kernel<<<dim3(R,rows),256,0,(cudaStream_t)stream>>>(x,out,n,rs,cs,nanmode,ws,R);
 return int(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Fused candidate scoring (score_backend "fused"), 16x16 tiles only. Additive exports; cuFFT (torch rfft2) sits between the two.
// pre : gather (clamped, reflect-padded neighbour P) + Hann-weighted mean removal + window -> FP32 tile, one warp per (tile, candidate).
// post: Hermitian half spectrum -> phase ramp -> |A-B|^2/(2 var) (DC bin 0) -> warp-sum / 255. No shifted spectrum is materialised.
// Arithmetic follows the torch ops of t2_score_cache.batched_chunk element by element (no fma: build uses --fmad=false); only the
// summation order of the 256-element sums differs.
__global__ void score_pre_kernel(const float* P,int H,int W,const long long* x0,const long long* y0,const float* V,const float* win,
 const float* winsum,float* inp,int C,int M,int a){
 const int tile=blockIdx.x,c=threadIdx.x>>5,lane=threadIdx.x&31,gt=a+tile;
 const long long vi=((long long)c*M+gt)*2;
 const long long bx=x0[gt]+(long long)rintf(V[vi])+96,by=y0[gt]+(long long)rintf(V[vi+1])+96;
 const int row=lane>>1,col0=(lane&1)*8;
 long long yy=by+row;yy=yy<0?0:(yy>H-1?H-1:yy);
 float g[8],w[8],s=0.f;
 #pragma unroll
 for(int j=0;j<8;++j){
  long long xx=bx+col0+j;xx=xx<0?0:(xx>W-1?W-1:xx);
  g[j]=P[yy*W+xx];w[j]=win[row*16+col0+j];s+=g[j]*w[j];
 }
 #pragma unroll
 for(int m=16;m>=1;m>>=1)s+=__shfl_xor_sync(0xffffffffu,s,m);
 const float mean=s/winsum[0];
 float* o=inp+((long long)tile*C+c)*256+row*16+col0;
 #pragma unroll
 for(int j=0;j<8;++j)o[j]=(g[j]-mean)*w[j];
}

__device__ __forceinline__ float score_bin(float2 s,float2 A,float v,float fu,float fv,float fx,float fy){
 const float ph=6.2831855f*(fv*fx+fu*fy);
 const float cs=cosf(ph),sn=sinf(ph);
 const float br=s.x*cs-s.y*sn,bi=s.x*sn+s.y*cs;
 const float dr=A.x-br,di=A.y-bi,m=hypotf(dr,di);
 return m*m/(2.f*v);
}

__global__ void score_post_kernel(const float2* S,const float2* A,const float* var,const float* V,const float* f,float* out,int C,int M,int a,int mc){
 const int tile=blockIdx.x,c=threadIdx.x>>5,lane=threadIdx.x&31,gt=a+tile;
 const long long vi=((long long)c*M+gt)*2;
 const float vx=V[vi],vy=V[vi+1],fx=vx-(float)(long long)rintf(vx),fy=vy-(float)(long long)rintf(vy);
 const float2* s=S+((long long)tile*C+c)*144;
 const float2* Ar=A+(long long)gt*256;const float* vr=var+(long long)gt*256;
 float acc=0.f;
 for(int idx=lane;idx<144;idx+=32){
  const int u=idx/9,v=idx%9;const float2 sv=s[idx];
  if(idx!=0)acc+=score_bin(sv,Ar[u*16+v],vr[u*16+v],f[u],f[v],fx,fy);
  if(v>=1&&v<=7){const int u2=(16-u)&15,v2=16-v;acc+=score_bin(make_float2(sv.x,-sv.y),Ar[u2*16+v2],vr[u2*16+v2],f[u2],f[v2],fx,fy);}
 }
 #pragma unroll
 for(int m=16;m>=1;m>>=1)acc+=__shfl_xor_sync(0xffffffffu,acc,m);
 if(lane==0)out[(long long)c*mc+tile]=acc/255.f;
}

extern "C" DL_EXPORT int dl_t2_score_pre(const float* P,int H,int W,const long long* x0,const long long* y0,const float* V,const float* win,
 const float* winsum,float* inp,int C,int M,int a,int mc,void* stream){
 if(C<1||C>32||mc<1||mc>2147483647)return int(cudaErrorInvalidValue);
 score_pre_kernel<<<mc,32*C,0,(cudaStream_t)stream>>>(P,H,W,x0,y0,V,win,winsum,inp,C,M,a);
 return int(cudaGetLastError());
}
extern "C" DL_EXPORT int dl_t2_score_post(const float2* S,const float2* A,const float* var,const float* V,const float* f,float* out,
 int C,int M,int a,int mc,void* stream){
 if(C<1||C>32||mc<1)return int(cudaErrorInvalidValue);
 score_post_kernel<<<mc,32*C,0,(cudaStream_t)stream>>>(S,A,var,V,f,out,C,M,a,mc);
 return int(cudaGetLastError());
}

// K7.4 FP64 Huber glT primitives. Exact selection remains outside these kernels.
__global__ void huber_residual_rows(const double* y,const double* beta,double* res,int rows,int N){
 long long i=(long long)blockIdx.x*256+threadIdx.x;if(i<(long long)rows*N)res[i]=y[i]-beta[i/N];
}
__global__ void huber_deviation(const double* res,const double* center,double* dev,int rows,int N){
 long long i=(long long)blockIdx.x*256+threadIdx.x;if(i<(long long)rows*N)dev[i]=fabs(res[i]-center[i/N]);
}
__global__ void huber_partials(const double* y,const double* res,const double* mad,double* parts,int rows,int N,int B){
 int row=blockIdx.y,tid=threadIdx.x;int start=blockIdx.x*1024+tid;
 __shared__ double num[256],den[256];double sn=0.,sd=0.;
 double scale=fmax(1.4826*mad[row],1e-4);
 for(int j=start;j<N&&j<((int)blockIdx.x+1)*1024;j+=256){
  long long at=(long long)row*N+j;double weight=fmin(1.345*scale/fmax(fabs(res[at]),1e-12),1.);
  sn+=weight*y[at];sd+=weight;
 }
 num[tid]=sn;den[tid]=sd;__syncthreads();
 for(int sh=128;sh>0;sh>>=1){if(tid<sh){num[tid]+=num[tid+sh];den[tid]+=den[tid+sh];}__syncthreads();}
 if(tid==0){parts[((long long)row*B+blockIdx.x)*2]=num[0];parts[((long long)row*B+blockIdx.x)*2+1]=den[0];}
}
__global__ void huber_finish(const double* parts,double* beta,int B){
 int row=blockIdx.x,tid=threadIdx.x;__shared__ double num[256],den[256];double sn=0.,sd=0.;
 for(int b=tid;b<B;b+=256){sn+=parts[((long long)row*B+b)*2];sd+=parts[((long long)row*B+b)*2+1];}
 num[tid]=sn;den[tid]=sd;__syncthreads();
 for(int sh=128;sh>0;sh>>=1){if(tid<sh){num[tid]+=num[tid+sh];den[tid]+=den[tid+sh];}__syncthreads();}
 if(tid==0)beta[row]=num[0]/(den[0]+1e-9);
}
extern "C" DL_EXPORT int dl_t2_huber_residual(const double*y,const double*beta,double*res,int rows,int N,void*stream){
 if(rows<1||rows>32||N<1)return int(cudaErrorInvalidValue);
 huber_residual_rows<<<((long long)rows*N+255)/256,256,0,(cudaStream_t)stream>>>(y,beta,res,rows,N);return int(cudaGetLastError());
}
extern "C" DL_EXPORT int dl_t2_huber_deviation(const double*res,const double*center,double*dev,int rows,int N,void*stream){
 if(rows<1||rows>32||N<1)return int(cudaErrorInvalidValue);
 huber_deviation<<<((long long)rows*N+255)/256,256,0,(cudaStream_t)stream>>>(res,center,dev,rows,N);return int(cudaGetLastError());
}
extern "C" DL_EXPORT int dl_t2_huber_solve(const double*y,const double*res,const double*mad,double*parts,double*beta,int rows,int N,void*stream){
 if(rows<1||rows>32||N<1)return int(cudaErrorInvalidValue);int B=(N+1023)/1024;
 huber_partials<<<dim3(B,rows),256,0,(cudaStream_t)stream>>>(y,res,mad,parts,rows,N,B);
 cudaError_t status=cudaGetLastError();if(status!=cudaSuccess)return int(status);
 huber_finish<<<rows,256,0,(cudaStream_t)stream>>>(parts,beta,B);return int(cudaGetLastError());
}

// K8.4 opt-in box-filtered fits (FP32 envelope), global-translation initialisation: the residual field starts at 0.
// Per iteration: own-position Tukey weight (x tile confidence), zero-padded (2r+1)^2 box sums of w, w*v0, w*v1 as exact
// separable direct window sums (fixed order, no atomics), prev = num/den where den>1e-12; final zero-weight -> raw residual Y.
// Scratch: P3 (B,3,N) weights, H (B,3,N) row sums, prev (B,2,N) floats (zeroed here) = 8*B*N floats; good (B,N) bytes.
__global__ void boxfit_weights(const double* Y,const double* W,const double* inv,const float* prev,float* P3,int B,int N){
 for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;i<(long long)B*N;i+=(long long)gridDim.x*blockDim.x){
  const int b=(int)(i/N),pos=(int)(i%N);
  const float v0=(float)Y[(long long)b*2*N+pos],v1=(float)Y[((long long)b*2+1)*N+pos];
  const float p0=prev[(long long)b*2*N+pos],p1=prev[((long long)b*2+1)*N+pos];
  const float d0=v0-p0,d1=v1-p1;
  float q=1.0f-(d0*d0+d1*d1)*(float)inv[i];q=q<0.0f?0.0f:q;
  const float wt=(float)W[pos]*(q*q);
  P3[(long long)b*3*N+pos]=wt;P3[((long long)b*3+1)*N+pos]=wt*v0;P3[((long long)b*3+2)*N+pos]=wt*v1;
 }
}
__global__ void boxfit_rows(const float* P3,float* H,int B,int ny,int nx,int r){
 const int N=ny*nx;
 for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;i<(long long)B*3*N;i+=(long long)gridDim.x*blockDim.x){
  const int x=(int)(i%nx);const long long row=i-x;float s=0.f;
  const int lo=x-r<0?0:x-r,hi=x+r>=nx?nx-1:x+r;
  for(int xx=lo;xx<=hi;++xx)s+=P3[row+xx];
  H[i]=s;
 }
}
__global__ void boxfit_cols_update(const float* H,const double* Y,float* prev,unsigned char* good,double* out,int B,int ny,int nx,int r,int last){
 const int N=ny*nx;
 for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;i<(long long)B*N;i+=(long long)gridDim.x*blockDim.x){
  const int b=(int)(i/N),pos=(int)(i%N),y=pos/nx,x=pos%nx;
  const int lo=y-r<0?0:y-r,hi=y+r>=ny?ny-1:y+r;
  float d=0.f,n0=0.f,n1=0.f;
  const float* h=H+(long long)b*3*N;
  for(int yy=lo;yy<=hi;++yy){const int o=yy*nx+x;d+=h[o];n0+=h[N+o];n1+=h[2*N+o];}
  const bool g=d>1e-12f;
  float* p=prev+(long long)b*2*N;
  if(g){p[pos]=n0/d;p[N+pos]=n1/d;}
  if(last){
   good[i]=g;
   out[(long long)b*2*N+pos]=g?(double)p[pos]:Y[(long long)b*2*N+pos];
   out[((long long)b*2+1)*N+pos]=g?(double)p[N+pos]:Y[((long long)b*2+1)*N+pos];
  }
 }
}
extern "C" DL_EXPORT int dl_t2_boxfit_f32(const double* Y,const double* W,const double* inv,double* out,float* scratch,unsigned char* good,
 int B,int ny,int nx,int r,int iters,void* stream){
 if(B<1||ny<1||nx<1||r<0||r>24||iters<1)return int(cudaErrorInvalidValue);
 cudaStream_t s=(cudaStream_t)stream;const long long N=(long long)ny*nx;
 float* P3=scratch;float* H=scratch+3*B*N;float* prev=scratch+6*B*N;
 if(cudaMemsetAsync(prev,0,sizeof(float)*2*B*N,s)!=cudaSuccess)return int(cudaGetLastError());
 const int T=256;
 const int g1=(int)((B*N+T-1)/T),g3=(int)((3*B*N+T-1)/T);
 for(int it=0;it<iters;++it){
  boxfit_weights<<<g1,T,0,s>>>(Y,W,inv,prev,P3,B,(int)N);
  boxfit_rows<<<g3,T,0,s>>>(P3,H,B,ny,nx,r);
  boxfit_cols_update<<<g1,T,0,s>>>(H,Y,prev,good,out,B,ny,nx,r,it==iters-1);
 }
 return int(cudaGetLastError());
}
