// Native FP64 definition box fit, used_by: t2_native.cu.
// rules: direct row/column sums, no SAT cancellation; six iterations, unchanged1e-12 final-weight threshold/raw fallback.
__global__ void native_boxfit_weights64(const double* Y,const double* W,const double* inv,const double* prev,double* P3,int B,int N){
 for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;i<(long long)B*N;i+=(long long)gridDim.x*blockDim.x){
  const int b=(int)(i/N),pos=(int)(i%N);
  const double v0=(double)Y[(long long)b*2*N+pos],v1=(double)Y[((long long)b*2+1)*N+pos];
  const double p0=prev[(long long)b*2*N+pos],p1=prev[((long long)b*2+1)*N+pos];
  const double d0=v0-p0,d1=v1-p1;
  double q=1.0f-(d0*d0+d1*d1)*(double)inv[i];q=q<0.0f?0.0f:q;
  const double wt=(double)W[pos]*(q*q);
  P3[(long long)b*3*N+pos]=wt;P3[((long long)b*3+1)*N+pos]=wt*v0;P3[((long long)b*3+2)*N+pos]=wt*v1;
 }
}
__global__ void native_boxfit_rows64(const double* P3,double* H,int B,int ny,int nx,int r){
 const int N=ny*nx;
 for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;i<(long long)B*3*N;i+=(long long)gridDim.x*blockDim.x){
  const int x=(int)(i%nx);const long long row=i-x;double s=0.f;
  const int lo=x-r<0?0:x-r,hi=x+r>=nx?nx-1:x+r;
  for(int xx=lo;xx<=hi;++xx)s+=P3[row+xx];
  H[i]=s;
 }
}
__global__ void native_boxfit_cols_update64(const double* H,const double* Y,double* prev,unsigned char* good,double* out,int B,int ny,int nx,int r,int last){
 const int N=ny*nx;
 for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;i<(long long)B*N;i+=(long long)gridDim.x*blockDim.x){
  const int b=(int)(i/N),pos=(int)(i%N),y=pos/nx,x=pos%nx;
  const int lo=y-r<0?0:y-r,hi=y+r>=ny?ny-1:y+r;
  double d=0.f,n0=0.f,n1=0.f;
  const double* h=H+(long long)b*3*N;
  for(int yy=lo;yy<=hi;++yy){const int o=yy*nx+x;d+=h[o];n0+=h[N+o];n1+=h[2*N+o];}
  const bool g=d>1e-12;
  double* p=prev+(long long)b*2*N;
  if(g){p[pos]=n0/d;p[N+pos]=n1/d;}
  if(last){
   good[i]=g;
   out[(long long)b*2*N+pos]=g?(double)p[pos]:Y[(long long)b*2*N+pos];
   out[((long long)b*2+1)*N+pos]=g?(double)p[N+pos]:Y[((long long)b*2+1)*N+pos];
  }
 }
}
extern "C" DL_EXPORT int dl_t2_native_boxfit_fp64(const double* Y,const double* W,const double* inv,double* out,double* scratch,unsigned char* good,
 int B,int ny,int nx,int r,int iters,void* stream){
 if(B<1||ny<1||nx<1||r<0||r>24||iters<1)return int(cudaErrorInvalidValue);
 cudaStream_t s=(cudaStream_t)stream;const long long N=(long long)ny*nx;
 double* P3=scratch;double* H=scratch+3*B*N;double* prev=scratch+6*B*N;
 if(cudaMemsetAsync(prev,0,sizeof(double)*2*B*N,s)!=cudaSuccess)return int(cudaGetLastError());
 const int T=256;
 const int g1=(int)((B*N+T-1)/T),g3=(int)((3*B*N+T-1)/T);
 for(int it=0;it<iters;++it){
  native_boxfit_weights64<<<g1,T,0,s>>>(Y,W,inv,prev,P3,B,(int)N);
  native_boxfit_rows64<<<g3,T,0,s>>>(P3,H,B,ny,nx,r);
  native_boxfit_cols_update64<<<g1,T,0,s>>>(H,Y,prev,good,out,B,ny,nx,r,it==iters-1);
 }
 return int(cudaGetLastError());
}
