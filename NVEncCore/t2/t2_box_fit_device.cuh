// Device-side box-mode fit preparation (perf/mcwiener-box-device-fit, 2026-11-23).
// Included inside namespace native_t2 after Buf/Image. Replaces the host Huber global translation
// (same definition: lower median (n-1)/2, MAD scale max(1e-4,1.4826*MAD), weights min(1,1.345*s/max(|r|,1e-12)),
// 21 reweighting iterations, weighted mean summed sequentially in tile order, den starts at 1e-9).
struct VecPtrs { const float* p[16]; };
__global__ void huber_gather(VecPtrs vp,const int* sel,int S,double* y){
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=32LL*S)return;
    int series=int(i/S),s=int(i%S),k=series>>1,ax=series&1;y[i]=double(vp.p[k][2*sel[s]+ax]);
}
__global__ void huber_res(const double* y,const double* beta,double* res,int S){
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=32LL*S)return;res[i]=y[i]-beta[i/S];
}
__global__ void huber_dev(const double* res,const double* med,double* dev,int S){
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=32LL*S)return;dev[i]=fabs(res[i]-med[i/S]);
}
__global__ void huber_wt(const double* res,const double* mad,double* wt,int S){
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=32LL*S)return;
    double s=fmax(1e-4,1.4826*mad[i/S]);wt[i]=fmin(1.,1.345*s/fmax(fabs(res[i]),1e-12));
}
__global__ void huber_mean(const double* y,const double* wt,double* beta,int S){
    int r=threadIdx.x;if(r>=32)return;double num=0,den=1e-9;const double* yy=y+(long long)r*S;const double* ww=wt+(long long)r*S;
    for(int j=0;j<S;++j){num+=ww[j]*yy[j];den+=ww[j];}beta[r]=num/den;
}
static Buf<double> huber_translations_device(ScratchPool& pool,cudaStream_t stream,std::vector<Buf<float>>& v,const std::vector<int>& sel,std::array<std::array<double,2>,16>& g){
    const int S=int(sel.size());VecPtrs vp;for(int k=0;k<16;++k)vp.p[k]=v[k].p;
    Buf<int> dsel(pool,S);ck(cudaMemcpyAsync(dsel.p,sel.data(),sizeof(int)*S,cudaMemcpyHostToDevice,stream));
    Buf<double> y(pool,32ull*S),res(pool,32ull*S),dev(pool,32ull*S),wt(pool,32ull*S),beta(pool,32),med(pool,32),mad(pool,32);
    const int nb=int((32LL*S+255)/256);
    huber_gather<<<nb,256,0,stream>>>(vp,dsel.p,S,y.p);
    ck((cudaError_t)dl_t2_row_median(y.p,beta.p,32,S,S,1,1,stream));
    auto scale=[&](){
        huber_res<<<nb,256,0,stream>>>(y.p,beta.p,res.p,S);
        ck((cudaError_t)dl_t2_row_median(res.p,med.p,32,S,S,1,1,stream));
        huber_dev<<<nb,256,0,stream>>>(res.p,med.p,dev.p,S);
        ck((cudaError_t)dl_t2_row_median(dev.p,mad.p,32,S,S,1,1,stream));
        huber_wt<<<nb,256,0,stream>>>(res.p,mad.p,wt.p,S);
    };
    scale();
    for(int it=0;it<21;++it){huber_mean<<<1,32,0,stream>>>(y.p,wt.p,beta.p,S);scale();}
    auto b=beta.get(32,stream);for(int k=0;k<16;++k)for(int ax=0;ax<2;++ax)g[k][ax]=b[2*k+ax];
    return beta;
}

// ---- fix 2/3: device preparation, candidate conversion and selection (box mode) ----
struct IntList16 { int k[16]; };
__global__ void box_s2(VecPtrs vp,IntList16 ks,const double* g,int N,double K,double sum2,double sum4,double det,double* s2){
    int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=N)return;double rss=0;
    for(int ax=0;ax<2;++ax){
        double b0=0,b1=0,b2=0;
        for(int i=0;i<16;++i){double z=double(vp.p[i][2*j+ax])-g[2*i+ax];b0+=z;b1+=ks.k[i]*z;b2+=ks.k[i]*ks.k[i]*z;}
        double c=(sum4*b0-sum2*b2)/det,u=b1/sum2,a=(K*b2-sum2*b0)/det;
        for(int i=0;i<16;++i){double z=double(vp.p[i][2*j+ax])-g[2*i+ax]-c-u*ks.k[i]-a*ks.k[i]*ks.k[i];rss+=z*z;}
    }
    s2[j]=rss/26;
}
__global__ void box_mark(const double* tex,const double* s2,const unsigned char* sel,const unsigned char* inner,int N,double* ta,double* va,double* fa,double* fb){
    int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=N)return;const double nan=__longlong_as_double(0x7ff8000000000000LL);
    ta[j]=sel[j]?tex[j]:nan;va[j]=(sel[j]&&tex[j]>0)?s2[j]*tex[j]:nan;fa[j]=inner[j]?s2[j]:nan;fb[j]=inner[j]?fmax(1.,tex[j]):nan;
}
__global__ void box_prep(VecPtrs vp,IntList16 targets,int B,const double* g,const double* tex,const double* s2,const unsigned char* inner,int N,double ac,double floorv,double* dy,double* dw,double* ds,double* sharedW,int* flag){
    int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=N)return;
    double s2c=fmax(1e-6,(26*fmax(1e-6,s2[j])+8*ac/fmax(floorv,tex[j]))/34),sig=sqrt(s2c),wgt=inner[j]?1/s2c:0.;
    sharedW[j]=wgt;int bad=0;
    bad|=!(isfinite(wgt)&&(wgt==0||(wgt>=1e-12&&wgt<=1e6)));bad|=!(isfinite(sig)&&sig>=1e-3&&sig<=1e6);
    for(int b=0;b<B;++b){
        int k=targets.k[b];dw[b*(long long)N+j]=wgt;ds[b*(long long)N+j]=sig;
        for(int ax=0;ax<2;++ax){double y=double(vp.p[k][2*j+ax])-g[2*k+ax];dy[(b*2+ax)*(long long)N+j]=y;bad|=!(isfinite(y)&&fabs(y)<=1e6);}
    }
    if(bad)atomicOr(flag,1);
}
__global__ void box_m(const double* med,int B,double twoLog2,double* m,int* flag){
    int b=threadIdx.x;if(b>=B)return;double md=isfinite(med[b])?med[b]:1.;double v=fmax(.05,md/twoLog2);m[b]=v;
    if(!(isfinite(v)&&v>=.05&&v<=1e20))atomicOr(flag,1);
}
__global__ void box_inv(const double* ds,const double* m,int B,int N,double* inv){
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=(long long)B*N)return;int b=int(i/N),j=int(i%N);
    double scale=(4.685*ds[b*(long long)N+j])*sqrt(m[b]);inv[i]=1/(scale*scale);
}
__global__ void box_cand(const double* out,const double* g,IntList16 targets,int B,int N,float* cand){
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=(long long)B*N)return;int b=int(i/N),j=int(i%N),k=targets.k[b];
    for(int ax=0;ax<2;++ax)cand[(long long)b*2*N+2*j+ax]=float(out[(b*2+ax)*(long long)N+j]+g[2*k+ax]);
}
// E layout: ((b*7)+c)*N+j; candidate c-1 of pair b: cand[c-1] + b*2N. In-place update of v[k] after all energies exist.
struct CandPtrs { const float* p[6]; };
__global__ void box_select(CandPtrs cp,VecPtrs vp,IntList16 targets,int B,int N,const float* E,const unsigned char* inner,double thr){
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=(long long)B*N)return;int b=int(i/N),j=int(i%N),k=targets.k[b];
    float* vv=const_cast<float*>(vp.p[k]);
    float c0x=cp.p[0][(long long)b*2*N+2*j],c0y=cp.p[0][(long long)b*2*N+2*j+1];
    if(!inner[j]){vv[2*j]=c0x;vv[2*j+1]=c0y;return;}
    const float* e=E+(long long)b*7*N+j;float best=e[0];for(int c=1;c<7;++c)best=fminf(best,e[(long long)c*N]);
    int choice=0;for(int c=1;c<7;++c)if((double(e[(long long)c*N])-double(best))*255<=thr){choice=c;break;}
    if(choice){vv[2*j]=cp.p[choice-1][(long long)b*2*N+2*j];vv[2*j+1]=cp.p[choice-1][(long long)b*2*N+2*j+1];}
}
