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
static void huber_translations_device(ScratchPool& pool,cudaStream_t stream,std::vector<Buf<float>>& v,const std::vector<int>& sel,std::array<std::array<double,2>,16>& g){
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
}
