// Box-mode fit with device-resident preparation and candidate selection (perf/mcwiener-box-device-fit).
// Returns false (no state changed except scratch) when the FP32 domain check fails, so the legacy path can take the FP64 route.
// level 2: device prep, host candidate selection; level >=3: device candidate selection; level >=4: reserved for the new score kernel.
static void fit_hash_report(std::vector<Buf<float>>& v,const std::vector<int>& targets,int N,int64_t index,cudaStream_t stream){
    static const bool on=std::getenv("T2_NATIVE_FITHASH")!=nullptr;if(!on)return;
    for(int k:targets){auto x=v[k].get(2*N,stream);unsigned long long hsh=1469598103934665603ULL;for(float f:x){unsigned u;memcpy(&u,&f,4);hsh=(hsh^u)*1099511628211ULL;}fprintf(stderr,"T2FITHASH frame=%lld k=%d %016llx\n",(long long)index,k,hsh);}
}
bool Engine::Impl::fit_box_device(std::vector<Buf<float>>& v,const std::vector<int>& ks,Source& ref,int oy,int ox,int level){
    int nx=(w+16+ox)/8+1,ny=(h+16+oy)/8+1,N=nx*ny;
    std::vector<unsigned char> innerMask(N,0),selMask(N,0);std::vector<int> inner;
    for(int j=0;j<N;++j){int x=j%nx*8-8-ox,y=j/nx*8-8-oy;if(x>=32&&w-x>=32&&y>=32&&h-y>=32){inner.push_back(j);innerMask[j]=1;}}
    if(inner.empty())return false;
    Buf<float> var=vars(ref.image[0],0,oy,ox,false),lvl(scratchPool,N);Buf<float2> centered(scratchPool,size_t(N)*256),spectra(scratchPool,size_t(N)*256);Buf<double> texture(scratchPool,N);
    center_tiles<<<N,256,0,stream>>>(ref.image[0].b.p,win.p,centered.p,lvl.p,h,w,oy,ox,nx);t2_fft16_kernel<<<N,256,0,stream>>>(centered.p,spectra.p,N,0);texture_power<<<blocks(N),256,0,stream>>>(spectra.p,var.p,texture.p,N);
    auto tex=texture.get(N,stream);
    std::vector<double> vals;for(int j:inner)vals.push_back(tex[j]);double tm=median(vals);std::vector<int> sel;for(int j:inner)if(tex[j]>=tm){sel.push_back(j);selMask[j]=1;}
    std::array<std::array<double,2>,16> g{};Buf<double> dg=huber_translations_device(scratchPool,stream,v,sel,g);
    VecPtrs vp;for(int k=0;k<16;++k)vp.p[k]=v[k].p;IntList16 kl;for(int k=0;k<16;++k)kl.k[k]=ks[k];
    double K=16,sum2=0,sum4=0;for(int k:ks){sum2+=k*k;sum4+=double(k)*k*k*k;}double det=K*sum4-sum2*sum2;
    Buf<unsigned char> dInner(scratchPool,N),dSel(scratchPool,N);ck(cudaMemcpyAsync(dInner.p,innerMask.data(),N,cudaMemcpyHostToDevice,stream));ck(cudaMemcpyAsync(dSel.p,selMask.data(),N,cudaMemcpyHostToDevice,stream));
    Buf<double> s2(scratchPool,N),ta(scratchPool,N),va(scratchPool,N),fa(scratchPool,N),fb(scratchPool,N),meds(scratchPool,4);
    box_s2<<<blocks(N),256,0,stream>>>(vp,kl,dg.p,N,K,sum2,sum4,det,s2.p);
    box_mark<<<blocks(N),256,0,stream>>>(texture.p,s2.p,dSel.p,dInner.p,N,ta.p,va.p,fa.p,fb.p);
    double* arrs[4]={va.p,fa.p,fb.p,ta.p};for(int a=0;a<4;++a)ck((cudaError_t)dl_t2_row_median(arrs[a],meds.p+a,1,N,N,1,1,stream));
    auto mv=meds.get(4,stream);double ac=std::isnan(mv[0])?mv[1]*mv[2]:mv[0],floorv=.05*std::max(1e-9,mv[3]);
    std::vector<int> targets;for(int i=0;i<16;++i)if(std::abs(ks[i])<=2)targets.push_back(i);int B=int(targets.size());IntList16 tl{};for(int b=0;b<B;++b)tl.k[b]=targets[b];
    Buf<double> dy(scratchPool,size_t(B)*2*N),dw(scratchPool,size_t(B)*N),ds(scratchPool,size_t(B)*N),sharedW(scratchPool,N),dz(scratchPool,size_t(B)*N),mm(scratchPool,B),mmed(scratchPool,B),dinv(scratchPool,size_t(B)*N);Buf<int> flag(scratchPool,1);
    ck(cudaMemsetAsync(flag.p,0,sizeof(int),stream));
    box_prep<<<blocks(N),256,0,stream>>>(vp,tl,B,dg.p,texture.p,s2.p,dInner.p,N,ac,floorv,dy.p,dw.p,ds.p,sharedW.p,flag.p);
    ck((cudaError_t)dl_t2_scale_z2(dy.p,dw.p,ds.p,dz.p,B,ny,nx,stream));
    ck((cudaError_t)dl_t2_row_median(dz.p,mmed.p,B,N,N,1,1,stream));
    box_m<<<1,32,0,stream>>>(mmed.p,B,2*std::log(2.),mm.p,flag.p);
    box_inv<<<blocks(B*N),256,0,stream>>>(ds.p,mm.p,B,N,dinv.p);
    if(flag.get(1,stream)[0])return false;
    const int radii[]={24,12,6,3,2,1};std::vector<Buf<float>> cand(6);
    for(int c=0;c<6;++c){
        Buf<double> out(scratchPool,size_t(B)*2*N);Buf<float> scratch(scratchPool,size_t(8)*B*N);Buf<unsigned char> good(scratchPool,size_t(B)*N);
        ck((cudaError_t)dl_t2_boxfit_f32(dy.p,sharedW.p,dinv.p,out.p,scratch.p,good.p,B,ny,nx,radii[c],6,stream));
        cand[c].size(scratchPool,size_t(B)*2*N);box_cand<<<blocks(B*N),256,0,stream>>>(out.p,dg.p,tl,B,N,cand[c].p);
    }
    if(level<3){
        for(int b=0;b<B;++b){int k=targets[b];std::array<std::vector<float>,7> energies;Source& nb=ring[(ref.index+ks[k])%17];
            for(int c=0;c<7;++c){Buf<float> e(scratchPool,N);score<<<N,256,0,stream>>>(ref.image[0].b.p,nb.image[0].b.p,c==0?v[k].p:cand[c-1].p+size_t(b)*2*N,win.p,var.p,e.p,h,w,oy,ox,nx);energies[c]=e.get(N,stream);}
            auto V0=v[k].get(2*N,stream);std::array<std::vector<float>,6> hc;for(int c=0;c<6;++c){auto all=cand[c].get(size_t(B)*2*N,stream);hc[c].assign(all.begin()+size_t(b)*2*N,all.begin()+size_t(b+1)*2*N);}
            auto chosen=hc[0];for(int j:inner){float best=energies[0][j];for(int c=1;c<7;++c)best=std::min(best,energies[c][j]);int choice=0;for(int c=1;c<7;++c)if((double(energies[c][j])-double(best))*255<=-2*std::log(.001)){choice=c;break;}
                if(choice==0){chosen[2*j]=V0[2*j];chosen[2*j+1]=V0[2*j+1];}else{chosen[2*j]=hc[choice-1][2*j];chosen[2*j+1]=hc[choice-1][2*j+1];}}
            v[k].put(scratchPool,chosen,stream);++refined;}
        fit_hash_report(v,targets,N,ref.index,stream);return true;
    }
    Buf<float> E(scratchPool,size_t(B)*7*N);
    CandPtrs cp;for(int c=0;c<6;++c)cp.p[c]=cand[c].p;
    if(level>=4){
        PairPtrs pp{};for(int b=0;b<B;++b){int k=targets[b];pp.nb[b]=ring[(ref.index+ks[k])%17].image[0].b.p;pp.v0[b]=v[k].p;}
        score_multi<<<N,256,0,stream>>>(ref.image[0].b.p,pp,cp,B,win.p,var.p,E.p,h,w,oy,ox,nx,N);
    }else
    for(int b=0;b<B;++b){int k=targets[b];Source& nb=ring[(ref.index+ks[k])%17];
        for(int c=0;c<7;++c)score<<<N,256,0,stream>>>(ref.image[0].b.p,nb.image[0].b.p,c==0?v[k].p:cand[c-1].p+size_t(b)*2*N,win.p,var.p,E.p+(size_t(b)*7+c)*N,h,w,oy,ox,nx);}
    const double thr=-2*std::log(.001);
    static const bool scoreCheck=std::getenv("T2_NATIVE_FITCHECK")!=nullptr;
    if(level>=4&&scoreCheck){ // compare choices of the new score against the legacy score on the same inputs, apply the new ones
        Buf<float> E0(scratchPool,size_t(B)*7*N);Buf<unsigned char> chA(scratchPool,size_t(B)*N),chB(scratchPool,size_t(B)*N);
        for(int b=0;b<B;++b){int k=targets[b];Source& nb=ring[(ref.index+ks[k])%17];
            for(int c=0;c<7;++c)score<<<N,256,0,stream>>>(ref.image[0].b.p,nb.image[0].b.p,c==0?v[k].p:cand[c-1].p+size_t(b)*2*N,win.p,var.p,E0.p+(size_t(b)*7+c)*N,h,w,oy,ox,nx);}
        box_select<<<blocks(B*N),256,0,stream>>>(cp,vp,tl,B,N,E0.p,dInner.p,thr,chA.p,0);
        box_select<<<blocks(B*N),256,0,stream>>>(cp,vp,tl,B,N,E.p,dInner.p,thr,chB.p,0);
        auto a=chA.get(size_t(B)*N,stream),c2=chB.get(size_t(B)*N,stream);size_t mism=0,tot=0;for(int b=0;b<B;++b)for(int j:inner){++tot;mism+=a[size_t(b)*N+j]!=c2[size_t(b)*N+j];}
        auto eA=E0.get(size_t(B)*7*N,stream),eB=E.get(size_t(B)*7*N,stream);double maxrel=0;for(size_t i=0;i<eA.size();++i)maxrel=std::max(maxrel,double(std::abs(eA[i]-eB[i]))/std::max(1e-6,double(std::abs(eA[i]))));
        fprintf(stderr,"T2SCORECHECK frame=%lld tiles=%zu choice_mismatch=%zu energy_max_rel_diff=%.3e\n",(long long)ref.index,tot,mism,maxrel);
    }
    box_select<<<blocks(B*N),256,0,stream>>>(cp,vp,tl,B,N,E.p,dInner.p,thr,nullptr,1);
    refined+=B;ck(cudaGetLastError());fit_hash_report(v,targets,N,ref.index,stream);return true;
}
