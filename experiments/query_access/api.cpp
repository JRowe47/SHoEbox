extern "C" {
void* qr_new(int parameters,uint32_t seed,int minimal,int kind,int joint){
    try{error.clear();return new shoe::QueryAdapter(parameters,seed,minimal,kind,joint);}
    catch(const std::exception& e){error=e.what();return nullptr;}
}
void qr_free(void* p){delete static_cast<shoe::QueryAdapter*>(p);}
int qr_configure(void* p,float rate,float scale,float auxiliary){return guard([&]{
    auto& a=*static_cast<shoe::QueryAdapter*>(p);
    if(!std::isfinite(rate)||rate<0||!std::isfinite(scale)||scale<=0||!std::isfinite(auxiliary)||auxiliary<0)
        throw std::invalid_argument("invalid rate");
    a.rate=rate;auto& m=*a.model;m.learning_rate=.008f*scale;m.slow_rate=.08f*scale;
    m.query_rate=.008f*scale;m.head_rate=.16f*scale;m.route_rate=.002f*scale;m.aux_weight=auxiliary;
});}
int qr_auxiliary(void* p,float value){return guard([&]{auto& a=*static_cast<shoe::QueryAdapter*>(p);
    if(!a.training||!std::isfinite(value)||value<0)throw std::invalid_argument("invalid auxiliary");a.auxiliary=value;});}
int qr_reset(void* p){return guard([&]{auto& a=*static_cast<shoe::QueryAdapter*>(p);a.model->reset(true);});}
int qr_finish(void* p){return guard([&]{static_cast<shoe::QueryAdapter*>(p)->model->finish();});}
int qr_observe(void* p,uint8_t b){return guard([&]{static_cast<shoe::QueryAdapter*>(p)->model->observe(b);});}
int qr_probs(void* p,float* out){return guard([&]{auto& a=*static_cast<shoe::QueryAdapter*>(p);shoe::QueryScope scope(a);a.model->log_probs(out);});}
int qr_train(void* p,const uint8_t* x,const uint8_t* y,const uint8_t* mask,const int16_t* labels,size_t n,double* loss,size_t* targets){return guard([&]{
    auto& a=*static_cast<shoe::QueryAdapter*>(p);if(!a.training)throw std::logic_error("frozen query adapter");
    if(labels)for(size_t i=0;i<n*5;++i)if(labels[i]<-1||labels[i]>255)throw std::invalid_argument("invalid prefix label");
    shoe::QueryScope scope(a);*loss=0;*targets=0;
    for(size_t t=0;t<n;++t){a.model->observe(x[t]);if(mask[t]){a.labels={{-1,-1,-1,-1,-1}};if(labels)std::copy_n(labels+5*t,5,a.labels.begin());*loss+=a.model->teach(y[t]);++*targets;}}
});}
int qr_score(void* p,const uint8_t* data,const uint8_t* mask,size_t n,int low,int classes,int rollout,double* out){return guard([&]{
    auto& a=*static_cast<shoe::QueryAdapter*>(p);shoe::QueryScope scope(a);auto& m=*a.model;
    if(a.training||n<2||low<0||classes<2||low+classes>256||rollout<0||rollout>2)throw std::invalid_argument("invalid score");
    for(size_t i=1;i<n;++i)if(mask[i]&&(data[i]<low||data[i]>=low+classes))throw std::invalid_argument("target outside alphabet");
    std::fill_n(out,9,0.);out[4]=out[5]=1;int answer_index=0;m.observe(data[0]);
    for(size_t t=1;t<n;++t){int observed=data[t];
        if(mask[t]){float lp[256];m.log_probs(lp);
            int full=int(std::max_element(lp,lp+256)-lp),restricted=int(std::max_element(lp+low,lp+low+classes)-lp);
            float mx=lp[restricted];double mass=0;for(int k=low;k<low+classes;++k)mass+=std::exp(double(lp[k]-mx));
            ++out[0];out[1]-=lp[data[t]];out[2]-=lp[data[t]]-mx-std::log(mass);
            out[3]+=restricted==data[t];out[4]*=restricted==data[t];out[5]*=full==data[t];out[6]+=full==data[t];if(answer_index<2)out[7+answer_index]+=full==data[t];++answer_index;
            if(rollout)observed=rollout==1?restricted:full;
        }if(t+1<n)m.observe(uint8_t(observed));
    }
});}
void* qr_clone(void* p,int frozen,int lesion){
    try{error.clear();auto a=std::make_unique<shoe::QueryAdapter>(*static_cast<shoe::QueryAdapter*>(p));
        if(lesion<0||lesion>8||(!frozen&&lesion))throw std::invalid_argument("frozen lesion required");
        if(frozen){a->training=false;a->model->inference(false);}a->lesion=lesion;return a.release();
    }catch(const std::exception& e){error=e.what();return nullptr;}
}
int qr_adopt(void* p,void* source){return guard([&]{
    auto& a=*static_cast<shoe::QueryAdapter*>(p);const auto& b=*static_cast<shoe::QueryAdapter*>(source);
    if(!a.training||a.targets||!b.training||!b.joint||a.model->minimal_core!=b.model->minimal_core||
       a.model->parameter_count()!=b.model->parameter_count())throw std::invalid_argument("incompatible source core");
    a.model=std::make_unique<shoe::Model>(*b.model);if(!a.joint)a.model->inference(false);
});}
int qr_features(void* p,float* out){return guard([&]{
    auto& a=*static_cast<shoe::QueryAdapter*>(p);a.prepare();
    for(const auto& raw:a.cache.raw){std::copy(raw.begin(),raw.end(),out);out+=shoe::D;}
    for(const auto& z:a.cache.z){std::copy(z.begin(),z.end(),out);out+=shoe::QRANK;}
    *out=a.cache.attention;
});}
int qr_info(void* p,double* out){return guard([&]{auto& a=*static_cast<shoe::QueryAdapter*>(p);auto& m=*a.model;
    out[0]=m.parameter_count();out[1]=a.kind?shoe::QPARAMS:0;out[2]=m.bytes;out[3]=a.targets;
    out[4]=a.training;out[5]=a.joint;out[6]=a.kind;out[7]=m.minimal_core;out[8]=sizeof(a.cache)+sizeof(a.errors);
    for(int l=0;l<4;++l)out[9+l]=m.levels[l].experts.size();
    out[13]=sizeof(shoe::QueryHead);out[14]=sizeof(shoe::QueryAdapter);out[15]=sizeof(a.norm);out[16]=a.norm[0].count;out[17]=0;out[18]=0;out[19]=sizeof(shoe::Stream)*4+sizeof(shoe::RecallStream)*4;out[20]=a.cache.sources.count;
});}
int qr_diagnostic(void* p,float* out){return guard([&]{
    // Inspection only: never called by the prediction or training byte loop.
    auto& a=*static_cast<shoe::QueryAdapter*>(p);a.prepare();out[0]=a.cache.attention;
    for(int lane=0;lane<3;++lane){float best=1e30f;int chosen=0;
        for(int b=0;b<256;++b){float d=0;for(int r=0;r<shoe::QRANK;++r){float x=a.cache.z[lane][r]-shoe::byte_code(b,r);d+=x*x;}
            if(d<best){best=d;chosen=b;}}out[lane+1]=chosen;}
    for(int lane=3;lane<5;++lane){std::array<float,511> lp{};
        for(int n=0;n<255;++n){float z=a.head.bias[n]+shoe::dot(a.head.output.data()+n*shoe::QRANK,a.cache.z[lane].data(),shoe::QRANK);
            lp[2*n+1]=lp[n]-shoe::softplus(z);lp[2*n+2]=lp[n]-shoe::softplus(-z);}
        out[lane+1]=std::max_element(lp.begin()+255,lp.end())-(lp.begin()+255);}
});}
int qr_benchmark(void* p,const uint8_t* data,size_t size,size_t count,double* checksum){return guard([&]{
    auto& a=*static_cast<shoe::QueryAdapter*>(p);if(a.training||!size)throw std::invalid_argument("frozen nonempty benchmark required");
    shoe::QueryScope scope(a);*checksum=0;float lp[256];
    for(size_t t=0;t<count;++t){a.model->observe(data[t%size]);a.model->log_probs(lp);*checksum+=lp[data[(t+1)%size]];}
});}
int qr_save(void* p,char* out,size_t capacity,size_t* size){return guard([&]{
    auto raw=shoe::query_save(*static_cast<shoe::QueryAdapter*>(p));*size=raw.size();
    if(out){if(capacity<raw.size())throw std::invalid_argument("small buffer");std::memcpy(out,raw.data(),raw.size());}
});}
void* qr_load(const char* raw,size_t n){try{error.clear();return shoe::query_load(raw,n).release();}catch(const std::exception& e){error=e.what();return nullptr;}}
}
