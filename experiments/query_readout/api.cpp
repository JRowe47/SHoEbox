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
int qr_reset(void* p){return guard([&]{auto& a=*static_cast<shoe::QueryAdapter*>(p);a.model->reset(true);});}
int qr_finish(void* p){return guard([&]{static_cast<shoe::QueryAdapter*>(p)->model->finish();});}
int qr_observe(void* p,uint8_t b){return guard([&]{static_cast<shoe::QueryAdapter*>(p)->model->observe(b);});}
int qr_probs(void* p,float* out){return guard([&]{auto& a=*static_cast<shoe::QueryAdapter*>(p);shoe::QueryScope scope(a);a.model->log_probs(out);});}
int qr_train(void* p,const uint8_t* x,const uint8_t* y,const uint8_t* mask,size_t n,double* loss,size_t* targets){return guard([&]{
    auto& a=*static_cast<shoe::QueryAdapter*>(p);if(!a.training)throw std::logic_error("frozen query adapter");
    shoe::QueryScope scope(a);*loss=0;*targets=0;
    for(size_t t=0;t<n;++t){a.model->observe(x[t]);if(mask[t]){*loss+=a.model->teach(y[t]);++*targets;}}
});}
int qr_score(void* p,const uint8_t* data,const uint8_t* mask,size_t n,int low,int classes,int rollout,double* out){return guard([&]{
    auto& a=*static_cast<shoe::QueryAdapter*>(p);shoe::QueryScope scope(a);auto& m=*a.model;
    if(a.training||n<2||low<0||classes<2||low+classes>256||rollout<0||rollout>2)throw std::invalid_argument("invalid score");
    for(size_t i=1;i<n;++i)if(mask[i]&&(data[i]<low||data[i]>=low+classes))throw std::invalid_argument("target outside alphabet");
    std::fill_n(out,7,0.);out[4]=out[5]=1;m.observe(data[0]);
    for(size_t t=1;t<n;++t){int observed=data[t];
        if(mask[t]){float lp[256];m.log_probs(lp);
            int full=int(std::max_element(lp,lp+256)-lp),restricted=int(std::max_element(lp+low,lp+low+classes)-lp);
            float mx=lp[restricted];double mass=0;for(int k=low;k<low+classes;++k)mass+=std::exp(double(lp[k]-mx));
            ++out[0];out[1]-=lp[data[t]];out[2]-=lp[data[t]]-mx-std::log(mass);
            out[3]+=restricted==data[t];out[4]*=restricted==data[t];out[5]*=full==data[t];out[6]+=full==data[t];
            if(rollout)observed=rollout==1?restricted:full;
        }if(t+1<n)m.observe(uint8_t(observed));
    }
});}
void* qr_clone(void* p,int frozen,int lesion){
    try{error.clear();auto a=std::make_unique<shoe::QueryAdapter>(*static_cast<shoe::QueryAdapter*>(p));
        if(lesion<0||lesion>3||(!frozen&&lesion))throw std::invalid_argument("frozen lesion required");
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
    // External inspection only. The model never calls a task parser.
    auto& a=*static_cast<shoe::QueryAdapter*>(p);a.prepare();
    std::copy(a.cache.q.begin(),a.cache.q.end(),out);out+=shoe::QINPUT;
    std::copy(a.cache.m.begin(),a.cache.m.end(),out);out+=shoe::MINPUT;
    for(int l=1;l<4;++l){auto& s=a.model->levels[l].s;
        // The persistent state is recomputed with the current trained query;
        // no write, clock advance, or synthetic observation is performed.
        shoe::AddressStream copy=s.addressed;shoe::AddressEvent e;
        shoe::address_forward(e,s.content,s.state,s.stable,copy,a.model->levels[l].address);
        std::copy(e.read.begin(),e.read.end(),out);out+=shoe::ADDRESS_V;
    }
});}
int qr_info(void* p,double* out){return guard([&]{auto& a=*static_cast<shoe::QueryAdapter*>(p);auto& m=*a.model;
    out[0]=m.parameter_count();out[1]=a.kind?shoe::QPARAMS:0;out[2]=m.bytes;out[3]=a.targets;
    out[4]=a.training;out[5]=a.joint;out[6]=a.kind;out[7]=m.minimal_core;out[8]=sizeof(a.cache)+sizeof(a.errors);
    for(int l=0;l<4;++l)out[9+l]=m.levels[l].experts.size();
    out[13]=sizeof(shoe::QueryHead);out[14]=sizeof(shoe::QueryAdapter);
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
