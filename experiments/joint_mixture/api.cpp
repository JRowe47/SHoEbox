extern "C" {
void* qr_new(int parameters,uint32_t seed,int minimal,int kind,int joint){
    try{error.clear();return new shoe::QueryAdapter(parameters,seed,minimal,kind,joint);}
    catch(const std::exception& e){error=e.what();return nullptr;}
}
void qr_free(void* p){delete static_cast<shoe::QueryAdapter*>(p);}
int qr_query_policy(void* p,int latest){return guard([&]{
    auto& a=*static_cast<shoe::QueryAdapter*>(p);
    if(a.training||(a.kind!=1&&a.kind!=2)||latest<0||latest>1)throw std::invalid_argument("frozen query policy required");
    a.kind=latest?2:1;
});}
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
int qr_train(void* p,const uint8_t* x,const uint8_t* y,const uint8_t* mask,const int32_t* labels,size_t n,double* loss,size_t* targets){return guard([&]{
    auto& a=*static_cast<shoe::QueryAdapter*>(p);if(!a.training)throw std::logic_error("frozen query adapter");
    if(labels)for(size_t i=0;i<n*5;++i)if(labels[i]<-1||labels[i]>(i%5==0?255:65535))throw std::invalid_argument("invalid prefix label");
    shoe::QueryScope scope(a);*loss=0;*targets=0;
    for(size_t t=0;t<n;++t){a.model->observe(x[t]);if(labels){shoe::Labels lab;std::copy_n(labels+5*t,5,lab.begin());a.local(lab);}if(mask[t]){*loss+=a.model->teach(y[t]);++*targets;}}
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
int qr_info(void* p,double* out){return guard([&]{auto& a=*static_cast<shoe::QueryAdapter*>(p);auto& m=*a.model;
    out[0]=m.parameter_count();out[1]=(a.kind?shoe::QPARAMS:0)+(a.kind==3?shoe::D+1:0);out[2]=m.bytes;out[3]=a.targets;
    out[4]=a.training;out[5]=a.joint;out[6]=a.kind;out[7]=m.minimal_core;out[8]=sizeof(a.cache)+sizeof(a.errors)+sizeof(a.gate_raw)+sizeof(a.gate_value)+sizeof(a.responsibility)+sizeof(a.mixture_loss);
    for(int l=0;l<4;++l)out[9+l]=m.levels[l].experts.size();
    out[13]=sizeof(shoe::QueryHead)+sizeof(a.applicability);out[14]=sizeof(shoe::QueryAdapter);out[15]=0;out[16]=a.local_updates;out[17]=0;out[18]=0;out[19]=sizeof(shoe::Stream)*4+sizeof(shoe::RecallStream)*4;out[20]=m.levels[1].retained.count*4;out[21]=a.gate_targets;out[22]=a.projection_steps;out[23]=a.presence_steps;
});}
int qr_diagnostic(void* p,float* out){return guard([&]{
    auto& a=*static_cast<shoe::QueryAdapter*>(p);a.prepare();out[0]=a.cache.attention;
    for(int lane=0;lane<3;++lane){const auto& z=lane==0?a.cache.query:a.cache.keys[lane-1];float best=1e30f;int chosen=0;
        for(int b=0;b<256;++b){float d=0;for(int r=0;r<shoe::QRANK;++r){float x=z[r]-shoe::byte_code(b,r);d+=x*x;}if(d<best){best=d;chosen=b;}}
        out[lane+1]=chosen;}
    for(int k=0;k<2;++k){int value=0;for(int r=0;r<shoe::VRANK;++r)if(a.cache.values[k][r]>0)value|=1<<r;out[k+4]=value;}
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

int qr_gate(void* p,float* out){return guard([&]{auto& a=*static_cast<shoe::QueryAdapter*>(p);a.prepare();*out=a.gate_value;});}
int qr_stream_score(void* p,const uint8_t* data,size_t n,double* out){return guard([&]{
 auto& a=*static_cast<shoe::QueryAdapter*>(p);if(a.training||n<2)throw std::invalid_argument("frozen stream required");
 shoe::QueryScope scope(a);*out=0;for(size_t i=0;i+1<n;++i){a.model->observe(data[i]);*out+=a.model->teach(data[i+1]);}*out/=n-1;
});}
int qr_features(void* p,float* out){return guard([&]{
 auto& m=*static_cast<shoe::QueryAdapter*>(p)->model;
 if(!m.bytes||m.bytes%64)throw std::logic_error("completed L3 required");
 std::copy(m.levels[3].s.state.begin(),m.levels[3].s.state.end(),out);
});}
int qr_credit(void* p,const float* dx){return guard([&]{
 auto& m=*static_cast<shoe::QueryAdapter*>(p)->model;
 if(!m.training||!m.bytes||m.bytes%64||!m.levels[3].s.count)throw std::logic_error("live L3 required");
 auto& s=m.levels[3].s;int uses=0;for(int t=0;t<s.count;++t)uses+=s.events[t].uses;
 for(int i=0;i<shoe::D;++i)if(!std::isfinite(dx[i]))throw std::invalid_argument("nonfinite credit");
 for(int i=0;i<shoe::D;++i)s.events[s.count-1].task[i]+=std::max(1,uses)*dx[i];m.finish();
});}
int qr_core_save(void* p,char* out,size_t capacity,size_t* size){return guard([&]{
 auto raw=shoe::save(*static_cast<shoe::QueryAdapter*>(p)->model);*size=raw.size();
 if(out){if(capacity<raw.size())throw std::invalid_argument("small buffer");std::memcpy(out,raw.data(),raw.size());}
});}
}
