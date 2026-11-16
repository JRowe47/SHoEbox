// Public lifecycle bridge. The E8 numeric implementation remains readout.hpp.
extern "C" {
void* pe_core(void* p){return static_cast<shoe::QueryAdapter*>(p)->model.get();}
void* pe_load(const char* raw,size_t n){try{
    error.clear();auto a=shoe::query_load(raw,n);
    if(a->kind!=4||!a->joint||!a->model->minimal_core)
        throw std::invalid_argument("public E8 requires the minimal joint local policy");
    return a.release();
}catch(const std::exception& e){error=e.what();return nullptr;}}
void* pe_clone(void* p,int frozen){try{
    error.clear();auto a=std::make_unique<shoe::QueryAdapter>(*static_cast<shoe::QueryAdapter*>(p));
    if(frozen){a->training=false;a->model->inference(false);}
    return a.release();
}catch(const std::exception& e){error=e.what();return nullptr;}}
int pe_observe(void* p,uint8_t byte,float* out){return guard([&]{
    auto& a=*static_cast<shoe::QueryAdapter*>(p);
    if(a.training)throw std::logic_error("observe requires inference; use run for training");
    shoe::QueryScope scope(a);a.model->observe(byte);a.model->log_probs(out);
});}
int pe_run(void* p,const uint8_t* x,const uint8_t* y,const uint8_t* mask,
           const int32_t* labels,size_t n,double* loss,size_t* targets){return guard([&]{
    auto& a=*static_cast<shoe::QueryAdapter*>(p);
    if(!n)throw std::invalid_argument("nonempty run required");
    if(labels&&!a.training)throw std::logic_error("local labels require training");
    if(labels)for(size_t i=0;i<n;++i)for(size_t j=0;j<6;++j)
        if(labels[6*i+j]<-1||labels[6*i+j]>(j==5?1:(j==0?255:65535)))
            throw std::invalid_argument("invalid prefix label");
    shoe::QueryScope scope(a);*loss=0;*targets=0;
    for(size_t t=0;t<n;++t){a.model->observe(x[t]);
        if(labels){shoe::Labels lab;std::copy_n(labels+6*t,6,lab.begin());a.local(lab);}
        if(!mask||mask[t]){*loss+=a.model->teach(y[t]);++*targets;}}
});}
}
