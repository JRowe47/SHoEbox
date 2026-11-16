// Appended to the instrumented translation unit by adapter.py.
namespace {
thread_local std::string task_error;
template<class F> int task_guard(F f){try{task_error.clear();f();return 0;}catch(const std::exception& e){task_error=e.what();return -1;}}
}
extern "C" {
const char* st_error(){return task_error.c_str();}
void* st_new(int parameters,uint32_t seed,int mode,uint64_t budget){try{task_error.clear();return new shoe::TaskAdapter(parameters,seed,mode,budget);}catch(const std::exception& e){task_error=e.what();return nullptr;}}
void st_free(void* ptr){delete static_cast<shoe::TaskAdapter*>(ptr);}
int st_run(void* ptr,const uint8_t* x,const uint8_t* y,size_t n,double* loss){return task_guard([&]{auto& a=*static_cast<shoe::TaskAdapter*>(ptr);shoe::TaskScope scope(a);*loss=a.model->run(x,y,n);});}
int st_supervise(void* ptr,const int* labels,int intermediate,double* loss){return task_guard([&]{auto& a=*static_cast<shoe::TaskAdapter*>(ptr);*loss=a.supervise(labels,intermediate);});}
int st_read(void* ptr,float* features,float* probs,const float* override_features){return task_guard([&]{auto& a=*static_cast<shoe::TaskAdapter*>(ptr);auto x=a.features();std::copy(x.begin(),x.end(),features);if(override_features)std::copy(override_features,override_features+128,x.begin());auto p=a.probabilities(x);std::copy(p.begin(),p.end(),probs);});}
int st_reset(void* ptr){return task_guard([&]{auto& a=*static_cast<shoe::TaskAdapter*>(ptr);shoe::TaskScope scope(a);a.model->reset();a.tape={};});}
int st_finish(void* ptr){return task_guard([&]{auto& a=*static_cast<shoe::TaskAdapter*>(ptr);shoe::TaskScope scope(a);a.model->finish();});}
void* st_clone(void* ptr,int frozen){try{task_error.clear();auto a=std::make_unique<shoe::TaskAdapter>(*static_cast<shoe::TaskAdapter*>(ptr));if(frozen){a->model->inference(true);a->tape={};}return a.release();}catch(const std::exception& e){task_error=e.what();return nullptr;}}
int st_info(void* ptr,double* out){return task_guard([&]{auto& a=*static_cast<shoe::TaskAdapter*>(ptr);auto& m=*a.model;out[0]=m.parameter_count()+sizeof(a.head)/sizeof(float);out[1]=m.route_learning.targets;out[2]=m.bytes;out[3]=a.labels[0];out[4]=a.labels[1];out[5]=m.training;for(int l=0;l<4;++l){out[6+l]=m.levels[l].s.executions;out[10+l]=m.levels[l].s.addressed.clock.ticks;}out[14]=sizeof(a.tape);});}
int st_permute(void* ptr,int scale,uint32_t seed,int* count){return task_guard([&]{auto& a=*static_cast<shoe::TaskAdapter*>(ptr);if(a.model->training||scale<2||scale>3)throw std::logic_error("permutation requires frozen L2/L3");auto& l=a.model->levels[scale];std::vector<int> ids;for(size_t k=0;k<l.experts.size();++k)if(l.routing.selections[k])ids.push_back(k);*count=ids.size();if(ids.size()<2)return;
    std::mt19937 rng(seed);std::shuffle(ids.begin(),ids.end(),rng);auto saved=l.experts;for(size_t j=0;j<ids.size();++j){auto& dst=l.experts[ids[j]];auto& src=saved[ids[(j+1)%ids.size()]];dst.w=src.w;dst.bias=src.bias;}});}
int st_native_probs(void* ptr,float* out){return task_guard([&]{static_cast<shoe::TaskAdapter*>(ptr)->model->log_probs(out);});}
int st_save(void* ptr,char* buffer,size_t capacity,size_t* size){return task_guard([&]{auto raw=shoe::task_save(*static_cast<shoe::TaskAdapter*>(ptr));*size=raw.size();if(buffer){if(capacity<raw.size())throw std::runtime_error("checkpoint buffer too small");std::memcpy(buffer,raw.data(),raw.size());}});}
void* st_load(const char* raw,size_t size){try{task_error.clear();return shoe::task_load(raw,size).release();}catch(const std::exception& e){task_error=e.what();return nullptr;}}
}
