// Two reusable jobs own immutable factors and inputs. Workers never touch Model.
struct PipelineInput {
  PipelineInput() noexcept {} // arrays are fully overwritten; do not zero-fill
  SemanticRef ref;
  float values[D];
};
struct PipelineFactor {
  PipelineFactor() noexcept {} // arrays are fully overwritten; do not zero-fill
  uint64_t mask;
  uint32_t input;
  float factor[D];
};
struct PipelineScope {
  PipelineScope() noexcept {} // arrays are fully overwritten; do not zero-fill
  int next=-1;
  bool route=false;
  float bias[D],keys[D];
  std::vector<PipelineFactor> factors;
};
struct PipelineWeightTile {
  PipelineWeightTile() noexcept {} // arrays are fully overwritten; do not zero-fill
  uint64_t bank;
  float weights[D*D];
};
struct PipelineEntry {
  PipelineEntry() noexcept {} // arrays are fully overwritten; do not zero-fill
  uint64_t bank;
  int first=-1,last=-1,weight=-1;
  bool route=false,weighted=false;
  float bias[D],keys[D];
};
// v1 is read only, preserving pending results from stage-6 checkpoints.
struct PipelineTileV1 { uint64_t bank;float weights[D*D],bias[D],keys[D]; };
struct PipelineBatch {
  uint64_t read_version=0,base_version=0,sequence=0,generation=0;
  size_t scope_count=0;
  std::vector<PipelineScope> scopes; // high-water storage; only scope_count live
  std::vector<PipelineInput> inputs;
  std::vector<PipelineEntry> entries;
  std::vector<PipelineWeightTile> weights;
  std::array<int,N*E> bank_index;
  struct InputLocation {uint64_t generation=0,serial=0;uint32_t index=0;int region=-1;};
  std::vector<InputLocation> input_index;
  PipelineBatch(){bank_index.fill(-1);}
  void reset(){
    for(const auto&e:entries)bank_index[e.bank]=-1;
    entries.clear();weights.clear();inputs.clear();scope_count=0;++generation;
  }
};
inline void pipeline_compute(PipelineBatch&batch,bool fused){
  for(auto&entry:batch.entries){
    float*weights=entry.weight<0?nullptr:batch.weights[entry.weight].weights;
    // Each bank walks only its own scopes, in the original normalization order.
    for(int k=entry.first;k>=0;k=batch.scopes[k].next){
      const auto&scope=batch.scopes[k];
      if(!scope.factors.empty()&&fused){
        for(int i=0;i<D;i++){
          float row[D]{};
          for(const auto&f:scope.factors){
            const float*input=batch.inputs[f.input].values;
            if(f.mask==~uint64_t(0))for(int j=0;j<D;j++)row[j]+=f.factor[i]*input[j];
            else {uint64_t mask=f.mask;while(mask){int j=__builtin_ctzll(mask);mask&=mask-1;row[j]+=f.factor[i]*input[j];}}
          }
          for(int j=0;j<D;j++)weights[i*D+j]-=row[j];
        }
      }else if(!scope.factors.empty()){
        float gradient[D*D]{};
        for(int i=0;i<D;i++)for(const auto&f:scope.factors){
          const float*input=batch.inputs[f.input].values;
          if(f.mask==~uint64_t(0))for(int j=0;j<D;j++)gradient[i*D+j]+=f.factor[i]*input[j];
          else {uint64_t mask=f.mask;while(mask){int j=__builtin_ctzll(mask);mask&=mask-1;gradient[i*D+j]+=f.factor[i]*input[j];}}
        }
        for(int j=0;j<D*D;j++)weights[j]-=gradient[j];
      }
      for(int j=0;j<D;j++)entry.bias[j]-=scope.bias[j];
      if(scope.route){
        float norm=1e-9f;
        for(int j=0;j<D;j++){entry.keys[j]-=scope.keys[j];norm+=entry.keys[j]*entry.keys[j];}
        norm=1/std::sqrt(norm);for(int j=0;j<D;j++)entry.keys[j]*=norm;
      }
    }
  }
}
class BankLearningWorker {
  std::mutex mutex;
  std::condition_variable wake,finished;
  bool stopping=false,busy=false,fused=true;
  PipelineBatch*job=nullptr;
  std::thread thread;
public:
  explicit BankLearningWorker(int cpu):thread([this,cpu]{
    if(cpu>=0){cpu_set_t set;CPU_ZERO(&set);CPU_SET(cpu,&set);pthread_setaffinity_np(pthread_self(),sizeof(set),&set);}
    std::unique_lock<std::mutex> lock(mutex);
    for(;;){
      wake.wait(lock,[&]{return stopping||busy;});if(stopping)return;
      auto*current=job;bool fusion=fused;lock.unlock();pipeline_compute(*current,fusion);lock.lock();
      job=nullptr;busy=false;finished.notify_all();
    }
  }){}
  ~BankLearningWorker(){wait();{std::lock_guard<std::mutex> lock(mutex);stopping=true;}wake.notify_one();thread.join();}
  void submit(PipelineBatch*batch,bool fusion){
    std::lock_guard<std::mutex> lock(mutex);job=batch;fused=fusion;busy=true;wake.notify_one();
  }
  void wait(){std::unique_lock<std::mutex> lock(mutex);finished.wait(lock,[&]{return !busy;});}
};
struct BankPipeline {
  bool enabled=false,fused=true;
  int workers=1;
  uint64_t version=0,sequence=0;
  std::unique_ptr<PipelineBatch> building=std::make_unique<PipelineBatch>();
  std::unique_ptr<PipelineBatch> spare=std::make_unique<PipelineBatch>(),pending;
  std::vector<PipelineWeightTile> visible;
  std::array<int,N*E> visible_index;
  // Declared last: worker joins before the jobs it can access are destroyed.
  std::unique_ptr<BankLearningWorker> worker;
  BankPipeline(){visible_index.fill(-1);building->reset();}
  void wait(){if(worker)worker->wait();}
};
