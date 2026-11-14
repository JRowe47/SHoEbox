// Stage 6: immutable, already differentiated factors cross the worker boundary.
// No worker touches the model, eligibility stores, Python arrays or router cache.
#include <chrono>
struct PipelineFactor {
  SemanticRef ref{};
  uint64_t mask=0;
  float input[D]{},factor[D]{};
};
struct PipelineScope {
  int bank=0;
  bool route=false;
  float bias[D]{},keys[D]{};
  std::vector<PipelineFactor> factors;
};
struct PipelineTile {
  uint64_t bank=0;
  float weights[D*D]{},bias[D]{},keys[D]{};
};
struct PipelineBatch {
  uint64_t read_version=0,base_version=0,sequence=0;
  std::vector<PipelineScope> scopes;
  std::vector<PipelineTile> tiles;
  uint64_t compute_ns=0;
};
inline void pipeline_compute(PipelineBatch&batch,bool fused){
  auto start=std::chrono::steady_clock::now();
  for(auto&tile:batch.tiles){
    // Scope order (including normalization at each scope) is part of the rule.
    for(const auto&scope:batch.scopes)if(scope.bank==int(tile.bank)){
      if(!scope.factors.empty()&&fused){
        // Keep one bank row hot: form its reduced update and apply it without
        // writing/rereading a full dense gradient matrix.
        for(int i=0;i<D;i++){
          float row[D]{};
          for(const auto&f:scope.factors){
            if(f.mask==~uint64_t(0))for(int j=0;j<D;j++)row[j]+=f.factor[i]*f.input[j];
            else {uint64_t mask=f.mask;while(mask){int j=__builtin_ctzll(mask);mask&=mask-1;row[j]+=f.factor[i]*f.input[j];}}
          }
          for(int j=0;j<D;j++)tile.weights[i*D+j]-=row[j];
        }
      }else if(!scope.factors.empty()){
        float gradient[D*D]{};
        for(int i=0;i<D;i++)for(const auto&f:scope.factors){
          if(f.mask==~uint64_t(0))for(int j=0;j<D;j++)gradient[i*D+j]+=f.factor[i]*f.input[j];
          else {uint64_t mask=f.mask;while(mask){int j=__builtin_ctzll(mask);mask&=mask-1;gradient[i*D+j]+=f.factor[i]*f.input[j];}}
        }
        for(int j=0;j<D*D;j++)tile.weights[j]-=gradient[j];
      }
      for(int j=0;j<D;j++)tile.bias[j]-=scope.bias[j];
      if(scope.route){
        float norm=1e-9f;
        for(int j=0;j<D;j++){tile.keys[j]-=scope.keys[j];norm+=tile.keys[j]*tile.keys[j];}
        norm=1/std::sqrt(norm);for(int j=0;j<D;j++)tile.keys[j]*=norm;
      }
    }
  }
  batch.scopes.clear();
  batch.compute_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
}
class BankLearningWorker {
  std::mutex mutex;
  std::condition_variable wake,finished;
  bool stopping=false,busy=false,fused=true;
  std::shared_ptr<PipelineBatch> job;
  std::thread thread;
public:
  explicit BankLearningWorker(int cpu):thread([this,cpu]{
    if(cpu>=0){cpu_set_t set;CPU_ZERO(&set);CPU_SET(cpu,&set);pthread_setaffinity_np(pthread_self(),sizeof(set),&set);}
    std::unique_lock<std::mutex> lock(mutex);
    for(;;){
      wake.wait(lock,[&]{return stopping||busy;});if(stopping)return;
      auto current=job;bool fusion=fused;lock.unlock();pipeline_compute(*current,fusion);lock.lock();
      job.reset();busy=false;finished.notify_all();
    }
  }){}
  ~BankLearningWorker(){wait();{std::lock_guard<std::mutex> lock(mutex);stopping=true;}wake.notify_one();thread.join();}
  void submit(std::shared_ptr<PipelineBatch> batch,bool fusion){
    std::lock_guard<std::mutex> lock(mutex);job=std::move(batch);fused=fusion;busy=true;wake.notify_one();
  }
  void wait(){std::unique_lock<std::mutex> lock(mutex);finished.wait(lock,[&]{return !busy;});}
};
struct BankPipeline {
  bool enabled=false,fused=true;
  int workers=1;
  uint64_t version=0,sequence=0;
  std::vector<PipelineScope> building;
  std::shared_ptr<PipelineBatch> pending;
  std::vector<PipelineTile> visible;
  std::array<int,N*E> visible_index;
  std::unique_ptr<BankLearningWorker> worker;
  // submissions, commits, scopes, factors, bank tiles, copied input bytes,
  // worker compute ns, blocking wait ns, fused forward rows, peak queued bytes.
  uint64_t metrics[10]{};
  BankPipeline(){visible_index.fill(-1);}
  void wait(){if(worker)worker->wait();}
};
