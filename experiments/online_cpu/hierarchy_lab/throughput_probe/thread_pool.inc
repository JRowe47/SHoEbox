#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <pthread.h>
#include <sched.h>
// One persistent team per model. Only disjoint rows execute concurrently.
// Workers sleep outside run(); no optimizer or temporal updates are reordered.
class RowWorkers {
  using Function=void(*)(void*,int,int);
  struct alignas(64) Done {std::atomic<uint64_t> value{0};};
  int count;std::vector<std::thread> threads;std::unique_ptr<Done[]> done;
  alignas(64) std::atomic<uint64_t> sequence{0};
  alignas(64) std::atomic<bool> active{false},stopping{false};
  std::mutex mutex;std::condition_variable wake;
  Function function=nullptr;void*context=nullptr;int length=0;
  void worker(int id,int cpu){
    if(cpu>=0){cpu_set_t set;CPU_ZERO(&set);CPU_SET(cpu,&set);pthread_setaffinity_np(pthread_self(),sizeof(set),&set);}
    uint64_t last=0;
    while(!stopping.load(std::memory_order_acquire)){
      if(!active.load(std::memory_order_acquire)){
        std::unique_lock<std::mutex> lock(mutex);wake.wait(lock,[&]{return active.load()||stopping.load();});continue;
      }
      uint64_t next=sequence.load(std::memory_order_acquire);
      if(next==last){_mm_pause();continue;}
      function(context,length*id/count,length*(id+1)/count);
      last=next;done[id].value.store(next,std::memory_order_release);
    }
  }
public:
  explicit RowWorkers(int n,const int*cpus):count(n),done(new Done[n]){
    for(int i=1;i<n;i++)threads.emplace_back([this,i,cpu=cpus?cpus[i]:-1]{worker(i,cpu);});
  }
  ~RowWorkers(){
    {std::lock_guard<std::mutex> lock(mutex);stopping.store(true,std::memory_order_release);}wake.notify_all();
    for(auto&t:threads)t.join();
  }
  void begin(){ {std::lock_guard<std::mutex> lock(mutex);active.store(true,std::memory_order_release);}wake.notify_all(); }
  void end(){std::lock_guard<std::mutex> lock(mutex);active.store(false,std::memory_order_release);}
  void run(Function fn,void*ctx,int n){
    function=fn;context=ctx;length=n;
    uint64_t next=sequence.fetch_add(1,std::memory_order_release)+1;
    fn(ctx,0,n/count);
    for(int i=1;i<count;i++)while(done[i].value.load(std::memory_order_acquire)!=next)_mm_pause();
  }
};
