// Pools recycle allocations; logical identity is always (region, tick), never
// an address. Shared publications can outlive their creating Model, so the
// synchronized resource has process lifetime and is safe across model threads.
inline std::pmr::synchronized_pool_resource& record_pool(){
  static std::pmr::synchronized_pool_resource pool;return pool;
}
// The training slab is absent until first use, and can be released when a
// stream permanently drops its derivative history. Wire arrays stay fixed.
template<class T,size_t Count> struct LazyRecordArray {
  std::unique_ptr<T[]> storage;
  T* data(){if(!storage)storage=std::make_unique<T[]>(Count);return storage.get();}
  const T* data()const{return storage.get();}
  T& operator[](size_t i){return data()[i];}
  const T& operator[](size_t i)const{static const T zero{};return storage?storage[i]:zero;}
  void release(){storage.reset();}
  size_t bytes()const{return storage?Count*sizeof(T):0;}
};
template<class T> struct RecordArchive:std::pmr::map<int64_t,T>{
  explicit RecordArchive(std::pmr::memory_resource*pool=&record_pool()):std::pmr::map<int64_t,T>(pool){}
};
template<class T,size_t...I> inline auto record_archives(std::pmr::memory_resource*pool,std::index_sequence<I...>){
  return std::array<RecordArchive<T>,sizeof...(I)>{((void)I,RecordArchive<T>(pool))...};
}
template<class T,size_t Count> inline auto record_archives(std::pmr::memory_resource*pool){
  return record_archives<T>(pool,std::make_index_sequence<Count>{});
}
// Reuse a bounded batch of archive nodes without retaining large, partly
// empty slabs after temporary checkpoint/evaluation copies. The owner-thread
// resource caches at most 64 blocks per allocation shape (neural/temporal).
class BoundedRecordPool:public std::pmr::memory_resource {
  struct Bin {size_t bytes,alignment,count=0;void*head=nullptr;};
  std::vector<Bin> bins;
  void*do_allocate(size_t bytes,size_t alignment)override{
    for(auto&bin:bins)if(bin.bytes==bytes&&bin.alignment==alignment){
      if(bin.head){void*p=bin.head;bin.head=*static_cast<void**>(p);--bin.count;return p;}
      return std::pmr::new_delete_resource()->allocate(bytes,alignment);
    }
    bins.push_back({bytes,alignment});return std::pmr::new_delete_resource()->allocate(bytes,alignment);
  }
  void do_deallocate(void*p,size_t bytes,size_t alignment)override{
    for(auto&bin:bins)if(bin.bytes==bytes&&bin.alignment==alignment&&bytes>=sizeof(void*)&&alignment>=alignof(void*)&&bin.count<64){
      *static_cast<void**>(p)=bin.head;bin.head=p;++bin.count;return;
    }
    std::pmr::new_delete_resource()->deallocate(p,bytes,alignment);
  }
  bool do_is_equal(const std::pmr::memory_resource&other)const noexcept override{return this==&other;}
public:
  BoundedRecordPool()=default;
  BoundedRecordPool(const BoundedRecordPool&)=delete;
  size_t reserved_bytes()const{size_t n=0;for(const auto&bin:bins)n+=bin.count*bin.bytes;return n;}
  void release(){
    for(auto&bin:bins)while(bin.head){void*p=bin.head;bin.head=*static_cast<void**>(p);
      std::pmr::new_delete_resource()->deallocate(p,bin.bytes,bin.alignment);}
    bins.clear();
  }
  ~BoundedRecordPool(){release();}
};
template<class T,class...Args> inline std::shared_ptr<T> publication_allocate(Args&&...args){
  return std::allocate_shared<T>(std::pmr::polymorphic_allocator<T>(&record_pool()),std::forward<Args>(args)...);
}

// Depth-sensitive incremental ownership. Each node propagates its maximum
// supported budget, retaining counts at every budget so removing one owner
// exposes the next owner's depth rather than losing a shared producer.
struct NeuralRetention {
  struct Edge {int node,budget;};
  struct Node {
    int region=-1;int64_t tick=0;
    int incoming[LACE_DEPTH+2]{},root=-2,next_root=-2,active=-2;
    uint64_t generation=0;bool queued=false;
    std::vector<Edge> edges;
  };
  std::unordered_map<int64_t,int> index[N];
  std::vector<Node> nodes;
  std::vector<int> free,roots,next_roots,work,candidates;
  uint64_t generation=0;
  int depth=-1,temporal=-1,stride=-1;
  bool initialized=false;
  void invalidate(){
    for(auto&i:index)i.clear();nodes.clear();free.clear();roots.clear();
    next_roots.clear();work.clear();candidates.clear();initialized=false;
  }
};
