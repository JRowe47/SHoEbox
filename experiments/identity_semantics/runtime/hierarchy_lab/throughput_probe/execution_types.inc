// Derived execution storage only. None of these indexes is checkpoint state.
using LocalKey=std::pair<int,int64_t>;
struct LocalSignal {
  std::array<float,D> neural{},input{};
  int neural_uses=0,input_uses=0;
};
struct LocalSignals {
  struct Entry {LocalKey first;LocalSignal second;};
  struct Slot {uint64_t generation=0;size_t index=0;};
  std::vector<Entry> entries;
  std::vector<Slot> table;
  std::vector<size_t> order;
  uint64_t generation=1;
  static size_t hash(LocalKey key){
    uint64_t h=uint64_t(key.second)*0x9e3779b97f4a7c15ULL+uint64_t(key.first);
    h^=h>>30;h*=0xbf58476d1ce4e5b9ULL;return size_t(h^(h>>27));
  }
  void reset(){
    entries.clear();order.clear();
    if(++generation==0){for(auto&s:table)s.generation=0;generation=1;}
  }
  void grow(){
    table.assign(std::max(size_t(32),table.size()*2),Slot{});
    for(size_t i=0;i<entries.size();i++){
      size_t at=hash(entries[i].first)&(table.size()-1);
      while(table[at].generation==generation)at=(at+1)&(table.size()-1);
      table[at]={generation,i};
    }
  }
  LocalSignal& operator[](LocalKey key){
    if(table.empty()||(entries.size()+1)*2>table.size())grow();
    size_t at=hash(key)&(table.size()-1);
    while(table[at].generation==generation){
      auto&v=entries[table[at].index];if(v.first==key)return v.second;
      at=(at+1)&(table.size()-1);
    }
    table[at]={generation,entries.size()};entries.push_back({key,{}});return entries.back().second;
  }
  void sort(){
    order.resize(entries.size());for(size_t i=0;i<order.size();i++)order[i]=i;
    std::sort(order.begin(),order.end(),[&](size_t a,size_t b){return entries[a].first<entries[b].first;});
  }
};
struct LocalAdjoint {
  LaceRecord*producer=nullptr;
  int region=0;
  float mean[D]{},du[D]{},dcue[D]{};
};
struct LocalCredit {
  LocalSignals pending[LACE_DEPTH],updates;
  std::vector<LocalAdjoint> adjoints;
  void reset(){for(auto&p:pending)p.reset();updates.reset();adjoints.clear();}
};
struct ReversePolicyKey {
  std::array<uint32_t,N+1> bits{};
  bool operator==(const ReversePolicyKey&other)const{return bits==other.bits;}
};
struct ReversePolicyHash {
  size_t operator()(const ReversePolicyKey&key)const{
    uint64_t h=1469598103934665603ULL;
    for(auto b:key.bits){h^=b;h*=1099511628211ULL;}return size_t(h);
  }
};
struct ReverseExecution {
  struct Policy {uint64_t generation=0;size_t group=0;};
  struct Group {size_t policy=0;std::vector<size_t> roots;float mass=0;};
  struct Plan {
    bool valid=false;size_t policy=0;
    int local=0,local_only=0,frontier=0;
    uint64_t ordinals[N]{};
    std::vector<SemanticRef> roots,nodes;
    std::vector<size_t> batch_ends;
  };
  std::unordered_map<ReversePolicyKey,size_t,ReversePolicyHash> keys;
  std::vector<Policy> policies;
  std::vector<Group> groups;
  std::array<Plan,32> plans;
  Plan scratch;
  uint64_t generation=0;
  size_t count=0;
  void invalidate(){
    keys.clear();policies.clear();count=0;
    for(auto&p:plans){p.valid=false;p.roots.clear();p.nodes.clear();p.batch_ends.clear();}
    scratch.valid=false;
  }
};
struct RegionalExecution {
  struct Recipe {int driver=0,multiplicity=0;float squared=0,scale=1;std::vector<int> immediate;};
  struct Frame {
    bool ready=false;int count=0,available=0;uint32_t fresh=0;
    ContentEvent children[COMP_MAX];
  };
  Recipe recipe[N];
  Frame bottom_family[3];
  int64_t family_tick=-1;
  bool family_compatible=false;
  uint32_t listeners[N]{},pending=0;
  bool valid=false;
  // A1/B1 share the observed byte's preparation, never recurrent state.
  float byte[D]{};
};
