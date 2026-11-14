#include "neural_fixture.cpp"
#include "retention_oracle.inc"
#include "semantic_retention_oracle.inc"
#include <cassert>
extern "C" {
int indexed_pool_bounds(){
  BoundedRecordPool pool;std::vector<void*> blocks;
  for(int i=0;i<1024;i++)blocks.push_back(pool.allocate(4096,16));
  for(auto*p:blocks)pool.deallocate(p,4096,16);
  if(pool.reserved_bytes()!=64*4096)return 0;
  blocks.clear();for(int i=0;i<64;i++)blocks.push_back(pool.allocate(4096,16));
  if(pool.reserved_bytes()!=0)return 0;
  for(auto*p:blocks)pool.deallocate(p,4096,16);
  return pool.reserved_bytes()==64*4096;
}
int indexed_verify(void*p){
  auto&m=*static_cast<Model*>(p);auto expected=reference_neural_marks(m);
  neural_collect(m);
  for(int r=0;r<N;r++){
    for(const auto&entry:expected.marks[r])if(!find_lace(m,r,entry.first,-1))return 0;
    for(const auto&entry:m.neural_archive[r])if(!expected.marks[r].count(entry.first))return 0;
    for(const auto&entry:m.neural_retention.index[r]){
      const auto&v=m.neural_retention.nodes[entry.second];
      auto it=expected.marks[r].find(entry.first);
      if(it==expected.marks[r].end()||it->second!=v.active)return 0;
      for(int count:v.incoming)if(count<0)return 0;
    }
  }
  return 1;
}
int indexed_semantic_verify(void*p){
  auto&m=*static_cast<Model*>(p);auto expected=reference_semantic_marks(m,nullptr,0);
  constituent_retain(m,nullptr,0);
  for(int i=0;i<constituent_capacity(m);i++)
    if(expected[i]!=constituent_retained(m,i)||m.semantic_retention.nodes[i].incoming<0)return 0;
  return 1;
}
int indexed_overlap(void*p){
  auto&m=*static_cast<Model*>(p);reset_context(p);m.s.tick=100;m.lcfg[0]=2;
  fixture_execute(m,2,2);fixture_execute(m,3,2);fixture_execute(m,4,2);
  current_lace(m,3)->nparents=1;current_lace(m,3)->parents[0]={2,-1,101,1,1};
  current_lace(m,4)->nparents=1;current_lace(m,4)->parents[0]={3,-1,102,1,1};
  // Two differently deep owners overlap. The direct root goes away first.
  auto&s=m.composition.stream;s.pending[13][0]=s.history[3][0];s.length[13]=1;
  s.pending[14][0]=s.history[4][0];s.length[14]=1;
  for(int r=2;r<=4;r++)for(int k=0;k<130;k++)fixture_execute(m,r,4);
  if(!indexed_verify(p)||!find_lace(m,2,101,-1))return 0;
  s.pending[13][0]=ContentEvent{};s.length[13]=0;
  if(!indexed_verify(p)||!find_lace(m,2,101,-1))return 0;
  m.lcfg[0]=1; // Rebuild derived ownership for an explicit depth change.
  if(!indexed_verify(p)||find_lace(m,2,101,-1)||!find_lace(m,3,102,-1))return 0;
  s.pending[14][0]=ContentEvent{};s.length[14]=0;
  if(!indexed_verify(p)||find_lace(m,3,102,-1)||find_lace(m,4,103,-1))return 0;
  reset_context(p);m.s.tick=100;m.lcfg[0]=1;
  fixture_execute(m,2,2);fixture_execute(m,3,2);fixture_execute(m,4,2);
  current_lace(m,3)->expert=-1; // Transparent producer consumes zero depth.
  current_lace(m,3)->nparents=1;current_lace(m,3)->parents[0]={2,-1,101,1,1};
  current_lace(m,4)->nparents=1;current_lace(m,4)->parents[0]={3,-1,102,1,1};
  s.pending[14][0]=s.history[4][0];s.length[14]=1;
  for(int r=2;r<=4;r++)for(int k=0;k<130;k++)fixture_execute(m,r,4);
  if(!indexed_verify(p)||!find_lace(m,2,101,-1))return 0;
  s.pending[14][0]=ContentEvent{};s.length[14]=0;
  if(!indexed_verify(p)||find_lace(m,2,101,-1))return 0;
  return 1;
}
int indexed_context_lookup(void*p){
  auto&m=*static_cast<Model*>(p);m.event.heads=1;m.s.tick=5;
  auto&v=m.laces.records[7][3];v.tick=5;float input[D]{},out[D];
  context_forward(m,7,input,out,m.event.context[7][3]);neural_invalidate(m);
  LaceRecord detached=v;
  if(context_record(m,7,detached)!=&m.event.context[7][3])return 0;
  neural_archive_slot(m,7,3);v.tick=69;m.event.context[7][3].prepare(1,69);
  if(!context_record(m,7,detached)||context_record(m,7,detached)->tick!=5)return 0;
  detached.tick=133;if(context_record(m,7,detached))return 0;
  return 1;
}
void indexed_storage(void*p,uint64_t*out){
  auto&m=*static_cast<Model*>(p);out[0]=out[1]=out[2]=0;
  for(int r=0;r<N;r++){
    for(const auto&v:m.event.context[r]){out[0]+=v.data.size()*sizeof(float);out[1]+=v.data.capacity()*sizeof(float);out[2]+=v.tick!=0;}
    for(const auto&entry:m.event.archive[r]){out[0]+=entry.second.data.size()*sizeof(float);out[1]+=entry.second.data.capacity()*sizeof(float);out[2]++;}
  }
  out[3]=sizeof(ContextHead);out[4]=sizeof(ContextEligibility);out[5]=sizeof(ContextEligibilityV1);
}
}
