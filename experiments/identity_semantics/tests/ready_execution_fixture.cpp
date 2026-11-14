#include "../runtime/hierarchy_lab/throughput_probe/completion.cpp"
#include <limits>
extern "C" int execution_accumulator_fixture(){
  LocalSignals work;std::map<LocalKey,LocalSignal> oracle;
  for(int repeat=0;repeat<5;repeat++){
    work.reset();oracle.clear();
    for(int i=0;i<3000;i++){
      LocalKey key{(i*13)%18,(i*37)%513};float value=float(i%19-9)*.03125f;
      auto&a=work[key];auto&b=oracle[key];a.neural[i%64]+=value;b.neural[i%64]+=value;
      a.neural_uses++;b.neural_uses++;
    }
    work.sort();if(work.order.size()!=oracle.size())return 0;size_t i=0;
    for(const auto&b:oracle){const auto&a=work.entries[work.order[i++]];
      if(a.first!=b.first||memcmp(&a.second,&b.second,sizeof(LocalSignal)))return 0;}
  }
  auto capacity=work.entries.capacity();work.generation=std::numeric_limits<uint64_t>::max();work.reset();
  if(work[{0,0}].neural_uses||work.generation!=1||work.entries.capacity()!=capacity)return 0;
  return 1;
}
extern "C" int execution_adjoint_fixture(void*p){
  auto&m=*static_cast<Model*>(p);int heads=m.event.heads;float sparse=m.lcfg[9];m.event.heads=0;m.lcfg[9]=64;
  std::array<LaceRecord,8> records;std::array<LocalAdjoint,8> jobs;LocalAdjoint*packet[8];float expected[8][D];
  for(int a=0;a<8;a++){
    auto&v=records[a];v.expert=2;v.gate=.3f+.1f*a;
    auto&job=jobs[a];job.producer=&v;job.region=3;packet[a]=&job;
    for(int j=0;j<D;j++){v.h[j]=.4f*std::sin(float(a*3+j));job.mean[j]=.3f*std::cos(float(a+j));}
    neural_input_gradient(m,3,v,job.mean,expected[a]);
  }
  int equal=1;
  for(int count:{1,3,4,7,8}){
    local_adjoint_packet(m,packet,count);
    for(int a=0;a<count;a++)equal&=!memcmp(expected[a],jobs[a].du,D*sizeof(float));
  }
  m.event.heads=heads;m.lcfg[9]=sparse;return equal;
}
extern "C" int execution_scheduler_fixture(void*p){
  auto&m=*static_cast<Model*>(p);execution_begin_run(m);
  m.execution.pending=1u<<3;
  if(execution_next(m,0)!=3)return 0;
  m.execution.listeners[5]=(1u<<2)|(1u<<3)|(1u<<9);execution_publish(m,5);
  if(execution_next(m,4)!=9||execution_next(m,10)!=N)return 0;
  if(m.execution.pending!=((1u<<2)|(1u<<3)))return 0;
  execution_begin_step(m,65);
  if(execution_next(m,0)!=0||execution_next(m,1)!=1||execution_next(m,2)!=2)return 0;
  m.composition.stream.draining=true;execution_begin_step(m,0);
  if(execution_next(m,0)!=2)return 0;
  m.composition.stream.draining=false;execution_begin_run(m);return 1;
}
extern "C" int execution_reverse_fixture(void*p){
  auto&m=*static_cast<Model*>(p);m.reverse_roots.clear();m.reverse_execution.invalidate();
  for(int q=0;q<5;q++){
    Model::ReverseRoot root{};root.ref={uint64_t(100+q%2),3,q%2};root.rate=q%2?-.2f:.1f;
    root.teaching=q==3?1:0;for(float&scale:root.scales)scale=1;
    if(q==4)root.scales[2]=-0.f;else root.scales[2]=0.f;
    m.reverse_roots.push_back(root);
  }
  reverse_group_roots(m);auto&x=m.reverse_execution;
  if(x.count!=3||x.groups[0].roots!=std::vector<size_t>({0,1,2})||x.groups[1].roots[0]!=3||x.groups[2].roots[0]!=4)return 0;
  for(int i=0;i<2;i++){auto&n=constituent_slot(m,i);n=ConstituentRecord{};n.ref={uint64_t(100+i),3,i};}
  auto&root=constituent_slot(m,1);root.edges[root.count++]={constituent_slot(m,0).ref,{},-100,1.f};
  auto&group=x.groups[0];auto&plan=reverse_schedule(m,group,++m.semantic_work_serial);
  if(plan.nodes.size()!=2)return 0;auto*storage=plan.nodes.data();
  if(reverse_schedule(m,group,++m.semantic_work_serial).nodes.data()!=storage)return 0;
  // Stale slab generations must not be interpreted as the former child.
  constituent_slot(m,0).ref.serial=999;
  if(reverse_schedule(m,group,++m.semantic_work_serial).nodes.size()!=1)return 0;
  constituent_slot(m,0).ref.serial=100;
  m.event.local=1;m.event.frontier=4;m.event.ordinal[3]=10;m.event.eligibility.resize(2);
  auto&saved=m.event.eligibility[1];saved.key.ref=root.ref;saved.children[0].ref=constituent_slot(m,0).ref;saved.children[0].ordinal=8;
  group.roots={1};
  if(reverse_schedule(m,group,++m.semantic_work_serial).nodes.size()!=2)return 0;
  m.event.ordinal[3]=12;
  if(reverse_schedule(m,group,++m.semantic_work_serial).nodes.size()!=1)return 0;
  m.event.ordinal[3]=10;m.event.local_only=true;
  if(reverse_schedule(m,group,++m.semantic_work_serial).nodes.size()!=1)return 0;
  m.event.local_only=false;m.event.local=0;m.reverse_roots.clear();return 1;
}
