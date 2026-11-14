#include <vector>
#include <fstream>
#include <iostream>
#include <cassert>
#include "../runtime/hierarchy_lab/throughput_probe/composition.cpp"
#ifdef SHOEBOX_LEARNED_COMPLETION
#define SHOEBOX_PUBLICATION_TYPES_READY 1
#include "publication_fixture.cpp"
#endif
template<class T> std::vector<T> read_array(std::ifstream&f){
  int64_t size;f.read(reinterpret_cast<char*>(&size),8);assert(size>=0&&size<10000000);
  std::vector<T> v(size);f.read(reinterpret_cast<char*>(v.data()),size*sizeof(T));assert(f.good());return v;
}
int main(int argc,char**argv){
  assert(argc==2);std::ifstream f(argv[1],std::ios::binary);assert(f.good());
  auto parameters=read_array<float>(f),embedding=read_array<float>(f),feedback=read_array<float>(f),slots=read_array<float>(f);
  auto starts=read_array<int>(f),perm=read_array<int>(f);auto signs=read_array<float>(f);auto fperm=read_array<int>(f);auto fsign=read_array<float>(f);auto cadence=read_array<int>(f);
  void*p=create(parameters.data(),embedding.data(),feedback.data(),slots.data(),starts.data(),perm.data(),signs.data(),fperm.data(),fsign.data(),cadence.data(),1);
#ifdef SHOEBOX_LEARNED_COMPLETION
  assert(publication_fixture_sparse());assert(publication_fixture_handles(p));
#endif
  float lcfg[12]{3,4,16,.1,.25,.5,0,1,2,64,4,1};configure_laces(p,lcfg);
  auto&m=*static_cast<Model*>(p);m.cfg[12]=256;m.cfg[13]=4;m.scfg[10]=.5;
#ifdef SHOEBOX_LEARNED_COMPLETION
  // Exercise the enabled reference paths, including learned bottom state and
  // the auxiliary objectives, rather than only the older comparison policy.
  lcfg[6]=1;lcfg[10]=1;configure_laces(p,lcfg);
  float cfg[24]{3,1,256,1,1,1,1,4,.6f,1,1,1,2048,4,.5f,.98f,1,1,1,.85f,.02f,0,.4f,.005f};
  configure(p,cfg);
  float scfg[24]{1,4,.5f,.125f,.03125f,4,1,64,1,.05f,.5f,1,512,2048,100000,8192,32768,.05f,16384,64,.1f,.5f,0,.01f};
  std::vector<float>extra(2*3*D+4*E*D,0);
  for(int r=0;r<2;r++)for(int j=0;j<D;j++){
    extra[(r*3)*D+j]=std::atanh(.1f/.5f);
    float write=r?.03125f:.125f;extra[(r*3+2)*D+j]=std::log(write/(1-write));
  }
  memcpy(extra.data()+2*3*D,m.keys,2*E*D*sizeof(float));
  configure_state(p,extra.data(),scfg);
  float utility[8]{1,1,1,1,1,.001f,0,0};utility_configure(p,utility);
#endif
#ifdef SHOEBOX_LEARNED_COMPLETION
  constexpr int length=1024; // Cache wraps plus the forced >16K live-DAG lifecycle below.
#else
  constexpr int length=10000;
#endif
  std::vector<unsigned char>x(length),y(length);for(int i=0;i<length;i++){x[i]=i%19<12?"abcdefghijkl"[i%12]:i%256;y[i]=x[i];}
  std::cerr<<"Training "<<length<<" bytes\n";
  float out=0;double loss=run(p,x.data(),y.data(),x.size(),6,.012,.004,.002,0,&out);assert(std::isfinite(loss));
  assert(composition_finish(p,6,.012,.004,.002));
  std::cerr<<"Training/drain passed; round-trip state\n";
  std::vector<unsigned char>saved(composition_state_size(p));composition_export(p,saved.data());
  #ifdef SHOEBOX_LEARNED_COMPLETION
  std::vector<unsigned char>content(content_expert_state_size(p));content_expert_export(p,content.data());
  std::vector<unsigned char>patterns(pattern_state_size(p,0));pattern_export(p,patterns.data(),0);
  std::vector<unsigned char>episodes(paired_memory_state_size(p));paired_memory_export(p,episodes.data());
  assert(paired_occupied(m)>0&&m.paired.metrics[10]>0&&m.paired.metrics[11]>0);
  assert(m.s.return_birth[0]==m.s.return_birth[1]);
  uint64_t content_updates=0;
  for(int r=0;r<N;r++){assert(m.content_expert_metrics[r][5]==0);content_updates+=m.content_expert_metrics[r][1];}
  assert(content_updates>0);
  std::vector<uint64_t>events(N*13),primary(N*65),coverage(N*65),lengths(N*9),recognized(N*9),updates(N*2);
  acceptance_status(p,events.data(),primary.data(),coverage.data(),lengths.data(),recognized.data(),updates.data());
  for(int r=0;r<N;r++){
    uint64_t pc=0,cc=0;for(int i=0;i<65;i++){pc+=primary[r*65+i];cc+=coverage[r*65+i];}
    assert(pc==events[r*13]&&cc==pc);
    assert(events[r*13+9]==0&&events[r*13+10]==0&&events[r*13+11]==0);
    assert(lengths[r*9]==0&&recognized[r*9]==0);
  }
  acceptance_reset(p);assert(m.acceptance.events[0][0]==0);
#endif
  assert(composition_import(p,saved.data(),saved.size(),0));
#ifdef SHOEBOX_LEARNED_COMPLETION
  assert(content_expert_import(p,content.data(),content.size()));
  assert(pattern_import(p,patterns.data(),patterns.size()));
  assert(paired_memory_import(p,episodes.data(),episodes.size()));
  assert(!paired_memory_import(p,episodes.data(),episodes.size()-1));
  assert(!pattern_import(p,patterns.data(),patterns.size()-1));
  assert(m.pattern.serial>0);
  assert(!content_expert_import(p,content.data(),content.size()-1));
  for(int r=0;r<N;r++){
    assert(m.laces.metrics[r][3]==0&&m.laces.metrics[r][11]==0);
    assert(m.composition.stream.message_metrics[r][3]==0);
    assert(m.composition.stream.metrics[r][10]==0&&m.composition.stream.metrics[r][11]==0);
    assert(m.neural_metrics[r][3]==0);
    assert(m.local_credit_metrics[r][7]==0);
  }
  uint64_t local_updates=0,feedback_deliveries=0;
  for(int r=0;r<N;r++){local_updates+=m.local_credit_metrics[r][4];feedback_deliveries+=m.local_credit_metrics[r][2];}
  assert(local_updates>0&&feedback_deliveries>0);
  std::vector<unsigned char>laces(lace_state_size());export_laces(p,laces.data());
  import_laces(p,laces.data());
  std::vector<unsigned char>neural(neural_state_size(p));neural_export(p,neural.data());
  assert(neural_import(p,neural.data(),neural.size()));
  assert(!neural_import(p,neural.data(),neural.size()-1));
  reset_context(p);
  assert(paired_occupied(m)>0&&!m.paired.stream.latest[0].stream&&!m.paired.stream.latest[1].stream);
  for(auto&archive:m.neural_archive)assert(archive.empty());
  for(auto&archive:m.neural_temporal)assert(archive.empty());
  for(int epoch:{4,8,16}){
    assert(reverse_configure(p,2,epoch,1.f));
    loss=run(p,x.data(),y.data(),65,6,.012,.004,.002,0,&out);assert(std::isfinite(loss));
    assert(m.reverse_epoch_position==1&&!m.reverse_roots.empty());
    std::vector<unsigned char>reverse(reverse_state_size(p));reverse_export(p,reverse.data());
    assert(reverse_import(p,reverse.data(),reverse.size()));
    assert(!reverse_import(p,reverse.data(),reverse.size()-1));
    assert(composition_finish(p,6,.012,.004,.002));
    assert(m.reverse_roots.empty()&&m.reverse_bank_version.empty());
    reset_context(p);
  }
  assert(reverse_configure(p,2,1,1.f));
  // Stage 5 shares the production native lifecycle and sanitizer harness.
  for(int heads:{0,1,2}){
    reset_context(p);assert(event_configure(p,heads,1,4,2,.05f,.1f,.25f));
    loss=run(p,x.data(),y.data(),513,6,.012,.004,.002,0,&out);assert(std::isfinite(loss));
    std::vector<unsigned char>event_state(event_state_size(p,0));event_export(p,event_state.data(),0);
    assert(event_import(p,event_state.data(),event_state.size(),0));
    assert(!event_import(p,event_state.data(),event_state.size()-1,0));
    for(int r=0;r<N;r++)assert(m.event.metrics[r][9]==0);
    assert(composition_finish(p,6,.012,.004,.002));
    for(int r=0;r<N;r++)for(int t=0;t<2;t++)assert(!m.event.replay[r][t].request.ref.serial);
  }
  reset_context(p);assert(event_configure(p,1,1,4,2,.05f,.1f,.25f));
  {
    ContentEvent child{};child.ref.region=3;semantic_encode(m,3,child);constituent_encode(m,3,child,nullptr,0,false);
    for(int i=0;i<SEM_RING+31;i++){
      ContentEvent next{};next.ref.region=3;semantic_encode(m,3,next);constituent_encode(m,3,next,&child,1,false);
      child=next;m.composition.stream.history[3][0]=child;
    }
    assert(constituent_capacity(m)==SEM_RING);float g[D]{};g[0]=.1f;
    constituent_update(m,child.semantic,g,.01f,3);event_drain(m);
    assert(m.event.metrics[3][8]>0&&m.event.metrics[3][3]>0&&m.event.metrics[3][9]==0);
  }
  reset_context(p);assert(event_configure(p,0,0,4,2,.05f,.1f,.25f));
  std::cerr<<"Growing live constituent DAG\n";
  ContentEvent child{};child.ref.region=3;semantic_encode(m,3,child);
  constituent_encode(m,3,child,nullptr,0,false);m.composition.stream.history[3][0]=child;
  for(int i=0;i<SEM_RING+31;i++){
    ContentEvent next{};next.ref.region=3;semantic_encode(m,3,next);
    constituent_encode(m,3,next,&child,1,false);child=next;
    m.composition.stream.history[3][0]=child;
  }
  assert(constituent_capacity(m)>SEM_RING);
  assert(reverse_configure(p,2,4,1.f));reverse_begin_epoch(m);m.reverse_epoch_position=1;
  float pending_gradient[D]{};pending_gradient[0]=.001f;
  constituent_update(m,child.semantic,pending_gradient,.01f,3);
  auto held=child.semantic;m.composition.stream.history[3][0]=ContentEvent{};
  m.semantic_free.clear();m.semantic_free_valid=true;
  constituent_allocate(m,3,nullptr,0);assert(constituent_record(m,held));
  m.composition.stream.history[3][0]=child;reverse_finish(p);
  assert(reverse_configure(p,2,1,1.f));

  std::cerr<<"Growth passed; credit and checkpoint continuation\n";
  float gradient[D];for(float&g:gradient)g=.1f;
  constituent_update(m,child.semantic,gradient,.01f,3);
  saved.resize(composition_state_size(p));composition_export(p,saved.data());
  content.resize(content_expert_state_size(p));content_expert_export(p,content.data());
  reset_context(p);assert(composition_import(p,saved.data(),saved.size(),0));
  assert(content_expert_import(p,content.data(),content.size()));
  assert(constituent_record(m,m.composition.stream.history[3][0].semantic));
  constituent_update(m,m.composition.stream.history[3][0].semantic,gradient,.01f,3);
  assert(m.content_expert_metrics[3][1]>SEM_RING&&m.content_expert_metrics[3][5]==0);
#endif
  std::cerr<<"Evaluating after reset\n";
  reset_context(p);loss=run(p,x.data(),y.data(),x.size(),0,0,0,0,0,&out);assert(std::isfinite(loss));
  destroy(p);std::cout<<"ASan/UBSan passed; train/eval/drain/reset/export/import (including semantic pool growth in completion); loss "<<loss<<"\n";
}
