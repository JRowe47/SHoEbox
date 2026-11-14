#include "pattern_fixture.cpp"
#include "semantic_retention_oracle.inc"

inline ContentEvent frame_fixture_chain(Model&m,int frames,ContentEvent*last_frame=nullptr){
  auto&s=m.composition.stream;int r=6;
  for(int f=0;f<frames;f++){
    ContentEvent children[COMP_MAX];int n=0;
    for(int k=m.starts[r];k<m.starts[r+1];k++)if(!m.slots[4*k+2]){
      int src=int(m.slots[4*k]),age=int(m.slots[4*k+1]);
      children[n]=completion_fixture_leaf(m,src,src*17+age+f*3,f*4+(1-age)*2,f*4+(2-age)*2);
      children[n].ref.seq=f*2+2-age;
      children[n++].ref.role=k;
    }
    auto frame=binding_compose(m,r,children,n,false,7,true);
    frame.ref.seq=++s.frame_serial[r];frame.ref.tick=m.s.tick;
    s.pending[r][f]=frame;if(last_frame)*last_frame=frame;
  }
  s.length[r]=frames;ContentEvent out{},parts[COMP_MAX];int n=0;
  completion_emit(m,r,false,out,parts,n);return out;
}
extern "C" {
int frame_contract_fixture(void*p){
  auto&m=*static_cast<Model*>(p);ContentEvent frame;
  auto out=frame_fixture_chain(m,1,&frame);
  auto*node=constituent_record(m,frame.semantic);auto*completed=constituent_record(m,out.semantic);
  if(!node||node->encoder!=2||content_expert_record(m,frame.semantic)||!completed||completed->encoder!=1||out.relayed)return 0;
  if(completed->count!=node->count||!(out.ref.id==frame.ref.id))return 0;
  for(int i=0;i<node->count;i++)if(completed->edges[i].source.serial!=node->edges[i].source.serial)return 0;
  if(frame.count!=3||node->count!=3||frame.present_mask!=7||out.clock_begin!=0||out.clock_end!=4)return 0;
  for(int i=0;i<node->count;i++){
    if(node->edges[i].slot!=m.starts[6]+i||node->edges[i].occurrence.role!=m.starts[6]+i)return 0;
    if(!constituent_record(m,node->edges[i].source))return 0;
  }
  // A direct teacher of the fixed descriptor cannot update its regional bank,
  // adapter or gate, but must still reach the actual child content encoders.
  auto bank=std::vector<float>(m.W+6*E*D*D,m.W+7*E*D*D);
  float symbols[D][8];memcpy(symbols,m.composition.symbols[4],sizeof(symbols));
  float child_symbols[D][8];memcpy(child_symbols,m.composition.symbols[0],sizeof(child_symbols));
  float g[D];for(int j=0;j<D;j++)g[j]=.02f*std::sin(j+1.f);
  constituent_update_reference(m,frame.semantic,g,.1f,6);
  return !memcmp(bank.data(),m.W+6*E*D*D,bank.size()*sizeof(float))&&
    !memcmp(symbols,m.composition.symbols[4],sizeof(symbols))&&
    memcmp(child_symbols,m.composition.symbols[0],sizeof(child_symbols));
}
float frame_vjp_fixture(void*p,int frames,int union_policy){
  auto&m=*static_cast<Model*>(p);float g[D];for(int j=0;j<D;j++)g[j]=.03f*std::cos(j+1.f);
  float worst=0,largest=0;
  // Verify downstream teaching through fixed role sums, including a frozen
  // completed recipient, against finite differences of the full learned chain.
  for(auto coordinate: {std::pair<int,int>{17,3},{23,4},{41,5}}){
    int j=coordinate.first,k=coordinate.second;float&weight=m.composition.symbols[0][j][k];
    float original=weight,loss[2];
    for(int side=0;side<2;side++){
      weight=original+(side?-.002f:.002f);auto out=frame_fixture_chain(m,frames);loss[side]=dot64(out.values().code,g);
    }
    weight=original;auto out=frame_fixture_chain(m,frames);
    m.frozen_region[6]=1;
    if(union_policy){constituent_update(m,out.semantic,g,.1f,6);}
    else constituent_update_reference(m,out.semantic,g,.1f,6);
    float actual=(original-weight)/.1f,expected=(loss[0]-loss[1])/.004f;
    largest=std::max(largest,std::abs(expected));
    worst=std::max(worst,std::abs(actual-expected));
    m.frozen_region[6]=0;
  }
  return largest>1e-7f?worst:1.f;
}
int frame_context_fixture(void*p){
  auto&m=*static_cast<Model*>(p);ContentEvent before,after;
  auto a=frame_fixture_chain(m,2,&before);
  for(int r=0;r<N;r++)for(int j=0;j<D;j++)m.s.mem[r][j]=.9f;
  for(int j=0;j<MK_DIM;j++)m.event.memory[6][0][j][j]=10;
  auto b=frame_fixture_chain(m,2,&after);
  return a.ref.id==b.ref.id&&before.ref.id==after.ref.id&&
    !memcmp(a.values().code,b.values().code,D*sizeof(float))&&
    !memcmp(before.values().code,after.values().code,D*sizeof(float));
}
int frame_local_frontier_fixture(void*p){
  auto&m=*static_cast<Model*>(p);ContentEvent frame;auto out=frame_fixture_chain(m,2,&frame);
  auto*node=constituent_record(m,out.semantic);if(!node||node->count!=2)return 0;
  m.event.local_only=true;
  for(int i=0;i<node->count;i++)if(!event_follow(m,node->ref,i))return 0;
  m.event.local_only=false;
  m.composition.stream.history[6][0]=out;constituent_retain(m,nullptr,0);
  auto oracle=reference_semantic_marks(m,nullptr,0);
  for(int i=0;i<constituent_capacity(m);i++)if(oracle[i]!=constituent_retained(m,i))return 0;
  return 1;
}
int frame_retention_fixture(void*p){
  auto&m=*static_cast<Model*>(p);constituent_retain(m,nullptr,0);
  auto oracle=reference_semantic_marks(m,nullptr,0);
  for(int i=0;i<constituent_capacity(m);i++)if(oracle[i]!=constituent_retained(m,i))return 0;
  return 1;
}
}
