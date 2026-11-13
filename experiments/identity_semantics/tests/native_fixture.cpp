#include "../runtime/hierarchy_lab/throughput_probe/composition.cpp"
extern "C" {
void fixture_memory(void*p,int r,int j,float delta){static_cast<Model*>(p)->s.mem[r][j]+=delta;}
void fixture_flush(void*p,int r){flush_credit(*static_cast<Model*>(p),r,0.f);}
void fixture_credit_clear(void*p){auto&m=*static_cast<Model*>(p);
  for(int r=0;r<N;r++)for(int i=0;i<LACE_RING;i++){
    auto&v=m.laces.records[r][i];memset(v.credit,0,sizeof(v.credit));memset(v.uses,0,sizeof(v.uses));
    auto&t=m.composition.stream.input[r][i];memset(t.gradient,0,sizeof(t.gradient));memset(t.uses,0,sizeof(t.uses));
  }
}
void fixture_lace_parents(void*p,int r,int64_t*out,float*weights){
  auto&m=*static_cast<Model*>(p);auto*v=current_lace(m,r);
  for(int i=0;i<LACE_LINKS;i++){
    out[4*i]=-1;out[4*i+1]=0;out[4*i+2]=0;out[4*i+3]=-1;weights[i]=0;
    if(v&&i<v->nparents){auto&e=v->parents[i];out[4*i]=e.region;out[4*i+1]=e.tick;out[4*i+2]=e.slot;out[4*i+3]=e.index;weights[i]=e.coefficient;}
  }
}
}
