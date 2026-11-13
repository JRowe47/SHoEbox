#include <cassert>
#include "completion_fixture.cpp"
extern "C" {
void content_fixture_build(void*p,int same_bank,int repeated,int relay,float*out,uint64_t*meta){
  auto&m=*static_cast<Model*>(p);
  ContentEvent nodes[3];nodes[0]=completion_fixture_leaf(m,3,21);
  for(int level=1;level<3;level++){
    auto child=nodes[level-1];
    if(relay)child=content_compose(m,child.ref.region,&child,1,1,false);
    ContentEvent parts[2]{child,child};
    if(!repeated)parts[1]=completion_fixture_leaf(m,0,70+level);
    nodes[level]=content_compose(m,same_bank?3:level==1?7:10,parts,2,2,false);
  }
  m.composition.stream.history[10][0]=nodes[2];
  for(int k=0;k<3;k++){
    auto&e=nodes[k];auto*v=content_expert_record(m,e.semantic);assert(v);
    float*a=out+k*(5*D+2);memcpy(a,e.code,sizeof(e.code));memcpy(a+D,e.structure,sizeof(e.structure));
    memcpy(a+2*D,v->input,sizeof(v->input));memcpy(a+3*D,v->residual,sizeof(v->residual));
    memcpy(a+4*D,v->route_derivative,sizeof(v->route_derivative));a[5*D]=v->gate;a[5*D+1]=v->gate_derivative;
    uint64_t*b=meta+10*k;b[0]=e.semantic.serial;b[1]=e.semantic.index;b[2]=v->expert;b[3]=v->runner;
    b[4]=v->input_mask;b[5]=e.ref.id.a;b[6]=e.ref.id.b;b[7]=e.clock_begin;b[8]=e.clock_end;b[9]=e.ref.region;
  }
}
void content_fixture_teach(void*p,const float*g,float rate,int teaching){
  auto&m=*static_cast<Model*>(p);auto ref=m.composition.stream.history[10][0].semantic;
  constituent_update(m,ref,g,rate,ref.region,teaching);
}
void content_fixture_history(void*p){
  auto&m=*static_cast<Model*>(p);
  for(int r=0;r<N;r++){
    m.s.exec[r]+=33;
    for(int e=0;e<E;e++)m.s.counts[r][e]=13*(e+1);
    for(int j=0;j<D;j++){
      m.s.h[r][j]=.9f*std::cos(float(r+j));m.s.u[r][j]=.7f*std::sin(float(r+j));
      m.probe.means[r][j]=.3f*std::cos(float(r-j));
      for(int k=0;k<2;k++){
        m.s.hist[r][k][j]=.6f*std::sin(float(r+j+k));
        m.composition.stream.history[r][k].context[j]=.8f*std::sin(float(r+j+k));
      }
    }
  }
}
void content_fixture_freeze(void*p,int r){static_cast<Model*>(p)->frozen_region[r]=1;}
void content_fixture_refresh(void*p){static_cast<Model*>(p)->refresh_keys();}
void content_fixture_route_config(void*p,int features){auto&m=*static_cast<Model*>(p);m.lcfg[2]=features;}
void content_fixture_saved_update(void*p){completion_fixture_update_grown(p);}
}
