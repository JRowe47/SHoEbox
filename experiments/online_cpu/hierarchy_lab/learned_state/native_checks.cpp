#include "native.cpp"
#include <vector>
#include <iostream>
#include <cassert>
#include <memory>
struct Fixture {
 std::vector<float> weights,extra;
 std::unique_ptr<Model> owner;
 Model& m;
 Fixture():weights(N*E*D*D+2*N*E*D+N*D*D+N*D+V*2*D+V),extra(6*D+4*E*D),owner(std::make_unique<Model>(weights.data(),nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,1)),m(*owner){
   m.state_parameters=extra.data();m.scfg[0]=1;m.scfg[1]=1;m.scfg[2]=.25;m.scfg[6]=1;m.scfg[7]=16;m.scfg[12]=4;m.scfg[13]=4;m.scfg[14]=1000;m.scfg[19]=64;m.scfg[21]=.5;
   for(int r=0;r<2;r++)for(int j=0;j<D;j++){state_parameter(m,r,0)[j]=.4;state_parameter(m,r,1)[j]=.3;state_parameter(m,r,2)[j]=-2;}
   for(int r=0;r<2;r++)for(int e=0;e<E;e++)prototype(m,r,e)[0]=m.keys[(r*E+e)*D]=1;
   initialize_learned(m);
 }
 void publish(int t){m.s.tick=t;m.s.chosen[0]=1;for(int j=0;j<D;j++){m.s.mem[0][j]=.2f+.01f*(t%3);m.s.u[0][j]=.25f;m.s.h[0][j]=.1f+.02f*t;}publish_state(m,0,1);}
};
float gate_value(int field,float perturb){Fixture f;state_parameter(f.m,0,field)[0]+=perturb;for(int t=1;t<=20;t++)f.publish(t);return f.m.learned.memory[0][0];}
int main(){
 {Fixture f;for(int t=1;t<=20;t++)f.publish(t);
  for(int field=1;field<=2;field++){float finite=(gate_value(field,.001)-gate_value(field,-.001))/.002f;float analytic=field==1?f.m.learned.eligibility_a[0][0]:f.m.learned.eligibility_b[0][0];assert(std::abs(finite-analytic)<.00015f);}}
 {Fixture f;f.m.scfg[5]=2;for(int t=1;t<=7;t++)f.publish(t);
  auto*now=temporal_record(f.m,0,7);auto*past=temporal_record(f.m,0,5);auto*middle=temporal_record(f.m,0,6);
  LaceRecord v;v.tick=7;v.expert=1;v.gate=1;memcpy(v.u,now->u,sizeof(v.u));memcpy(v.h,now->h,sizeof(v.h));
  float g[D];std::fill(g,g+D,.3f);int j=__builtin_ctzll(past->input_mask);
  float expected=-.01f*2/norm(past->u)*(.3f*(1-v.h[0]*v.h[0]))*now->read[0]*past->write[0]*(1-middle->write[0])*(1-past->h[0]*past->h[0])*past->u[j];
  learn_bottom_state(f.m,0,&v,g,.01f);assert(std::abs(f.m.W[D*D+j]-expected)<1e-8f);assert(f.m.learned.metrics[0][1]==1);
  auto before=f.weights;f.m.learned.generation[0][1]++;learn_bottom_state(f.m,0,&v,g,.01f);assert(before==f.weights);}
 {Fixture f;f.m.scfg[11]=1;f.m.scfg[15]=0;f.m.s.u[0][1]=1;task_offset(f.m,0,0)[2]=.12;
  double mass=4,s0=4,s1=0,decay=1.-1./(E*4);float cue=1/std::sqrt(1.000001f);
  for(int t=0;t<100;t++){update_bottom_prototype(f.m,0,0,~uint64_t(0),.002f);mass=decay*mass+1;s0*=decay;s1=decay*s1+cue;}
  assert(std::abs(f.m.learned.mass[0][0]-mass)<1e-4);assert(std::abs(f.m.learned.sums[0][0][1]-s1)<1e-4);assert(task_offset(f.m,0,0)[2]==.12f);
  assert(std::abs(prototype(f.m,0,0)[0]-s0/std::sqrt(s0*s0+s1*s1))<1e-5);}
 {Fixture f;f.publish(1);auto old=f.m.learned.records[0][1];LaceRecord v;v.tick=1;v.expert=1;
  assert(valid_bottom_publication(f.m,0,&v));for(int i=0;i<D*D;i++)f.m.W[2*D*D+i]=float(i)*.00001;
  assert(revive_expert(f.m,0,1,2));assert(!valid_bottom_publication(f.m,0,&v));assert(f.m.learned.generation[0][1]==1);
  assert(memcmp(f.m.W+D*D,f.m.W+2*D*D,D*D*sizeof(float))==0);
  assert(!revive_expert(f.m,0,2,2));}
 {Fixture f;f.m.scfg[8]=1;f.publish(1);LaceRecord v;v.tick=1;v.expert=1;v.runner=0;v.gate_derivative=1;
  for(int j=0;j<D;j++){v.u[j]=.2;v.h[j]=.1;v.residual[j]=.2;}
  float g[D];std::fill(g,g+D,.2);f.m.learned.generation[0][0]++;auto before=f.extra;
  bottom_task_update(f.m,0,v,g,.01);assert(before==f.extra);}
 std::cout<<"PASS: gate finite differences, temporal write credit, stale generation rejection, count-normalized EMA, expert cloning\n";
}
