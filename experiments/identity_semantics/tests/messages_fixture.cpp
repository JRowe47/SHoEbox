#define SHOEBOX_LEARNED_MESSAGES 1
#include "relays_fixture.cpp"
extern "C" {
void message_fixture_weights(void*p){
 auto&m=*static_cast<Model*>(p);
 for(int r=2;r<N;r++)for(int j=0;j<D;j++)for(int k=0;k<8;k++)m.composition.symbols[r-2][j][k]=.15f*std::sin(float(r*37+j*7+k));
}
void message_fixture_payload(void*p,int r,int src,int coordinate,float delta,int reencode,float*out,uint64_t*ids){
 auto&m=*static_cast<Model*>(p);ContentEvent parts[COMP_MAX]{};int n=0;uint32_t fresh=0;
 for(int k=m.starts[r];k<m.starts[r+1];k++)if(!m.slots[4*k+2]){
   auto child=m.composition.stream.history[int(m.slots[4*k])][int(m.slots[4*k+1])];
   if(!child.ref.seq)continue;child.ref.role=k;
   if(reencode&&child.origin.region>=2)for(int j=0;j<D;j++){
     float z=0;for(int f=0;f<8;f++)z+=m.composition.symbols[child.origin.region-2][j][f]*child.symbol_features[f];
     float v=std::tanh(z);child.code[j]=child.structure[j]+.25f*v;child.symbol_derivative[j]=.25f*(1-v*v);
   }
   if(child.ref.region==src)child.code[coordinate]+=delta;
   parts[n++]=child;fresh|=1u<<(k-m.starts[r]);
 }
 auto event=binding_compose(m,r,parts,n,false,fresh);float q[D]{},c[D]{},qw=0,cw=0;LaceParent parents[LACE_LINKS];int np=0;
 binding_bundle(m,r,event,parts,n,q,c,qw,cw,parents,np);
 MessageInput msg{};message_payload(m,r,parts,n,q,msg);msg.qa=1;msg.alpha=1;
 auto&v=m.laces.records[r][m.laces.cursor[r]];v.u[0]=0;memset(v.u,0,sizeof(v.u));
 auto&state=m.composition.stream.input[r][m.laces.cursor[r]];state.tick=v.tick;state.symbol_scale=1;state.message=msg;
#ifdef SHOEBOX_LEARNED_COMPLETION
 state.semantic=event.semantic;
#else
 memcpy(state.symbol_features,event.symbol_features,sizeof(state.symbol_features));memcpy(state.symbol_derivative,event.symbol_derivative,sizeof(state.symbol_derivative));
#endif
 memcpy(out,q,D*sizeof(float));memcpy(out+D,event.structure,D*sizeof(float));memcpy(out+2*D,c,D*sizeof(float));ids[0]=event.ref.id.a;ids[1]=event.ref.id.b;
}
void message_fixture_semantic_credit(void*p,int r,const float*g,float rate){
 auto&m=*static_cast<Model*>(p);auto&v=m.laces.records[r][m.laces.cursor[r]];semantic_input_update(m,r,v,g,rate);
}
void message_fixture_pulse(void*p,int src,int j,float delta){
 auto&m=*static_cast<Model*>(p);m.s.hist[src][0][j]+=delta;m.composition.stream.history[src][0].context[j]+=delta;
}
void message_fixture_state(void*p,int r,float*out,int*choices){
 auto&m=*static_cast<Model*>(p);auto*v=current_lace(m,r);memcpy(out,m.s.h[r],D*sizeof(float));
 memcpy(out+D,m.probe.cues[r][m.laces.cursor[r]],D*sizeof(float));
 memcpy(out+2*D,m.s.u[r],D*sizeof(float));choices[0]=v->expert;choices[1]=v->runner;
}
void message_fixture_feedback_isolate(void*p,int receiver){
 auto&m=*static_cast<Model*>(p);auto*slots=const_cast<float*>(m.slots);
 for(int r=2;r<N;r++)if(r!=receiver)for(int k=m.starts[r];k<m.starts[r+1];k++)if(int(slots[4*k])==14&&slots[4*k+2])slots[4*k+3]=0;
}
void message_fixture_credit(void*p,int r,int src,const float*g,float*out){
 auto&m=*static_cast<Model*>(p);for(int a=0;a<LACE_RING;a++)memset(m.laces.records[src][a].credit,0,sizeof(LaceRecord::credit));
 distribute_credit(m,r,*current_lace(m,r),g,0);
 // Delay edge is the preceding source execution, since the full tick has now run.
 auto&input=m.composition.stream.input[r][m.laces.cursor[r]];
 for(int k=0;k<input.message.feedback_count;k++){auto&e=input.message.feedback[k];if(e.region==src){auto*v=find_lace(m,src,e.tick,e.index);if(v)memcpy(out,v->credit[0],D*sizeof(float));}}
}
void message_fixture_route_keys(void*p,int r,const float*direction){
 auto&m=*static_cast<Model*>(p);for(int e=0;e<E;e++)for(int j=0;j<D;j++)m.keys[(r*E+e)*D+j]=(e==0?1.f:e==1?-1.f:0.f)*direction[j];
 m.refresh_keys();memset(m.s.counts[r],0,sizeof(m.s.counts[r]));m.s.exec[r]=0;
}
void message_fixture_gate_gradient(void*p,int r,const float*g,float*out){auto&m=*static_cast<Model*>(p);message_route_gradient(m,r,*current_lace(m,r),g,out);}
void message_fixture_gate_objective(void*p,int r,int j,float delta,const float*g,float*out){
 auto&m=*static_cast<Model*>(p);auto&v=*current_lace(m,r);
 // Perturb the saved cue within one selected/runner/mask cell. All other
 // forward inputs stay fixed. Reconstruct the margin from the actual gate.
 float p0=v.gate*.5f;float margin=m.lcfg[4]*std::log(p0/(1-p0));
 margin+=delta*(m.keys[(r*E+v.expert)*D+j]-m.keys[(r*E+v.runner)*D+j])*((v.input_mask>>j)&1);
 float gate=2/(1+std::exp(-margin/m.lcfg[4]));float value=0;
 for(int i=0;i<D;i++)value+=g[i]*std::tanh(std::atanh(v.h[i])+(gate-v.gate)*v.residual[i]);out[0]=value;
}
}
