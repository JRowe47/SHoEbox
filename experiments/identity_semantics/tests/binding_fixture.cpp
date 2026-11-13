#include "../runtime/hierarchy_lab/throughput_probe/binding.cpp"
extern "C" {
void fixture_stage(void*p,int src,int age,int value,int64_t seq,int64_t begin,int64_t end,float context){
 auto&m=*static_cast<Model*>(p);ContentEvent e{};e.ref.id={content_mix(value+1),content_mix(value+257)};
 e.ref.region=src;e.ref.seq=seq;e.ref.begin=begin;e.ref.end=end;e.ref.tick=seq;e.ref.index=int(seq)&15;e.ref.recognized=1;e.origin=e.ref;e.count=2;e.kind=1;
 memcpy(e.code,m.emb+(value&255)*D,D*sizeof(float));
 for(int j=0;j<D;j++)e.context[j]=context+.01f*j;
 m.composition.stream.history[src][age]=e;m.s.valid[src]=2;
 memcpy(m.s.hist[src][age],e.context,D*sizeof(float));
 auto&v=m.laces.records[src][e.ref.index];v=LaceRecord{};v.tick=seq;v.expert=0;
 m.laces.cursor[src]=e.ref.index;
}
int fixture_prepare(void*p,int r,int learn,uint64_t*meta,float*code){
 auto&m=*static_cast<Model*>(p);ContentEvent e{},parts[COMP_MAX]{};int n=0;
 bool yes=content_prepare(m,r,learn,e,parts,n);if(!yes)return 0;
 meta[0]=e.ref.id.a;meta[1]=e.ref.id.b;meta[2]=n;meta[3]=e.present_mask;meta[4]=e.fresh_mask;
 for(int i=0;i<n;i++){meta[5+3*i]=parts[i].ref.region;meta[6+3*i]=parts[i].ref.seq;meta[7+3*i]=parts[i].ref.role;}
 memcpy(code,e.code,D*sizeof(float));content_publish(m,r,e);return 1;
}
void fixture_inputs(void*p,int r,float*qout,float*cout){
 auto&m=*static_cast<Model*>(p);ContentEvent e{},parts[COMP_MAX]{};int n=0;
 for(int k=m.starts[r];k<m.starts[r+1];k++)if(!m.slots[4*k+2]){int src=m.slots[4*k],age=m.slots[4*k+1];parts[n]=m.composition.stream.history[src][age];parts[n++].ref.role=k;}
 e=binding_compose(m,r,parts,n,false,0);float q[D]{},c[D]{},qw=0,cw=0;LaceParent parents[LACE_LINKS];int np=0;
 binding_bundle(m,r,e,parts,n,q,c,qw,cw,parents,np);memcpy(qout,q,sizeof(q));memcpy(cout,c,sizeof(c));
}
void fixture_bundle_gradient(void*p,int r,const float*g,float*out){
 auto&m=*static_cast<Model*>(p);ContentEvent e{},parts[COMP_MAX]{};int n=0;
 for(int k=m.starts[r];k<m.starts[r+1];k++)if(!m.slots[4*k+2]){int src=m.slots[4*k],age=m.slots[4*k+1];parts[n]=m.composition.stream.history[src][age];parts[n++].ref.role=k;}
 e=binding_compose(m,r,parts,n,false,0);float q[D]{},c[D]{},qw=0,cw=0;LaceParent parents[LACE_LINKS];int np=0;
 binding_bundle(m,r,e,parts,n,q,c,qw,cw,parents,np);
 memset(out,0,N*2*D*sizeof(float));
 for(int i=0;i<np;i++){auto&edge=parents[i];int age=int(m.slots[4*edge.slot+1]);for(int j=0;j<D;j++)out[(edge.region*2+age)*D+m.perm[edge.slot*D+j]]+=-edge.coefficient*m.signs[edge.slot*D+j]*g[j];}
}
void fixture_context_delta(void*p,int src,int age,int j,float delta){static_cast<Model*>(p)->composition.stream.history[src][age].context[j]+=delta;}
void fixture_current_cues(void*p,int r,float*out){auto&m=*static_cast<Model*>(p);memcpy(out,m.probe.cues[r][m.laces.cursor[r]],D*sizeof(float));}
void fixture_driver_code(void*p,int r,float*out){
 auto&m=*static_cast<Model*>(p);int src=int(m.slots[4*m.starts[r]]);ContentEvent parts[2]{m.composition.stream.history[src][1],m.composition.stream.history[src][0]};
 auto e=content_compose(m,r,parts,2,1,false);memcpy(out,e.code,D*sizeof(float));
}
}
