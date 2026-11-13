#define SHOEBOX_PRESERVE_RELAYS 1
#include "semantics_fixture.cpp"
extern "C" {
void relay_fixture_schedule(void*p,int r){
 auto&m=*static_cast<Model*>(p);auto*v=current_lace(m,r);
 m.composition.stream.teaching[r]=PendingTeaching{};
 relay_schedule(m,r,m.probe.cues[r][m.laces.cursor[r]]);
}
void relay_fixture_teach(void*p,int r,const float*q,int mode,float rate,int relay){
 relay_teach(*static_cast<Model*>(p),r,q,mode,rate,relay);
}
void relay_fixture_public_output(void*p,int r,float value){
 auto&m=*static_cast<Model*>(p);
 for(int j=0;j<D;j++){m.s.h[r][j]=value;m.s.u[r][j]=-value;m.s.pred[r][j]=value;}
 m.s.chosen[r]=(m.s.chosen[r]+1)%E;
}
void relay_fixture_expire(void*p,int r){
 auto&m=*static_cast<Model*>(p);auto&t=m.composition.stream.teaching[r];
 m.laces.records[r][t.index].tick++;
}
void relay_fixture_forward(void*p,int source,int receiver){
 auto&m=*static_cast<Model*>(p);auto child=m.composition.stream.history[source][0];
 auto event=content_compose(m,receiver,&child,1,1,false);
 content_passthrough(m,receiver,event,child);
}
void relay_fixture_recognition(void*p,int r,int recognized){
 auto&e=static_cast<Model*>(p)->composition.stream.history[r][0];
 e.ref.recognized=recognized;e.origin.recognized=recognized;
 e.confidence=recognized?.875f:.25f;
}
void relay_fixture_snapshot(void*p,int r,float*out){
 auto&m=*static_cast<Model*>(p);const auto&t=m.composition.stream.teaching[r];
 const auto&v=m.laces.records[r][t.index];
 memcpy(out,t.prediction,D*sizeof(float));memcpy(out+D,t.input,D*sizeof(float));
 memcpy(out+2*D,v.h,D*sizeof(float));memcpy(out+3*D,v.u,D*sizeof(float));
 out[4*D]=v.expert;
}
void relay_fixture_familiar_pair(void*p,int r,int*result){
 auto&m=*static_cast<Model*>(p);auto&z=m.composition.stream;
 int driver=int(m.slots[4*m.starts[r]]);
 auto first=z.history[driver][0];first.ref.seq=1;first.ref.begin=0;first.ref.end=3;
 auto next=first;next.ref.seq=2;next.ref.begin=3;next.ref.end=6;
 next.ref.id=content_pair(first.ref.id,{7,9},11);
 z.pending[r][0]=first;z.length[r]=1;z.seen[r]=1;z.history[driver][0]=next;
 ContentEvent out{},parts[COMP_MAX]{};int n=0;
 result[0]=content_prepare(m,r,false,out,parts,n);result[1]=z.length[r];
 z.draining=1;result[2]=content_prepare(m,r,false,out,parts,n);z.draining=0;
 result[3]=out.count;result[4]=out.ref.recognized;
}
}
