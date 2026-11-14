#ifndef SHOEBOX_PUBLICATION_TYPES_READY
#include "../runtime/hierarchy_lab/throughput_probe/completion_checks.cpp"
#endif
#include <cassert>

extern "C" int publication_fixture_handles(void*p){
  auto&m=*static_cast<Model*>(p);ContentEvent a{};
  a.ref.region=3;a.ref.id={13,29};a.ref.seq=5;a.semantic={100,3,4};
  for(int j=0;j<D;j++){
    a.edit_values().code[j]=.01f*(j-31);a.edit_values().structure[j]=.003f*(j+1);
    a.edit_context().context[j]=.02f*(j-9);
  }
  ContentEvent b=a;b.ref.role=m.starts[6];b.ref.seq=8;b.origin.region=2;
  if(a.payload!=b.payload||a.contextual!=b.contextual||a.ref.seq!=5)return 0;
  const int slot=m.starts[6];const auto&first=publication_role(m,slot,a);
  float old=first.delta[0];uint64_t misses=m.publication_cache_metrics[1];
  publication_role(m,slot,b);
  if(m.publication_cache_metrics[1]!=misses||!m.publication_cache_metrics[0])return 0;
  publication_role(m,slot,b,true);publication_role(m,slot,a,true);
  if(!m.publication_cache_metrics[2])return 0;
  // Same exact identity and semantic handle, but a deliberately detached vector.
  int coordinate=m.perm[slot*D];b.edit_values().code[coordinate]+=.25f;
  if(a.payload==b.payload||a.values().code[coordinate]==b.values().code[coordinate])return 0;
  const auto&second=publication_role(m,slot,b);
  if(second.delta[0]==old||m.publication_cache_metrics[1]!=misses+1)return 0;
  b.edit_context().context[coordinate]-=.5f;
  if(a.contextual==b.contextual)return 0;
  const auto&context=publication_role(m,slot,b,true);
  if(context.context[0]!=b.context_data().context[coordinate])return 0;
  // Compare role arithmetic at several changed gains, using the same cache.
  for(float weight:{-.37f,0.f,.2f,1.f}){
    const auto&cached=publication_role(m,slot,b);float direct[D],reused[D];
    for(int j=0;j<D;j++){
      direct[j]=reused[j]=.017f*j;int k=m.perm[slot*D+j];
      direct[j]+=weight*m.signs[slot*D+j]*(b.values().code[k]-b.values().structure[k]);
      reused[j]+=weight*m.signs[slot*D+j]*cached.delta[j];
    }
    if(memcmp(direct,reused,sizeof(direct)))return 0;
  }
  // Expanded checkpoint export/import does not serialize pointers or alias edits.
  ContentEventWire wire{};copy_publication(wire,b);ContentEvent restored{};copy_publication(restored,wire);
  if(memcmp(&restored.values(),&b.values(),sizeof(PublicationPayload)))return 0;
  if(restored.ref.seq!=b.ref.seq||restored.semantic.serial!=b.semantic.serial)return 0;
  constituent_invalidate(m);
  return m.publication_roles.empty()&&m.publication_cache_metrics[0]==0;
}
extern "C" int publication_fixture_sparse(){
  LaceRecord a{};InputCredit input{};
  const LaceRecord&read=a;const InputCredit&ri=input;
  if(read.credit[0][7]!=0||ri.gradient[2][5]!=0||a.credit.rows||input.gradient.rows)return 0;
  a.credit[1][7]=.5f;a.uses[1]=2;input.gradient[2][5]=-.25f;input.uses[2]=1;
  LaceRecord b=a;b.credit[1][7]+=.2f;
  if(a.credit[1][7]!=.5f||a.credit.rows==b.credit.rows)return 0;
  LaceRecordWire wire{};copy_lace(wire,a);LaceRecord restored{};copy_lace(restored,wire);
  if(restored.credit[1][7]!=.5f||restored.uses[1]!=2)return 0;
  LegacyInputCredit iw{};completion_copy_input(iw,input);InputCredit ir{};completion_copy_input(ir,iw);
  if(ir.gradient[2][5]!=-.25f||ir.uses[2]!=1)return 0;
  memset(a.credit[1],0,D*sizeof(float));a.credit.release_if_zero();
  if(a.credit.rows)return 0;
  // Preserve every wire bit, including negative zero, on an explicit import.
  wire.credit[0][0]=-0.f;LaceRecord z{};copy_lace(z,wire);LaceRecordWire round{};copy_lace(round,z);
  if(memcmp(wire.credit,round.credit,sizeof(wire.credit)))return 0;
  return 1;
}
