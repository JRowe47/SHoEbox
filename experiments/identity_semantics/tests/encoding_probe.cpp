#ifdef SEMANTIC_PROBE
#define SHOEBOX_IDENTITY_SEMANTICS 1
#endif
#define SHOEBOX_MULTI_BINDING 1
#include "../runtime/hierarchy_lab/throughput_probe/composition.cpp"
extern "C" void probe_encode(void*p,int r,int gap,int duration,int source,int offset,uint64_t*ids,float*code){
 auto&m=*static_cast<Model*>(p);ContentEvent parts[2]{};
 for(int i=0;i<2;i++){
  auto&e=parts[i];e.ref.id={content_mix(65+i),content_mix(321+i)};e.ref.region=i?source:0;e.ref.seq=i+1;e.ref.begin=offset+(i?gap:0);e.ref.end=e.ref.begin+(i?duration:1);
  memcpy(e.code,m.emb+(65+i)*D,sizeof(e.code));
#ifdef SHOEBOX_IDENTITY_SEMANTICS
  memcpy(e.structure,e.code,sizeof(e.structure));
#endif
 }
 auto e=content_compose(m,r,parts,2,1,false);ids[0]=e.ref.id.a;ids[1]=e.ref.id.b;memcpy(code,e.code,sizeof(e.code));
}
