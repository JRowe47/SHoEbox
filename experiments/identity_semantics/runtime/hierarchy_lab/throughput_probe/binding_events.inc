// A role names a declared recipe slot, not a compacted list position.
// Temporal slots refer to distinct completed producer occurrences. A fresh
// occurrence is admitted once per role; unchanged roles are explicitly held.
inline bool binding_source_removed(Model&m,int src){
  return (m.s.ablation==1&&src>=2&&src!=17)||(m.s.ablation>=10&&src==m.s.ablation-10);
}
inline bool binding_ready(Model&m,int r){
  auto&s=m.composition.stream;bool fresh=false;int n=0;
  for(int k=m.starts[r];k<m.starts[r+1];k++){
    const float*slot=m.slots+4*k;if(slot[2])continue;
    int src=int(slot[0]),age=int(slot[1]),role=k-m.starts[r];
    if(binding_source_removed(m,src))continue;
    const auto&child=s.history[src][age];
    if(!child.ref.seq){if(r<15)return false;continue;}
    n++;fresh|=s.fusion_seen[r][role]!=child.ref.seq;
  }
  return fresh&&n>=(r<15?2:1);
}
inline ContentEvent binding_compose(Model&m,int r,const ContentEvent*children,int n,bool learn,uint32_t fresh_mask){
  ContentEvent out{};out.kind=4;out.count=n;out.ref.region=r;out.fresh_mask=fresh_mask;
#ifdef SHOEBOX_PRESERVE_RELAYS
  out.local_arity=n;
#endif
  out.ref.begin=children[0].ref.begin;out.ref.end=children[0].ref.end;
  for(int i=1;i<n;i++){out.ref.begin=std::min(out.ref.begin,children[i].ref.begin);out.ref.end=std::max(out.ref.end,children[i].ref.end);}
  ContentID id{content_mix(uint64_t(r)+0x62696e64),content_mix(0x726f6c657631)};
  float squared=0;
  for(int k=m.starts[r];k<m.starts[r+1];k++)if(!m.slots[4*k+2])squared+=m.slots[4*k+3]*m.slots[4*k+3];
  float scale=1.f/std::sqrt(std::max(1.f,squared));
  for(int i=0;i<n;i++){
    const auto&child=children[i];int k=child.ref.role,role=k-m.starts[r];
    out.children[i]=child.ref;out.present_mask|=uint32_t(1)<<role;
    uint64_t start=uint64_t(child.ref.begin-out.ref.begin),duration=uint64_t(child.ref.end-child.ref.begin);
    uint64_t shape=content_mix(uint64_t(k))^content_mix(start+0x1234)^content_mix(duration+0x9876);
#ifdef SHOEBOX_IDENTITY_SEMANTICS
    shape^=content_mix(uint64_t(child.ref.region)+0x737263);
#endif
    id=content_pair(id,child.ref.id,shape);
    // Preserve compositional child features, and make source identity and
    // relative geometry visible to routing/teaching through a small fixed tag.
    uint64_t tag=content_mix(child.ref.id.a^content_mix(child.ref.id.b)^shape);
    float weight=scale*m.slots[4*k+3]*m.regional_gain[child.ref.region];
#ifdef SHOEBOX_IDENTITY_SEMANTICS
    for(int j=0;j<D;j++)out.structure[j]+=weight*m.signs[k*D+j]*child.structure[m.perm[k*D+j]];
#else
    for(int j=0;j<D;j++)out.code[j]+=weight*(m.signs[k*D+j]*child.code[m.perm[k*D+j]]+.125f*((tag>>j)&1?1.f:-1.f));
#endif
  }
  // The present-role mask distinguishes a missing role from a shifted role.
  out.ref.id=content_pair(id,{out.present_mask,0x70726573656e74},0x62696e64);
#ifdef SHOEBOX_IDENTITY_SEMANTICS
  semantic_encode(m,r,out);
#endif
  uint32_t count=content_count(m.composition.patterns[r],out.ref.id,learn,m.composition.stream.metrics[r][6]);
  out.ref.recognized=count>=m.composition.min_count;out.confidence=std::min(1.f,float(count)/m.composition.min_count);
  return out;
}
inline bool binding_prepare(Model&m,int r,bool learn,ContentEvent&event,ContentEvent*parts,int&n){
  auto&s=m.composition.stream;n=0;uint32_t fresh=0;
  for(int k=m.starts[r];k<m.starts[r+1];k++){
    const float*slot=m.slots+4*k;if(slot[2])continue;
    int src=int(slot[0]),age=int(slot[1]),role=k-m.starts[r];
    if(binding_source_removed(m,src))continue;
    const auto&child=s.history[src][age];
    if(!child.ref.seq){
      if(r<15){s.metrics[r][5]++;s.metrics[r][17]++;return false;}
      continue;
    }
    if(n>=COMP_MAX||role>=COMP_MAX)return false; // Schema validation also runs at configuration.
    parts[n]=child;parts[n].ref.role=k;n++;
    if(s.fusion_seen[r][role]!=child.ref.seq)fresh|=uint32_t(1)<<role;
  }
  if((r<15&&n<2)||((r!=17||s.draining)&&(!n||!fresh))){s.metrics[r][5]++;return false;}
  if(!fresh){event=n?s.history[r][0]:ContentEvent{};return true;}
  event=binding_compose(m,r,parts,n,learn,fresh);
  for(int i=0;i<n;i++)s.fusion_seen[r][parts[i].ref.role-m.starts[r]]=parts[i].ref.seq;
  int fresh_count=__builtin_popcount(fresh);
  s.metrics[r][0]+=fresh_count;s.metrics[r][14]++;s.metrics[r][15]+=n-fresh_count;s.metrics[r][16]+=fresh_count;
  return true;
}
inline void binding_bundle(Model&m,int r,const ContentEvent&event,const ContentEvent*parts,int n,
                           float*q,float*c,float&qw,float&cw,LaceParent*parents,int&nparents){
  if(n){memcpy(q,event.code,D*sizeof(float));qw=1;}
  float squared=0;for(int k=m.starts[r];k<m.starts[r+1];k++)if(!m.slots[4*k+2])squared+=m.slots[4*k+3]*m.slots[4*k+3];
  float scale=1.f/std::sqrt(std::max(1.f,squared));
  for(int i=0;i<n;i++){
    const auto&child=parts[i];int k=child.ref.role;
    float weight=scale*m.slots[4*k+3]*m.regional_gain[child.ref.region];
    for(int j=0;j<D;j++)c[j]+=weight*m.signs[k*D+j]*child.context[m.perm[k*D+j]];
    if(weight&&nparents<LACE_LINKS){const auto&origin=child.origin;
      parents[nparents++]={origin.region,k,origin.tick,-weight,origin.index};
      m.composition.stream.metrics[r][12]+=origin.region!=child.ref.region;
    }
  }
  cw=n?1.f:0.f;
}
