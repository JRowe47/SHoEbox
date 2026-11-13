inline uint64_t content_mix(uint64_t x){
  x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;return x^(x>>31);
}
inline ContentID content_pair(ContentID a,ContentID b,uint64_t role){
  return {content_mix(a.a^content_mix(b.a+role+0x9e3779b97f4a7c15ULL)),
          content_mix(a.b+content_mix(b.b^role^0xd6e8feb86659fd93ULL))};
}
inline int content_coord(int j,int position){return (j+11*position+7)&63;}
inline float content_sign(int j,int position){return ((j*13+position*17)%11)<5?-1.f:1.f;}
inline uint32_t content_count(CountEntry*table,ContentID key,bool learn,double&evictions){
  int base=int(key.a%(COMP_TABLE/COMP_WAYS))*COMP_WAYS,best=base;
  for(int i=base;i<base+COMP_WAYS;i++){
    if(table[i].count&&table[i].key==key){
      if(learn&&table[i].count<UINT32_MAX)table[i].count++;
      return table[i].count;
    }
    if(table[i].count<table[best].count)best=i;
  }
  if(!learn)return 0;
  evictions+=table[best].count>0;table[best]={key,1,0};return 1;
}
inline float content_transition(Model&m,int r,ContentID from,ContentID to,bool learn){
  auto&z=m.composition;auto&metric=z.stream.metrics[r];
  ContentID pair=content_pair(from,to,0x70616972);
  // Decide using previously learned evidence; mutation follows the decision.
  uint32_t denominator=content_count(z.totals[r],from,false,metric[6]);
  uint32_t numerator=content_count(z.pairs[r],pair,false,metric[6]);
  float confidence=numerator>=z.min_count?std::min(1.f,float(numerator)/std::max(1u,denominator)):0.f;
  if(learn){content_count(z.totals[r],from,true,metric[6]);content_count(z.pairs[r],pair,true,metric[6]);}
  return confidence;
}
inline ContentEvent content_compose(Model&m,int r,const ContentEvent*children,int n,int kind,bool learn){
  ContentEvent out{};out.count=n;out.kind=n==1&&kind==1?3:kind;
  out.ref.region=r;out.ref.begin=children[0].ref.begin;out.ref.end=children[0].ref.end;
  ContentID id{content_mix(uint64_t(r)+0x636f6d70),content_mix(uint64_t(kind)+0x726f6c65)};
  float scale=1.f/std::sqrt(float(n));
  for(int i=0;i<n;i++){
    out.children[i]=children[i].ref;out.children[i].role=i;
    out.ref.begin=std::min(out.ref.begin,children[i].ref.begin);out.ref.end=std::max(out.ref.end,children[i].ref.end);
    // Relative geometry is semantic; absolute byte offsets and event ticks are not.
    uint64_t geometry=uint64_t(children[i].ref.begin-children[0].ref.begin);
    id=content_pair(id,children[i].ref.id,content_mix(uint64_t(i)^geometry));
    for(int j=0;j<D;j++)out.code[j]+=scale*content_sign(j,i)*children[i].code[content_coord(j,i)];
  }
  out.ref.id=id;
  if(n==1&&kind==1){out.ref.id=children[0].ref.id;memcpy(out.code,children[0].code,sizeof(out.code));out.confidence=0;}
  else {
    auto&z=m.composition;uint32_t count=content_count(z.patterns[r],id,learn,z.stream.metrics[r][6]);
    out.ref.recognized=count>=z.min_count;out.confidence=std::min(1.f,float(count)/z.min_count);
  }
  return out;
}
inline void content_publish(Model&m,int r,ContentEvent event,bool relay=false){
  auto&z=m.composition.stream;event.ref.seq=z.history[r][0].ref.seq+1;
  event.ref.tick=m.s.tick;event.ref.index=m.laces.cursor[r];
  if(!relay)event.origin=event.ref;
  memcpy(event.context,m.s.h[r],sizeof(event.context));
  z.history[r][1]=z.history[r][0];z.history[r][0]=event;
  auto&d=z.metrics[r];d[1]++;d[2]+=event.count>=2;d[3]+=event.count>=2&&event.ref.recognized;
  d[4]+=event.count>=2&&!event.ref.recognized;d[7]+=event.ref.end-event.ref.begin;
}
inline void content_atom(Model&m,int r,int byte){
  ContentEvent event{};event.ref.id={content_mix(uint64_t(byte)+1),content_mix(uint64_t(byte)+257)};
  event.ref.region=r;event.ref.begin=m.composition.stream.byte_position-1;event.ref.end=m.composition.stream.byte_position;event.ref.recognized=1;
  memcpy(event.code,m.emb+byte*D,sizeof(event.code));content_publish(m,r,event);
}
inline void content_passthrough(Model&m,int r,ContentEvent event,const ContentEvent&child){
  auto&s=m.s;auto&z=m.composition.stream;
  // No recognition winner, learned transform or recurrent-state update for an
  // unresolved singleton. The exact child output and its gradient pass through.
  memcpy(s.h[r],child.context,D*sizeof(float));s.pending[r]=0;
  memcpy(s.hist[r][1],s.hist[r][0],D*sizeof(float));memcpy(s.hist[r][0],s.h[r],D*sizeof(float));s.valid[r]++;
  // A relay is not a neural execution. Keeping the actual producer avoids
  // both false expert teaching and eviction of useful recurrent eligibility.
  event.origin=child.origin;z.metrics[r][13]++;
  content_publish(m,r,event,true);
}
inline bool content_prepare(Model&m,int r,bool learn,ContentEvent&event,ContentEvent*parts,int&n){
  auto&z=m.composition;auto&s=z.stream;n=0;
  if(r<2)return true;
  if(r<=14){
    int driver=int(m.slots[4*m.starts[r]]);const auto&next=s.history[driver][0];
    if((m.s.ablation==1&&driver>=2&&driver!=17)||(m.s.ablation>=10&&driver==m.s.ablation-10))return false;
    if(!next.ref.seq||next.ref.seq==s.seen[r]){
      if(s.draining&&s.length[r]){
        n=s.length[r];for(int i=0;i<n;i++)parts[i]=s.pending[r][i];s.length[r]=0;
        event=content_compose(m,r,parts,n,1,learn);return true;
      }
      s.metrics[r][5]++;return false;
    }
    s.seen[r]=next.ref.seq;s.metrics[r][0]++;
    int&length=s.length[r];bool boundary=false;
    if(length){
      const auto&last=s.pending[r][length-1];
      float score=content_transition(m,r,last.ref.id,next.ref.id,learn);
      boundary=score<z.boundary_threshold||last.ref.end!=next.ref.begin;
      // Familiar compounds can form a novel relation without pretending that
      // the relation was already recognized. Keep the proposal explicitly
      // provisional until the independent pattern dictionary has evidence.
      if(length==1&&last.count>=2&&next.count>=2&&last.ref.recognized&&next.ref.recognized&&last.ref.end==next.ref.begin)boundary=false;
    }
    if(boundary){n=length;for(int i=0;i<n;i++)parts[i]=s.pending[r][i];length=0;}
    s.pending[r][length++]=next;
    if(!n&&length>=int(z.max_children)){n=length;for(int i=0;i<n;i++)parts[i]=s.pending[r][i];length=0;}
    if(!n){s.metrics[r][5]++;return false;}
    event=content_compose(m,r,parts,n,1,learn);return true;
  }
  // Associative EC input is a stable fusion of content events. Y continues to
  // predict every byte, reading held content plus a separately credited context.
  bool fresh=false;
  for(int k=m.starts[r];k<m.starts[r+1];k++){
    const float*slot=m.slots+4*k;if(slot[2])continue;
    int src=int(slot[0]),age=int(slot[1]);const auto&child=s.history[src][age];
    if((m.s.ablation==1&&src>=2&&src!=17)||(m.s.ablation>=10&&src==m.s.ablation-10))continue;
    if(!child.ref.seq)continue;
    if(n>=COMP_MAX)break;
    parts[n]=child;parts[n].ref.role=k;
    fresh|=s.fusion_seen[r][n]!=child.ref.seq;s.fusion_seen[r][n]=child.ref.seq;n++;
  }
  if((r!=17||s.draining)&&(!n||!fresh)){s.metrics[r][5]++;return false;}
  if(n)event=(r==17&&!fresh)?s.history[r][0]:content_compose(m,r,parts,n,2,learn);
  return true;
}
inline void content_bundle(Model&m,int r,const ContentEvent&event,const ContentEvent*parts,int n,
                           float*q,float*c,float&qw,float&cw,LaceParent*parents,int&nparents){
  if(n){memcpy(q,event.code,D*sizeof(float));qw=1;}
  float weight=1.f/std::sqrt(float(std::max(1,n)));
  for(int i=0;i<n;i++){
    const auto&child=parts[i];
    float gain=m.regional_gain[child.ref.region];
    for(int j=0;j<D;j++)c[j]+=weight*gain*content_sign(j,i)*child.context[content_coord(j,i)];
    if(gain&&nparents<LACE_LINKS){const auto&origin=child.origin;
      parents[nparents++]={origin.region,-100-i,origin.tick,-weight*gain,origin.index};
      m.composition.stream.metrics[r][12]+=origin.region!=child.ref.region;
    }
  }
  cw=n?1.f:0.f;
}
