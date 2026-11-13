// Class descriptors contain learned child content in separate role slots.
// Primary occurrences anchor geometry; held all-source coverage cannot move
// that anchor. Side roles remain in the schema, prototypes and exact ledger.
inline void pattern_normalize(const float*input,float*out){
  float norm=1e-12f;for(int j=0;j<D;j++)norm+=input[j]*input[j];
  norm=1.f/std::sqrt(norm);for(int j=0;j<D;j++)out[j]=input[j]*norm;
}
inline PatternDescriptor pattern_descriptor(Model&m,int r,const ContentEvent&event,
                                             const ContentEvent*parts,int n,bool binding){
  PatternDescriptor x{};x.shape.kind=event.kind;x.shape.count=n;x.shape.mask=event.present_mask;
  int driver=binding?int(m.slots[4*m.starts[r]]):-1;
  int64_t begin=event.clock_begin,end=event.clock_end;
  float duration=float(std::max<int64_t>(1,end-begin));
  for(int i=0;i<n;i++){
    const auto&child=parts[i];int role=binding?child.ref.role-m.starts[r]:i;
    x.shape.roles[i]=role;x.shape.sources[i]=child.ref.region;
    int repeats=0;for(int k=0;k<n;k++)repeats+=parts[k].ref.region==child.ref.region;
    bool core=!binding||child.ref.region==driver||repeats>1;
    if(core)x.shape.core|=1u<<i;
    // Discrete relations protect reversal, overlap and gaps between primary
    // occurrences. Held-context age/duration are soft, bounded features.
    if(core&&i){
      int previous=i-1;
      while(previous>=0&&!(x.shape.core&(1u<<previous)))previous--;
      if(previous>=0){
        const auto&prior=parts[previous];
        x.shape.relations[i]=child.clock_begin>=prior.clock_end?1:
          child.clock_end<=prior.clock_begin?-1:2;
      }
    }
    pattern_normalize(child.code,x.content[i]);
    x.geometry[i][0]=std::tanh(float(child.clock_begin-begin)/duration);
    x.geometry[i][1]=std::tanh(float(child.clock_end-begin)/duration);
    x.geometry[i][2]=std::tanh(std::log1p(float(std::max<int64_t>(0,child.clock_end-child.clock_begin))/duration));
  }
  return x;
}
inline float pattern_distance(const PatternDescriptor&a,const PatternDescriptor&b){
  if(memcmp(&a.shape,&b.shape,sizeof(PatternShape)))return 1e9f;
  float core=0,side=0,geometry=0;int nc=0,ns=0;
  for(unsigned i=0;i<a.shape.count;i++){
    float distance=0;for(int j=0;j<D;j++){float d=a.content[i][j]-b.content[i][j];distance+=.5f*d*d;}
    if(a.shape.core&(1u<<i)){
      // A near match in other roles cannot hide a changed primary constituent.
      if(distance>.35f)return 1e9f;
      core+=distance;nc++;
    }else{side+=distance;ns++;}
    for(int k=0;k<3;k++){float d=a.geometry[i][k]-b.geometry[i][k];geometry+=(a.shape.core&(1u<<i)?1.f:.1f)*d*d;}
  }
  return core/std::max(1,nc)+.1f*side/std::max(1,ns)+.025f*geometry/std::max(1u,a.shape.count);
}
inline PatternAssignment* pattern_assignment(Model&m,const ContentEvent&e){
  auto ref=e.semantic;
  if(!ref.serial||ref.index<0||size_t(ref.index)>=m.pattern.assignments.size())return nullptr;
  auto&saved=m.pattern.assignments[ref.index];
  return saved.ref.serial==ref.serial&&saved.ref.region==ref.region?&saved:nullptr;
}
inline ContentID pattern_key(Model&m,const ContentEvent&e){
  // Atom classes are byte identities; transparent relays keep the same class.
  if(!e.semantic.serial&&e.kind==0)return e.ref.id;
  auto*saved=pattern_assignment(m,e);
  return saved&&saved->id?ContentID{saved->id,PATTERN_TAG+uint64_t(saved->ref.region)}:ContentID{};
}
inline PatternClass* pattern_find(Model&m,int r,uint64_t id){
  for(auto&entry:m.pattern.classes[r])if(entry.id==id)return &entry;
  return nullptr;
}
inline void pattern_classify(Model&m,int r,ContentEvent&out,const ContentEvent*parts,int n,bool binding,bool learn){
  auto&z=m.pattern;if(!z.enabled)return;
  auto&metric=z.metrics[r];metric[0]++;
  learn=learn&&!m.frozen_region[r];
  PatternDescriptor x=pattern_descriptor(m,r,out,parts,n,binding);
  PatternClass*match=nullptr;float best=.22f;
  for(auto&entry:z.classes[r]){
    float d=pattern_distance(x,entry.prototype);
    if(d<best){best=d;match=&entry;}
  }
  if(learn)z.clock[r]++;
  if(!match&&learn){
    if(z.classes[r].size()<PATTERN_CAP){z.classes[r].emplace_back();match=&z.classes[r].back();}
    else{
      match=&*std::min_element(z.classes[r].begin(),z.classes[r].end(),[](const auto&a,const auto&b){return a.last_used<b.last_used;});
      metric[6]++;
    }
    *match=PatternClass{};match->id=++z.serial;match->prototype=x;metric[1]++;
  }else if(match)metric[2]++;
  else metric[7]++;
  if(match&&learn){
    uint64_t prior=match->count;match->count++;match->last_used=z.clock[r];
    if(prior&&match->count==m.composition.min_count)metric[3]++;
    // Bounded online adaptation follows slowly changing child encoders. Class
    // serials stay stable as prototypes move; eviction never reuses a serial.
    float rate=1.f/float(std::min<uint64_t>(match->count,64));
    for(int i=0;i<n;i++){
      for(int j=0;j<D;j++)match->prototype.content[i][j]+=rate*(x.content[i][j]-match->prototype.content[i][j]);
      for(int k=0;k<3;k++)match->prototype.geometry[i][k]+=rate*(x.geometry[i][k]-match->prototype.geometry[i][k]);
    }
  }
  out.ref.recognized=match&&match->count>=m.composition.min_count;
  out.confidence=match?std::min(1.f,float(match->count)/m.composition.min_count):0.f;
  metric[out.ref.recognized?4:5]++;
  if(out.semantic.serial){
    if(size_t(out.semantic.index)>=z.assignments.size())z.assignments.resize(constituent_capacity(m));
    auto&saved=z.assignments[out.semantic.index];saved=PatternAssignment{};
    saved.ref=out.semantic;saved.id=match?match->id:0;
    float cue[D]{},exported[D];pattern_normalize(out.code,exported);
    int core=__builtin_popcount(x.shape.core),side=n-core;
    for(int i=0;i<n;i++){
      float weight=x.shape.core&(1u<<i)?1.f/std::sqrt(float(std::max(1,core))):.1f/std::sqrt(float(std::max(1,side)));
      for(int j=0;j<D;j++){
        int coord=binding?m.perm[parts[i].ref.role*D+j]:(j+11*i+7)&63;
        float sign=binding?m.signs[parts[i].ref.role*D+j]:((j*13+i*17)%11<5?-1.f:1.f);
        cue[j]+=weight*sign*x.content[i][coord];
      }
    }
    for(int j=0;j<D;j++)cue[j]+=.15f*exported[j];
    pattern_normalize(cue,saved.prediction);
  }
}
inline float pattern_transition(Model&m,int r,const ContentEvent&from,const ContentEvent&to,bool learn){
  auto&z=m.pattern;auto&metric=z.metrics[r];metric[8]++;
  ContentID a=pattern_key(m,from),b=pattern_key(m,to);
  if(!a.a&&!a.b)return 0;
  double evictions=0;
  ContentID pair=content_pair(a,b,0x636c617373706169ULL);
  uint32_t total=content_count(z.totals[r],a,false,evictions);
  uint32_t count=(b.a||b.b)?content_count(z.pairs[r],pair,false,evictions):0;
  float categorical=count>=m.composition.min_count?std::min(1.f,float(count)/std::max(1u,total)):0.f;
  metric[9]+=categorical>0;
  float prediction=0,next[D];
  if(auto*saved=pattern_assignment(m,to))memcpy(next,saved->prediction,sizeof(next));
  else pattern_normalize(to.code,next);
  PatternClass*entry=a.b==PATTERN_TAG+uint64_t(r)?pattern_find(m,r,a.a):nullptr;
  if(entry&&entry->next_count>=m.composition.min_count){
    float distance=0;for(int j=0;j<D;j++){float d=next[j]-entry->next[j];distance+=.5f*d*d;}
    // A diffuse/multimodal forecast receives low confidence without renormalizing
    // its mean. Evidence is measured before the current target is learned.
    prediction=std::max(0.f,1.f-distance/.25f);metric[10]++;metric[11]+=prediction>.7f;
  }
  float score=entry?.5f*(categorical+prediction):categorical;
  if(learn&&!m.frozen_region[r]){
    content_count(z.totals[r],a,true,evictions);
    if(b.a||b.b)content_count(z.pairs[r],pair,true,evictions);
    if(entry){
      entry->next_count++;float rate=1.f/float(std::min<uint64_t>(entry->next_count,64));
      for(int j=0;j<D;j++)entry->next[j]+=rate*(next[j]-entry->next[j]);
    }
    metric[15]+=uint64_t(evictions);
  }
  return score;
}
inline void pattern_reset_stream(Model&m){
  m.pattern.assignments.clear();memset(m.pattern.metrics,0,sizeof(m.pattern.metrics));
}
