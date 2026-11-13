struct PatternHeader {
  uint64_t version=1,serial=0,assignments=0;
  uint64_t counts[N]{},clock[N]{};
};
extern "C" {
int pattern_configure(void*p,int enabled){
  auto&m=*static_cast<Model*>(p);
  if((enabled!=0&&enabled!=1)||m.composition.stream.semantic_serial||m.composition.stream.byte_position)return 0;
  m.pattern.enabled=enabled;return 1;
}
void pattern_status(void*p,uint64_t*out){
  auto&z=static_cast<Model*>(p)->pattern;
  for(int r=0;r<N;r++){
    memcpy(out+r*(PATTERN_METRICS+2),z.metrics[r],sizeof(z.metrics[r]));
    out[r*(PATTERN_METRICS+2)+PATTERN_METRICS]=z.classes[r].size();
    uint64_t count=0;for(const auto&entry:z.classes[r])count+=entry.count;
    out[r*(PATTERN_METRICS+2)+PATTERN_METRICS+1]=count;
  }
}
void pattern_events(void*p,uint64_t*out){
  auto&m=*static_cast<Model*>(p);
  for(int r=0;r<N;r++){
    auto key=pattern_key(m,m.composition.stream.history[r][0]);out[2*r]=key.a;out[2*r+1]=key.b;
  }
}
size_t pattern_state_size(void*p,int fitted){
  auto&z=static_cast<Model*>(p)->pattern;size_t count=0,assignments=0;
  for(auto&classes:z.classes)count+=classes.size();
  if(!fitted)for(const auto&entry:z.assignments)assignments+=entry.ref.serial!=0;
  return sizeof(PatternHeader)+sizeof(z.totals)+sizeof(z.pairs)+count*sizeof(PatternClass)+assignments*sizeof(PatternAssignment);
}
void pattern_export(void*p,void*out,int fitted){
  auto&z=static_cast<Model*>(p)->pattern;PatternHeader header{};header.serial=z.serial;
  memcpy(header.clock,z.clock,sizeof(header.clock));
  for(int r=0;r<N;r++)header.counts[r]=z.classes[r].size();
  if(!fitted)for(const auto&entry:z.assignments)header.assignments+=entry.ref.serial!=0;
  auto*cursor=static_cast<unsigned char*>(out);
  auto write=[&](const void*source,size_t size){if(size)memcpy(cursor,source,size);cursor+=size;};
  write(&header,sizeof(header));write(z.totals,sizeof(z.totals));write(z.pairs,sizeof(z.pairs));
  for(auto&classes:z.classes)write(classes.data(),classes.size()*sizeof(PatternClass));
  if(!fitted)for(const auto&entry:z.assignments)if(entry.ref.serial)write(&entry,sizeof(entry));
}
int pattern_import(void*p,const void*data,size_t size){
  auto&m=*static_cast<Model*>(p);PatternHeader header{};
  size_t base=sizeof(header)+sizeof(m.pattern.totals)+sizeof(m.pattern.pairs);
  if(size<base)return 0;memcpy(&header,data,sizeof(header));
  if(header.version!=1||header.assignments>uint64_t(constituent_capacity(m)))return 0;
  size_t expected=base+header.assignments*sizeof(PatternAssignment);
  for(auto count:header.counts){if(count>PATTERN_CAP)return 0;expected+=count*sizeof(PatternClass);}
  if(expected!=size)return 0;
  auto next=std::make_unique<PatternRuntime>();next->enabled=m.pattern.enabled;next->serial=header.serial;
  memcpy(next->clock,header.clock,sizeof(header.clock));
  auto*cursor=static_cast<const unsigned char*>(data)+sizeof(header);
  auto read=[&](void*target,size_t bytes){if(bytes)memcpy(target,cursor,bytes);cursor+=bytes;};
  read(next->totals,sizeof(next->totals));read(next->pairs,sizeof(next->pairs));
  std::map<uint64_t,int> ids;
  for(int r=0;r<N;r++){
    next->classes[r].resize(header.counts[r]);read(next->classes[r].data(),header.counts[r]*sizeof(PatternClass));
    for(const auto&entry:next->classes[r]){
      const auto&shape=entry.prototype.shape;
      if(!entry.id||entry.id>header.serial||!entry.count||entry.last_used>header.clock[r]||
         !ids.emplace(entry.id,r).second||shape.count>COMP_MAX||
         (shape.kind!=1&&shape.kind!=2&&shape.kind!=4)||shape.core>>shape.count)return 0;
      for(unsigned i=0;i<shape.count;i++){
        if(shape.roles[i]<0||shape.roles[i]>=COMP_MAX||shape.sources[i]<0||shape.sources[i]>=N||
           shape.relations[i]<-1||shape.relations[i]>2)return 0;
        for(float value:entry.prototype.content[i])if(!std::isfinite(value))return 0;
        for(float value:entry.prototype.geometry[i])if(!std::isfinite(value))return 0;
      }
      for(float value:entry.next)if(!std::isfinite(value))return 0;
    }
  }
  if(header.assignments)next->assignments.resize(constituent_capacity(m));
  int previous=-1;
  for(uint64_t i=0;i<header.assignments;i++){
    PatternAssignment entry{};read(&entry,sizeof(entry));auto*node=constituent_record(m,entry.ref);
    if(!node||node->ref.region!=entry.ref.region||entry.ref.index<=previous||entry.id>header.serial)return 0;
    for(float value:entry.prediction)if(!std::isfinite(value))return 0;
    auto found=ids.find(entry.id);if(found!=ids.end()&&found->second!=entry.ref.region)return 0;
    previous=entry.ref.index;next->assignments[entry.ref.index]=entry;
  }
  m.pattern=std::move(*next);return 1;
}
}
