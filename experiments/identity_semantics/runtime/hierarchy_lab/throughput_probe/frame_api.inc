// Frame policy v1 is immutable during a live stream. Recognition statistics
// belong to the representation: switching an already fitted dictionary is not
// a continuation conversion. Instantiate a fresh model for that comparison.
extern "C" {
int frame_configure(void*p,int enabled){
  auto&m=*static_cast<Model*>(p);
  if((enabled!=0&&enabled!=1)||m.composition.stream.byte_position||
     m.composition.stream.semantic_serial||m.content_credit_depth||
     (enabled&&(!m.composition.enabled||!m.content_experts)))return 0;
  for(int r=0;r<N;r++)if(m.s.valid[r])return 0;
  if(m.lightweight_frames!=bool(enabled)){
    for(int r=3;r<=14;r++)for(const auto&entry:m.composition.patterns[r])if(entry.count)return 0;
    if(m.pattern.serial)return 0;
  }
  m.lightweight_frames=enabled;return 1;
}
// External memory inventory only; no added training-loop observers/counters.
void frame_inspect(void*p,uint64_t*out){
  auto&m=*static_cast<Model*>(p);memset(out,0,N*6*sizeof(uint64_t));
  for(int i=0;i<constituent_capacity(m);i++){
    const auto&node=constituent_slot(m,i);if(!node.ref.serial)continue;
    auto*row=out+6*node.ref.region;
    row[0]+=node.encoder==2;row[1]+=node.encoder==1;
    if(node.encoder==2)row[2]+=node.count;
  }
  for(int r=0;r<N;r++){
    auto*row=out+6*r;row[3]=m.composition.stream.frame_serial[r];
    row[4]=m.composition.stream.history[r][0].ref.seq;
    row[5]=m.composition.stream.length[r];
  }
}
}
