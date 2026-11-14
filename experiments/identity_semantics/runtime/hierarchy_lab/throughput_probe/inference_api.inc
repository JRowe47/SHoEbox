// Action 7. Explicit one-way eligibility boundary; resuming needs a context reset.
// Only publication-owned pattern/descriptor values survive the transition.
inline void inference_finish_training(Model&m){
  if(m.forward_only)return;
  prediction_slow_flush(m);reverse_finish_epoch(m);event_drain(m);
  reverse_finish_epoch(m);pipeline_drain(m);
}
inline void inference_release_derivatives(Model&m){
  auto&s=m.composition.stream;
  for(int r=0;r<N;r++){
    for(auto&v:m.laces.records[r])v=LaceRecord{};
    for(auto&v:s.input[r])v=InputCredit{};
    s.teaching[r]=PendingTeaching{};
    m.neural_archive[r].clear();m.event.archive[r].clear();
    for(auto&v:m.event.context[r]){v.clear();std::pmr::vector<float> empty{&record_pool()};v.data.swap(empty);}
  }
  for(auto&v:m.neural_temporal)v.clear();
  m.neural_record_pool.release();neural_invalidate(m);
  m.neural_retention=NeuralRetention{};m.semantic_retention=SemanticRetention{};
  s.semantic_records.release();m.semantic_mark.release();m.semantic_gradient.release();
  m.semantic_overflow={};m.semantic_overflow.shrink_to_fit();
  m.semantic_overflow_mark={};m.semantic_overflow_mark.shrink_to_fit();
  m.semantic_overflow_gradient={};m.semantic_overflow_gradient.shrink_to_fit();s.reserved=0;
  m.content_records={};m.content_records.shrink_to_fit();
  m.content_input_cache={};m.content_input_cache.shrink_to_fit();
  m.pattern.assignments={};m.pattern.assignments.shrink_to_fit();
  m.event.eligibility={};m.event.eligibility.shrink_to_fit();
  m.event.new_events={};m.event.new_events.shrink_to_fit();
  m.event.workspace.clear();std::pmr::vector<float> empty{&record_pool()};m.event.workspace.data.swap(empty);
  memset(m.event.replay,0,sizeof(m.event.replay));m.event.training=false;
  memset(m.learned.records,0,sizeof(m.learned.records));
  memset(m.learned.eligibility_a,0,sizeof(m.learned.eligibility_a));
  memset(m.learned.eligibility_b,0,sizeof(m.learned.eligibility_b));
  m.local_credit_work=LocalCredit{};m.reverse_execution=ReverseExecution{};
  m.reverse_roots={};m.reverse_roots.shrink_to_fit();m.reverse_work={};m.reverse_work.shrink_to_fit();
  m.reverse_bank_version={};m.reverse_bank_version.shrink_to_fit();
  for(auto&bank:m.content_banks)bank.reset();m.content_dirty={};m.content_dirty.shrink_to_fit();
  m.content_workers.reset();m.bank_pipeline.wait();m.bank_pipeline.worker.reset();m.bank_pipeline.workers=1;
  m.bank_pipeline.building=std::make_unique<PipelineBatch>();m.bank_pipeline.spare=std::make_unique<PipelineBatch>();
  m.bank_pipeline.visible={};m.bank_pipeline.visible.shrink_to_fit();
  constituent_invalidate(m);m.semantic_free.shrink_to_fit();
  m.prediction.stream.samples=0;m.prediction.stream.rate=0;m.prediction.stream.mode=0;
  memset(m.prediction.stream.gradient,0,sizeof(m.prediction.stream.gradient));
  memset(m.prediction.stream.current,0,sizeof(m.prediction.stream.current));
  memset(m.prediction.stream.pending,0,sizeof(m.prediction.stream.pending));
  memset(m.s.pending,0,sizeof(m.s.pending));memset(m.s.credit,0,sizeof(m.s.credit));m.s.credit_count=0;
}
struct InferenceWire {
  uint64_t version=1,experts=E,policy=1,reserved=0;
  State state{};
  int64_t block_before=0;int utility_ablation=0;
  float reference[2][D]{};
  ProbeRuntime probe{};UtilityRuntime utility{};
  float bottom_memory[2][D]{};
  int cursor[N]{};
  int64_t byte_position=0;
  uint64_t semantic_serial=0,semantic_cursor=0,frame_serial[N]{};
  ContentEventWire history[N][2]{},pending[N][COMP_MAX]{};
  ForwardPublication forward[N][COMP_MAX+2]{};
  uint64_t present[N][COMP_MAX+2]{};
  int length[N]{};
  int64_t seen[N]{},fusion_seen[N][COMP_MAX]{},memory_tick[N]{},event_serial[N]{};
  int memory_index[N]{};
  float context_memory[N][MK_HEADS][MK_DIM][MK_DIM]{};
  int64_t slow_tick=0,slow_byte=0;
  float slow[D]{},output[D]{},feedback[D]{};
};
extern "C" {
int inference_finish_boundary(void*p){
  auto&m=*static_cast<Model*>(p);if(!m.composition.enabled||m.content_credit_depth||m.composition.stream.draining)return 0;
  inference_finish_training(m);return 1;
}
int inference_enter(void*p){
  auto&m=*static_cast<Model*>(p);if(m.forward_only)return 1;
  if(!m.composition.enabled||m.content_credit_depth||m.composition.stream.draining)return 0;
  inference_finish_training(m);
  auto retain=[&](ContentEvent&e){
    if(!e.semantic.serial)return;
    auto&v=e.edit_forward();
    if(auto*node=constituent_record(m,e.semantic)){
      v.encoder=node->encoder;
      if(node->encoder==2)memcpy(v.frame_payload,node->payload,sizeof(v.frame_payload));
    }
    if(auto*content=content_expert_record(m,e.semantic))v.expert=content->expert;
    if(auto*assignment=pattern_assignment(m,e)){v.pattern=*assignment;v.has_pattern=true;}
  };
  for(int r=0;r<N;r++){
    for(auto&e:m.composition.stream.history[r])retain(e);
    for(auto&e:m.composition.stream.pending[r])retain(e);
  }
  inference_release_derivatives(m);m.forward_only=true;return 1;
}
int inference_resume(void*p,int reset_stream){
  auto&m=*static_cast<Model*>(p);if(!m.forward_only)return 1;if(reset_stream!=1)return 0;
  reset_context(p);initialize_learned(m);
  m.bank_pipeline.bytes=m.bank_pipeline.version=m.bank_pipeline.sequence=0;
  m.bank_pipeline.position=0;m.reverse_version=0;
  m.forward_only=false;return 1;
}
size_t inference_state_size(){return sizeof(InferenceWire);}
int inference_export(void*p,void*out){
  auto&m=*static_cast<Model*>(p);if(!m.forward_only)return 0;
  auto&w=*static_cast<InferenceWire*>(out);memset(&w,0,sizeof(w));w.version=1;w.experts=E;w.policy=1;
  w.state=m.s;w.probe=m.probe;w.utility=m.utility;w.block_before=m.block_before;w.utility_ablation=m.utility_ablation;
  memcpy(w.reference,m.reference,sizeof(w.reference));
  memcpy(w.bottom_memory,m.learned.memory,sizeof(w.bottom_memory));memcpy(w.cursor,m.laces.cursor,sizeof(w.cursor));
  const auto&s=m.composition.stream;
  w.byte_position=s.byte_position;w.semantic_serial=s.semantic_serial;w.semantic_cursor=s.semantic_cursor;
  memcpy(w.frame_serial,s.frame_serial,sizeof(w.frame_serial));
  for(int r=0;r<N;r++){
    auto put=[&](ContentEventWire&to,const ContentEvent&from,int k){
      copy_publication(to,from);w.present[r][k]=bool(from.forward);
      if(from.forward)w.forward[r][k]=*from.forward;
    };
    for(int i=0;i<2;i++)put(w.history[r][i],s.history[r][i],i);
    for(int i=0;i<COMP_MAX;i++)put(w.pending[r][i],s.pending[r][i],i+2);
  }
#define COPY_FIELD(field) memcpy(w.field,s.field,sizeof(w.field))
  COPY_FIELD(length);COPY_FIELD(seen);COPY_FIELD(fusion_seen);COPY_FIELD(memory_tick);COPY_FIELD(event_serial);COPY_FIELD(memory_index);
#undef COPY_FIELD
  memcpy(w.context_memory,m.event.memory,sizeof(w.context_memory));
  const auto&y=m.prediction.stream;w.slow_tick=y.slow_tick;w.slow_byte=y.slow_byte;
  memcpy(w.slow,y.slow,sizeof(w.slow));memcpy(w.output,y.output,sizeof(w.output));memcpy(w.feedback,y.feedback,sizeof(w.feedback));return 1;
}
int inference_import(void*p,const void*input,size_t size){
  auto&m=*static_cast<Model*>(p);if(!m.forward_only||size!=sizeof(InferenceWire))return 0;
  const auto&w=*static_cast<const InferenceWire*>(input);
  if(w.version!=1||w.experts!=E||w.policy!=1||w.reserved||w.utility_ablation<0||w.utility_ablation>1||w.state.tick<0||w.byte_position<0||w.byte_position>w.state.tick||
     w.slow_tick<0||w.slow_tick>w.state.tick||w.slow_byte<0||w.slow_byte>w.byte_position)return 0;
  auto finite=[](const float*a,size_t n){for(size_t i=0;i<n;i++)if(!std::isfinite(a[i]))return false;return true;};
  if(!finite(&w.context_memory[0][0][0][0],N*MK_HEADS*MK_DIM*MK_DIM)||!finite(&w.bottom_memory[0][0],2*D)||
     !finite(w.slow,D)||!finite(w.output,D)||!finite(w.feedback,D))return 0;
  for(int r=0;r<N;r++){
    if(w.length[r]<0||w.length[r]>COMP_MAX||w.cursor[r]<0||w.cursor[r]>=LACE_RING)return 0;
    for(int k=0;k<COMP_MAX+2;k++){
      const auto&e=k<2?w.history[r][k]:w.pending[r][k-2];const auto&v=w.forward[r][k];
      if(w.present[r][k]>1||e.count<0||e.count>COMP_MAX||e.context_count<0||e.context_count>COMP_MAX||
         e.semantic.serial>w.semantic_serial||e.kind<0||e.kind>4||e.ref.tick>w.state.tick||
         !finite(e.code,D)||!finite(e.structure,D)||!finite(e.context,D)||!std::isfinite(e.confidence))return 0;
      if(e.semantic.serial&&(!w.present[r][k]||e.semantic.region<2||e.semantic.region>=N))return 0;
      if(w.present[r][k]&&(v.encoder<0||v.encoder>2||v.expert<0||v.expert>=E||v.has_pattern>1||
         !finite(v.frame_payload,D)||!finite(v.pattern.prediction,D)||
         (v.has_pattern&&(v.pattern.ref.serial!=e.semantic.serial||v.pattern.ref.region!=e.semantic.region||v.pattern.id>m.pattern.serial))))return 0;
    }
  }
  // Validate all fields before changing the live stream.
  reset_composition_stream(m.composition.stream);auto&s=m.composition.stream;
  m.s=w.state;m.probe=w.probe;m.utility=w.utility;m.block_before=w.block_before;m.utility_ablation=w.utility_ablation;
  memcpy(m.reference,w.reference,sizeof(w.reference));
  memcpy(m.learned.memory,w.bottom_memory,sizeof(w.bottom_memory));memcpy(m.laces.cursor,w.cursor,sizeof(w.cursor));
  s.byte_position=w.byte_position;s.semantic_serial=w.semantic_serial;s.semantic_cursor=w.semantic_cursor;
  memcpy(s.frame_serial,w.frame_serial,sizeof(w.frame_serial));
  for(int r=0;r<N;r++){
    auto get=[&](ContentEvent&to,const ContentEventWire&from,int k){
      copy_publication(to,from);if(w.present[r][k])to.forward=publication_allocate<ForwardPublication>(w.forward[r][k]);
    };
    for(int i=0;i<2;i++)get(s.history[r][i],w.history[r][i],i);
    for(int i=0;i<COMP_MAX;i++)get(s.pending[r][i],w.pending[r][i],i+2);
  }
#define COPY_FIELD(field) memcpy(s.field,w.field,sizeof(w.field))
  COPY_FIELD(length);COPY_FIELD(seen);COPY_FIELD(fusion_seen);COPY_FIELD(memory_tick);COPY_FIELD(event_serial);COPY_FIELD(memory_index);
#undef COPY_FIELD
  memcpy(m.event.memory,w.context_memory,sizeof(w.context_memory));
  auto&y=m.prediction.stream;y.slow_tick=w.slow_tick;y.slow_byte=w.slow_byte;
  memcpy(y.slow,w.slow,sizeof(w.slow));memcpy(y.output,w.output,sizeof(w.output));memcpy(y.feedback,w.feedback,sizeof(w.feedback));
  constituent_invalidate(m);m.refresh_keys();m.cached_read_valid[0]=m.cached_read_valid[1]=false;return 1;
}
// Inventory is derived on demand, never collected on the byte path.
void inference_inspect(void*p,uint64_t*out){
  auto&m=*static_cast<Model*>(p);memset(out,0,16*sizeof(uint64_t));out[0]=m.forward_only;out[1]=sizeof(Model);
  out[2]=m.composition.stream.semantic_records.bytes()+m.semantic_mark.bytes()+m.semantic_gradient.bytes();
  out[3]=m.content_records.capacity()*sizeof(ContentExpertRecord)+m.event.eligibility.capacity()*sizeof(EventEligibility)+m.pattern.assignments.capacity()*sizeof(PatternAssignment);
  for(int r=0;r<N;r++){
    out[4]+=m.neural_archive[r].size();out[5]+=m.event.archive[r].size();
    for(const auto&v:m.laces.records[r])out[6]+=v.tick!=0;
    for(const auto&v:m.event.context[r])out[7]+=v.tick!=0;
    for(const auto&e:m.composition.stream.history[r])out[8]+=bool(e.forward);
    for(const auto&e:m.composition.stream.pending[r])out[8]+=bool(e.forward);
  }
  for(const auto&v:m.neural_temporal)out[9]+=v.size();
  out[10]=m.reverse_roots.size();out[11]=m.bank_pipeline.pending?1:0;
  out[12]=m.bank_pipeline.building->scope_count;out[13]=m.prediction.stream.samples;
  out[14]=sizeof(InferenceWire);out[15]=m.neural_record_pool.reserved_bytes();
}
}
