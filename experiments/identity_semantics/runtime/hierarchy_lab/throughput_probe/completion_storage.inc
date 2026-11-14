// Explicit readers/writers for composition ABI 9/10 and neural archive ABI 1.
// Checkpoint layout is independent of the compact in-memory eligibility.
struct LegacyInputCredit {
  int64_t tick=0,previous_tick=0;
  int previous_index=-1;
  float retention=0;
  SemanticRef semantic{};
  float symbol_features[8]{},symbol_derivative[D]{},symbol_scale=0;
  LegacyMessageInput message{};
  float gradient[LACE_DEPTH][D]{};
  int uses[LACE_DEPTH]{};
};
using CompositionCheckpoint=CompositionRuntimeLayout<LegacyInputCredit,ContentEventWire>;
using CompositionCheckpointStream=CompositionStreamLayout<LegacyInputCredit,ContentEventWire>;
static_assert(sizeof(LegacyMessageInput)==4368 && sizeof(LegacyInputCredit)==5488,
              "Historical completion checkpoint layout changed");
static_assert(sizeof(ContentEventWire)==1960,"Historical publication layout changed");
static_assert(offsetof(CompositionRuntime,stream)==offsetof(CompositionCheckpoint,stream));
static_assert(offsetof(CompositionStream,history)==offsetof(CompositionCheckpointStream,history));

template<class Target,class Source> inline void completion_copy_input(Target&to,const Source&from){
  to.tick=from.tick;to.previous_tick=from.previous_tick;
  to.previous_index=from.previous_index;to.retention=from.retention;
  to.semantic=from.semantic;to.symbol_scale=from.symbol_scale;
  to.message.feedback_count=from.message.feedback_count;
  memcpy(to.message.feedback,from.message.feedback,sizeof(to.message.feedback));
  memcpy(to.message.feedback_derivative,from.message.feedback_derivative,sizeof(to.message.feedback_derivative));
  memcpy(to.message.route_derivative,from.message.route_derivative,sizeof(to.message.route_derivative));
  to.message.qa=from.message.qa;to.message.alpha=from.message.alpha;
  copy_credit(to.gradient,from.gradient);memcpy(to.uses,from.uses,sizeof(to.uses));
}
template<class Target,class Source> inline void copy_publication(Target&to,const Source&from){
  to.ref=from.ref;to.origin=from.origin;to.count=from.count;to.kind=from.kind;
  to.confidence=from.confidence;to.local_arity=from.local_arity;to.relayed=from.relayed;
  to.present_mask=from.present_mask;to.fresh_mask=from.fresh_mask;to.semantic=from.semantic;
  to.clock_begin=from.clock_begin;to.clock_end=from.clock_end;
  auto&v=to.edit_values();const auto&s=from.values();
  memcpy(v.children,s.children,sizeof(v.children));memcpy(v.code,s.code,sizeof(v.code));
  memcpy(v.structure,s.structure,sizeof(v.structure));
  memcpy(v.symbol_features,s.symbol_features,sizeof(v.symbol_features));
  memcpy(v.symbol_derivative,s.symbol_derivative,sizeof(v.symbol_derivative));
  auto&c=to.edit_context();const auto&sc=from.context_data();
  memcpy(c.context,sc.context,sizeof(c.context));c.context_count=sc.context_count;
  memcpy(c.context_edges,sc.context_edges,sizeof(c.context_edges));
}
inline void completion_export_storage(const CompositionRuntime&source,void*data){
  auto&wire=*static_cast<CompositionCheckpoint*>(data);
  constexpr size_t prefix=offsetof(CompositionRuntime,stream)+offsetof(CompositionStream,history);
  memcpy(&wire,&source,prefix);
  for(int r=0;r<N;r++){
    for(int i=0;i<2;i++)copy_publication(wire.stream.history[r][i],source.stream.history[r][i]);
    for(int i=0;i<COMP_MAX;i++)copy_publication(wire.stream.pending[r][i],source.stream.pending[r][i]);
    for(int i=0;i<LACE_RING;i++){
      auto&input=wire.stream.input[r][i];memset(&input,0,sizeof(input));
      completion_copy_input(input,source.stream.input[r][i]);
    }
  }
  constexpr size_t middle=offsetof(CompositionStream,input)-offsetof(CompositionStream,length);
  static_assert(middle==offsetof(CompositionCheckpointStream,input)-offsetof(CompositionCheckpointStream,length));
  memcpy(&wire.stream.length,&source.stream.length,middle);
  constexpr size_t suffix=sizeof(CompositionStream)-offsetof(CompositionStream,metrics);
  static_assert(suffix==sizeof(CompositionCheckpointStream)-offsetof(CompositionCheckpointStream,metrics));
  memcpy(&wire.stream.metrics,&source.stream.metrics,suffix);
}
inline void completion_import_storage(CompositionRuntime&target,const CompositionCheckpoint&source){
  constexpr size_t prefix=offsetof(CompositionRuntime,stream)+offsetof(CompositionStream,history);
  memcpy(&target,&source,prefix);
  // Intern identical payloads from expanded old checkpoints. Compare bytes as
  // well as identity: equal structural IDs can carry different learned codes.
  std::vector<std::shared_ptr<PublicationPayload>> payloads;
  std::vector<std::shared_ptr<PublicationContext>> contexts;
  auto restore=[&](ContentEvent&to,const ContentEventWire&from){
    to=ContentEvent{};copy_publication(to,from);
    bool found=false;for(const auto&p:payloads)if(!memcmp(p.get(),to.payload.get(),sizeof(*p))){to.payload=p;found=true;break;}
    if(!found)payloads.push_back(to.payload);
    found=false;for(const auto&p:contexts)if(!memcmp(p.get(),to.contextual.get(),sizeof(*p))){to.contextual=p;found=true;break;}
    if(!found)contexts.push_back(to.contextual);
  };
  for(int r=0;r<N;r++){
    for(int i=0;i<2;i++)restore(target.stream.history[r][i],source.stream.history[r][i]);
    for(int i=0;i<COMP_MAX;i++)restore(target.stream.pending[r][i],source.stream.pending[r][i]);
    for(int i=0;i<LACE_RING;i++){
      auto&input=target.stream.input[r][i];input=InputCredit{};
      completion_copy_input(input,source.stream.input[r][i]);
    }
  }
  constexpr size_t middle=offsetof(CompositionStream,input)-offsetof(CompositionStream,length);
  memcpy(&target.stream.length,&source.stream.length,middle);
  constexpr size_t suffix=sizeof(CompositionStream)-offsetof(CompositionStream,metrics);
  memcpy(&target.stream.metrics,&source.stream.metrics,suffix);
}
