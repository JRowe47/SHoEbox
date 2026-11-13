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
using CompositionCheckpoint=CompositionRuntimeLayout<LegacyInputCredit>;
using CompositionCheckpointStream=CompositionStreamLayout<LegacyInputCredit>;
static_assert(sizeof(LegacyMessageInput)==4368 && sizeof(LegacyInputCredit)==5488,
              "Historical completion checkpoint layout changed");
static_assert(offsetof(CompositionRuntime,stream)==offsetof(CompositionCheckpoint,stream));
static_assert(offsetof(CompositionStream,input)==offsetof(CompositionCheckpointStream,input));
static_assert(sizeof(CompositionStream)-offsetof(CompositionStream,metrics)==
              sizeof(CompositionCheckpointStream)-offsetof(CompositionCheckpointStream,metrics));

template<class Target,class Source> inline void completion_copy_input(Target&to,const Source&from){
  to.tick=from.tick;to.previous_tick=from.previous_tick;
  to.previous_index=from.previous_index;to.retention=from.retention;
  to.semantic=from.semantic;to.symbol_scale=from.symbol_scale;
  to.message.feedback_count=from.message.feedback_count;
  memcpy(to.message.feedback,from.message.feedback,sizeof(to.message.feedback));
  memcpy(to.message.feedback_derivative,from.message.feedback_derivative,sizeof(to.message.feedback_derivative));
  memcpy(to.message.route_derivative,from.message.route_derivative,sizeof(to.message.route_derivative));
  to.message.qa=from.message.qa;to.message.alpha=from.message.alpha;
  memcpy(to.gradient,from.gradient,sizeof(to.gradient));
  memcpy(to.uses,from.uses,sizeof(to.uses));
}
inline void completion_export_storage(const CompositionRuntime&source,void*data){
  auto*out=static_cast<unsigned char*>(data);
  constexpr size_t prefix=offsetof(CompositionRuntime,stream)+offsetof(CompositionStream,input);
  memcpy(out,&source,prefix);
  for(int r=0;r<N;r++)for(int i=0;i<LACE_RING;i++){
    LegacyInputCredit wire{};
    // Canonical zero values for fields that completion never consumes.
    memset(&wire,0,sizeof(wire));completion_copy_input(wire,source.stream.input[r][i]);
    memcpy(out+prefix+(r*LACE_RING+i)*sizeof(wire),&wire,sizeof(wire));
  }
  constexpr size_t suffix=sizeof(CompositionStream)-offsetof(CompositionStream,metrics);
  memcpy(out+offsetof(CompositionCheckpoint,stream)+offsetof(CompositionCheckpointStream,metrics),
         &source.stream.metrics,suffix);
}
inline void completion_import_storage(CompositionRuntime&target,const CompositionCheckpoint&source){
  constexpr size_t prefix=offsetof(CompositionRuntime,stream)+offsetof(CompositionStream,input);
  memcpy(&target,&source,prefix);
  for(int r=0;r<N;r++)for(int i=0;i<LACE_RING;i++){
    auto&input=target.stream.input[r][i];input=InputCredit{};
    completion_copy_input(input,source.stream.input[r][i]);
  }
  constexpr size_t suffix=sizeof(CompositionStream)-offsetof(CompositionStream,metrics);
  memcpy(&target.stream.metrics,&source.stream.metrics,suffix);
}
