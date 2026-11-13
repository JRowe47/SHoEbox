// Reusable classes are learned prototypes, never expert numbers or exact IDs.
// Keep this sidecar separate from composition ABI 10 for lossless old imports.
constexpr int PATTERN_CAP=256, PATTERN_METRICS=16;
constexpr uint64_t PATTERN_TAG=0x636c617373000000ULL;
struct PatternShape {
  uint32_t kind=0,count=0,mask=0,core=0;
  int roles[COMP_MAX]{},sources[COMP_MAX]{},relations[COMP_MAX]{};
};
struct PatternDescriptor {
  PatternShape shape{};
  float content[COMP_MAX][D]{},geometry[COMP_MAX][3]{};
};
struct PatternClass {
  uint64_t id=0,count=0,last_used=0,next_count=0;
  PatternDescriptor prototype{};
  float next[D]{};
};
struct PatternAssignment {
  SemanticRef ref{};uint64_t id=0;
  float prediction[D]{}; // Saved role-aware target, independent of recurrent context.
};
struct PatternRuntime {
  bool enabled=true;
  uint64_t serial=0,clock[N]{};
  std::vector<PatternClass> classes[N];
  std::vector<PatternAssignment> assignments;
  CountEntry totals[N][COMP_TABLE]{},pairs[N][COMP_TABLE]{};
  // observed, created, reused, promoted, recognized, provisional, evicted,
  // unknown_eval, transitions, class_hits, forecasts, forecast_hits,
  // sequence_publications, primary_span_sum, sequence_frames, table_evictions.
  uint64_t metrics[N][PATTERN_METRICS]{};
};
