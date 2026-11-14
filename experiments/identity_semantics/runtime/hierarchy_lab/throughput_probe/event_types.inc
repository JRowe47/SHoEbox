// Stage 5 learning policy: local exact eligibility plus directional interface
// credit. This is separate from the ancestry policy and its checkpoint ABI.
constexpr int MK_HEADS=2,MK_DIM=8,MK_FEATURES=MK_HEADS*MK_DIM;
struct ContextEligibility {
  int64_t tick=0;
  float before[MK_HEADS][MK_DIM][MK_DIM]{};
  float query_derivative[MK_HEADS][MK_DIM]{},read[MK_FEATURES]{};
};
struct EventKey {
  SemanticRef ref{};
  uint64_t ordinal=0;
  float feature[MK_DIM]{};
};
struct EventEligibility {
  EventKey key{},children[COMP_MAX]{};
};
struct EventCreditRequest {
  SemanticRef ref{};
  float gradient[D]{},rate=0,scales[N]{};
  int receiver=0,teaching=0;
  uint32_t reserved=0;
};
struct EventReplay {
  EventCreditRequest request{};
  float priority=0;
  uint32_t reserved=0;
};
struct EventParameters {
  float readout[N][D][MK_FEATURES]{},bias[N][D]{};
  // Independent local/task targets; never replace a vector teacher by reward.
  float credit[N][2][D][MK_DIM]{};
};
struct EventRuntime {
  int heads=0,local=0,frontier=4,replay_budget=2;
  float credit_lr=.05f,synthetic_scale=.1f,replay_scale=.25f;
  EventParameters parameters{};
  float memory[N][MK_HEADS][MK_DIM][MK_DIM]{};
  uint64_t ordinal[N]{};
  ContextEligibility context[N][LACE_RING]{};
  std::map<int64_t,ContextEligibility> archive[N];
  std::vector<EventEligibility> eligibility;
  EventReplay replay[N][2]{};
  // Observations are excluded from checkpoints, including per-region coverage.
  uint64_t metrics[N][16]{};
  double teacher_norm[N][2]{};
  std::vector<SemanticRef> new_events;
  bool initialized=false,training=false,local_only=false,synthetic=false;
  float rates[2]{};
};
