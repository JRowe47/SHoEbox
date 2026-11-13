// Composition ABI: v2 legacy, v3 role binding, v4 semantics, v5 relay preservation.
// Pattern identity is independent of compute expert, stream position and context.
constexpr int COMP_TABLE=2048, COMP_WAYS=4, COMP_MAX=8, COMP_METRICS=20;
struct ContentID { uint64_t a=0,b=0; };
inline bool operator==(ContentID x,ContentID y){return x.a==y.a&&x.b==y.b;}
struct CountEntry { ContentID key{}; uint32_t count=0,pad=0; };
struct ChildEvent {
  ContentID id{};
  int64_t seq=0,begin=0,end=0,tick=0;
  int region=-1,index=-1,role=0,recognized=0;
};
#ifdef SHOEBOX_LEARNED_COMPLETION
#include "completion_types.inc"
#endif
struct ContentEvent {
  ChildEvent ref{};
  ChildEvent origin{}; // Actual neural producer; identity relays keep this intact.
  int count=0,kind=0; // 0 atom, 1 sequence, 2 fusion, 3 legacy singleton, 4 role binding
  float confidence=0;
#ifdef SHOEBOX_PRESERVE_RELAYS
  // count/kind/children describe the content itself, including through relays.
  // These fields describe only what this receiving region did on this event.
  int local_arity=0,relayed=0;
#endif
#ifdef SHOEBOX_MULTI_BINDING
  uint32_t present_mask=0,fresh_mask=0;
#endif
  ChildEvent children[COMP_MAX]{};
  float code[D]{},context[D]{};
#ifdef SHOEBOX_LEARNED_COMPLETION
  SemanticRef semantic{};
  int64_t clock_begin=0,clock_end=0; // Primary occurrences only; ref spans cover all roles.
  int context_count=0;
  LaceParent context_edges[COMP_MAX]{};
#endif
#ifdef SHOEBOX_IDENTITY_SEMANTICS
  // Immutable descriptor, distinct from the learned semantic code and context.
  // This prevents stale child embeddings from changing a parent's symbol meaning.
  float structure[D]{},symbol_features[8]{},symbol_derivative[D]{};
#endif
};
#ifdef SHOEBOX_LEARNED_MESSAGES
#include "message_types.inc"
#endif
struct InputCredit {
  int64_t tick=0,previous_tick=0;
  int previous_index=-1;
  float retention=0;
#ifdef SHOEBOX_LEARNED_COMPLETION
  SemanticRef semantic{};
#endif
#ifdef SHOEBOX_IDENTITY_SEMANTICS
  float symbol_features[8]{},symbol_derivative[D]{},symbol_scale=0;
#endif
#ifdef SHOEBOX_LEARNED_MESSAGES
  MessageInput message{};
#endif
  float gradient[LACE_DEPTH][D]{};
  int uses[LACE_DEPTH]{};
};
#ifdef SHOEBOX_PRESERVE_RELAYS
struct PendingTeaching {
  // Relays never advance this region's eligibility ring. Consume at the next
  // content event, before a neural execution can overwrite the saved producer.
  int64_t tick=0;
  int index=-1;
  float prediction[D]{},input[D]{};
};
#endif
struct CompositionStream {
  int64_t byte_position=0;
  uint32_t draining=0,reserved=0;
#ifdef SHOEBOX_LEARNED_COMPLETION
  uint64_t semantic_serial=0,semantic_cursor=0,frame_serial[N]{};
  ConstituentRecord semantic_records[SEM_RING]{};
  // visited nodes, expired handles, work-budget omissions, completed sequences
  uint64_t completion_metrics[N][5]{}; // last: capacity prunes of live eligibility
#endif
#ifdef SHOEBOX_LEARNED_MESSAGES
  // Semantic deliveries, gradient norm, feedback deliveries, missing feedback.
  double message_metrics[N][4]{};
#endif
  ContentEvent history[N][2]{},pending[N][COMP_MAX]{};
  int length[N]{};
  int64_t seen[N]{},fusion_seen[N][COMP_MAX]{};
  int64_t memory_tick[N]{},event_serial[N]{};
  int memory_index[N]{};
  InputCredit input[N][LACE_RING]{};
  double metrics[N][COMP_METRICS]{};
#ifdef SHOEBOX_PRESERVE_RELAYS
  PendingTeaching teaching[N]{};
  // scheduled, neural targets, relay targets, nonlearning targets,
  // expired provenance, duplicate schedules (must stay zero).
  uint64_t teaching_metrics[N][6]{};
#endif
};
struct CompositionRuntime {
#ifdef SHOEBOX_LEARNED_COMPLETION
  uint32_t version=10,enabled=1; // Explicit content encoder tags; v9 live import is lossless.
#elif defined(SHOEBOX_LEARNED_MESSAGES)
  uint32_t version=6,enabled=1;
#elif defined(SHOEBOX_PRESERVE_RELAYS)
  uint32_t version=5,enabled=1;
#elif defined(SHOEBOX_IDENTITY_SEMANTICS)
  uint32_t version=4,enabled=1;
#elif defined(SHOEBOX_MULTI_BINDING)
  uint32_t version=3,enabled=1;
#else
  uint32_t version=2,enabled=1;
#endif
  float boundary_threshold=.35f,context_gain=.2f;
  uint32_t min_count=3,max_children=8;
  CountEntry totals[N][COMP_TABLE]{},pairs[N][COMP_TABLE]{},patterns[N][COMP_TABLE]{};
#ifdef SHOEBOX_IDENTITY_SEMANTICS
  // Eight fixed features per local symbol; 8,192 learned coefficients total.
  float symbols[N-2][D][8]{};
  uint64_t symbol_updates[N-2]{};
#endif
#ifdef SHOEBOX_LEARNED_COMPLETION
  float payload_gates[N-2]{}; // gain = 1 + 0.5*tanh(parameter)
#endif
  CompositionStream stream{};
};
