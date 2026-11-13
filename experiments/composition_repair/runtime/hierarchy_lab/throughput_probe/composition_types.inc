// Composition ABI v1: fixed-capacity learned dictionaries and causal events.
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
struct ContentEvent {
  ChildEvent ref{};
  ChildEvent origin{}; // Actual neural producer; identity relays keep this intact.
  int count=0,kind=0; // 0 atom, 1 sequential compound, 2 fusion, 3 unresolved singleton
  float confidence=0;
  ChildEvent children[COMP_MAX]{};
  float code[D]{},context[D]{};
};
struct InputCredit {
  int64_t tick=0,previous_tick=0;
  int previous_index=-1;
  float retention=0;
  float gradient[LACE_DEPTH][D]{};
  int uses[LACE_DEPTH]{};
};
struct CompositionStream {
  int64_t byte_position=0;
  uint32_t draining=0,reserved=0;
  ContentEvent history[N][2]{},pending[N][COMP_MAX]{};
  int length[N]{};
  int64_t seen[N]{},fusion_seen[N][COMP_MAX]{};
  int64_t memory_tick[N]{},event_serial[N]{};
  int memory_index[N]{};
  InputCredit input[N][LACE_RING]{};
  double metrics[N][COMP_METRICS]{};
};
struct CompositionRuntime {
  uint32_t version=2,enabled=1;
  float boundary_threshold=.35f,context_gain=.2f;
  uint32_t min_count=3,max_children=8;
  CountEntry totals[N][COMP_TABLE]{},pairs[N][COMP_TABLE]{},patterns[N][COMP_TABLE]{};
  CompositionStream stream{};
};
