// Separate semantic eligibility from the much faster neural execution ring.
// Handles are generation checked. Relays copy a handle, never reassign credit.
// Initial slab; live ancestry can extend into the checkpointed overflow pool.
constexpr int SEM_RING=16384;
struct SemanticRef { uint64_t serial=0; int region=0,index=0; };
struct ConstituentEdge {
  SemanticRef source{};
  ChildEvent occurrence{};
  int slot=0; float weight=0;
};
struct ConstituentRecord {
  SemanticRef ref{};
  int count=0;
  uint32_t encoder=0; // Occupies v9 padding; 1 requires saved content eligibility.
  ConstituentEdge edges[COMP_MAX]{};
  float features[8]{},derivative[D]{},payload[D]{},payload_derivative[D]{};
  float gain_derivative=0;
};
// Preserve the byte layout for explicit, lossless v9 live-state imports. Never
// interpret the old padding as an encoder flag; the importer clears it.
struct ConstituentRecordV9 {
  SemanticRef ref{};int count=0;ConstituentEdge edges[COMP_MAX]{};
  float features[8]{},derivative[D]{},payload[D]{},payload_derivative[D]{};
  float gain_derivative=0;
};
static_assert(sizeof(ConstituentRecord)==sizeof(ConstituentRecordV9));
static_assert(offsetof(ConstituentRecord,edges)==offsetof(ConstituentRecordV9,edges));
struct ContentExpertRecord {
  SemanticRef ref{};
  int expert=0,runner=0;
  uint64_t input_mask=0;
  float gate=1,gate_derivative=0;
  float input[D]{},residual[D]{},route_derivative[D]{};
};
struct ContentExpertGradient {
  float weights[D*D]{},bias[D]{},keys[D]{};
  bool route=false;
};
