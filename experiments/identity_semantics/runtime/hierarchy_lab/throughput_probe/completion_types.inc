// Separate semantic eligibility from the much faster neural execution ring.
// Handles are generation checked. Relays copy a handle, never reassign credit.
constexpr int SEM_RING=16384,SEM_WORK=SEM_RING;
struct SemanticRef { uint64_t serial=0; int region=0,index=0; };
struct ConstituentEdge {
  SemanticRef source{};
  ChildEvent occurrence{};
  int slot=0; float weight=0;
};
struct ConstituentRecord {
  SemanticRef ref{};
  int count=0;
  ConstituentEdge edges[COMP_MAX]{};
  float features[8]{},derivative[D]{},payload[D]{},payload_derivative[D]{};
  float gain_derivative=0;
};
