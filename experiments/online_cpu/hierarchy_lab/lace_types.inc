// Bounded publication eligibility. Structural route identity and feature SDRs
// are separate fields. A saved expert index is never treated as semantic data.
constexpr int LACE_RING=16,LACE_LINKS=16,LACE_DEPTH=3;
// `index` occupies former structure padding. Legacy records remain readable:
// every fast lookup validates the immutable tick and falls back to a scan.
struct LaceParent { int region=-1,slot=-1; int64_t tick=0; float coefficient=0; int index=-1; };
struct LaceRecord {
  int64_t tick=0;
  int expert=0,runner=0,nparents=0,recall_slot=-1;
  int64_t recall_birth=0;
  float gate=1,gate_derivative=0;
  uint64_t input_mask=0,feature_mask=0;
  float u[D]{},h[D]{},residual[D]{},returned[D]{};
  LaceParent parents[LACE_LINKS]{};
  float credit[LACE_DEPTH][D]{};
  int uses[LACE_DEPTH]{};
};
struct LaceState {
  LaceRecord records[N][LACE_RING]{};
  int cursor[N]{};
  // publications, credits, flushes, missed parent IDs, route-gradient updates,
  // credited gradient norm, feature overlap, prior masks, and depth counts.
  double metrics[N][12]{};
};
