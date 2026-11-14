// Stage 5 learning policy: local exact eligibility plus directional interface
// credit. This is separate from the ancestry policy and its checkpoint ABI.
constexpr int MK_HEADS=2,MK_DIM=8,MK_FEATURES=MK_HEADS*MK_DIM;
struct ContextEligibilityV1 {
  int64_t tick=0;
  float before[MK_HEADS][MK_DIM][MK_DIM]{};
  float query_derivative[MK_HEADS][MK_DIM]{},read[MK_FEATURES]{};
};
// Runtime allocates precisely the active heads. The expanded v1 wire format
// remains explicit at the checkpoint boundary; inactive values are zero.
struct ContextHead {
  float before[MK_DIM][MK_DIM]{},query_derivative[MK_DIM]{},read[MK_DIM]{};
};
struct ContextEligibility {
  int64_t tick=0;
  std::pmr::vector<float> data{&record_pool()};
  void prepare(int heads,int64_t at){data.resize(heads*(MK_DIM*MK_DIM+2*MK_DIM));tick=at;}
  void clear(){tick=0;data.clear();}
  int heads()const{return int(data.size())/(MK_DIM*MK_DIM+2*MK_DIM);}
  using Matrix=float[MK_DIM][MK_DIM];
  Matrix& before(int h){return *reinterpret_cast<Matrix*>(data.data()+h*MK_DIM*MK_DIM);}
  const Matrix& before(int h)const{return *reinterpret_cast<const Matrix*>(data.data()+h*MK_DIM*MK_DIM);}
  float* query(int h){return data.data()+heads()*MK_DIM*MK_DIM+h*MK_DIM;}
  const float* query(int h)const{return data.data()+heads()*MK_DIM*MK_DIM+h*MK_DIM;}
  float* reads(){return data.data()+heads()*(MK_DIM*MK_DIM+MK_DIM);}
  const float* reads()const{return data.data()+heads()*(MK_DIM*MK_DIM+MK_DIM);}
};
static_assert(sizeof(ContextEligibilityV1)==648,"Historical event extension changed");
inline ContextEligibilityV1 context_wire(const ContextEligibility&v){
  ContextEligibilityV1 out{};out.tick=v.tick;
  for(size_t h=0;h<size_t(v.heads());h++){
    memcpy(out.before[h],v.before(h),sizeof(out.before[h]));
    memcpy(out.query_derivative[h],v.query(h),sizeof(out.query_derivative[h]));
    memcpy(out.read+h*MK_DIM,v.reads()+h*MK_DIM,MK_DIM*sizeof(float));
  }
  return out;
}
inline void context_from_wire(ContextEligibility&v,const ContextEligibilityV1&wire,int heads){
  v.prepare(heads,wire.tick);
  for(int h=0;h<heads;h++){
    memcpy(v.before(h),wire.before[h],sizeof(wire.before[h]));
    memcpy(v.query(h),wire.query_derivative[h],sizeof(wire.query_derivative[h]));
    memcpy(v.reads()+h*MK_DIM,wire.read+h*MK_DIM,MK_DIM*sizeof(float));
  }
}
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
  RecordArchive<ContextEligibility> archive[N];
  ContextEligibility workspace;
  std::vector<EventEligibility> eligibility;
  EventReplay replay[N][2]{};
  // Observations are excluded from checkpoints, including per-region coverage.
  uint64_t metrics[N][16]{};
  double teacher_norm[N][2]{};
  std::vector<SemanticRef> new_events;
  bool initialized=false,training=false,local_only=false,synthetic=false;
  float rates[2]{};
};
