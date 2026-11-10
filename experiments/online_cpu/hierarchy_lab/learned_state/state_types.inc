// Learned bottom state and bounded temporal publication eligibility.
constexpr int STATE_REGIONS=2,TEMP_RING=64,STATE_CONFIG=24;
struct TemporalRecord {
  int64_t tick=0;
  int expert=0,generation=0,runner=0,runner_generation=0;
  float response_gate=1;
  uint64_t input_mask=0;
  float u[D]{},h[D]{},write[D]{},before[D]{};
  float eligibility_a[D]{},eligibility_b[D]{},read[D]{},read_derivative[D]{};
};
struct LearnedRuntime {
  float memory[2][D]{},eligibility_a[2][D]{},eligibility_b[2][D]{};
  TemporalRecord records[2][TEMP_RING]{};
  int cursor[2]{},generation[2][E]{};
  int64_t events[2]{},last_used[2][E]{};
  double decay_clock[2]{},last_decay[2][E]{};
  float sums[2][E][D]{},mass[2][E]{},utility[2][E]{};
  double metrics[2][12]{};
};
