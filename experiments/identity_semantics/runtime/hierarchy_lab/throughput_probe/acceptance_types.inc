// Observations only: never serialized, consulted by learning, or used as gates.
// Each stream/import starts a new observation window. Checkpoint ABI stays 10.
struct AcceptanceObservations {
  // publications, primary sum/max, coverage sum/max, formed, recognized,
  // provisional, relays, invalid primary, invalid coverage, outside coverage,
  // partial EC/Y fusions without their designated primary source yet.
  uint64_t events[N][13]{};
  // Bin 0 is zero; bin b>0 covers [2^(b-1), 2^b-1] bytes.
  uint64_t primary_hist[N][65]{},coverage_hist[N][65]{};
  // Locally completed temporal publications only (L2-G5), excluding relays.
  // A binding emitted alone is one frame, not its number of source roles.
  uint64_t lengths[N][COMP_MAX+1]{},recognized_lengths[N][COMP_MAX+1]{};
  // Calls which pass rate/freeze/producer guards, not a claim of nonzero delta.
  uint64_t expert_update_calls[N][2]{}; // language/DFA, local prediction
};
