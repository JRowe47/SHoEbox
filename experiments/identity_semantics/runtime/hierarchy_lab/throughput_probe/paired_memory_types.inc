// Detached EC publications form one immutable content/context episode. Storage
// is content addressed; no sliding window, FIFO or age-based expiry is used.
struct PairedObservation {
  ChildEvent occurrence{};
  uint64_t stream=0;
  int64_t clock_begin=0,clock_end=0;
  float value[D]{},key[D]{};
};
struct PairedEpisode {
  uint64_t id=0;
  int64_t born=0,updated=0;
  float priority=0,utility=0;
  PairedObservation members[2]{};
};
struct PairedStream {
  uint64_t epoch=1;
  PairedObservation latest[2]{},query[2]{};
  uint64_t considered_stream[2]{};
  int64_t considered_seq[2]{};
  int64_t read_tick=0;
  uint64_t read_episode=0;
  int read_slot=-1;
  float similarity=0,gate=0;
};
constexpr int PAIRED_METRICS=16;
struct PairedMemory {
  bool enabled=true;
  uint64_t serial=0;
  std::vector<PairedEpisode> slots;
  PairedStream stream{};
  // C/X publications, candidates, admissions, replacements, reinforcements,
  // novelty/priority rejections, joint reads, comparisons, hits, utility credit,
  // retired utility handles, incomplete reads, cached reads, incomplete writes.
  uint64_t metrics[PAIRED_METRICS]{};
};
