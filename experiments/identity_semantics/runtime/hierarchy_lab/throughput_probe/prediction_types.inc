// Action 4: an explicit prediction policy, independent of directional credit.
// Feature snapshots contain values and immutable identities, never pointers.
constexpr int FORECAST_REGIONS=14;
struct CompactFeatureView {
  int64_t tick=0;
  uint64_t publication=0,bank_version=0,response_version=0,forecast_version=0;
  SemanticRef semantic{};
  float response[D]{},content[D]{},context[MK_FEATURES]{};
};
struct PredictionStream {
  CompactFeatureView current[N]{},pending[N]{};
  int64_t slow_tick=0,slow_byte=0;
  uint64_t samples=0;
  float gradient[D]{},rate=0;
  float slow[D]{},output[D]{},feedback[D]{};
  int mode=0,reserved=0;
};
struct PredictionRuntime {
  bool compact=false;
  int rank=16,stride=1;
  float slow_scale=1.f;
  std::vector<float> forecast;
  // Bias, observed-prefix diagonal, A1 diagonal, delayed-feedback diagonal.
  float fast[4][D]{};
  uint64_t forecast_version[N]{},response_version[N]{};
  PredictionStream stream{};
};
