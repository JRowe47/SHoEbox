// Semantic codes are local constituent adapters, not recurrent responses.
// A consumer saves the exact occurrence's adapter eligibility. Holding or
// relaying a publication cannot redirect credit to a newer occurrence/expert.
struct SemanticEdge {
  ChildEvent producer{},occurrence{};
  int slot=-1;
  float coefficient=0,features[8]{},derivative[D]{};
};
// Frozen v6/v10 checkpoint layout, also used by historical message kernels.
struct LegacyMessageInput {
  int semantic_count=0,feedback_count=0;
  SemanticEdge semantic[COMP_MAX]{};
  LaceParent feedback[COMP_MAX]{};
  float payload_derivative[D]{},feedback_derivative[D]{};
  float route_derivative[D]{}; // Saved selected-minus-runner gate Jacobian.
  float qa=0,alpha=0;
};
#ifdef SHOEBOX_LEARNED_COMPLETION
// Completion teaches through SemanticRef. No per-child adapter copies or
// payload-addition derivative are read on this path.
struct MessageInput {
  int feedback_count=0;
  LaceParent feedback[COMP_MAX]{};
  float feedback_derivative[D]{},route_derivative[D]{};
  float qa=0,alpha=0;
};
#else
using MessageInput=LegacyMessageInput;
#endif
