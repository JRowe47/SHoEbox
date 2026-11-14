// Semantic handles already have stable slab indices and serial generations.
// Roots include in-flight encoding, queued reverse objectives and replay.
// The ancestry DAG is acyclic by serial; event-frontier edges can only expire
// between collections and are reconciled separately from owner changes.
struct SemanticRetention {
  struct Node {
    uint64_t serial=0,root_generation=0,expiry_generation=0;
    int incoming=0;bool root=false,active=false,queued=false;
    unsigned edges=0;
  };
  std::vector<Node> nodes;
  std::vector<int> roots,next_roots,work,expiring,next_expiring;
  uint64_t generation=0;
  int local=-1,frontier=-1,local_only=-1;
  void invalidate(){nodes.clear();roots.clear();next_roots.clear();work.clear();expiring.clear();next_expiring.clear();}
};
