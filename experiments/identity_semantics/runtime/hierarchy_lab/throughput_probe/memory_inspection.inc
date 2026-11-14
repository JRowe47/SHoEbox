// Explicit, read-only API-boundary inspection. Never called by run().
extern "C" {
size_t diagnostic_memory_ranges(uint64_t*out){
  size_t count=0;
  auto add=[&](uint64_t section,size_t offset,size_t bytes){
    if(out){out[count*3]=section;out[count*3+1]=offset;out[count*3+2]=bytes;}count++;
  };
#define RANGE(section,type,field) add(section,offsetof(type,field),sizeof(type::field))
  RANGE(0,State,reads);RANGE(0,State,writes);RANGE(0,State,correct);RANGE(0,State,mse);
  RANGE(0,State,pc);RANGE(0,State,max_age);RANGE(0,State,age_sum);RANGE(0,State,similarity_sum);
  RANGE(1,LaceStateWire,metrics);RANGE(2,LearnedRuntime,metrics);
  RANGE(3,CompositionCheckpoint,symbol_updates);
#define STREAM_RANGE(field) add(3,offsetof(CompositionCheckpoint,stream)+offsetof(CompositionCheckpointStream,field),sizeof(CompositionCheckpointStream::field))
  STREAM_RANGE(completion_metrics);STREAM_RANGE(message_metrics);STREAM_RANGE(metrics);STREAM_RANGE(teaching_metrics);
  RANGE(4,NeuralSnapshotHeader,metrics);
#undef STREAM_RANGE
#undef RANGE
  return count;
}
void inspect_live_memory(void*ptr,uint64_t*out){
  auto&m=*static_cast<Model*>(ptr);m.bank_pipeline.wait();memset(out,0,N*12*sizeof(uint64_t));
  for(int r=0;r<N;r++){
    auto*row=out+r*12;row[0]=m.s.exec[r];row[1]=m.composition.stream.event_serial[r];
    row[2]=m.neural_archive[r].size();row[3]=m.pending_neural[r].size();row[4]=m.pending_input[r].size();
    row[5]=m.event.ordinal[r];row[6]=m.pattern.classes[r].size();
    for(const auto&v:m.laces.records[r])row[7]+=v.tick!=0;
    for(const auto&e:m.composition.stream.history[r])if(e.ref.seq){
      row[8]++;row[9]+=e.ref.begin<0||e.ref.end<=e.ref.begin;
      // EC/Y partial fusions can have no designated primary constituent yet.
      bool unclocked=r>=15&&e.kind==4&&e.clock_begin==0&&e.clock_end==0;
      row[9]+=!unclocked&&(e.clock_begin<e.ref.begin||e.clock_end>e.ref.end||e.clock_begin>=e.clock_end);
    }
  }
  for(int k=0;k<constituent_capacity(m);k++){
    const auto&v=constituent_slot(m,k);if(!v.ref.serial||v.ref.region<0||v.ref.region>=N)continue;
    auto*row=out+v.ref.region*12;row[10]++;
    row[11]+=v.encoder==1&&!content_expert_record(m,v.ref);
  }
}
}

// Quiescent inventory only: no counters, timers or observers in training.
extern "C" void indexed_storage_status(void*p,uint64_t*out){
  auto&m=*static_cast<Model*>(p);for(int i=0;i<13;i++)out[i]=0;
  out[0]=m.event.heads;out[1]=sizeof(m.event.context);out[5]=N*LACE_RING*sizeof(ContextEligibilityV1);
  for(int r=0;r<N;r++){
    for(const auto&v:m.event.context[r]){out[2]+=v.data.size()*sizeof(float);out[3]+=v.data.capacity()*sizeof(float);out[4]+=v.tick!=0;}
    for(const auto&entry:m.event.archive[r]){
      out[1]+=sizeof(ContextEligibility);out[2]+=entry.second.data.size()*sizeof(float);
      out[3]+=entry.second.data.capacity()*sizeof(float);out[4]++;out[5]+=sizeof(ContextEligibilityV1);
    }
  }
  const auto&n=m.neural_retention;out[6]=n.nodes.size()-n.free.size();out[7]=n.nodes.capacity()*sizeof(NeuralRetention::Node);
  for(const auto&v:n.nodes)out[7]+=v.edges.capacity()*sizeof(NeuralRetention::Edge);
  for(const auto*v:{&n.free,&n.roots,&n.next_roots,&n.work,&n.candidates})out[7]+=v->capacity()*sizeof(int);
  const auto&s=m.semantic_retention;out[8]=s.nodes.capacity()*sizeof(SemanticRetention::Node);
  for(const auto*v:{&s.roots,&s.next_roots,&s.work,&s.expiring,&s.next_expiring})out[8]+=v->capacity()*sizeof(int);
  for(const auto&v:s.nodes)out[9]+=v.active;
  out[10]=sizeof(ContextHead);out[11]=sizeof(ContextEligibilityV1);
  out[12]=m.neural_record_pool.reserved_bytes();
}

// Inspect retained workspace and plans, not historical counters. Batch-size
// bins describe currently valid cached traversal plans at this API boundary.
extern "C" void execution_storage_status(void*p,uint64_t*out){
  auto&m=*static_cast<Model*>(p);memset(out,0,32*sizeof(uint64_t));
  auto signals=[&](const LocalSignals&s){
    out[0]+=s.entries.capacity()*sizeof(LocalSignals::Entry)+s.table.capacity()*sizeof(LocalSignals::Slot)+s.order.capacity()*sizeof(size_t);
    out[1]+=s.entries.size();
  };
  auto&local=m.local_credit_work;for(const auto&s:local.pending)signals(s);signals(local.updates);
  out[0]+=local.adjoints.capacity()*sizeof(LocalAdjoint);out[2]=local.adjoints.size();
  auto&reverse=m.reverse_execution;out[3]=reverse.policies.size();out[4]=reverse.count;
  out[5]=reverse.groups.capacity()*sizeof(ReverseExecution::Group)+reverse.policies.capacity()*sizeof(ReverseExecution::Policy);
  for(const auto&g:reverse.groups)out[5]+=g.roots.capacity()*sizeof(size_t);
  auto plan=[&](const ReverseExecution::Plan&p,bool cached){
    out[5]+=(p.roots.capacity()+p.nodes.capacity())*sizeof(SemanticRef)+p.batch_ends.capacity()*sizeof(size_t);
    if(!cached||!p.valid)return;
    for(auto ref:p.nodes)if(!constituent_record(m,ref))return;
    out[6]++;out[7]+=p.nodes.size();size_t begin=0;
    for(auto end:p.batch_ends){if(end>begin&&end-begin<=16)out[16+end-begin-1]++;begin=end;}
  };
  for(const auto&p:reverse.plans)plan(p,true);plan(reverse.scratch,false);
  out[8]=m.execution.pending;
  for(const auto&r:m.execution.recipe)out[9]+=r.immediate.capacity()*sizeof(int);
}
