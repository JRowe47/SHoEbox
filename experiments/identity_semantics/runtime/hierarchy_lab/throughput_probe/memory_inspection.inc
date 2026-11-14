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
    row[11]+=v.encoder&&!content_expert_record(m,v.ref);
  }
}
}
