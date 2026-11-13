// Archive serialization is separate from the unchanged hot-cache layouts.
// No pointers, map internals or transient reachability marks enter a checkpoint.
struct NeuralSnapshotHeader {
  uint64_t version=1,counts[N]{},metrics[N][4]{};
};
// Frozen archive v1 wire record. Unused fields are materialized only at I/O.
struct RetainedNeuralV1 {
  LaceRecord lace{};
  LegacyInputCredit input{};
  float cue[D]{};
  TemporalRecord temporal{};
};
static_assert(sizeof(RetainedNeuralV1)==11424,"Historical neural checkpoint layout changed");
extern "C" {
// Payload sizes exclude allocator/map overhead and transient work buffers.
void completion_storage_status(void*p,uint64_t*out){
  auto&m=*static_cast<Model*>(p);
  out[0]=sizeof(InputCredit);out[1]=sizeof(LegacyInputCredit);
  out[2]=sizeof(RetainedNeural);out[3]=sizeof(RetainedNeuralV1);
  out[4]=sizeof(CompositionRuntime);out[5]=sizeof(CompositionCheckpoint);
  out[6]=out[7]=0;
  for(const auto&archive:m.neural_archive)out[6]+=archive.size();
  for(const auto&archive:m.neural_temporal)out[7]+=archive.size();
  out[8]=out[6]*sizeof(RetainedNeural)+out[7]*sizeof(TemporalRecord);
  out[9]=out[6]*sizeof(RetainedNeuralV1);
}
void local_credit_configure(void*p,int enabled){static_cast<Model*>(p)->local_neural_credit=enabled!=0;}
void local_credit_status(void*p,uint64_t*out){memcpy(out,static_cast<Model*>(p)->local_credit_metrics,sizeof(Model::local_credit_metrics));}
size_t neural_state_size(void*p){
  auto&m=*static_cast<Model*>(p);size_t size=sizeof(NeuralSnapshotHeader);
  for(const auto&archive:m.neural_archive)size+=archive.size()*sizeof(RetainedNeuralV1);
  return size;
}
void neural_export(void*p,void*out){
  auto&m=*static_cast<Model*>(p);NeuralSnapshotHeader header{};
  for(int r=0;r<N;r++)header.counts[r]=m.neural_archive[r].size();
  memcpy(header.metrics,m.neural_metrics,sizeof(header.metrics));
  auto*cursor=static_cast<unsigned char*>(out);memcpy(cursor,&header,sizeof(header));cursor+=sizeof(header);
  for(int r=0;r<N;r++)for(const auto&entry:m.neural_archive[r]){
    RetainedNeuralV1 wire{};memset(&wire,0,sizeof(wire));
    wire.lace=entry.second.lace;completion_copy_input(wire.input,entry.second.input);
    memcpy(wire.cue,entry.second.cue,sizeof(wire.cue));
    if(r<STATE_REGIONS){auto found=m.neural_temporal[r].find(entry.first);
      if(found!=m.neural_temporal[r].end())wire.temporal=found->second;
    }
    memcpy(cursor,&wire,sizeof(wire));cursor+=sizeof(wire);
  }
}
int neural_import(void*p,const void*data,size_t size){
  if(size<sizeof(NeuralSnapshotHeader))return 0;
  NeuralSnapshotHeader header{};memcpy(&header,data,sizeof(header));if(header.version!=1)return 0;
  size_t remaining=(size-sizeof(header))/sizeof(RetainedNeuralV1),count=0;
  for(auto n:header.counts){if(n>remaining-count)return 0;count+=n;}
  if(size!=sizeof(header)+count*sizeof(RetainedNeuralV1))return 0;
  auto&m=*static_cast<Model*>(p);std::map<int64_t,RetainedNeural> archives[N];
  std::map<int64_t,TemporalRecord> temporal[STATE_REGIONS];
  auto*cursor=static_cast<const unsigned char*>(data)+sizeof(header);
  for(int r=0;r<N;r++){
    int64_t previous=0;
    for(uint64_t i=0;i<header.counts[r];i++){
      RetainedNeuralV1 wire{};memcpy(&wire,cursor,sizeof(wire));cursor+=sizeof(wire);
      auto&v=wire.lace;auto&input=wire.input;
      if(v.tick<=previous||v.tick>m.s.tick||v.expert<-1||v.expert>=E||v.runner<0||v.runner>=E||
         v.nparents<0||v.nparents>LACE_LINKS||input.tick!=v.tick||input.previous_tick>=v.tick||
         input.message.feedback_count<0||input.message.feedback_count>COMP_MAX||
         input.message.semantic_count<0||input.message.semantic_count>COMP_MAX)return 0;
      for(const auto&recent:m.laces.records[r])if(recent.tick==v.tick)return 0;
      for(int k=0;k<v.nparents;k++)if(v.parents[k].region<0||v.parents[k].region>=N)return 0;
      for(int k=0;k<input.message.feedback_count;k++)if(input.message.feedback[k].region<0||input.message.feedback[k].region>=N)return 0;
      if(wire.temporal.tick&&wire.temporal.tick!=v.tick)return 0;
      RetainedNeural saved{};saved.lace=v;completion_copy_input(saved.input,input);
      memcpy(saved.cue,wire.cue,sizeof(saved.cue));
      if(r<STATE_REGIONS&&wire.temporal.tick)temporal[r].emplace(v.tick,wire.temporal);
      previous=v.tick;archives[r].emplace(v.tick,saved);
    }
  }
  for(int r=0;r<N;r++)m.neural_archive[r].swap(archives[r]);
  for(int r=0;r<STATE_REGIONS;r++)m.neural_temporal[r].swap(temporal[r]);
  neural_invalidate(m);
  memcpy(m.neural_metrics,header.metrics,sizeof(m.neural_metrics));return 1;
}
void neural_metrics(void*p,uint64_t*out){
  auto&m=*static_cast<Model*>(p);
  for(int r=0;r<N;r++){
    memcpy(out+r*5,m.neural_metrics[r],4*sizeof(uint64_t));
    out[r*5+4]=m.neural_archive[r].size();
  }
}
}
