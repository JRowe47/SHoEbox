// Archive serialization is separate from the unchanged hot-cache layouts.
// No pointers, map internals or transient reachability marks enter a checkpoint.
struct NeuralSnapshotHeader {
  uint64_t version=1,counts[N]{},metrics[N][4]{};
};
extern "C" {
size_t neural_state_size(void*p){
  auto&m=*static_cast<Model*>(p);size_t size=sizeof(NeuralSnapshotHeader);
  for(const auto&archive:m.neural_archive)size+=archive.size()*sizeof(RetainedNeural);
  return size;
}
void neural_export(void*p,void*out){
  auto&m=*static_cast<Model*>(p);NeuralSnapshotHeader header{};
  for(int r=0;r<N;r++)header.counts[r]=m.neural_archive[r].size();
  memcpy(header.metrics,m.neural_metrics,sizeof(header.metrics));
  auto*cursor=static_cast<unsigned char*>(out);memcpy(cursor,&header,sizeof(header));cursor+=sizeof(header);
  for(const auto&archive:m.neural_archive)for(const auto&entry:archive){
    memcpy(cursor,&entry.second,sizeof(RetainedNeural));cursor+=sizeof(RetainedNeural);
  }
}
int neural_import(void*p,const void*data,size_t size){
  if(size<sizeof(NeuralSnapshotHeader))return 0;
  NeuralSnapshotHeader header{};memcpy(&header,data,sizeof(header));if(header.version!=1)return 0;
  size_t remaining=(size-sizeof(header))/sizeof(RetainedNeural),count=0;
  for(auto n:header.counts){if(n>remaining-count)return 0;count+=n;}
  if(size!=sizeof(header)+count*sizeof(RetainedNeural))return 0;
  auto&m=*static_cast<Model*>(p);std::map<int64_t,RetainedNeural> archives[N];
  auto*cursor=static_cast<const unsigned char*>(data)+sizeof(header);
  for(int r=0;r<N;r++){
    int64_t previous=0;
    for(uint64_t i=0;i<header.counts[r];i++){
      RetainedNeural saved{};memcpy(&saved,cursor,sizeof(saved));cursor+=sizeof(saved);
      auto&v=saved.lace;auto&input=saved.input;
      if(v.tick<=previous||v.tick>m.s.tick||v.expert<-1||v.expert>=E||v.runner<0||v.runner>=E||
         v.nparents<0||v.nparents>LACE_LINKS||input.tick!=v.tick||input.previous_tick>=v.tick||
         input.message.feedback_count<0||input.message.feedback_count>COMP_MAX||
         input.message.semantic_count<0||input.message.semantic_count>COMP_MAX)return 0;
      for(const auto&recent:m.laces.records[r])if(recent.tick==v.tick)return 0;
      for(int k=0;k<v.nparents;k++)if(v.parents[k].region<0||v.parents[k].region>=N)return 0;
      for(int k=0;k<input.message.feedback_count;k++)if(input.message.feedback[k].region<0||input.message.feedback[k].region>=N)return 0;
      if(saved.temporal.tick&&saved.temporal.tick!=v.tick)return 0;
      previous=v.tick;archives[r].emplace(v.tick,saved);
    }
  }
  for(int r=0;r<N;r++)m.neural_archive[r].swap(archives[r]);
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
