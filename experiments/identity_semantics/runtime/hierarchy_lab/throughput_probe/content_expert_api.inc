struct ContentExpertHeader {uint64_t version=1,count=0;};
extern "C" {
int content_workers_configure(void*p,int count,const int*cpus){
  auto&m=*static_cast<Model*>(p);
  if(count<1||count>32||m.content_worker_depth||m.content_credit_depth)return 0;
  if(cpus)for(int i=0;i<count;i++)if(cpus[i]<0||cpus[i]>=CPU_SETSIZE)return 0;
  m.content_workers.reset();
  if(count>1)m.content_workers=std::make_unique<RowWorkers>(count,cpus);
  return 1;
}
void content_expert_configure(void*p,int enabled){static_cast<Model*>(p)->content_experts=enabled!=0;}
void content_expert_status(void*p,uint64_t*out){memcpy(out,static_cast<Model*>(p)->content_expert_metrics,sizeof(Model::content_expert_metrics));}
size_t content_expert_state_size(void*p){
  auto&m=*static_cast<Model*>(p);size_t count=0;
  for(const auto&record:m.content_records)count+=record.ref.serial!=0;
  return sizeof(ContentExpertHeader)+count*sizeof(ContentExpertRecord);
}
void content_expert_export(void*p,void*out){
  auto&m=*static_cast<Model*>(p);ContentExpertHeader header{};
  for(const auto&record:m.content_records)header.count+=record.ref.serial!=0;
  auto*cursor=static_cast<unsigned char*>(out);memcpy(cursor,&header,sizeof(header));cursor+=sizeof(header);
  for(const auto&record:m.content_records)if(record.ref.serial){
    memcpy(cursor,&record,sizeof(record));cursor+=sizeof(record);
  }
}
int content_expert_import(void*p,const void*data,size_t size){
  if(size<sizeof(ContentExpertHeader))return 0;
  ContentExpertHeader header{};memcpy(&header,data,sizeof(header));
  if(header.version!=1||header.count!=(size-sizeof(header))/sizeof(ContentExpertRecord)||
     size-sizeof(header)!=header.count*sizeof(ContentExpertRecord))return 0;
  auto&m=*static_cast<Model*>(p);int capacity=constituent_capacity(m);size_t required=0;
  for(int i=0;i<capacity;i++){const auto&node=constituent_slot(m,i);required+=node.ref.serial&&node.encoder==1;}
  if(header.count!=required)return 0;
  std::vector<ContentExpertRecord> records(header.count?capacity:0);
  auto*cursor=static_cast<const unsigned char*>(data)+sizeof(header);int previous=-1;
  for(uint64_t i=0;i<header.count;i++){
    ContentExpertRecord record{};memcpy(&record,cursor,sizeof(record));cursor+=sizeof(record);
    auto*node=constituent_record(m,record.ref);
    if(!node||node->ref.region!=record.ref.region||node->ref.index!=record.ref.index||
       node->encoder!=1||record.ref.index<=previous||record.expert<0||record.expert>=E||
       record.runner<0||record.runner>=E||!std::isfinite(record.gate)||!std::isfinite(record.gate_derivative))return 0;
    for(const float*values:{record.input,record.residual,record.route_derivative})
      for(int j=0;j<D;j++)if(!std::isfinite(values[j]))return 0;
    previous=record.ref.index;records[record.ref.index]=record;
  }
  m.content_records.swap(records);m.content_input_cache.clear();return 1;
}
// v9 used these bytes only as padding. Import the original event values and
// mark every historical encoder as legacy; no fitted values or clocks change.
int composition_import_v9(void*p,const void*data,size_t size){
  if(size<sizeof(CompositionRuntime)||(size-sizeof(CompositionRuntime))%sizeof(ConstituentRecord))return 0;
  uint32_t version;memcpy(&version,data,sizeof(version));if(version!=9)return 0;
  std::vector<unsigned char> converted(size);memcpy(converted.data(),data,size);
  auto*state=reinterpret_cast<CompositionRuntime*>(converted.data());state->version=10;
  for(auto&node:state->stream.semantic_records)node.encoder=0;
  auto*overflow=reinterpret_cast<ConstituentRecord*>(converted.data()+sizeof(CompositionRuntime));
  for(size_t i=0;i<(size-sizeof(CompositionRuntime))/sizeof(ConstituentRecord);i++)overflow[i].encoder=0;
  return composition_import(p,converted.data(),size,0);
}
}
