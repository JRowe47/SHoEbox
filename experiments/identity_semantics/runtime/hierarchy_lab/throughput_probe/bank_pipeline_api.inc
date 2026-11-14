// Extension 1 stores a quiescent pending result, not thread implementation state.
// Export waits for computation without making the next version visible.
extern "C" {
int bank_pipeline_configure(void*ptr,int enabled){
  auto&m=*static_cast<Model*>(ptr);auto&p=m.bank_pipeline;
  if((enabled!=0&&enabled!=1)||m.composition.stream.byte_position||m.composition.stream.semantic_serial||
     p.pending||!p.building.empty()||m.content_credit_depth||
     (enabled&&(!m.event.heads||!m.event.local||m.reverse_epoch_bytes!=1||m.reverse_policy!=2)))return 0;
  for(int r=0;r<N;r++)if(m.s.valid[r])return 0;
  p.enabled=enabled;return 1;
}
int bank_pipeline_workers(void*ptr,int count,int cpu,int fused){
  auto&m=*static_cast<Model*>(ptr);auto&p=m.bank_pipeline;
  if((count!=1&&count!=2)||cpu < -1||cpu>=CPU_SETSIZE||(fused!=0&&fused!=1)||
     m.content_credit_depth||(!p.enabled&&count!=1))return 0;
  p.wait();p.worker.reset();p.workers=count;p.fused=fused;
  if(count==2)p.worker=std::make_unique<BankLearningWorker>(cpu);
  return 1;
}
void bank_pipeline_status(void*ptr,uint64_t*out){
  auto&p=static_cast<Model*>(ptr)->bank_pipeline;memcpy(out,p.metrics,sizeof(p.metrics));
  out[10]=p.version;out[11]=p.sequence;out[12]=p.pending?1:0;out[13]=p.workers;out[14]=p.enabled;out[15]=p.fused;
}
void bank_pipeline_reset_status(void*ptr){auto&p=static_cast<Model*>(ptr)->bank_pipeline;memset(p.metrics,0,sizeof(p.metrics));}
void bank_pipeline_finish(void*ptr){pipeline_drain(*static_cast<Model*>(ptr));}
size_t bank_pipeline_state_size(void*ptr){
  auto&p=static_cast<Model*>(ptr)->bank_pipeline;p.wait();
  return 12*sizeof(uint64_t)+(p.pending?p.pending->tiles.size()*sizeof(PipelineTile):0);
}
void bank_pipeline_export(void*ptr,void*out){
  auto&m=*static_cast<Model*>(ptr);auto&p=m.bank_pipeline;p.wait();pipeline_install(m);
  uint64_t h[12]{1,uint64_t(E),uint64_t(p.enabled),p.version,p.sequence,uint64_t(bool(p.pending)),0,0,0,0,0,0};
  if(p.pending){auto&b=*p.pending;h[6]=b.read_version;h[7]=b.base_version;h[8]=b.sequence;h[9]=b.tiles.size();}
  memcpy(out,h,sizeof(h));if(p.pending)memcpy(static_cast<unsigned char*>(out)+sizeof(h),p.pending->tiles.data(),h[9]*sizeof(PipelineTile));
}
int bank_pipeline_import(void*ptr,const void*input,size_t size){
  auto&m=*static_cast<Model*>(ptr);auto&p=m.bank_pipeline;
  if(size<12*sizeof(uint64_t)||m.content_credit_depth||!p.building.empty())return 0;
  uint64_t h[12];memcpy(h,input,sizeof(h));
  if(h[0]!=1||h[1]!=E||h[2]!=uint64_t(p.enabled)||h[5]>1||h[9]>(N-2)*E||h[10]||h[11]||
     size!=sizeof(h)+h[9]*sizeof(PipelineTile)||h[3]>h[4]||
     (!h[5]&&(h[3]!=h[4]||h[6]||h[7]||h[8]||h[9]))||
     (h[5]&&(!p.enabled||h[4]!=h[3]+1||h[7]!=h[3]||h[8]!=h[4]||h[6]>h[7]||h[7]-h[6]>1)))return 0;
  std::shared_ptr<PipelineBatch> next;
  if(h[5]){
    next=std::make_shared<PipelineBatch>();next->read_version=h[6];next->base_version=h[7];next->sequence=h[8];next->tiles.resize(h[9]);
    memcpy(next->tiles.data(),static_cast<const unsigned char*>(input)+sizeof(h),h[9]*sizeof(PipelineTile));
    uint64_t last=0;
    for(const auto&t:next->tiles){
      if(t.bank<2*E||t.bank>=N*E||t.bank<=last)return 0;last=t.bank;
      for(float v:t.weights)if(!std::isfinite(v))return 0;
      for(float v:t.bias)if(!std::isfinite(v))return 0;
      for(float v:t.keys)if(!std::isfinite(v))return 0;
    }
  }
  p.wait();pipeline_install(m);p.pending=std::move(next);p.version=h[3];p.sequence=h[4];return 1;
}
}
