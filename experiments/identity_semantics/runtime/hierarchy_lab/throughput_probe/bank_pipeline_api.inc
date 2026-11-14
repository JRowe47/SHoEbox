// Boundary-only inspection and persistence. No clocks or counter collection in
// capture/compute/commit. Extension 2 stores compact entries plus optional W.
constexpr size_t PIPELINE_HEADER=12*sizeof(uint64_t),PIPELINE_ENTRY=2*sizeof(uint64_t)+2*D*sizeof(float);
inline size_t pipeline_wire_size(const BankPipeline&p){
  return PIPELINE_HEADER+(p.pending?p.pending->entries.size()*PIPELINE_ENTRY+p.pending->weights.size()*D*D*sizeof(float):0);
}
extern "C" {
int bank_pipeline_configure(void*ptr,int enabled){
  auto&m=*static_cast<Model*>(ptr);auto&p=m.bank_pipeline;
  if((enabled!=0&&enabled!=1)||m.composition.stream.byte_position||m.composition.stream.semantic_serial||
     p.pending||p.building->scope_count||m.content_credit_depth||
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
  auto&p=static_cast<Model*>(ptr)->bank_pipeline;
  memset(out,0,16*sizeof(uint64_t));
  out[10]=p.version;out[11]=p.sequence;out[12]=p.pending?1:0;out[13]=p.workers;out[14]=p.enabled;out[15]=p.fused;
}
void bank_pipeline_reset_status(void*){} // retained ABI; no observation window
void bank_pipeline_finish(void*ptr){pipeline_drain(*static_cast<Model*>(ptr));}
size_t bank_pipeline_state_size(void*ptr){auto&p=static_cast<Model*>(ptr)->bank_pipeline;p.wait();return pipeline_wire_size(p);}
void bank_pipeline_export(void*ptr,void*out){
  auto&m=*static_cast<Model*>(ptr);auto&p=m.bank_pipeline;p.wait();pipeline_install(m);
  uint64_t h[12]{2,uint64_t(E),uint64_t(p.enabled),p.version,p.sequence,uint64_t(bool(p.pending)),0,0,0,0,0,0};
  if(p.pending){auto&b=*p.pending;h[6]=b.read_version;h[7]=b.base_version;h[8]=b.sequence;h[9]=b.entries.size();h[10]=b.weights.size();}
  memcpy(out,h,sizeof(h));auto*dst=static_cast<unsigned char*>(out)+sizeof(h);
  if(p.pending)for(const auto&e:p.pending->entries){
    uint64_t record[2]{e.bank,uint64_t(e.weight>=0)|(uint64_t(e.route)<<1)};
    memcpy(dst,record,sizeof(record));dst+=sizeof(record);
    memcpy(dst,e.bias,sizeof(e.bias));dst+=sizeof(e.bias);memcpy(dst,e.keys,sizeof(e.keys));dst+=sizeof(e.keys);
    if(e.weight>=0){const auto&w=p.pending->weights[e.weight];memcpy(dst,w.weights,sizeof(w.weights));dst+=sizeof(w.weights);}
  }
}
int bank_pipeline_import(void*ptr,const void*input,size_t size){
  auto&m=*static_cast<Model*>(ptr);auto&p=m.bank_pipeline;
  if(size<PIPELINE_HEADER||m.content_credit_depth||p.building->scope_count)return 0;
  uint64_t h[12];memcpy(h,input,sizeof(h));bool legacy=h[0]==1;
  if((!legacy&&h[0]!=2)||h[1]!=E||h[2]!=uint64_t(p.enabled)||h[5]>1||h[9]>(N-2)*E||h[11]||
     (legacy?h[10]!=0:h[10]>h[9])||h[3]>h[4]||
     (!h[5]&&(h[3]!=h[4]||h[6]||h[7]||h[8]||h[9]||h[10]))||
     (h[5]&&(!p.enabled||h[4]!=h[3]+1||h[7]!=h[3]||h[8]!=h[4]||h[6]>h[7]||h[7]-h[6]>1)))return 0;
  size_t expected=sizeof(h)+(legacy?h[9]*sizeof(PipelineTileV1):h[9]*PIPELINE_ENTRY+h[10]*D*D*sizeof(float));
  if(size!=expected)return 0;
  std::unique_ptr<PipelineBatch> next;
  if(h[5]){
    next=std::make_unique<PipelineBatch>();next->read_version=h[6];next->base_version=h[7];next->sequence=h[8];
    next->entries.reserve(h[9]);next->weights.reserve(legacy?h[9]:h[10]);
    const auto*src=static_cast<const unsigned char*>(input)+sizeof(h);const auto*end=static_cast<const unsigned char*>(input)+size;
    uint64_t last=0;
    for(size_t i=0;i<h[9];i++){
      next->entries.emplace_back();auto&e=next->entries.back();uint64_t flags=3;
      if(legacy){
        PipelineTileV1 wire;memcpy(&wire,src,sizeof(wire));src+=sizeof(wire);e.bank=wire.bank;
        memcpy(e.bias,wire.bias,sizeof(e.bias));memcpy(e.keys,wire.keys,sizeof(e.keys));
        next->weights.emplace_back();auto&w=next->weights.back();w.bank=e.bank;memcpy(w.weights,wire.weights,sizeof(w.weights));
      }else{
        if(size_t(end-src)<PIPELINE_ENTRY)return 0;
        uint64_t record[2];memcpy(record,src,sizeof(record));src+=sizeof(record);e.bank=record[0];flags=record[1];
        if(flags>3)return 0;
        memcpy(e.bias,src,sizeof(e.bias));src+=sizeof(e.bias);memcpy(e.keys,src,sizeof(e.keys));src+=sizeof(e.keys);
        if(flags&1){
          if(size_t(end-src)<D*D*sizeof(float)||next->weights.size()>=h[10])return 0;
          next->weights.emplace_back();auto&w=next->weights.back();w.bank=e.bank;
          memcpy(w.weights,src,sizeof(w.weights));src+=sizeof(w.weights);
        }
      }
      if(e.bank<2*E||e.bank>=N*E||e.bank<=last)return 0;last=e.bank;
      e.route=flags&2;e.weighted=flags&1;e.weight=e.weighted?int(next->weights.size()-1):-1;
      if(e.weighted)for(float v:next->weights[e.weight].weights)if(!std::isfinite(v))return 0;
      for(float v:e.bias)if(!std::isfinite(v))return 0;
      for(float v:e.keys)if(!std::isfinite(v))return 0;
      next->bank_index[e.bank]=int(i);
    }
    if(src!=end||(!legacy&&next->weights.size()!=h[10]))return 0;
  }
  p.wait();pipeline_install(m);p.pending=std::move(next);p.version=h[3];p.sequence=h[4];return 1;
}
// The caller owns the API boundary. Entry rows: bank, has W, routes, scopes,
// factors. All aggregate calculations are here, never in the training loop.
void bank_pipeline_inspect(void*ptr,uint64_t*out,uint64_t*rows){
  auto&p=static_cast<Model*>(ptr)->bank_pipeline;p.wait();memset(out,0,24*sizeof(uint64_t));
  out[0]=p.version;out[1]=p.sequence;out[2]=bool(p.pending);out[3]=p.workers;out[4]=p.enabled;out[5]=p.fused;
  out[6]=pipeline_wire_size(p);out[7]=uintptr_t(p.building.get());out[8]=uintptr_t(p.pending.get());
  for(const auto&t:p.visible)out[9]+=p.visible_index[t.bank]>=0;
  auto storage=[&](const PipelineBatch*b){if(!b)return;
    out[10]+=sizeof(*b)+b->scopes.capacity()*sizeof(PipelineScope)+b->inputs.capacity()*sizeof(PipelineInput)+
      b->entries.capacity()*sizeof(PipelineEntry)+b->weights.capacity()*sizeof(PipelineWeightTile)+b->input_index.capacity()*sizeof(PipelineBatch::InputLocation);
    for(const auto&s:b->scopes)out[10]+=s.factors.capacity()*sizeof(PipelineFactor);
  };
  storage(p.building.get());storage(p.pending.get());storage(p.spare.get());out[10]+=p.visible.capacity()*sizeof(PipelineWeightTile);
  if(!p.pending)return;const auto&b=*p.pending;
  out[11]=b.entries.size();out[12]=b.weights.size();out[13]=b.scope_count;out[14]=b.inputs.size();
  out[16]=b.read_version;out[17]=b.base_version;
  out[18]=b.entries.size()*sizeof(PipelineEntry)+b.weights.size()*sizeof(PipelineWeightTile)+b.inputs.size()*sizeof(PipelineInput)+b.scope_count*sizeof(PipelineScope);
  for(size_t i=0;i<b.entries.size();i++){
    const auto&e=b.entries[i];uint64_t scopes=0,factors=0;
    for(int k=e.first;k>=0;k=b.scopes[k].next){scopes++;factors+=b.scopes[k].factors.size();}
    out[15]+=factors;
    if(rows){uint64_t row[5]{e.bank,uint64_t(e.weight>=0),uint64_t(e.route),scopes,factors};memcpy(rows+i*5,row,sizeof(row));}
  }
  out[18]+=out[15]*sizeof(PipelineFactor);out[19]=(out[11]-out[12])*D*D*sizeof(float);
  out[20]=(out[15]-out[14])*D*sizeof(float);
}
}
