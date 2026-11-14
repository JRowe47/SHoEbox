// Action 6 wire 3: a completed pending job plus an unfinished factor arena.
// Read/compute/commit versions are schedule state; worker count is execution.
constexpr size_t EPOCH_HEADER=20*sizeof(uint64_t),EPOCH_INPUT=24+D*sizeof(float),
  EPOCH_SCOPE=16+2*D*sizeof(float),EPOCH_FACTOR=16+D*sizeof(float);
inline size_t epoch_wire_size(const BankPipeline&p){
  const auto&b=*p.building;size_t n=EPOCH_HEADER+pipeline_wire_size(p)+b.inputs.size()*EPOCH_INPUT+b.entries.size()*16;
  for(size_t i=0;i<b.scope_count;i++)n+=EPOCH_SCOPE+b.scopes[i].factors.size()*EPOCH_FACTOR;
  return n;
}
extern "C" {
int bank_epoch_configure(void*ptr,int epoch,int lag,int dense){
  auto&m=*static_cast<Model*>(ptr);auto&p=m.bank_pipeline;
  if((epoch!=1&&epoch!=4&&epoch!=8&&epoch!=16)||(lag!=0&&lag!=1)||(dense!=0&&dense!=1)||
     m.composition.stream.byte_position||m.composition.stream.semantic_serial||p.pending||p.position||p.building->scope_count||
     m.content_credit_depth||!m.event.heads||m.reverse_epoch_bytes!=1||m.reverse_policy!=2)return 0;
  for(int r=0;r<N;r++)if(m.s.valid[r])return 0;
  p.enabled=true;p.epochs=true;p.epoch_bytes=epoch;p.lag_epochs=lag;p.dense=dense;return 1;
}
size_t bank_pipeline_state_size(void*ptr){
  auto&p=static_cast<Model*>(ptr)->bank_pipeline;p.wait();return p.epochs?epoch_wire_size(p):pipeline_wire_size(p);
}
void bank_pipeline_export(void*ptr,void*out){
  auto&m=*static_cast<Model*>(ptr);auto&p=m.bank_pipeline;
  if(!p.epochs){pipeline_v2_export(ptr,out);return;}
  p.wait();pipeline_install(m);const auto&b=*p.building;uint64_t factors=0;
  for(size_t i=0;i<b.scope_count;i++)factors+=b.scopes[i].factors.size();
  uint64_t h[20]{3,E,uint64_t(p.epoch_bytes),uint64_t(p.lag_epochs),uint64_t(p.dense),uint64_t(p.position),p.bytes,
    pipeline_wire_size(p),b.inputs.size(),b.entries.size(),b.scope_count,factors,b.objectives,0,0,0,0,0,0,0};
  if(p.pending){h[13]=p.pending->first_byte;h[14]=p.pending->last_byte;h[15]=p.pending->objectives;}
  auto*dst=static_cast<unsigned char*>(out);auto put=[&](const void*data,size_t n){memcpy(dst,data,n);dst+=n;};
  put(h,sizeof(h));pipeline_v2_export(ptr,dst);dst+=h[7];
  for(const auto&i:b.inputs){uint64_t ref[3]{i.ref.serial,uint64_t(i.ref.region),uint64_t(i.ref.index)};put(ref,sizeof(ref));put(i.values,sizeof(i.values));}
  for(const auto&e:b.entries){
    uint64_t count=0;for(int k=e.first;k>=0;k=b.scopes[k].next)++count;
    uint64_t entry[2]{e.bank,count};put(entry,sizeof(entry));
    for(int k=e.first;k>=0;k=b.scopes[k].next){const auto&s=b.scopes[k];
      uint64_t scope[2]{uint64_t(s.route)|(uint64_t(s.objectives)<<1),s.factors.size()};put(scope,sizeof(scope));
      put(s.bias,sizeof(s.bias));put(s.keys,sizeof(s.keys));
      for(const auto&f:s.factors){uint64_t factor[2]{f.mask,f.input};put(factor,sizeof(factor));put(f.factor,sizeof(f.factor));}
    }
  }
}
int bank_pipeline_import(void*ptr,const void*input,size_t size){
  auto&m=*static_cast<Model*>(ptr);auto&p=m.bank_pipeline;
  if(!p.epochs)return pipeline_v2_import(ptr,input,size);
  if(size<EPOCH_HEADER||m.content_credit_depth)return 0;
  const auto*src=static_cast<const unsigned char*>(input);const auto*end=src+size;
  auto get=[&](void*dst,size_t n){if(size_t(end-src)<n)return false;memcpy(dst,src,n);src+=n;return true;};
  uint64_t h[20];if(!get(h,sizeof(h)))return 0;
  if(h[0]!=3||h[1]!=E||h[2]!=uint64_t(p.epoch_bytes)||h[3]!=uint64_t(p.lag_epochs)||h[4]!=uint64_t(p.dense)||
     h[5]>=h[2]||h[6]<h[5]||h[7]<PIPELINE_HEADER||h[7]>size_t(end-src)||h[8]>size/EPOCH_INPUT||h[8]>h[11]||
     h[9]>(N-2)*E||h[10]>size/EPOCH_SCOPE||h[11]>size/EPOCH_FACTOR||h[12]>3||h[15]>3||h[16]||h[17]||h[18]||h[19])return 0;
  const auto*pending=src;src+=h[7];uint64_t ph[12];memcpy(ph,pending,sizeof(ph));
  if(ph[0]!=2||(!h[3]&&ph[5])||(!ph[5]&&(h[13]||h[14]||h[15]))||
     (ph[5]&&(h[13]>h[14]||h[14]>h[6]-h[5]||h[14]-h[13]>h[2])))return 0;
  auto building=std::make_unique<PipelineBatch>();building->reset();building->inputs.reserve(h[8]);
  for(size_t i=0;i<h[8];i++){
    uint64_t ref[3];if(!get(ref,sizeof(ref))||!ref[0]||ref[1]<2||ref[1]>=N||ref[2]>=m.content_records.size())return 0;
    building->inputs.emplace_back();auto&v=building->inputs.back();v.ref={ref[0],int(ref[1]),int(ref[2])};
    if(!get(v.values,sizeof(v.values)))return 0;for(float f:v.values)if(!std::isfinite(f))return 0;
    if(ref[2]>=building->input_index.size())building->input_index.resize(ref[2]+1);
    building->input_index[ref[2]]={building->generation,ref[0],uint32_t(i),int(ref[1])};
  }
  uint64_t factors=0,objectives=0;
  for(size_t i=0;i<h[9];i++){
    uint64_t entry[2];if(!get(entry,sizeof(entry))||entry[0]<2*E||entry[0]>=N*E||building->bank_index[entry[0]]>=0||!entry[1]||entry[1]>h[10]-building->scope_count)return 0;
    building->bank_index[entry[0]]=int(i);building->entries.emplace_back();auto&e=building->entries.back();e.bank=entry[0];
    for(size_t k=0;k<entry[1];k++){
      uint64_t scope[2];if(!get(scope,sizeof(scope))||scope[0]<2||scope[0]>7||scope[1]>h[11]-factors)return 0;
      int index=int(building->scope_count++);building->scopes.emplace_back();auto&s=building->scopes.back();
      if(e.last<0)e.first=index;else building->scopes[e.last].next=index;e.last=index;
      s.route=scope[0]&1;s.objectives=uint32_t(scope[0]>>1);objectives|=s.objectives;e.route|=s.route;e.weighted|=scope[1]!=0;
      if(!get(s.bias,sizeof(s.bias))||!get(s.keys,sizeof(s.keys)))return 0;
      for(float f:s.bias)if(!std::isfinite(f))return 0;for(float f:s.keys)if(!std::isfinite(f))return 0;
      for(size_t q=0;q<scope[1];q++){
        uint64_t factor[2];if(!get(factor,sizeof(factor))||factor[1]>=h[8])return 0;
        s.factors.emplace_back();auto&f=s.factors.back();f.mask=factor[0];f.input=uint32_t(factor[1]);
        if(!get(f.factor,sizeof(f.factor)))return 0;for(float v:f.factor)if(!std::isfinite(v))return 0;
      }
      factors+=scope[1];
    }
  }
  if(src!=end||building->scope_count!=h[10]||factors!=h[11]||objectives!=h[12])return 0;
  building->objectives=objectives;
  // The nested decoder validates its entire result before mutation. Only then
  // replace the partial arena, so every rejection is atomic.
  if(!pipeline_v2_import(ptr,pending,h[7]))return 0;
  p.building=std::move(building);p.position=int(h[5]);p.bytes=h[6];
  if(p.pending){p.pending->first_byte=h[13];p.pending->last_byte=h[14];p.pending->objectives=h[15];}
  return 1;
}
void bank_epoch_inspect(void*ptr,uint64_t*out){
  auto&p=static_cast<Model*>(ptr)->bank_pipeline;p.wait();memset(out,0,20*sizeof(uint64_t));
  out[0]=p.epochs;out[1]=p.epoch_bytes;out[2]=p.lag_epochs;out[3]=p.dense;out[4]=p.position;out[5]=p.bytes;
  out[6]=p.building->entries.size();out[7]=p.building->scope_count;out[8]=p.building->inputs.size();out[10]=p.building->objectives;
  for(size_t i=0;i<p.building->scope_count;i++)out[9]+=p.building->scopes[i].factors.size();
  if(p.pending){out[11]=p.pending->first_byte;out[12]=p.pending->last_byte;out[13]=p.pending->objectives;}
  out[14]=p.epochs?epoch_wire_size(p):pipeline_wire_size(p);out[15]=(N-2)*E*D*(D+2)*sizeof(float);
}
}
