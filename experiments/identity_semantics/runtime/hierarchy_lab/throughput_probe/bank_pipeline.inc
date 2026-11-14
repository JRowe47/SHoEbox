inline void pipeline_install(Model&m,const PipelineBatch*next=nullptr){
  auto&p=m.bank_pipeline;
  for(const auto&tile:p.visible){
    if(p.visible_index[tile.bank]<0)continue;
    int replacement=next?next->bank_index[tile.bank]:-1;
    // A successor already incorporates this version. Do not install a row
    // that would immediately be superseded by that successor's weight tile.
    if(replacement<0||next->entries[replacement].weight<0)
      memcpy(m.W+tile.bank*D*D,tile.weights,sizeof(tile.weights));
    p.visible_index[tile.bank]=-1;
  }
}
inline void pipeline_capture(Model&m,int index,Model::ContentBankWorkspace&bank){
  auto&batch=*m.bank_pipeline.building;
  size_t position=batch.scope_count++;
  if(position==batch.scopes.size())batch.scopes.emplace_back();
  auto&scope=batch.scopes[position];scope.next=-1;scope.route=bank.gradient.route;
  memcpy(scope.bias,bank.gradient.bias,sizeof(scope.bias));memcpy(scope.keys,bank.gradient.keys,sizeof(scope.keys));
  scope.factors.resize(bank.requests.size());
  for(size_t i=0;i<bank.requests.size();i++){
    const auto&request=bank.requests[i];auto&f=scope.factors[i];f.mask=request.mask;
    memcpy(f.factor,request.factor,sizeof(f.factor));
    size_t slot=request.ref.index;
    if(slot>=batch.input_index.size())batch.input_index.resize(slot+1);
    auto&location=batch.input_index[slot];
    if(location.generation!=batch.generation||location.serial!=request.ref.serial||location.region!=request.ref.region){
      location={batch.generation,request.ref.serial,uint32_t(batch.inputs.size()),request.ref.region};
      batch.inputs.emplace_back();auto&input=batch.inputs.back();input.ref=request.ref;
      memcpy(input.values,m.content_records[slot].input,sizeof(input.values));
    }
    f.input=location.index;
  }
  for(const auto&request:bank.routes){
    const float*input=m.content_records[request.ref.index].input;uint64_t mask=request.mask;
    while(mask){int j=__builtin_ctzll(mask);mask&=mask-1;scope.keys[j]+=request.step*input[j];}
  }
  int&location=batch.bank_index[index];
  if(location<0){location=int(batch.entries.size());batch.entries.emplace_back();batch.entries.back().bank=index;}
  auto&entry=batch.entries[location];
  if(entry.last<0)entry.first=int(position);else batch.scopes[entry.last].next=int(position);
  entry.last=int(position);entry.route|=scope.route;entry.weighted|=!scope.factors.empty();
  SHOEBOX_DIAGNOSTIC(m.content_factor_metrics[1]+=bank.requests.size();m.content_factor_metrics[3]+=bank.routes.size();
    m.content_factor_metrics[5]=std::max(m.content_factor_metrics[5],uint64_t(bank.requests.size()+bank.routes.size()));)
  bank.requests.clear();bank.routes.clear();bank.active=false;
}
inline void pipeline_commit(Model&m){
  auto&p=m.bank_pipeline;if(!p.pending)return;
  p.wait();auto&batch=*p.pending;pipeline_install(m,&batch);
  p.visible.swap(batch.weights);
  for(size_t i=0;i<p.visible.size();i++)p.visible_index[p.visible[i].bank]=int(i);
  for(const auto&e:batch.entries){
    memcpy(m.bias+e.bank*D,e.bias,sizeof(e.bias));
    if(e.route){memcpy(m.keys+e.bank*D,e.keys,sizeof(e.keys));m.refresh_key(e.bank/E,e.bank%E);}
  }
  p.version++;p.spare=std::move(p.pending);
}
inline void pipeline_finish_step(Model&m){
  auto&p=m.bank_pipeline;if(!p.enabled)return;
  uint64_t read_version=p.version;pipeline_commit(m);
  auto batch=std::move(p.building);p.building=std::move(p.spare);p.building->reset();
  batch->read_version=read_version;batch->base_version=p.version;batch->sequence=++p.sequence;
  std::sort(batch->entries.begin(),batch->entries.end(),[](const auto&a,const auto&b){return a.bank<b.bank;});
  for(size_t i=0;i<batch->entries.size();i++){
    auto&e=batch->entries[i];batch->bank_index[e.bank]=int(i);
    if(e.weighted){
      e.weight=int(batch->weights.size());batch->weights.emplace_back();auto&w=batch->weights.back();w.bank=e.bank;
      memcpy(w.weights,m.weight_bank(e.bank),sizeof(w.weights));
    }
    memcpy(e.bias,m.bias+e.bank*D,sizeof(e.bias));memcpy(e.keys,m.keys+e.bank*D,sizeof(e.keys));
  }
  // Empty bytes still advance the same delayed version boundary.
  p.pending=std::move(batch);
  if(p.worker)p.worker->submit(p.pending.get(),p.fused);else pipeline_compute(*p.pending,p.fused);
}
inline void pipeline_drain(Model&m){
  if(m.bank_pipeline.enabled&&m.bank_pipeline.building->scope_count)pipeline_finish_step(m);
  pipeline_commit(m);pipeline_install(m);
}
// Same AVX reduction as dot64, with the version install using those same loads.
inline float pipeline_dot_copy(float*dest,const float*source,const float*input){
#ifdef __AVX2__
  __m256 v=_mm256_setzero_ps(),w=_mm256_setzero_ps();
  for(int j=0;j<D;j+=16){
    __m256 a=_mm256_loadu_ps(source+j),b=_mm256_loadu_ps(source+j+8);
    _mm256_storeu_ps(dest+j,a);_mm256_storeu_ps(dest+j+8,b);
    v=_mm256_add_ps(v,_mm256_mul_ps(a,_mm256_loadu_ps(input+j)));
    w=_mm256_add_ps(w,_mm256_mul_ps(b,_mm256_loadu_ps(input+j+8)));
  }
  v=_mm256_add_ps(v,w);__m128 x=_mm_add_ps(_mm256_castps256_ps128(v),_mm256_extractf128_ps(v,1));
  x=_mm_hadd_ps(x,x);x=_mm_hadd_ps(x,x);return _mm_cvtss_f32(x);
#else
  float z=0;for(int j=0;j<D;j++){float a=source[j];dest[j]=a;z+=a*input[j];}return z;
#endif
}
inline void pipeline_content_forward(Model&m,int index,const float*input,float*output){
  auto&p=m.bank_pipeline;const float*w=m.weight_bank(index);
  bool install=p.enabled&&p.fused&&p.visible_index[index]>=0;
  for(int i=0;i<D;i++)output[i]=install?pipeline_dot_copy(m.W+index*D*D+i*D,w+i*D,input):dot64(w+i*D,input);
  if(install){p.visible_index[index]=-1;}
}
