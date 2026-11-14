inline void pipeline_install(Model&m){
  auto&p=m.bank_pipeline;
  for(const auto&tile:p.visible)if(p.visible_index[tile.bank]>=0)
    memcpy(m.W+tile.bank*D*D,tile.weights,sizeof(tile.weights));
  p.visible.clear();p.visible_index.fill(-1);
}
inline void pipeline_capture(Model&m,int index,Model::ContentBankWorkspace&bank){
  auto&p=m.bank_pipeline;p.building.emplace_back();auto&scope=p.building.back();
  scope.bank=index;scope.route=bank.gradient.route;
  memcpy(scope.bias,bank.gradient.bias,sizeof(scope.bias));memcpy(scope.keys,bank.gradient.keys,sizeof(scope.keys));
  scope.factors.reserve(bank.requests.size());
  for(const auto&request:bank.requests){
    scope.factors.emplace_back();auto&f=scope.factors.back();f.ref=request.ref;f.mask=request.mask;
    memcpy(f.factor,request.factor,sizeof(f.factor));
    memcpy(f.input,m.content_records[request.ref.index].input,sizeof(f.input));
  }
  for(const auto&request:bank.routes){
    const float*input=m.content_records[request.ref.index].input;uint64_t mask=request.mask;
    while(mask){int j=__builtin_ctzll(mask);mask&=mask-1;scope.keys[j]+=request.step*input[j];}
  }
  p.metrics[2]++;p.metrics[3]+=scope.factors.size();p.metrics[5]+=scope.factors.size()*D*sizeof(float);
  m.content_factor_metrics[1]+=bank.requests.size();m.content_factor_metrics[3]+=bank.routes.size();
  m.content_factor_metrics[5]=std::max(m.content_factor_metrics[5],uint64_t(bank.requests.size()+bank.routes.size()));
  bank.requests.clear();bank.routes.clear();bank.active=false;
}
inline void pipeline_commit(Model&m){
  auto&p=m.bank_pipeline;if(!p.pending)return;
  auto start=std::chrono::steady_clock::now();p.wait();
  p.metrics[7]+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
  pipeline_install(m);auto batch=std::move(p.pending);
  p.metrics[6]+=batch->compute_ns;p.visible=std::move(batch->tiles);
  for(size_t i=0;i<p.visible.size();i++){
    const auto&t=p.visible[i];p.visible_index[t.bank]=int(i);
    memcpy(m.bias+t.bank*D,t.bias,sizeof(t.bias));memcpy(m.keys+t.bank*D,t.keys,sizeof(t.keys));
    m.refresh_key(t.bank/E,t.bank%E);
  }
  p.version++;p.metrics[1]++;
}
inline void pipeline_finish_step(Model&m){
  auto&p=m.bank_pipeline;if(!p.enabled)return;
  uint64_t read_version=p.version;
  pipeline_commit(m);
  // Even an empty byte has a pending boundary: chunking must not alter time.
  auto batch=std::make_shared<PipelineBatch>();batch->read_version=read_version;
  batch->base_version=p.version;batch->sequence=++p.sequence;batch->scopes=std::move(p.building);
  std::array<bool,N*E> used{};
  for(const auto&s:batch->scopes)used[s.bank]=true;
  batch->tiles.reserve(std::count(used.begin(),used.end(),true));
  uint64_t bytes=0;
  for(int index=0;index<N*E;index++)if(used[index]){
    batch->tiles.emplace_back();auto&t=batch->tiles.back();t.bank=index;
    memcpy(t.weights,m.weight_bank(index),sizeof(t.weights));
    memcpy(t.bias,m.bias+index*D,sizeof(t.bias));memcpy(t.keys,m.keys+index*D,sizeof(t.keys));
  }
  for(const auto&s:batch->scopes)bytes+=sizeof(s)+s.factors.size()*sizeof(PipelineFactor);
  bytes+=batch->tiles.size()*sizeof(PipelineTile);p.metrics[9]=std::max(p.metrics[9],bytes);
  p.metrics[0]++;p.metrics[4]+=batch->tiles.size();p.pending=batch;
  if(p.worker)p.worker->submit(batch,p.fused);else pipeline_compute(*batch,p.fused);
}
inline void pipeline_drain(Model&m){
  if(m.bank_pipeline.enabled&&!m.bank_pipeline.building.empty())pipeline_finish_step(m);
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
  if(install){p.visible_index[index]=-1;p.metrics[8]+=D;}
}
