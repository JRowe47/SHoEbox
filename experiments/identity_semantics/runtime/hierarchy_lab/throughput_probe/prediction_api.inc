// Prediction extension ABI 1. Import validates a detached candidate first.
extern "C" {
int prediction_configure(void*p,int compact,int rank,int stride,float scale){
  auto&m=*static_cast<Model*>(p);auto&z=m.prediction;
  if((compact!=0&&compact!=1)||(rank!=8&&rank!=16&&rank!=32&&rank!=64)||
     (stride!=1&&stride!=2&&stride!=4&&stride!=8&&stride!=16)||!std::isfinite(scale)||scale<=0||scale>4||
     ((compact||stride>1)&&(!m.composition.enabled||m.reverse_epoch_bytes!=1))||
     m.composition.stream.byte_position||m.composition.stream.semantic_serial||z.stream.samples)return 0;
  for(int r=0;r<N;r++)if(m.s.valid[r])return 0;
  bool initialize=compact&&(!z.compact||rank!=z.rank);
  if(initialize){
    z.forecast.assign(FORECAST_REGIONS*D*rank,0.f);float pool[D];
    for(int r=2;r<N;r++)if(prediction_region(r)>=0)for(int i=0;i<D;i++){
      prediction_pool(m.P+(r*D+i)*D,pool,rank);
      memcpy(z.forecast.data()+(prediction_region(r)*D+i)*rank,pool,rank*sizeof(float));
    }
    memset(z.forecast_version,0,sizeof(z.forecast_version));
  }
  if(!compact)z.forecast.clear();
  if(stride>1&&z.stride==1)for(int j=0;j<D;j++){
    z.fast[0][j]=0;z.fast[1][j]=.125f;z.fast[2][j]=.125f;z.fast[3][j]=.1f;
  }
  z.compact=compact;z.rank=rank;z.stride=stride;z.slow_scale=scale;
  prediction_reset_stream(m);return 1;
}
size_t prediction_state_size(void*p,int fitted){
  const auto&z=static_cast<Model*>(p)->prediction;
  return 8*sizeof(uint64_t)+z.forecast.size()*sizeof(float)+sizeof(z.fast)+
    sizeof(z.forecast_version)+sizeof(z.response_version)+(fitted?0:sizeof(PredictionStream));
}
void prediction_export(void*p,void*out,int fitted){
  auto&z=static_cast<Model*>(p)->prediction;
  uint64_t h[8]{1,uint64_t(fitted),uint64_t(z.compact),uint64_t(z.rank),uint64_t(z.stride),z.forecast.size(),0,E};
  uint32_t bits;memcpy(&bits,&z.slow_scale,4);h[6]=bits;
  auto*dest=static_cast<unsigned char*>(out);
  auto put=[&](const void*src,size_t n){if(n)memcpy(dest,src,n);dest+=n;};
  put(h,sizeof(h));put(z.forecast.data(),z.forecast.size()*sizeof(float));put(z.fast,sizeof(z.fast));
  put(z.forecast_version,sizeof(z.forecast_version));put(z.response_version,sizeof(z.response_version));
  if(!fitted)put(&z.stream,sizeof(z.stream));
}
int prediction_import(void*p,const void*input,size_t size,int fitted){
  auto&m=*static_cast<Model*>(p);auto&z=m.prediction;
  if(size!=prediction_state_size(p,fitted)||m.content_credit_depth)return 0;
  const auto*src=static_cast<const unsigned char*>(input);
  auto get=[&](void*out,size_t n){if(n)memcpy(out,src,n);src+=n;};
  uint64_t h[8];get(h,sizeof(h));uint32_t bits;memcpy(&bits,&z.slow_scale,4);
  if(h[0]!=1||h[1]!=uint64_t(fitted)||h[2]!=uint64_t(z.compact)||h[3]!=uint64_t(z.rank)||
     h[4]!=uint64_t(z.stride)||h[5]!=z.forecast.size()||h[6]!=bits||h[7]!=E)return 0;
  auto fresh=std::make_unique<PredictionRuntime>();auto&v=*fresh;v.forecast.resize(z.forecast.size());
  get(v.forecast.data(),v.forecast.size()*sizeof(float));get(v.fast,sizeof(v.fast));
  get(v.forecast_version,sizeof(v.forecast_version));get(v.response_version,sizeof(v.response_version));
  auto finite=[](const float*a,size_t n){for(size_t i=0;i<n;i++)if(!std::isfinite(a[i]))return false;return true;};
  if(!finite(v.forecast.data(),v.forecast.size())||!finite(&v.fast[0][0],4*D))return 0;
  if(!fitted){
    get(&v.stream,sizeof(v.stream));const auto&s=v.stream;
    auto valid_view=[&](const CompactFeatureView&f,int r){
      return f.tick>=0&&f.tick<=m.s.tick&&f.forecast_version<=v.forecast_version[r]&&f.response_version<=v.response_version[r]&&
        finite(f.response,D)&&finite(f.content,D)&&finite(f.context,MK_FEATURES)&&
        (!f.semantic.serial||(f.semantic.region>=2&&f.semantic.region<N&&f.semantic.index>=0));
    };
    for(int r=0;r<N;r++){
      if(!valid_view(s.current[r],r)||!valid_view(s.pending[r],r))return 0;
      const auto&t=m.composition.stream.teaching[r];
      if(z.compact&&prediction_region(r)>=0&&t.tick&&
        (s.pending[r].tick!=t.tick||s.pending[r].forecast_version!=v.forecast_version[r]))return 0;
    }
    auto*slow=current_lace(m,17);
    if(s.reserved||s.slow_tick<0||s.slow_tick>m.s.tick||s.slow_byte<0||s.slow_byte>m.composition.stream.byte_position||
       (s.slow_tick&&(!slow||slow->tick!=s.slow_tick||s.current[17].tick!=s.slow_tick))||
       s.samples>uint64_t(z.stride)||(!s.slow_tick&&s.samples)||
       (s.samples&&(z.stride==1||s.mode<1||s.mode>6||s.mode==4||s.mode==5))||
       !std::isfinite(s.rate)||s.rate<0||!finite(s.gradient,D)||!finite(s.slow,D)||!finite(s.output,D)||!finite(s.feedback,D))return 0;
  }
  z.forecast=std::move(v.forecast);memcpy(z.fast,v.fast,sizeof(z.fast));
  memcpy(z.forecast_version,v.forecast_version,sizeof(z.forecast_version));memcpy(z.response_version,v.response_version,sizeof(z.response_version));
  if(!fitted)z.stream=v.stream;return 1;
}
void prediction_finish(void*p){auto&m=*static_cast<Model*>(p);prediction_slow_flush(m);pipeline_drain(m);}
void prediction_inspect(void*p,uint64_t*out,float*features){
  const auto&z=static_cast<Model*>(p)->prediction;const auto&s=z.stream;
  out[0]=z.compact;out[1]=z.rank;out[2]=z.stride;out[3]=s.slow_tick;out[4]=s.slow_byte;out[5]=s.samples;
  out[6]=z.forecast.size();out[7]=sizeof(PredictionStream);
  for(int r=0;r<N;r++){
    const auto&v=s.current[r];auto*row=out+8+r*8;
    row[0]=v.tick;row[1]=v.publication;row[2]=v.semantic.serial;row[3]=v.bank_version;
    row[4]=v.response_version;row[5]=v.forecast_version;row[6]=s.pending[r].tick;row[7]=z.forecast_version[r];
    memcpy(features+r*(2*D+MK_FEATURES),v.response,D*sizeof(float));
    memcpy(features+r*(2*D+MK_FEATURES)+D,v.content,D*sizeof(float));
    memcpy(features+r*(2*D+MK_FEATURES)+2*D,v.context,MK_FEATURES*sizeof(float));
  }
}
}
