// Routing uses only this constituent's descriptor and learned child payload.
// In particular, usage pressure, feedback, running means and recurrent memory
// cannot choose or change its content transform at fixed parameters.
inline void content_expert_encode(Model&m,int r,ConstituentRecord&node,const float*structure,float gain,float*out){
  if(size_t(node.ref.index)>=m.content_records.size())m.content_records.resize(constituent_capacity(m));
  auto&saved=m.content_records[node.ref.index];saved=ContentExpertRecord{};
  saved.ref=node.ref;node.encoder=1;
  for(int j=0;j<D;j++)saved.input[j]=structure[j]+gain*node.payload[j];
  int sparse=r>=int(m.lcfg[8])?int(m.lcfg[2]):D;
  saved.input_mask=feature_mask(saved.input,sparse);
  int indices[D],count=0;uint64_t mask=saved.input_mask;
  while(mask){indices[count++]=__builtin_ctzll(mask);mask&=mask-1;}
  float scores[E];router_scores(m,r,saved.input,indices,count,scores);
  int best=0,runner=0;float first=-1e30f,second=-1e30f;
  for(int e=0;e<E;e++){
    if(scores[e]>first){second=first;runner=best;first=scores[e];best=e;}
    else if(scores[e]>second){second=scores[e];runner=e;}
  }
  saved.expert=best;saved.runner=runner;
  if(m.lcfg[3]>0){
    float p=1.f/(1.f+std::exp(-(first-second)/m.lcfg[4]));
    saved.gate=2*p;saved.gate_derivative=2*p*(1-p)/m.lcfg[4];
    for(int j=0;j<D;j++)if(saved.input_mask&(uint64_t(1)<<j))
      saved.route_derivative[j]=saved.gate_derivative*(m.read_keys()[(r*E+best)*D+j]-m.read_keys()[(r*E+runner)*D+j]);
  }
  const float*w=m.read_weights()+(r*E+best)*D*D,*b=m.read_bias()+(r*E+best)*D;
  for(int i=0;i<D;i++){
    saved.residual[i]=dot64(w+i*D,saved.input);
    out[i]=b[i]+saved.gate*saved.residual[i];
  }
  m.content_expert_metrics[r][0]++;
}
