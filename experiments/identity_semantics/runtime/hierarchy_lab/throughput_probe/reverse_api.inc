// Checkpoint ABI 1 is independent of the constituent/neural record layouts.
// Partial epochs own both their fixed bank version and every pending root.
extern "C" {
int reverse_configure(void*p,int policy,int epoch,float scale){
  auto&m=*static_cast<Model*>(p);
  if((policy!=1&&policy!=2)||(epoch!=1&&epoch!=4&&epoch!=8&&epoch!=16)||
     (policy==1&&epoch!=1)||!std::isfinite(scale)||scale<0||scale>8||
     m.content_credit_depth||!m.reverse_roots.empty()||!m.reverse_bank_version.empty())return 0;
  m.reverse_policy=policy;m.reverse_epoch_bytes=epoch;m.reverse_update_scale=scale;return 1;
}
void reverse_finish(void*p){reverse_finish_epoch(*static_cast<Model*>(p));}
void reverse_status(void*p,uint64_t*out){
  auto&m=*static_cast<Model*>(p);memcpy(out,m.reverse_metrics,sizeof(m.reverse_metrics));
  out[8]=m.reverse_version;out[9]=m.reverse_epoch_position;out[10]=m.reverse_roots.size();
  out[11]=m.reverse_bank_version.size()*sizeof(float);
}
void reverse_reset_status(void*p){auto&m=*static_cast<Model*>(p);memset(m.reverse_metrics,0,sizeof(m.reverse_metrics));}
size_t reverse_state_size(void*p){auto&m=*static_cast<Model*>(p);
  return 6*sizeof(uint64_t)+m.reverse_bank_version.size()*sizeof(float)+m.reverse_roots.size()*sizeof(Model::ReverseRoot);
}
void reverse_export(void*p,void*out){
  auto&m=*static_cast<Model*>(p);uint64_t header[6]{1,uint64_t(m.reverse_policy),uint64_t(m.reverse_epoch_bytes),uint64_t(m.reverse_epoch_position),m.reverse_version,m.reverse_roots.size()};
  auto*dest=static_cast<unsigned char*>(out);memcpy(dest,header,sizeof(header));dest+=sizeof(header);
  if(!m.reverse_bank_version.empty()){size_t n=m.reverse_bank_version.size()*sizeof(float);memcpy(dest,m.reverse_bank_version.data(),n);dest+=n;}
  if(!m.reverse_roots.empty())memcpy(dest,m.reverse_roots.data(),m.reverse_roots.size()*sizeof(Model::ReverseRoot));
}
int reverse_import(void*p,const void*input,size_t size){
  auto&m=*static_cast<Model*>(p);if(size<6*sizeof(uint64_t)||m.content_credit_depth)return 0;
  auto*data=static_cast<const unsigned char*>(input);uint64_t header[6];memcpy(header,data,sizeof(header));data+=sizeof(header);
  if(header[0]!=1||header[1]!=uint64_t(m.reverse_policy)||header[2]!=uint64_t(m.reverse_epoch_bytes)||
     header[3]>=header[2]||(header[2]==1&&(header[3]||header[5])))return 0;
  size_t floats=header[3]?N*E*D*(D+2):0,bytes=sizeof(header)+floats*sizeof(float);
  if(size<bytes||header[5]>(size-bytes)/sizeof(Model::ReverseRoot)||size-bytes!=header[5]*sizeof(Model::ReverseRoot)||(!header[3]&&header[5]))return 0;
  std::vector<float> banks(floats);if(floats){memcpy(banks.data(),data,floats*sizeof(float));data+=floats*sizeof(float);}
  for(float value:banks)if(!std::isfinite(value))return 0;
  std::vector<Model::ReverseRoot> roots(header[5]);if(!roots.empty())memcpy(roots.data(),data,roots.size()*sizeof(Model::ReverseRoot));
  for(const auto&root:roots){
    if(!constituent_record(m,root.ref)||root.receiver<2||root.receiver>=N||root.teaching<0||root.teaching>1||!std::isfinite(root.rate)||root.rate==0||root.reserved||
       constituent_record(m,root.ref)->ref.region!=root.ref.region)return 0;
    for(float v:root.gradient)if(!std::isfinite(v))return 0;
    for(float v:root.scales)if(!std::isfinite(v))return 0;
  }
  m.reverse_bank_version=std::move(banks);m.reverse_roots=std::move(roots);
  m.reverse_epoch_position=int(header[3]);m.reverse_version=header[4];m.refresh_keys();return 1;
}
void reverse_visible_banks(void*p,float*out){auto&m=*static_cast<Model*>(p);
  memcpy(out,m.read_weights(),N*E*D*D*sizeof(float));out+=N*E*D*D;
  memcpy(out,m.read_bias(),N*E*D*sizeof(float));out+=N*E*D;
  memcpy(out,m.read_keys(),N*E*D*sizeof(float));
}
}
