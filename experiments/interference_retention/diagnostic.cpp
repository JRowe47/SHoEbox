// Read-only state instrumentation and externally timed freeze for diagnosis only.
// No function below is called by candidate inference.
extern "C" {
int ir_state(void* p,double* out){return guard([&]{
 auto& a=*static_cast<shoe::QueryAdapter*>(p);auto& s=a.model->levels[1].retained;
 out[0]=s.count;out[1]=s.clock;
 for(int j=0;j<64;++j)out[2+j]=s.last[j];
 for(int k=0;k<16;++k){out[66+k*65]=s.units[k].born;for(int j=0;j<64;++j)out[67+k*65+j]=s.units[k].stable[j];}
});}
int ir_decode(void* p,double* out){return guard([&]{
 auto& a=*static_cast<shoe::QueryAdapter*>(p);a.prepare();auto& s=a.model->levels[1].retained;
 out[0]=a.cache.slots[0];out[1]=a.cache.phases[0];out[2]=a.cache.slots[1];out[3]=a.cache.phases[1];
 for(int k=0;k<16;++k)for(int ph=0;ph<4;++ph){int base=4+(k*4+ph)*5;
  auto x=shoe::source_vector(s.units[k].stable);out[base]=shoe::affine(a.head.gate[ph],x);
  shoe::LowRank key{};for(int r=0;r<16;++r)key[r]=std::tanh(shoe::affine(a.head.key[ph*16+r],x));
  float best=1e30f,dist=0;int chosen=0;for(int r=0;r<16;++r){float d=key[r]-a.cache.query[r];dist+=d*d;}
  for(int b=0;b<256;++b){float sum=0;for(int r=0;r<16;++r){float d=key[r]-shoe::byte_code(b,r);sum+=d*d;}if(sum<best){best=sum;chosen=b;}}
  int value=0;for(int r=0;r<8;++r)if(shoe::affine(a.head.value[ph*8+r],x)>0)value|=1<<r;
  out[base+1]=chosen;out[base+2]=value;out[base+3]=best;out[base+4]=dist;
 }
});}
int ir_frozen_span(void* p,const uint8_t* data,size_t n){return guard([&]{
 auto& a=*static_cast<shoe::QueryAdapter*>(p);if(a.training)throw std::logic_error("frozen diagnostic only");
 auto saved=a.model->levels[1].retained;
 for(size_t i=0;i<n;++i){a.model->observe(data[i]);auto& s=a.model->levels[1].retained;s.units=saved.units;s.count=saved.count;s.clock=saved.clock;}
});}
}
