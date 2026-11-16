// Read-only diagnostic export for an independent Python arithmetic check.
// This entry point is never called by fitting or candidate inference.
extern "C" int a2_probe(void* ptr,double* out){return guard([&]{
 auto& a=*static_cast<shoe::QueryAdapter*>(ptr);a.prepare();int i=0;
 for(const auto& row:a.head.gate)for(float v:row)out[i++]=v;
 for(const auto& row:a.head.key)for(float v:row)out[i++]=v;
 for(const auto& row:a.head.value)for(float v:row)out[i++]=v;
 for(int l=0;l<2;++l){const auto& s=a.model->levels[l].retained;out[i++]=s.count;out[i++]=s.clock;
  for(const auto& u:s.units){out[i++]=u.born;for(float v:u.stable)out[i++]=v;}}
 out[i++]=a.cache.query_slot;
 for(float v:a.cache.query)out[i++]=v;
 for(int k=0;k<2;++k){out[i++]=a.cache.slots[k];out[i++]=a.cache.phases[k];
  for(float v:a.cache.keys[k])out[i++]=v;for(float v:a.cache.values[k])out[i++]=v;}
 out[i++]=a.cache.attention;for(float v:a.cache.h)out[i++]=v;
});}
