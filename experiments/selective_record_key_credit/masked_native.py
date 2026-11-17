"""Test-only scalar A8 local reference and exact hybrid-call instrumentation."""
MASKED=r'''
#include <iostream>
int main(){using namespace shoe;int n=0;auto ck=[&](bool ok){++n;if(!ok)throw std::runtime_error("A9 mask check "+std::to_string(n));};
 std::mt19937 rng(9147);std::uniform_real_distribution<float> u(-.2f,.2f);
 int actual_calls=0,reference_calls=0,omitted=0,executed=0,local_calls=0;
 for(int bytes:{1,2,7,8,12})for(int mask=0;mask<16;++mask)for(int query=0;query<2;++query)for(int request=0;request<2;++request)for(bool allow:{false,true}){
  QueryAdapter a(500000,7,true,4,true);a.rate=.16f;a.auxiliary=1.f;a.model->bytes=bytes;
  for(auto& v:a.model->levels[0].retained.last)v=u(rng);for(auto& v:a.model->levels[1].retained.last)v=u(rng);
  for(auto& row:a.head.gate)for(auto& v:row)v=u(rng);
  QueryAdapter b(a);auto keys=a.head.key;Labels labels;labels.fill(-1);labels[5]=request;
  if(query)labels[0]=255;int positive=0;for(int ph=0;ph<4;++ph)if(mask&(1<<ph)){labels[ph+1]=(31+ph)*256+73;++positive;}
  bool event=bytes>=8&&bytes%4==0;int q=bytes>=2&&query?1:0;int rec=event?positive:0;
  test_hybrid_calls=0;a.local(labels,allow);ck(test_hybrid_calls==q+(allow?rec:0));actual_calls+=test_hybrid_calls;
  test_hybrid_calls=0;b.a8_local_reference(labels);ck(test_hybrid_calls==q+rec);reference_calls+=test_hybrid_calls;local_calls+=2;
  ck(a.head.value==b.head.value);ck(a.head.gate==b.head.gate);ck(a.applicability==b.applicability);ck(save(*a.model)==save(*b.model));
  ck(a.local_updates==b.local_updates&&a.presence_steps==b.presence_steps&&a.gate_targets==b.gate_targets);
  ck(a.projection_steps+(allow?0:16*rec)==b.projection_steps);
  for(int r=0;r<80;++r){
   if(r>=64||allow)ck(a.head.key[r]==b.head.key[r]);else ck(a.head.key[r]==keys[r]);
  }
  if(allow)ck(query_save(a)==query_save(b));
  // Equal exogenous responsibility preserves the untouched native update.
  a.responsibility=b.responsibility=.375f;std::array<RecallValue,4> av{},bv{};a.backward(av);b.backward(bv);
  ck(save(*a.model)==save(*b.model));
  omitted+=allow?0:rec;executed+=allow?rec:0;
 }
 std::cout<<n<<" A9 masked credit checks passed; local calls "<<local_calls<<"; actual/reference hybrids "<<actual_calls<<"/"<<reference_calls<<"; executed/omitted record hybrids "<<executed<<"/"<<omitted<<"\n";
}
'''
