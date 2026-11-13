#define SHOEBOX_CONTRIBUTION_CHECKS 1
#include "../runtime/hierarchy_lab/throughput_probe/completion.cpp"
extern "C" {
void acceptance_fixture_publish(void*p,int r,int64_t begin,int64_t end,int64_t primary_begin,
                                int64_t primary_end,int kind,int count,int recognized,int relay){
  auto&m=*static_cast<Model*>(p);ContentEvent e{};
  e.ref.begin=begin;e.ref.end=end;e.clock_begin=primary_begin;e.clock_end=primary_end;
  e.ref.region=r;e.ref.recognized=recognized;e.kind=kind;e.count=count;e.local_arity=count;
  content_publish(m,r,e,relay);
}
uint64_t acceptance_fixture_hash(int64_t begin,int64_t end){
  // Use a heap model: the reference owns substantial live eligibility state.
  std::vector<float>p(N*E*D*D+2*N*E*D+N*D*D+N*D+V*2*D+V,0);
  auto m=std::make_unique<Model>(p.data(),nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,1);
  ContentEvent e{};e.ref.begin=0;e.ref.end=100;e.clock_begin=begin;e.clock_end=end;
  contribution_event(*m,6,e);return m->contribution_hashes[6];
}
}
