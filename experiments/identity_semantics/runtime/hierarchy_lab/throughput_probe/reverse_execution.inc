inline size_t reverse_policy_key(Model&m,const Model::ReverseRoot&root){
  ReversePolicyKey key;key.bits[0]=uint32_t(root.teaching);
  memcpy(key.bits.data()+1,root.scales,sizeof(root.scales));
  auto&state=m.reverse_execution;auto found=state.keys.find(key);
  if(found!=state.keys.end())return found->second;
  size_t id=state.policies.size();state.policies.emplace_back();state.keys.emplace(key,id);return id;
}
inline void reverse_group_roots(Model&m){
  auto&state=m.reverse_execution;state.count=0;
  if(++state.generation==0){for(auto&p:state.policies)p.generation=0;state.generation=1;}
  for(size_t q=0;q<m.reverse_roots.size();q++){
    size_t id=reverse_policy_key(m,m.reverse_roots[q]);auto&policy=state.policies[id];
    if(policy.generation!=state.generation){
      policy.generation=state.generation;policy.group=state.count++;
      if(state.groups.size()<state.count)state.groups.emplace_back();
      auto&group=state.groups[policy.group];group.roots.clear();group.mass=0;group.policy=id;
    }
    auto&group=state.groups[policy.group];group.roots.push_back(q);group.mass+=std::abs(m.reverse_roots[q].rate);
  }
}
inline bool execution_same_ref(SemanticRef a,SemanticRef b){
  return a.serial==b.serial&&a.index==b.index&&a.region==b.region;
}
// Reuse only topology: derivatives always read the current scope's bank
// version. Event traversal additionally depends on the regional frontier clocks.
inline ReverseExecution::Plan& reverse_schedule(Model&m,const ReverseExecution::Group&group,uint64_t mark){
  auto&state=m.reverse_execution;uint64_t hash=group.policy;
  for(auto q:group.roots){const auto&r=m.reverse_roots[q].ref;hash=(hash^r.serial)*1099511628211ULL;}
  auto&plan=state.plans[hash%state.plans.size()];
  bool cached=plan.valid&&plan.policy==group.policy&&plan.local==m.event.local&&
    plan.local_only==m.event.local_only&&plan.frontier==m.event.frontier&&plan.roots.size()==group.roots.size();
  if(cached&&m.event.local)cached=!memcmp(plan.ordinals,m.event.ordinal,sizeof(plan.ordinals));
  if(cached)for(size_t q=0;q<group.roots.size();q++)
    if(!execution_same_ref(plan.roots[q],m.reverse_roots[group.roots[q]].ref)){cached=false;break;}
  if(cached)for(auto ref:plan.nodes)if(!constituent_record(m,ref)){cached=false;break;}
  if(cached){
    for(auto ref:plan.nodes){constituent_mark(m,ref.index)=mark;memset(constituent_gradient(m,ref.index),0,D*sizeof(float));}
    return plan;
  }
  auto&work=state.scratch;work.nodes.clear();work.batch_ends.clear();bool complete=true;
  int receiver=m.reverse_roots[group.roots[0]].receiver;
  auto admit=[&](SemanticRef ref){
    if(!ref.serial)return;
    if(!constituent_record(m,ref)){
      SHOEBOX_DIAGNOSTIC(m.composition.stream.completion_metrics[receiver][1]++;)complete=false;return;
    }
    if(constituent_mark(m,ref.index)==mark)return;
    constituent_mark(m,ref.index)=mark;memset(constituent_gradient(m,ref.index),0,D*sizeof(float));work.nodes.push_back(ref);
  };
  for(auto q:group.roots)admit(m.reverse_roots[q].ref);
  for(size_t i=0;i<work.nodes.size();i++){
    const auto&node=*constituent_record(m,work.nodes[i]);
    for(int e=0;e<node.count;e++)if(event_follow(m,node.ref,e))admit(node.edges[e].source);
  }
  std::sort(work.nodes.begin(),work.nodes.end(),[](SemanticRef a,SemanticRef b){return a.serial>b.serial;});
  for(size_t begin=0;begin<work.nodes.size();){
    size_t end=begin;uint64_t newest_child=0;
    while(end<work.nodes.size()&&end-begin<16&&(end==begin||work.nodes[end].serial>newest_child)){
      const auto&node=*constituent_record(m,work.nodes[end]);
      for(int e=0;e<node.count;e++)newest_child=std::max(newest_child,node.edges[e].source.serial);
      ++end;
    }
    work.batch_ends.push_back(end);begin=end;
  }
  // Bound cached handles. An unusually deep DAG uses the retained scratch plan.
  if(!complete||work.nodes.size()>4096||group.roots.size()>4096){plan.valid=false;return work;}
  plan.valid=true;plan.policy=group.policy;plan.local=m.event.local;plan.local_only=m.event.local_only;
  plan.frontier=m.event.frontier;memcpy(plan.ordinals,m.event.ordinal,sizeof(plan.ordinals));
  plan.roots.clear();for(auto q:group.roots)plan.roots.push_back(m.reverse_roots[q].ref);
  plan.nodes.swap(work.nodes);plan.batch_ends.swap(work.batch_ends);return plan;
}
