// EXP03/05-A1: keep learned record evidence in the existing cache.
// No byte tests, selected-record labels, extra slots, weights or sequence state.
// Selective update motivation: Schlag et al., ICML 2021,
// https://proceedings.mlr.press/v139/schlag21a.html. This cache rule is ours.
namespace shoe {
inline bool ir_recognized(const QueryHead& head,const Vec& stable){
 Vec x=source_vector(stable);
 for(int phase=0;phase<4;++phase)if(affine(head.gate[phase],x)>0)return true;
 return false;
}
inline int ir_admit(Model* m,int scale,const Vec& current,const Vec& content){
 auto* a=query_adapter(m);auto& s=m->levels[scale].retained;
 if(scale!=1||!a||!a->kind||s.count<RECALL_SLOTS)return s.admit(current,content);
 // This is RecallStream::admit's unchanged pair formation and occurrence update.
 if(!s.have_previous){s.previous=current;s.previous_content=content;s.have_previous=true;return -1;}
 Vec stable{},combined{};
 for(int i=0;i<D;++i){int j=(3*i+13)%D;stable[i]=current[i]+s.previous[j];combined[i]=std::tanh(.5f*(content[i]+s.previous_content[j]));}
 normalize(stable);s.previous=current;s.previous_content=content;
 if(s.clock&&stable==s.last)return -1;
 s.last=stable;int slot=0;
 for(;slot<s.count;++slot)if(s.units[slot].stable==stable)break;
 if(slot==s.count){
  int oldest=0,unrecognized=-1;
  for(int i=0;i<s.count;++i){
   if(s.units[i].born<s.units[oldest].born)oldest=i;
   if(!ir_recognized(a->head,s.units[i].stable)&&(unrecognized<0||s.units[i].born<s.units[unrecognized].born))unrecognized=i;
  }
  if(unrecognized>=0)slot=unrecognized;
  else {if(!ir_recognized(a->head,stable))return -1;slot=oldest;}
 }
 auto& u=s.units[slot];u.stable=stable;u.key=recall_pool(stable);u.born=++s.clock;
 float norm=std::sqrt(dot(u.key.data(),u.key.data(),RECALL_K)+1e-8f);
 for(float& x:u.key)x/=norm;
 auto learned=recall_pool(combined);
 for(int i=0;i<RECALL_K;++i){u.value[i]=std::tanh(learned[i]);u.value[RECALL_K+i]=u.key[i];}
 return slot;
}
}
