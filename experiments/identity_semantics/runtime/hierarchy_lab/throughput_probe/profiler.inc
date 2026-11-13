#include <chrono>
enum ProfileCategory {
  P_BYTE,P_INPUT,P_MEMORY,P_FLUSH,P_LOCAL_TEACHER,P_STATE_INPUT,P_ROUTING,
  P_FORWARD,P_PUBLICATION,P_PROTOTYPE,P_PREDICTOR,P_HEAD,P_TASK,P_READOUT,
  P_EXPERT_UPDATE,P_CREDIT,P_STATE_LEARN,P_ROUTE_UPDATE,P_BOTTOM_ROUTE,
  P_FEATURE_MASK,P_PUBLISH_STATE,P_MATERIALIZE,P_CATEGORIES
};
struct ProfileData {
  uint64_t ns[P_CATEGORIES]{},calls[P_CATEGORIES]{};
  uint64_t total_bytes=0,sampled_bytes=0,last=0;
  int current=-1;bool active=false;
};
static thread_local ProfileData profile_data;
inline uint64_t profile_now(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
inline void profile_switch(int next){
  auto&p=profile_data;if(!p.active)return;
  auto now=profile_now();if(p.current>=0)p.ns[p.current]+=now-p.last;
  p.last=now;p.current=next;
}
struct ProfileScope {
  int previous;
  explicit ProfileScope(int category):previous(profile_data.current){
    if(profile_data.active){profile_switch(category);profile_data.calls[category]++;}
  }
  void stage(int category){if(profile_data.active){profile_switch(category);profile_data.calls[category]++;}}
  ~ProfileScope(){profile_switch(previous);}
};
inline void profile_sample(int64_t tick){
  auto&p=profile_data;p.total_bytes++;
  uint64_t x=uint64_t(tick)+0x9e3779b97f4a7c15ULL;
  x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;x^=x>>31;
  p.active=(x&63)==0;p.current=-1;if(p.active){p.sampled_bytes++;p.last=profile_now();}
}
extern "C" void profile_reset(){profile_data=ProfileData{};}
extern "C" void profile_export(uint64_t*out){
  auto&p=profile_data;out[0]=p.total_bytes;out[1]=p.sampled_bytes;
  for(int i=0;i<P_CATEGORIES;i++){out[2+i]=p.ns[i];out[2+P_CATEGORIES+i]=p.calls[i];}
}
