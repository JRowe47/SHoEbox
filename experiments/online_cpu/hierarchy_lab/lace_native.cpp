// Streaming local updates. No autograd, temporal tape, dense optimizer moments,
// second graph sweep, or hidden label input. All mutation is after publication.
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cstdint>
#include <immintrin.h>
#ifndef SHOEBOX_EXPERTS
#define SHOEBOX_EXPERTS 12
#endif
constexpr int D=64,E=SHOEBOX_EXPERTS,N=18,V=256,CAP=512,PCAP=8192;
// Bounded publication eligibility. Structural route identity and feature SDRs
// are separate fields. A saved expert index is never treated as semantic data.
constexpr int LACE_RING=16,LACE_LINKS=16,LACE_DEPTH=3;
// `index` occupies former structure padding. Legacy records remain readable:
// every fast lookup validates the immutable tick and falls back to a scan.
struct LaceParent { int region=-1,slot=-1; int64_t tick=0; float coefficient=0; int index=-1; };
struct LaceRecord {
  int64_t tick=0;
  int expert=0,runner=0,nparents=0,recall_slot=-1;
  int64_t recall_birth=0;
  float gate=1,gate_derivative=0;
  uint64_t input_mask=0,feature_mask=0;
  float u[D]{},h[D]{},residual[D]{},returned[D]{};
  LaceParent parents[LACE_LINKS]{};
  float credit[LACE_DEPTH][D]{};
  int uses[LACE_DEPTH]{};
};
struct LaceState {
  LaceRecord records[N][LACE_RING]{};
  int cursor[N]{};
  // publications, credits, flushes, missed parent IDs, route-gradient updates,
  // credited gradient norm, feature overlap, prior masks, and depth counts.
  double metrics[N][12]{};
};

struct State {
  float hist[N][2][D]{},mem[N][D]{},u[N][D]{},h[N][D]{},pred[N][D]{};
  float episodic[CAP][2][D]{},ekey[CAP][D]{};
  int valid[N]{},chosen[N]{},pending[N]{},used=0,cursor=0,ablation=0;
  int64_t tick=0,exec[N]{},counts[N][E]{},reads=0,writes=0,correct=0;
  double mse[N]{};
  float address[CAP][2][D]{},query[2][D]{},qmean[2][D]{},vmean[2][D]{},previous[2][D]{};
  int64_t written[CAP]{};
  int previous_valid=0,credit_count=0;
  float credit[D]{};
  float returned[2][D]{};
  // Content-addressed two-choice set associative store; no FIFO cursor.
  float pk[2][PCAP][D]{},pv[2][PCAP][D]{};
  float priority[2][PCAP]{},utility[2][PCAP]{};
  int64_t born[2][PCAP]{},updated[2][PCAP]{};
  float pending_key[2][D]{},key_trace[2][D]{},contribution[2][D]{};
  int64_t return_birth[2]{};
  int pending_buckets[2][4]{},pending_valid[2]{},recall_slot[2]{-1,-1};
  uint32_t rng=7131587;
  bool freeze_writes=false;
  int64_t pc[16]{},max_age=0;
  double age_sum=0,similarity_sum=0;

};
struct Model {
  float *W,*bias,*keys,*P,*pbias,*readout,*rbias;
  const float *emb,*feedback,*slots,*signs,*fsign;
  const int *starts,*perm,*fperm,*cadence;
  int head;
  State s;
  LaceState laces;
  // depth, credit gain, SDR k, task router gain, gate temperature, reuse exponent, DFA gain, task sampling stride, first sparse region, credit feature budget, third-hop stride, gated credit flush
  float lcfg[12]{0,1,64,0,.25f,.5f,1,1,0,64,1,0};
  // Observational counters are outside State: original checkpoint ABI is kept.
  // They never enter inference or learning and restart at each logging window.
  bool observe=false;
  float reference[2][D]{};
  int64_t block_before=0;
  double hc[2][80]{};
  // key mode, gain, capacity, temporal association, value mode (h or q),
  // value centering, EC-C direct credit, credit multiplier, match threshold,
  // write stride. These are configuration, never target-dependent inputs.
  float cfg[24]{2,1,256,1,1,1,1,4,.6f,1,1,1,256,4,.04f,.98f,1,1,0,0,.02f,0,.4f,.02f};
  Model(float *p,const float *em,const float *fb,const float *sl,const int *st,const int *pe,const float *si,const int *fp,const float *fs,const int *ca,int ht):emb(em),feedback(fb),slots(sl),signs(si),fsign(fs),starts(st),perm(pe),fperm(fp),cadence(ca),head(ht){
    W=p;p+=N*E*D*D;bias=p;p+=N*E*D;keys=p;p+=N*E*D;P=p;p+=N*D*D;pbias=p;p+=N*D;readout=p;p+=V*2*D;rbias=p;
  }
};
inline float dot64(const float*a,const float*b){
#ifdef __AVX2__
  __m256 v=_mm256_setzero_ps(),w=_mm256_setzero_ps();
  for(int j=0;j<D;j+=16){v=_mm256_add_ps(v,_mm256_mul_ps(_mm256_loadu_ps(a+j),_mm256_loadu_ps(b+j)));w=_mm256_add_ps(w,_mm256_mul_ps(_mm256_loadu_ps(a+j+8),_mm256_loadu_ps(b+j+8)));}
  v=_mm256_add_ps(v,w);__m128 x=_mm_add_ps(_mm256_castps256_ps128(v),_mm256_extractf128_ps(v,1));
  x=_mm_hadd_ps(x,x);x=_mm_hadd_ps(x,x);return _mm_cvtss_f32(x);
#else
  float z=0;for(int j=0;j<D;j++)z+=a[j]*b[j];return z;
#endif
}
// Walsh-Hadamard sign projections cost O(D log D); each query examines
// two primary buckets plus one low-margin neighbour per bucket.
inline uint32_t random32(State&s){uint32_t x=s.rng;x^=x<<13;x^=x>>17;x^=x<<5;return s.rng=x;}
inline void buckets(const float*q,int count,int*out){
  float v[D];for(int j=0;j<D;j++)v[j]=q[j]*((j*37%17)<8?-1.f:1.f);
  for(int n=1;n<D;n*=2)for(int i=0;i<D;i+=2*n)for(int j=0;j<n;j++){float a=v[i+j],b=v[i+j+n];v[i+j]=a+b;v[i+j+n]=a-b;}
  int half=count/2;int bits=0;while((1<<bits)<half)bits++;
  for(int t=0;t<2;t++){
    int code=0,flip=0;float margin=1e30f;
    for(int b=0;b<bits;b++){float z=v[(b*7+t*23)%D];code|=int(z>0)<<b;if(std::abs(z)<margin){margin=std::abs(z);flip=b;}}
    out[t]=code+t*half;out[t+2]=(code^(1<<flip))+t*half;
  }
}
inline void normalize_key(Model&m,int a,const float*q,float*out){
  float n=1e-9f;for(int j=0;j<D;j++){out[j]=q[j]-(int(m.cfg[0])==3?m.reference[a][j]:(int(m.cfg[0])==2?m.s.qmean[a][j]:0))+m.cfg[18]*m.s.key_trace[a][(j*13+7)%D];n+=out[j]*out[j];}
  n=1/std::sqrt(n);for(int j=0;j<D;j++)out[j]*=n;
}
inline int lookup(Model&m,int a,const float*q,float&sim,bool extra,const int*bs){
  State&s=m.s;int ways=int(m.cfg[13]),best=-1;sim=-1;
  for(int i=0;i<(extra?4:2);i++){
    bool duplicate=false;for(int j=0;j<i;j++)duplicate|=bs[j]==bs[i];if(duplicate)continue;
    for(int w=0;w<ways;w++){int k=bs[i]*ways+w;s.pc[1]++;
      if(!s.born[a][k])continue;s.pc[2]++;float z=dot64(q,s.pk[a][k]);
      if(z>sim){sim=z;best=k;}
    }
  }return best;
}
inline void commit(Model&m,int a,const float*q,const float*current_key,const int*current_buckets){
  State&s=m.s;
  if(s.pending_valid[a]&&!s.freeze_writes&&s.ablation!=2){
    // Stable and adaptive associations use disjoint bucket namespaces.
    // Both are content addressed. Neither has a FIFO or a time-to-live.
    for(int tier=0;tier<2;tier++){
      s.pc[3]++;int ways=int(m.cfg[13]),base=s.pending_buckets[a][tier]*ways,match=-1;float sim=-1;
      for(int w=0;w<ways;w++){int k=base+w;s.pc[1]++;if(!s.born[a][k])continue;s.pc[2]++;float z=dot64(s.pending_key[a],s.pk[a][k]);if(z>sim){sim=z;match=k;}}
      float surprise=0,value_norm=1e-6f;
      if(match>=0)for(int j=0;j<D;j++){float d=q[j]-s.pv[a][match][j];surprise+=d*d;value_norm+=q[j]*q[j]+s.pv[a][match][j]*s.pv[a][match][j];}
      surprise/=value_norm;
      if(match>=0&&sim>=m.cfg[15]&&surprise<m.cfg[22]){
        for(int j=0;j<D;j++)s.pv[a][match][j]+=.0625f*(q[j]-s.pv[a][match][j]);
        s.updated[a][match]=s.tick;s.pc[4]++;
      }else if(1-sim>=(tier?.05f:m.cfg[14])||(tier&&surprise>=m.cfg[22])){
        int victim=-1;float weakest=1e30f;
        for(int w=0;w<ways;w++){
          int k=base+w;
          if(tier&&s.born[a][k])s.priority[a][k]*=1-m.cfg[23]*(1-std::clamp(s.utility[a][k],0.f,1.f));
          float keep=s.born[a][k]?s.priority[a][k]+.1f*s.utility[a][k]:-1;
          if(keep<weakest){weakest=keep;victim=k;}
        }
        float transition=0;for(int j=0;j<D;j++){float d=q[j]-s.previous[a][j];transition+=d*d;}
        float importance=1+m.cfg[16]*std::min(3.f,transition+surprise);
        float u=(random32(s)+1.0)/4294967297.0,priority=std::pow(u,1.f/importance);
        if(priority>weakest){
          if(s.born[a][victim])s.pc[6]++;else {s.used++;s.pc[7]++;}
          memcpy(s.pk[a][victim],s.pending_key[a],D*sizeof(float));memcpy(s.pv[a][victim],q,D*sizeof(float));
          s.born[a][victim]=s.updated[a][victim]=s.tick;s.priority[a][victim]=priority;s.utility[a][victim]=0;s.pc[5]++;s.writes++;
        }else s.pc[8]++;
      }else s.pc[9]++;
    }
  }
  memcpy(s.pending_key[a],current_key,D*sizeof(float));memcpy(s.pending_buckets[a],current_buckets,4*sizeof(int));memcpy(s.previous[a],q,D*sizeof(float));s.pending_valid[a]=1;
  for(int j=0;j<D;j++)s.key_trace[a][j]+=.25f*(q[j]-s.qmean[a][j]-s.key_trace[a][j]);
  for(int j=0;j<D;j++)s.vmean[a][j]+=.01f*(q[j]-s.vmean[a][j]);
}
inline LaceRecord* current_lace(Model&,int);
inline void route_task_update(Model&,int,LaceRecord&,const float*,float);
inline void sparse_weight_update(float*,const float*,float,uint64_t);
inline float norm(const float*x){float z=1.f;for(int i=0;i<D;i++)z+=x[i]*x[i];return z;}
inline void expert_update(Model&m,int r,const float*g,float rate,LaceRecord*record=nullptr){
  State&s=m.s;
  LaceRecord*v=record?record:current_lace(m,r);
  int e=record?record->expert:s.chosen[r];
  const float*u=record?record->u:s.u[r],*h=record?record->h:s.h[r];
  const float*returned=record?record->returned:s.returned[std::clamp(r-15,0,1)];
  float gate=v?v->gate:1.f,a=rate/norm(u);
  float*w=m.W+(r*E+e)*D*D,*b=m.bias+(r*E+e)*D;
  for(int i=0;i<D;i++){float d=std::clamp(g[i]*(1.f-h[i]*h[i]),-2.f,2.f);b[i]-=a*d;
    if(record&&m.lcfg[9]<D){
      if(d!=0)sparse_weight_update(w+i*D,u,a*d*gate,record->input_mask);
    }else for(int j=0;j<D;j++)w[i*D+j]-=a*d*gate*u[j];
    // DFA leaves these optional predictor matrices unused. In this variant,
    // the two EC matrices are ordinary, independently learned recall ports.
    if(m.cfg[10]&&(r==15||r==16)){
      float*port=m.P+r*D*D;
      if(int(m.cfg[10])==2)port[i*D+i]-=a*d*returned[i];
      else for(int j=0;j<D;j++)port[i*D+j]-=a*d*returned[j];
    }
  }
  if(!record&&(r==15||r==16)&&m.cfg[10]&&s.recall_slot[r-15]>=0&&s.return_birth[r-15]==s.born[r-15][s.recall_slot[r-15]]){
    int a=r-15,k=s.recall_slot[a];float influence=0;
    for(int i=0;i<D;i++)influence-=g[i]*(1-h[i]*h[i])*s.contribution[a][i];
    s.utility[a][k]+=.01f*(std::clamp(influence,-1.f,1.f)-s.utility[a][k]);
  }
  if(record&&(r==15||r==16)&&m.cfg[10]&&record->recall_slot>=0&&record->recall_birth==s.born[r-15][record->recall_slot]){
    int a=r-15,k=record->recall_slot;float influence=0;
    for(int i=0;i<D;i++){
      float contribution=int(m.cfg[10])==2?m.P[(r*D+i)*D+i]*record->returned[i]:dot64(m.P+(r*D+i)*D,record->returned);
      influence-=g[i]*(1-h[i]*h[i])*contribution;
    }
    s.utility[a][k]+=.01f*(std::clamp(influence,-1.f,1.f)-s.utility[a][k]);
  }
  if(v)route_task_update(m,r,*v,g,rate);
}

inline void sparse_weight_update(float*w,const float*u,float factor,uint64_t mask){
#ifdef __AVX512F__
  __m512 f=_mm512_set1_ps(factor);
  for(int j=0;j<D;j+=16){__mmask16 k=__mmask16(mask>>j);if(!k)continue;
    __m512 current=_mm512_loadu_ps(w+j),input=_mm512_loadu_ps(u+j);
    _mm512_storeu_ps(w+j,_mm512_mask3_fnmadd_ps(f,input,current,k));
  }
#else
  while(mask){int j=__builtin_ctzll(mask);mask&=mask-1;w[j]-=factor*u[j];}
#endif
}
inline uint64_t feature_mask(const float*u,int k){
  if(k>=D)return ~uint64_t(0);
  int ids[D];for(int j=0;j<D;j++)ids[j]=j;
  std::nth_element(ids,ids+k,ids+D,[&](int a,int b){
    float x=std::abs(u[a]),y=std::abs(u[b]);return x==y?a<b:x>y;
  });
  uint64_t mask=0;for(int j=0;j<k;j++)mask|=uint64_t(1)<<ids[j];return mask;
}
inline LaceRecord* current_lace(Model&m,int r){
  auto&v=m.laces.records[r][m.laces.cursor[r]];return v.tick?&v:nullptr;
}
inline LaceRecord* find_lace(Model&m,int r,int64_t tick,int index=-1){
  if(!tick)return nullptr;
  if(index>=0&&index<LACE_RING&&m.laces.records[r][index].tick==tick)return &m.laces.records[r][index];
  for(int a=0;a<LACE_RING;a++){
    auto&v=m.laces.records[r][(m.laces.cursor[r]-a)&(LACE_RING-1)];
    if(v.tick==tick)return &v;
  }return nullptr;
}
inline void route_task_update(Model&m,int r,LaceRecord&v,const float*g,float rate){
  if(m.lcfg[3]<=0||v.gate_derivative==0)return;
  float dg=0;for(int j=0;j<D;j++)dg+=g[j]*(1-v.h[j]*v.h[j])*v.residual[j];
  float step=std::clamp(rate*m.lcfg[3]*dg*v.gate_derivative,-.05f,.05f);
  float *chosen=m.keys+(r*E+v.expert)*D,*runner=m.keys+(r*E+v.runner)*D;
  uint64_t mask=v.input_mask;
  while(mask){int j=__builtin_ctzll(mask);mask&=mask-1;float d=step*v.u[j];chosen[j]-=d;runner[j]+=d;}
  for(float*k:{chosen,runner}){float n=1e-9f;for(int j=0;j<D;j++)n+=k[j]*k[j];n=1/std::sqrt(n);for(int j=0;j<D;j++)k[j]*=n;}
  m.laces.metrics[r][4]++;
}
inline void distribute_credit(Model&m,int r,LaceRecord&v,const float*g,int depth){
  if(depth>=int(m.lcfg[0]))return;
  if(depth>=2&&m.lcfg[10]>1&&(v.tick/m.cadence[r])%int(m.lcfg[10]))return;
  float pre[D],du[D];
  bool sparse=m.lcfg[9]<D;
  for(int i=0;i<D;i++){pre[i]=g[i]*(1-v.h[i]*v.h[i]);du[i]=(!sparse||(v.input_mask&(uint64_t(1)<<i)))?pre[i]:0;}
  float*w=m.W+(r*E+v.expert)*D*D;
  if(sparse){
    // Input SDR coordinates admit synapses for both upstream task credit and
    // plasticity. The forward expert still receives the complete payload.
#ifdef __AVX512F__
    __m512 gate=_mm512_set1_ps(v.gate);
    for(int i=0;i<D;i++)if(pre[i]!=0){__m512 error=_mm512_set1_ps(pre[i]);
      for(int j=0;j<D;j+=16){__mmask16 k=__mmask16(v.input_mask>>j);if(!k)continue;
        __m512 scaled=_mm512_mul_ps(gate,_mm512_loadu_ps(w+i*D+j));
        __m512 current=_mm512_loadu_ps(du+j);
        _mm512_storeu_ps(du+j,_mm512_mask3_fmadd_ps(scaled,error,current,k));
      }
    }
#else
    int ids[D],count=0;uint64_t mask=v.input_mask;
    while(mask){ids[count++]=__builtin_ctzll(mask);mask&=mask-1;}
    for(int i=0;i<D;i++)if(pre[i]!=0)for(int a=0;a<count;a++){int j=ids[a];du[j]+=v.gate*w[i*D+j]*pre[i];}
#endif
  }else for(int i=0;i<D;i++)for(int j=0;j<D;j++)du[j]+=v.gate*w[i*D+j]*pre[i];
  for(int k=0;k<v.nparents;k++){
    const auto&edge=v.parents[k];
    // A1 has a direct byte loss; avoid duplicating its dominant shortcut.
    if(edge.region==0)continue;
    LaceRecord*p=find_lace(m,edge.region,edge.tick,edge.index);
    if(!p){m.laces.metrics[r][3]++;continue;}
    if(edge.slot<0){for(int j=0;j<D;j++)p->credit[depth][j]+=edge.coefficient*du[j];}
    else for(int j=0;j<D;j++)p->credit[depth][m.perm[edge.slot*D+j]]+=edge.coefficient*m.signs[edge.slot*D+j]*du[j];
    p->uses[depth]++;m.laces.metrics[edge.region][1]++;m.laces.metrics[edge.region][8+depth]++;
  }
}
// Apply accumulated downstream teaching only when this region next runs.
// Local Jacobians use current weights and saved publication activations:
// this is bounded delayed local credit, not an exact temporal gradient.
inline void flush_credit(Model&m,int r,float lr){
  if(!m.lcfg[0])return;
  for(int a=LACE_RING-1;a>=0;a--){
    auto&v=m.laces.records[r][(m.laces.cursor[r]-a)&(LACE_RING-1)];
    if(!v.tick)continue;
    float total[D]{};bool any=false;
    for(int d=0;d<int(m.lcfg[0]);d++)if(v.uses[d]){
      float mean[D],scale=1.f/std::pow(float(v.uses[d]),m.lcfg[5]);
      for(int j=0;j<D;j++){mean[j]=v.credit[d][j]/v.uses[d];total[j]+=v.credit[d][j]*scale;}
      // Clear before propagation so delayed cycles cannot consume new credit
      // recursively in this same depth or bypass the explicit depth budget.
      memset(v.credit[d],0,D*sizeof(float));v.uses[d]=0;
      distribute_credit(m,r,v,mean,d+1);any=true;
    }
    if(any){
      double normg=0;for(float z:total)normg+=z*z;
      m.laces.metrics[r][2]++;m.laces.metrics[r][5]+=std::sqrt(normg);
      expert_update(m,r,total,lr*m.lcfg[1],&v);
    }
  }
}

extern "C" {
void* create(float*p,float*em,float*fb,float*sl,int*st,int*pe,float*si,int*fp,float*fs,int*ca,int head){return new Model(p,em,fb,sl,st,pe,si,fp,fs,ca,head);}
void destroy(void*p){delete static_cast<Model*>(p);}
void reset(void*p){auto&m=*static_cast<Model*>(p);m.laces=LaceState{};auto&s=m.s;memset(&s,0,sizeof(s));s.rng=7131587;s.recall_slot[0]=s.recall_slot[1]=-1;}
size_t state_size(){return sizeof(State);}
void export_state(void*p,void*out){memcpy(out,&static_cast<Model*>(p)->s,sizeof(State));}
void import_state(void*p,void*in){memcpy(&static_cast<Model*>(p)->s,in,sizeof(State));}
void set_ablation(void*p,int a){static_cast<Model*>(p)->s.ablation=a;}
void publications(void*p,float*out){memcpy(out,static_cast<Model*>(p)->s.h,N*D*sizeof(float));}
void hc_observe(void*p,int enabled){auto&m=*static_cast<Model*>(p);m.observe=enabled;memset(m.hc,0,sizeof(m.hc));}
void hc_metrics(void*p,double*out){memcpy(out,static_cast<Model*>(p)->hc,2*80*sizeof(double));}
void set_reference(void*p,float*v){memcpy(static_cast<Model*>(p)->reference,v,2*D*sizeof(float));}
void configure(void*p,float*cfg){memcpy(static_cast<Model*>(p)->cfg,cfg,24*sizeof(float));}
size_t lace_state_size(){return sizeof(LaceState);}
void export_laces(void*p,void*out){memcpy(out,&static_cast<Model*>(p)->laces,sizeof(LaceState));}
void import_laces(void*p,void*in){memcpy(&static_cast<Model*>(p)->laces,in,sizeof(LaceState));}
void configure_laces(void*p,float*cfg){memcpy(static_cast<Model*>(p)->lcfg,cfg,12*sizeof(float));}
void lace_metrics(void*p,double*out){memcpy(out,static_cast<Model*>(p)->laces.metrics,N*12*sizeof(double));}
void lace_codes(void*p,uint64_t*out){auto&m=*static_cast<Model*>(p);for(int r=0;r<N;r++){auto*v=current_lace(m,r);out[3*r]=v?v->input_mask:0;out[3*r+1]=v?v->feature_mask:0;out[3*r+2]=v?v->expert:0;}}
void query_vectors(void*p,float*out){memcpy(out,static_cast<Model*>(p)->s.query,2*D*sizeof(float));}
void persistent_metrics(void*p,double*out){auto&s=static_cast<Model*>(p)->s;for(int i=0;i<16;i++)out[i]=s.pc[i];out[16]=s.age_sum;out[17]=s.max_age;out[18]=s.similarity_sum;out[19]=s.used;}
void block_before(void*p,int64_t cutoff){static_cast<Model*>(p)->block_before=cutoff;}
void freeze_writes(void*p,int value){static_cast<Model*>(p)->s.freeze_writes=value;}
void clear_memory(void*p){auto&s=static_cast<Model*>(p)->s;memset(s.born,0,sizeof(s.born));memset(s.pending_valid,0,sizeof(s.pending_valid));s.used=0;s.recall_slot[0]=s.recall_slot[1]=-1;}
void reset_context(void*p){auto&m=*static_cast<Model*>(p);m.laces=LaceState{};auto&s=m.s;
 memset(s.hist,0,sizeof(s.hist));memset(s.mem,0,sizeof(s.mem));memset(s.u,0,sizeof(s.u));memset(s.h,0,sizeof(s.h));memset(s.pred,0,sizeof(s.pred));
 memset(s.valid,0,sizeof(s.valid));memset(s.pending,0,sizeof(s.pending));memset(s.pending_valid,0,sizeof(s.pending_valid));memset(s.key_trace,0,sizeof(s.key_trace));
 memset(s.credit,0,sizeof(s.credit));s.credit_count=0;memset(s.returned,0,sizeof(s.returned));s.recall_slot[0]=s.recall_slot[1]=-1;
}
int draw(void*p,float temp,float*uniform){
  Model&m=*static_cast<Model*>(p);int node=0;
  for(int d=0;d<8;d++){
    float z=m.rbias[node];for(int j=0;j<D;j++)z+=m.readout[node*2*D+j]*m.s.h[0][j]+m.readout[node*2*D+D+j]*m.s.h[17][j];
    float prob=1.f/(1.f+std::exp(-z/temp));node=2*node+1+int(uniform[d]<prob);
  }return node-255;
}
void info(void*p,float*out){auto&s=static_cast<Model*>(p)->s;for(int r=0;r<N;r++){out[r]=s.exec[r];out[N+r]=s.exec[r]>1?s.mse[r]/(s.exec[r]-1):0;for(int e=0;e<E;e++)out[2*N+r*E+e]=s.counts[r][e];}int i=N*(E+2);out[i]=s.reads;out[i+1]=s.writes;out[i+2]=s.correct;out[i+3]=s.tick;out[i+4]=sizeof(State);}
double run(void*p,const unsigned char*x,const unsigned char*y,int length,int mode,float lr,float lp,float route,int emit,float*out){
  Model&m=*static_cast<Model*>(p);State&s=m.s;double loss=0;
  for(int t=0;t<length;t++){
    s.tick++;float snapshot[N][D];int valid[N];int64_t snapshot_tick[N]{};int snapshot_index[N]{};
    bool tracing=m.lcfg[0]>0||m.lcfg[2]<D||m.lcfg[3]>0;
    for(int r=0;r<N;r++){memcpy(snapshot[r],s.hist[r][0],D*sizeof(float));valid[r]=s.valid[r];auto*v=current_lace(m,r);snapshot_tick[r]=v?v->tick:0;snapshot_index[r]=m.laces.cursor[r];}
    bool active[N]{};float qgain[N]{},hcderiv[N][D]{},hcvalue[N][D]{};
    for(int r=0;r<N;r++){
      bool scheduled=s.tick%m.cadence[r]==0;
      bool event_ec=(r==15||r==16)&&m.cfg[19]>0;
      if(!scheduled&&!event_ec)continue;
      active[r]=true;
      if(mode&&!m.lcfg[11])flush_credit(m,r,lr);
      LaceParent parents[LACE_LINKS];int nparents=0;
      float q[D]{},c[D]{};float qw=0,cw=0;
      if(r<2){memcpy(q,m.emb+x[t]*D,D*sizeof(float));qw=1;}
      for(int k=m.starts[r];k<m.starts[r+1];k++){
        const float*slot=m.slots+4*k;int src=int(slot[0]),age=int(slot[1]),delay=int(slot[2]);float w=slot[3];
        if((delay?valid[src]:s.valid[src])<=age)continue;
        if((s.ablation==1&&src>=2&&src!=17)||(s.ablation>=10&&src==s.ablation-10))continue;
        const float*v=delay?snapshot[src]:s.hist[src][age];float*dest=delay?c:q;
        for(int j=0;j<D;j++)dest[j]+=w*m.signs[k*D+j]*v[m.perm[k*D+j]];
        if(delay)cw+=w*w;else qw+=w*w;
        if(tracing){auto&v=m.laces.records[src][(m.laces.cursor[src]-age)&(LACE_RING-1)];
          parents[nparents++]={src,k,delay?snapshot_tick[src]:v.tick,w*(delay?-1.f:1.f),delay?snapshot_index[src]:(m.laces.cursor[src]-age)&(LACE_RING-1)};
        }
      }
      // Previous emitted byte in Y is the currently observed prefix byte.
      if(r==17){for(int j=0;j<D;j++)q[j]+=.25f*m.emb[x[t]*D+j];qw+=.0625f;}
      // An optional ordinary A1 publication gives EC a byte-clock cue while
      // preserving all original higher-order source connections.
      if((r==15||r==16)&&m.cfg[11]){
        for(int j=0;j<D;j++)q[j]+=m.cfg[11]*s.h[0][j];qw+=m.cfg[11]*m.cfg[11];
      }
      float qa=1.f/std::sqrt(std::max(1.f,qw)),ca=.2f/std::sqrt(std::max(1.f,cw));
      qgain[r]=qa;
      for(int k=0;k<nparents;k++)parents[k].coefficient*=parents[k].coefficient<0?-ca:qa;
      if(tracing&&(r==15||r==16)&&m.cfg[11]){auto*v=current_lace(m,0);parents[nparents++]={0,-1,v?v->tick:0,qa*m.cfg[11],m.laces.cursor[0]};}
      for(int j=0;j<D;j++){q[j]*=qa;c[j]*=ca;}
      if(r==15||r==16){
        int a=r-15;
        for(int j=0;j<D;j++){
          s.query[a][j]=q[j];
          s.qmean[a][j]+=.01f*(q[j]-s.qmean[a][j]);
        }
        if(r==15&&mode&&int(m.cfg[6])==2&&s.credit_count){
          float g[D];for(int j=0;j<D;j++)g[j]=s.credit[j]/s.credit_count;
          expert_update(m,15,g,lr*m.cfg[7]);memset(s.credit,0,sizeof(s.credit));s.credit_count=0;
        }
        memset(s.returned[a],0,D*sizeof(float));
      }
      // Sparse content lookup: read existing associations before committing
      // the previous cue -> current observed EC input transition.
      if(r==15||r==16){
        int a=r-15;s.recall_slot[a]=-1;float key[D];normalize_key(m,a,q,key);int bs[4];buckets(key,int(m.cfg[12]),bs);
        float change=s.pending_valid[a]?1-dot64(key,s.pending_key[a]):1;
        bool wake=false;
        bool attempt=s.used&&s.ablation!=2&&s.tick%int(m.cfg[17])==0&&(m.cfg[1]>0||m.cfg[21])&&(scheduled||change>=m.cfg[20]);
        if(attempt){
          s.pc[0]++;float sim;int best=lookup(m,a,key,sim,true,bs);
          wake=best>=0&&sim>=m.cfg[19];
          float gate=best<0?0:std::clamp((sim-m.cfg[8])/std::max(1e-6f,1-m.cfg[8]),0.f,1.f)*m.cfg[1];
          if(gate>0){
            s.pc[10]++;int64_t age=s.tick-s.updated[a][best];s.age_sum+=age;s.max_age=std::max(s.max_age,age);s.similarity_sum+=sim;
            s.pc[11]+=age>256;s.pc[12]+=age>4096;s.pc[13]+=age>16384;
            if(s.ablation==3||m.cfg[21]||(m.block_before&&s.born[a][best]<=m.block_before))gate=0;
            if(s.ablation==4){int count=int(m.cfg[12])*int(m.cfg[13]);int wrong=(best+count/2)%count;for(int k=0;k<count&&!s.born[a][wrong];k++)wrong=(wrong+1)%count;best=wrong;}
            s.recall_slot[a]=best;s.return_birth[a]=s.born[a][best];
            for(int j=0;j<D;j++){float v=gate*(s.pv[a][best][j]-(m.cfg[5]?s.vmean[a][j]:0));s.returned[a][j]=v;if(!m.cfg[10])c[j]+=v;}
            s.reads++;
          }
        }
        if(s.tick%int(m.cfg[9])==0)commit(m,a,q,key,bs);
        if(event_ec&&!scheduled&&!wake){active[r]=false;s.pc[14]++;continue;}
        if(event_ec&&!scheduled)s.pc[15]++;
      }
      if(mode&&m.lcfg[11])flush_credit(m,r,lr);
      // A future observed bottom-up bundle teaches the previous publication.
      // Context/recall are excluded from the teacher; they still condition u.
      if((mode==1||mode==3||mode==6)&&s.pending[r]&&!(m.cfg[10]&&(r==15||r==16))){
        float err[D],g[D]{};double mse=0;
        for(int i=0;i<D;i++){err[i]=std::clamp(s.pred[r][i]-q[i],-1.f,1.f);mse+=err[i]*err[i];}
        float*P=m.P+r*D*D,*pb=m.pbias+r*D;
        for(int i=0;i<D;i++)for(int j=0;j<D;j++)g[j]+=P[i*D+j]*err[i];
        expert_update(m,r,g,lp);
        float a=lp/norm(s.h[r]);
        for(int i=0;i<D;i++){pb[i]-=a*err[i];for(int j=0;j<D;j++)P[i*D+j]-=a*err[i]*s.h[r][j];}
        s.mse[r]+=mse/D;
      }
      float alpha=r==1?.125f:.5f;
      for(int j=0;j<D;j++){s.u[r][j]=(1-alpha)*s.mem[r][j]+alpha*(q[j]+c[j]);s.mem[r][j]=s.u[r][j];}
      int best=0,runner=0;float score=-1e30f,second=-1e30f;
      int sparse_k=r>=int(m.lcfg[8])?int(m.lcfg[2]):D;
      bool task_route=m.lcfg[3]>0&&r>=int(m.lcfg[8]);
      uint64_t routing_mask=feature_mask(s.u[r],sparse_k);
      int indices[D],nindices=0;uint64_t bits=routing_mask;
      while(bits){indices[nindices++]=__builtin_ctzll(bits);bits&=bits-1;}
      for(int e=0;e<E;e++){
        float v=0;
        if(sparse_k>=D)for(int j=0;j<D;j++)v+=s.u[r][j]*m.keys[(r*E+e)*D+j];
        else for(int a=0;a<nindices;a++){int j=indices[a];v+=s.u[r][j]*m.keys[(r*E+e)*D+j];}
        // Small fixed competition pressure; counts are training/inference state.
        v-=.02f*float(s.counts[r][e])/float(1+s.exec[r]);
        if(v>score){second=score;runner=best;score=v;best=e;}else if(v>second){second=v;runner=e;}
      }
      s.chosen[r]=best;s.exec[r]++;s.counts[r][best]++;
      float gate=1,gate_derivative=0;
      if(task_route){float p=1.f/(1.f+std::exp(-(score-second)/m.lcfg[4]));gate=2*p;gate_derivative=2*p*(1-p)/m.lcfg[4];}
      float residual[D];
      float*w=m.W+(r*E+best)*D*D,*b=m.bias+(r*E+best)*D;
      for(int i=0;i<D;i++){
        float z=s.u[r][i]+b[i];
        if(!task_route){for(int j=0;j<D;j++)z+=w[i*D+j]*s.u[r][j];residual[i]=0;}
        else {residual[i]=dot64(w+i*D,s.u[r]);z+=gate*residual[i];}
        if(m.cfg[10]&&(r==15||r==16)){
          float recall=0;
          if(int(m.cfg[10])==2)recall=m.P[(r*D+i)*D+i]*s.returned[r-15][i];
          else for(int j=0;j<D;j++)recall+=m.P[(r*D+i)*D+j]*s.returned[r-15][j];
          s.contribution[r-15][i]=recall;z+=recall;
        }
        s.h[r][i]=std::tanh(z);
      }
      if(tracing){
        auto*previous=current_lace(m,r);uint64_t prior=previous?previous->feature_mask:0;
        int cursor=(m.laces.cursor[r]+1)&(LACE_RING-1);m.laces.cursor[r]=cursor;
        auto&v=m.laces.records[r][cursor];
        for(int d=0;d<LACE_DEPTH;d++)if(v.uses[d])m.laces.metrics[r][11]+=v.uses[d];
        v=LaceRecord{};v.tick=s.tick;v.expert=best;v.runner=runner;v.gate=gate;v.gate_derivative=gate_derivative;
        v.input_mask=routing_mask;v.feature_mask=feature_mask(s.h[r],16);v.nparents=nparents;
        memcpy(v.u,s.u[r],D*sizeof(float));memcpy(v.h,s.h[r],D*sizeof(float));memcpy(v.residual,residual,D*sizeof(float));
        for(int k=0;k<nparents;k++){v.parents[k]=parents[k];v.parents[k].coefficient*=alpha;}
        if(r==15||r==16){memcpy(v.returned,s.returned[r-15],D*sizeof(float));v.recall_slot=s.recall_slot[r-15];v.recall_birth=s.return_birth[r-15];}
        m.laces.metrics[r][0]++;m.laces.metrics[r][6]+=__builtin_popcountll(prior&v.feature_mask)/16.;
      }
      memcpy(s.hist[r][1],s.hist[r][0],D*sizeof(float));memcpy(s.hist[r][0],s.h[r],D*sizeof(float));s.valid[r]++;
      if(mode&&mode!=4&&route>0){
        float*k=m.keys+(r*E+best)*D;float n=std::sqrt(norm(s.u[r])-1.f+1e-6f),kn=0;
        if(sparse_k<D){n=0;for(int a=0;a<nindices;a++){float z=s.u[r][indices[a]];n+=z*z;}n=std::sqrt(n+1e-6f);}
        for(int j=0;j<D;j++){float cue=(routing_mask&(uint64_t(1)<<j))?s.u[r][j]/n:0;k[j]+=(cue-k[j])*route;kn+=k[j]*k[j];}
        kn=1.f/std::sqrt(kn+1e-9f);for(int j=0;j<D;j++)k[j]*=kn;
      }
      if((mode==1||mode==3||mode==6)&&(mode!=6||s.exec[r]%4==0)&&!(m.cfg[10]&&(r==15||r==16))){for(int i=0;i<D;i++){float z=m.pbias[r*D+i];for(int j=0;j<D;j++)z+=m.P[(r*D+i)*D+j]*s.h[r][j];s.pred[r][i]=z;}s.pending[r]=1;}
      else if(mode==6)s.pending[r]=0;
    }
    float features[2*D],logits[V],prob[V],maxlog=-1e30f;int selected[V],nout=V;
    for(int j=0;j<D;j++){features[j]=s.h[0][j];features[D+j]=s.h[17][j];}
    int guess=0;
    double denom=0;
    if(m.head==0){
      for(int v=0;v<V;v++){selected[v]=v;float z=m.rbias[v];for(int j=0;j<2*D;j++)z+=m.readout[v*2*D+j]*features[j];logits[v]=z;if(z>maxlog){maxlog=z;guess=v;}}
      for(int v=0;v<V;v++){prob[v]=std::exp(logits[v]-maxlog);denom+=prob[v];}
      loss+=std::log(denom)+maxlog-logits[y[t]];
      if(emit)memcpy(out+t*V,logits,V*sizeof(float));
    }else{
      // Balanced binary hierarchical softmax. Only target-path nodes train.
      // A greedy path supplies a cheap diagnostic; it is NOT joint MAP.
      bool cached[V]{};auto score=[&](int v){
        if(!cached[v]){float z=m.rbias[v];for(int j=0;j<2*D;j++)z+=m.readout[v*2*D+j]*features[j];logits[v]=z;cached[v]=true;}return logits[v];};
      nout=8;int node=0;
      for(int d=0;d<8;d++){
        int bit=(y[t]>>(7-d))&1;float z=score(node);selected[d]=node;
        loss+=std::max(0.f,z)-z*bit+std::log1p(std::exp(-std::abs(z)));
        prob[node]=1.f/(1.f+std::exp(-z))-bit;node=2*node+1+bit;
      }
      node=0;for(int d=0;d<8;d++)node=2*node+1+int(score(node)>0);guess=node-255;
      if(emit){
        float mass[511]{};for(int v=0;v<255;v++){
          float z=score(v),left=-std::max(z,0.f)-std::log1p(std::exp(-std::abs(z)));
          mass[2*v+1]=mass[v]+left;mass[2*v+2]=mass[v]+left+z;
        }memcpy(out+t*V,mass+255,V*sizeof(float));
      }
    }
    s.correct+=guess==y[t];
    if(mode){
      if(m.head==0){for(int v=0;v<V;v++)prob[v]/=denom;prob[y[t]]-=1;}
      float direct[2*D]{},broadcast[D]{};
      for(int k=0;k<nout;k++){int v=selected[k];for(int j=0;j<2*D;j++)direct[j]+=m.readout[v*2*D+j]*prob[v];}
      // Exact local derivative through the current Y -> EC-C edge, with
      // cached upstream state detached. Mode 2 accumulates across publication
      // reuse and teaches that publication before its next event.
      float gec[D]{};
      if(!m.lcfg[0]&&m.cfg[6]&&s.valid[15]&&(int(m.cfg[6])==2||active[15])){
        float pre[D],du[D];for(int j=0;j<D;j++){pre[j]=direct[D+j]*(1-s.h[17][j]*s.h[17][j]);du[j]=pre[j];}
        float*w=m.W+(17*E+s.chosen[17])*D*D;
        for(int i=0;i<D;i++)for(int j=0;j<D;j++)du[j]+=w[i*D+j]*pre[i];
        for(int k=m.starts[17];k<m.starts[18];k++)if(int(m.slots[4*k])==15){
          for(int j=0;j<D;j++)gec[m.perm[k*D+j]]+=.5f*qgain[17]*m.slots[4*k+3]*m.signs[k*D+j]*du[j];
        }
        if(int(m.cfg[6])==2){for(int j=0;j<D;j++)s.credit[j]+=gec[j];s.credit_count++;}
        else if(active[15])expert_update(m,15,gec,lr*m.cfg[7]);
      }
      if(m.lcfg[0]&&s.tick%int(m.lcfg[7])==0){auto*v=current_lace(m,17);if(v)distribute_credit(m,17,*v,direct+D,0);}
      // Exact local task derivative at A1 and Y; DFA elsewhere uses an
      // independent random error sketch and region-specific signed permutation.
      if(mode!=4&&mode!=5){expert_update(m,0,direct,lr);expert_update(m,17,direct+D,lr);}
      if(mode==5){
        // Exact current-tick spatial chain, with previous events detached.
        // The router is discrete and uses the same competitive update as DFA.
        float g[N][D]{};memcpy(g[0],direct,D*sizeof(float));memcpy(g[17],direct+D,D*sizeof(float));
        for(int r=N-1;r>=0;r--)if(active[r]){
          float pre[D],du[D];for(int j=0;j<D;j++){pre[j]=g[r][j]*(1-s.h[r][j]*s.h[r][j]);du[j]=pre[j];}
          float*w=m.W+(r*E+s.chosen[r])*D*D;
          for(int i=0;i<D;i++)for(int j=0;j<D;j++)du[j]+=w[i*D+j]*pre[i];
          float hcdot=0;for(int j=0;j<D;j++)hcdot+=du[j]*hcvalue[r][j];
          float alpha=r==1?.125f:.5f;
          for(int k=m.starts[r];k<m.starts[r+1];k++){
            const float*sl=m.slots+4*k;int src=int(sl[0]);
            if(sl[1]!=0||sl[2]!=0||!active[src])continue;
            if((s.ablation==1&&src>=2&&src!=17)||(s.ablation>=10&&src==s.ablation-10))continue;
            for(int j=0;j<D;j++)g[src][m.perm[k*D+j]]+=alpha*qgain[r]*sl[3]*m.signs[k*D+j]*(du[j]+hcdot*hcderiv[r][j]);
          }
          expert_update(m,r,g[r],lr);
        }
      }
      if(mode==2||mode==3||mode==6){
        for(int k=0;k<nout;k++){int v=selected[k];for(int j=0;j<D;j++)broadcast[j]+=m.feedback[v*D+j]*prob[v];}
        for(int r=1;r<N-1;r++)if(active[r]&&!(r==15&&m.cfg[6])&&(!m.lcfg[0]||m.lcfg[6]>0)){float g[D];for(int j=0;j<D;j++)g[j]=m.fsign[r*D+j]*broadcast[m.fperm[r*D+j]]*(m.lcfg[0]?m.lcfg[6]:1.f);expert_update(m,r,g,lr);}
      }
      float fn=1.f;for(float v:features)fn+=v*v;float a=lr/fn;
      for(int k=0;k<nout;k++){int v=selected[k];m.rbias[v]-=lr*.05f*prob[v];for(int j=0;j<2*D;j++)m.readout[v*2*D+j]-=a*prob[v]*features[j];}
    }
  }
  return loss/std::max(1,length);
}
}
