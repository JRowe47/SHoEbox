extern "C" {
int composition_configure(void*p,int enabled,float threshold,int count,int maximum,float context){
  if((enabled!=0&&enabled!=1)||!std::isfinite(threshold)||threshold<=0||threshold>=1||count<1||count>1000000||maximum<2||maximum>COMP_MAX||!std::isfinite(context)||context<0||context>1)return 0;
  auto&z=static_cast<Model*>(p)->composition;
#ifdef SHOEBOX_MULTI_BINDING
  auto&m=*static_cast<Model*>(p);
  for(int r=3;r<N;r++){
    if(m.starts[r+1]-m.starts[r]>COMP_MAX)return 0;
    for(int k=m.starts[r];k<m.starts[r+1];k++)if(!m.slots[4*k+2]){
      if(m.slots[4*k]<0||m.slots[4*k]>=r||m.slots[4*k+1]<0||m.slots[4*k+1]>1)return 0;
    }
#ifdef SHOEBOX_LEARNED_COMPLETION
    if(r<=14){
      if(m.starts[r]==m.starts[r+1]||m.slots[4*m.starts[r]+2])return 0;
      int driver=int(m.slots[4*m.starts[r]]),count=0,ages=0;
      for(int k=m.starts[r];k<m.starts[r+1];k++)if(!m.slots[4*k+2]&&int(m.slots[4*k])==driver){
        int bit=1<<int(m.slots[4*k+1]);if(ages&bit)return 0;ages|=bit;count++;
      }
      if(count<1||count>2||ages!=(1<<count)-1)return 0;
    }
#endif
  }
#endif
  z.enabled=enabled;z.boundary_threshold=threshold;z.min_count=count;z.max_children=maximum;z.context_gain=context;return 1;
}
size_t composition_state_size(void*p=nullptr){
#ifdef SHOEBOX_LEARNED_COMPLETION
  if(p)return sizeof(CompositionRuntime)+static_cast<Model*>(p)->semantic_overflow.size()*sizeof(ConstituentRecord);
#endif
  return sizeof(CompositionRuntime);
}
size_t composition_fitted_size(){return offsetof(CompositionRuntime,stream);}
void composition_export(void*p,void*out){
  auto&m=*static_cast<Model*>(p);memcpy(out,&m.composition,sizeof(CompositionRuntime));
#ifdef SHOEBOX_LEARNED_COMPLETION
  if(!m.semantic_overflow.empty())memcpy(static_cast<unsigned char*>(out)+sizeof(CompositionRuntime),
    m.semantic_overflow.data(),m.semantic_overflow.size()*sizeof(ConstituentRecord));
#endif
}
int composition_import(void*p,const void*in,size_t size,int fitted){
#ifdef SHOEBOX_LEARNED_COMPLETION
  if(fitted?size!=offsetof(CompositionRuntime,stream):size<sizeof(CompositionRuntime))return 0;
  size_t overflow=fitted?0:(size-sizeof(CompositionRuntime))/sizeof(ConstituentRecord);
  if(!fitted&&size!=sizeof(CompositionRuntime)+overflow*sizeof(ConstituentRecord))return 0;
#else
  if(size!=(fitted?offsetof(CompositionRuntime,stream):sizeof(CompositionRuntime)))return 0;
#endif
  const auto*z=static_cast<const CompositionRuntime*>(in);
  if(z->version!=static_cast<Model*>(p)->composition.version||z->enabled>1||!std::isfinite(z->boundary_threshold)||z->boundary_threshold<=0||z->boundary_threshold>=1||z->min_count<1||z->min_count>1000000||z->max_children<2||z->max_children>COMP_MAX||!std::isfinite(z->context_gain)||z->context_gain<0||z->context_gain>1)return 0;
  auto&target=static_cast<Model*>(p)->composition;
#ifdef SHOEBOX_LEARNED_COMPLETION
  if(!fitted&&z->stream.reserved!=overflow)return 0;
  auto&m=*static_cast<Model*>(p);constituent_resize(m,overflow);
  memcpy(&target,in,fitted?size:sizeof(CompositionRuntime));
  if(overflow)memcpy(m.semantic_overflow.data(),static_cast<const unsigned char*>(in)+sizeof(CompositionRuntime),overflow*sizeof(ConstituentRecord));
#else
  memcpy(&target,in,size);
#endif
  if(fitted)memset(&target.stream,0,sizeof(target.stream));return 1;
}
void composition_metrics(void*p,double*out){memcpy(out,static_cast<Model*>(p)->composition.stream.metrics,sizeof(CompositionStream::metrics));}
void composition_events(void*p,uint64_t*out,float*code){
  auto&z=static_cast<Model*>(p)->composition.stream;
  for(int r=0;r<N;r++){
    const auto&e=z.history[r][0];uint64_t*row=out+r*(12+COMP_MAX*9);
    row[0]=e.ref.seq;row[1]=e.ref.id.a;row[2]=e.ref.id.b;row[3]=e.ref.begin;row[4]=e.ref.end;
    row[5]=e.ref.tick;row[6]=e.count;row[7]=e.ref.recognized;row[8]=e.kind;row[9]=z.length[r];row[10]=z.seen[r];row[11]=z.byte_position;
    for(int i=0;i<COMP_MAX;i++){const auto&c=e.children[i];uint64_t*v=row+12+9*i;
      v[0]=c.id.a;v[1]=c.id.b;v[2]=c.seq;v[3]=c.begin;v[4]=c.end;v[5]=c.tick;v[6]=c.region;v[7]=c.role;v[8]=c.recognized;}
    memcpy(code+r*D,e.code,D*sizeof(float));
  }
}
int composition_finish(void*p,int mode,float lr,float lp,float route){
#ifdef SHOEBOX_CONTRIBUTION_CHECKS
  if(mode)return 0;
#endif
  auto&m=*static_cast<Model*>(p);if(!m.composition.enabled)return 1;
  if(mode<0||mode>6||mode==5)return 0;
  auto&z=m.composition.stream;z.draining=1;unsigned char byte=0;float unused=0;
  int passes=0;
  for(;passes<2*N+2;passes++){
    bool pending=false;
    for(int r=2;r<=14;r++){
#ifdef SHOEBOX_MULTI_BINDING
#ifdef SHOEBOX_LEARNED_COMPLETION
      if(r>=3){pending|=completion_ready(m,r);continue;}
#else
      if(r>=3){pending|=binding_ready(m,r);continue;}
#endif
#endif

      int driver=int(m.slots[4*m.starts[r]]);
#ifdef SHOEBOX_CONTRIBUTION_CHECKS
      if(contribution_selected(m,CHECK_SOURCE,driver)){z.length[r]=0;z.seen[r]=z.history[driver][0].ref.seq;continue;}
#endif
      if((m.s.ablation==1&&driver>=2&&driver!=17)||(m.s.ablation>=10&&driver==m.s.ablation-10)){z.length[r]=0;z.seen[r]=z.history[driver][0].ref.seq;continue;}
      pending|=z.length[r]>0||z.seen[r]!=z.history[driver][0].ref.seq;
    }
#ifdef SHOEBOX_MULTI_BINDING
    for(int r=15;r<N;r++)pending|=binding_ready(m,r);
#endif
    if(!pending)break;
    run(p,&byte,&byte,1,mode,lr,lp,route,0,&unused);
  }
  if(mode)for(int i=0;i<LACE_DEPTH+1;i++)for(int r=N-1;r>=0;r--)flush_credit(m,r,lr);
  z.draining=0;return passes<2*N+2;
}
void composition_credit_probe(void*p,int region,const float*gradient,int input_only,int depth){
  auto&m=*static_cast<Model*>(p);if(region<0||region>=N||depth<0||depth>=LACE_DEPTH)return;
  auto*v=current_lace(m,region);if(!v)return;
  if(input_only)deliver_input_credit(m,region,*v,gradient,depth);else distribute_credit(m,region,*v,gradient,depth);
}
void composition_credit_snapshot(void*p,int r,double*out){
  auto&m=*static_cast<Model*>(p);if(r<0||r>=N)return;
  for(int a=0;a<LACE_RING;a++){
    int index=(m.laces.cursor[r]-a)&(LACE_RING-1);auto&v=m.laces.records[r][index];auto&t=m.composition.stream.input[r][index];
    double*row=out+a*(8+2*D);row[0]=v.tick;row[1]=t.previous_tick;row[2]=t.retention;row[3]=v.nparents;
    row[4]=t.uses[0];row[5]=v.uses[0];row[6]=v.expert;row[7]=t.tick;
    for(int j=0;j<D;j++){row[8+j]=t.gradient[0][j];row[8+D+j]=v.credit[0][j];}
  }
}
}

#ifdef SHOEBOX_MULTI_BINDING
extern "C" void binding_role_status(void*p,uint64_t*out){
 auto&m=*static_cast<Model*>(p);
 for(int r=0;r<N;r++){const auto&e=m.composition.stream.history[r][0];out[3*r]=e.present_mask;out[3*r+1]=e.fresh_mask;out[3*r+2]=e.kind;}
}
#endif

#ifdef SHOEBOX_IDENTITY_SEMANTICS
extern "C" {
void semantic_parameters(void*p,float*out){memcpy(out,static_cast<Model*>(p)->composition.symbols,sizeof(CompositionRuntime::symbols));}
void semantic_updates(void*p,uint64_t*out){memcpy(out,static_cast<Model*>(p)->composition.symbol_updates,sizeof(CompositionRuntime::symbol_updates));}
}
#endif

#ifdef SHOEBOX_PRESERVE_RELAYS
extern "C" {
// Fresh-stream evaluation may retain the associative store without retaining
// forecasts, recurrent state, routing counts, or eligibility. Preserve memory
// coordinate centers and its clock so stored timestamps keep their meaning.
void evaluation_copy_memory(void*target,const void*source){
  auto&d=static_cast<Model*>(target)->s;const auto&s=static_cast<const Model*>(source)->s;
  memcpy(d.pk,s.pk,sizeof(d.pk));memcpy(d.pv,s.pv,sizeof(d.pv));
  memcpy(d.priority,s.priority,sizeof(d.priority));memcpy(d.utility,s.utility,sizeof(d.utility));
  memcpy(d.born,s.born,sizeof(d.born));memcpy(d.updated,s.updated,sizeof(d.updated));
  memcpy(d.qmean,s.qmean,sizeof(d.qmean));memcpy(d.vmean,s.vmean,sizeof(d.vmean));
  d.used=s.used;d.tick=s.tick;
}
void evaluation_memory_info(void*p,uint64_t*out){
  auto&s=static_cast<Model*>(p)->s;out[0]=out[1]=0;
  for(int a=0;a<2;a++)for(int i=0;i<PCAP;i++)out[a]+=s.born[a][i]!=0;
  uint64_t hash=14695981039346656037ULL;
  auto add=[&](const void*p,size_t n){auto*b=static_cast<const unsigned char*>(p);for(size_t i=0;i<n;i++){hash^=b[i];hash*=1099511628211ULL;}};
  add(s.pk,sizeof(s.pk));add(s.pv,sizeof(s.pv));add(s.priority,sizeof(s.priority));
  add(s.utility,sizeof(s.utility));add(s.born,sizeof(s.born));add(s.updated,sizeof(s.updated));
  out[2]=hash;out[3]=s.freeze_writes;out[4]=s.tick;
}
void relay_teaching_status(void*p,double*out){
 auto&z=static_cast<Model*>(p)->composition.stream;
 for(int r=0;r<N;r++){
  for(int k=0;k<6;k++)out[r*8+k]=double(z.teaching_metrics[r][k]);
  out[r*8+6]=double(z.teaching[r].tick);out[r*8+7]=z.teaching[r].index;
 }
}
void relay_event_status(void*p,uint64_t*out,float*confidence){
 auto&z=static_cast<Model*>(p)->composition.stream;
 for(int r=0;r<N;r++){
  const auto&e=z.history[r][0];auto*v=out+r*8;
  v[0]=e.local_arity;v[1]=e.relayed;v[2]=e.origin.region;v[3]=e.origin.tick;
  v[4]=e.origin.index;v[5]=e.origin.seq;v[6]=e.origin.id.a;v[7]=e.origin.id.b;
  confidence[r]=e.confidence;
 }
}
}
#endif

#ifdef SHOEBOX_CONTRIBUTION_CHECKS
#include "contribution_api.inc"
#endif

#ifdef SHOEBOX_LEARNED_MESSAGES
extern "C" {
int message_configure(void*p,const float*values){
  for(int i=0;i<4;i++)if(!std::isfinite(values[i])||values[i]<0||values[i]>(i==3?8.f:1.f))return 0;
  auto&m=*static_cast<Model*>(p);
  // Saved derivatives belong to this policy; only change at a stream boundary.
  for(int r=0;r<N;r++)if(m.composition.stream.history[r][0].ref.seq)return 0;
  memcpy(m.message_config,values,sizeof(m.message_config));return 1;
}
void message_metrics(void*p,double*out){memcpy(out,static_cast<Model*>(p)->composition.stream.message_metrics,sizeof(CompositionStream::message_metrics));}
}
#endif

#ifdef SHOEBOX_LEARNED_COMPLETION
extern "C" {
void completion_metrics(void*p,uint64_t*out){memcpy(out,static_cast<Model*>(p)->composition.stream.completion_metrics,sizeof(CompositionStream::completion_metrics));}
void completion_gates(void*p,float*out){memcpy(out,static_cast<Model*>(p)->composition.payload_gates,sizeof(CompositionRuntime::payload_gates));}
size_t completion_capacity(void*p){return constituent_capacity(*static_cast<Model*>(p));}
void completion_clocks(void*p,int64_t*out){
  auto&s=static_cast<Model*>(p)->composition.stream;
  for(int r=0;r<N;r++){out[2*r]=s.history[r][0].clock_begin;out[2*r+1]=s.history[r][0].clock_end;}
}
}
#endif
