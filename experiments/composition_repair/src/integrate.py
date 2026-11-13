"""Reproducibly apply the composition repair to the preserved native runtime."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'runtime/hierarchy_lab/throughput_probe'
s=(root/'native.cpp').read_text()
def change(old,new,count=1):
    global s
    if s.count(old)!=count:raise RuntimeError((old[:100],s.count(old),count))
    s=s.replace(old,new)
change('struct Model {','#include "composition_types.inc"\nstruct Model {\n  CompositionRuntime composition;')
# Teaching-source gain governs every error-driven update, including routing and
# bottom-state eligibility. Prototype competition has its own independent rate.
change('  if(m.frozen_region[r])return;\n  State&s=m.s;',
       '  if(m.frozen_region[r])return;\n  rate*=m.teaching_scale[r][teaching];\n  if(rate==0)return;\n  State&s=m.s;')
change('  LaceRecord*v=record?record:current_lace(m,r);',
       '  LaceRecord*v=record?record:current_lace(m,r);\n  if(v&&v->expert<0)return;')
change('  if(m.teaching_scale[r][teaching]!=1.f)a*=m.teaching_scale[r][teaching];','')
change('        float a=local_rate/norm(s.h[r]);','        float a=local_rate*m.teaching_scale[r][1]/norm(s.h[r]);')
# The composition router sees only constituent content; saved cues are needed by
# its task update even when the legacy centering probe is disabled.
change('  if(r>=2&&m.probe.config[1]>0){int id=int(&v-&m.laces.records[r][0]);',
       '  if(r>=2&&(m.composition.enabled||m.probe.config[1]>0)){int id=int(&v-&m.laces.records[r][0]);')
# Separate gradient of retained INPUT from the publication's OUTPUT gradient.
start=s.index('  for(int k=0;k<v.nparents;k++){',s.index('inline void distribute_credit'))
end=s.index('\n}\n// Apply accumulated',start)
original=s[start:end]
new='''  deliver_input_credit(m,r,v,du,depth);'''
s=s[:start]+new+s[end:]
helper='''inline void deliver_input_credit(Model&m,int r,LaceRecord&v,const float*du,int depth){
  if(depth>=int(m.lcfg[0]))return;
  for(int k=0;k<v.nparents;k++){
    const auto&edge=v.parents[k];
    if(edge.region==0&&!m.composition.enabled)continue;
    LaceRecord*p=find_lace(m,edge.region,edge.tick,edge.index);
    if(!p){m.laces.metrics[r][3]++;continue;}
    if(m.composition.enabled&&p->expert<0){
      float gradient[D]{};
      if(edge.slot<=-100){int position=-100-edge.slot;for(int j=0;j<D;j++)gradient[content_coord(j,position)]+=edge.coefficient*content_sign(j,position)*du[j];}
      else if(edge.slot<0){for(int j=0;j<D;j++)gradient[j]=edge.coefficient*du[j];}
      else for(int j=0;j<D;j++)gradient[m.perm[edge.slot*D+j]]+=edge.coefficient*m.signs[edge.slot*D+j]*du[j];
      // Transparent identity relays do not consume a learned-composition hop.
      deliver_input_credit(m,edge.region,*p,gradient,depth);m.composition.stream.metrics[edge.region][12]++;continue;
    }
    if(edge.slot<=-100){int position=-100-edge.slot;
      for(int j=0;j<D;j++)p->credit[depth][content_coord(j,position)]+=edge.coefficient*content_sign(j,position)*du[j];
    }else if(edge.slot<0){for(int j=0;j<D;j++)p->credit[depth][j]+=edge.coefficient*du[j];}
    else for(int j=0;j<D;j++)p->credit[depth][m.perm[edge.slot*D+j]]+=edge.coefficient*m.signs[edge.slot*D+j]*du[j];
    p->uses[depth]++;m.laces.metrics[edge.region][1]++;m.laces.metrics[edge.region][8+depth]++;
  }
  if(m.composition.enabled&&r>=2){
    int index=int(&v-&m.laces.records[r][0]);auto&z=m.composition.stream;
    auto&state=z.input[r][index];
    if(state.tick!=v.tick||!state.previous_tick||state.retention==0)return;
    auto*previous=find_lace(m,r,state.previous_tick,state.previous_index);
    if(!previous){z.metrics[r][10]++;return;}
    int pindex=int(previous-&m.laces.records[r][0]);auto&target=z.input[r][pindex];
    if(target.tick!=previous->tick){z.metrics[r][10]++;return;}
    for(int j=0;j<D;j++)target.gradient[depth][j]+=state.retention*du[j];
    target.uses[depth]++;z.metrics[r][8]++;
  }
}
'''
# Coordinate maps are shared by transport and its transpose.
change('inline void distribute_credit(Model&m,int r,LaceRecord&v,const float*g,int depth){',
       'inline int content_coord(int,int);\ninline float content_sign(int,int);\n'+helper+'inline void distribute_credit(Model&m,int r,LaceRecord&v,const float*g,int depth){')
change('  if(depth>=2&&m.lcfg[10]>1&&(v.tick/m.cadence[r])%int(m.lcfg[10]))return;',
       '  if(m.composition.enabled&&v.expert<0){deliver_input_credit(m,r,v,g,depth);return;}\n  if(!m.composition.enabled&&depth>=2&&m.lcfg[10]>1&&(v.tick/m.cadence[r])%int(m.lcfg[10]))return;')
change('  if(!m.lcfg[0])return;\n  for(int a=LACE_RING-1;a>=0;a--){',
'''  if(!m.lcfg[0])return;
  if(m.composition.enabled&&r>=2){
    auto&z=m.composition.stream;
    // Newest to oldest: a bounded depth budget, no spatial expert update for
    // input-only state dependence, and no gradient through a fictitious h edge.
    for(int a=0;a<LACE_RING;a++){
      int index=(m.laces.cursor[r]-a)&(LACE_RING-1);
      auto&v=m.laces.records[r][index];auto&state=z.input[r][index];
      if(!v.tick||state.tick!=v.tick)continue;
      for(int d=0;d<int(m.lcfg[0]);d++)if(state.uses[d]){
        float gradient[D];float scale=1.f/std::pow(float(state.uses[d]),m.lcfg[5]);
        for(int j=0;j<D;j++)gradient[j]=state.gradient[d][j]*scale;
        memset(state.gradient[d],0,D*sizeof(float));state.uses[d]=0;
        deliver_input_credit(m,r,v,gradient,d+1);z.metrics[r][9]++;
      }
    }
  }
  for(int a=LACE_RING-1;a>=0;a--){''')
change('extern "C" {\nvoid* create(', '#include "composition_events.inc"\nextern "C" {\nvoid* create(')
change('void reset(void*p){auto&m=*static_cast<Model*>(p);m.laces=LaceState{};',
       'void reset(void*p){auto&m=*static_cast<Model*>(p);memset(&m.composition.stream,0,sizeof(m.composition.stream));m.laces=LaceState{};')
change('void reset_context(void*p){auto&m=*static_cast<Model*>(p);m.laces=LaceState{};',
       'void reset_context(void*p){auto&m=*static_cast<Model*>(p);memset(&m.composition.stream,0,sizeof(m.composition.stream));m.laces=LaceState{};')
change('    bool tracing=m.lcfg[0]>0||m.lcfg[2]<D||m.lcfg[3]>0;',
       '    bool tracing=m.composition.enabled||m.lcfg[0]>0||m.lcfg[2]<D||m.lcfg[3]>0;')
change('    s.tick++;float snapshot[N][D];',
       '    s.tick++;if(!m.composition.stream.draining)m.composition.stream.byte_position++;float snapshot[N][D];')
change('int snapshot_index[N]{};', 'int snapshot_index[N]{},snapshot_region[N]{};')
change('snapshot_index[r]=m.laces.cursor[r];}',
'''snapshot_index[r]=m.laces.cursor[r];snapshot_region[r]=r;
      if(m.composition.enabled){const auto&origin=m.composition.stream.history[r][0].origin;
        snapshot_region[r]=origin.region;snapshot_tick[r]=origin.tick;snapshot_index[r]=origin.index;}
    }''')
change('parents[nparents++]={src,k,delay?snapshot_tick[src]:v.tick,',
       'parents[nparents++]={delay?snapshot_region[src]:src,k,delay?snapshot_tick[src]:v.tick,')
change('State&s=m.s;double loss=0;m.refresh_keys();',
       'State&s=m.s;double loss=0;ContentEvent composed{},parts[COMP_MAX]{};m.refresh_keys();')
change('      bool scheduled=s.tick%m.cadence[r]==0;',
'''      if(m.composition.stream.draining&&r<2)continue;
      int nparts=0;
      bool composition_learning=mode!=0&&mode!=4&&!m.frozen_region[r];
      if(m.composition.enabled&&r>=2&&!content_prepare(m,r,composition_learning,composed,parts,nparts))continue;
      if(m.composition.enabled&&r>=2&&r<=14&&nparts==1){
        if(mode)flush_credit(m,r,lr);
        content_passthrough(m,r,composed,parts[0]);continue;
      }
      bool scheduled=m.composition.enabled||s.tick%m.cadence[r]==0;''')
change('      bool event_ec=(r==15||r==16)&&m.cfg[19]>0;',
       '      bool event_ec=!m.composition.enabled&&(r==15||r==16)&&m.cfg[19]>0;')
change('      if(r<2){memcpy(q,m.emb+x[t]*D,D*sizeof(float));qw=1;}',
'''      if(r<2){memcpy(q,m.emb+x[t]*D,D*sizeof(float));qw=1;}
      if(m.composition.enabled&&r>=2)content_bundle(m,r,composed,parts,nparts,q,c,qw,cw,parents,nparents);''')
change('        const float*slot=m.slots+4*k;int src=int(slot[0]),age=int(slot[1]),delay=int(slot[2]);float w=slot[3];',
'''        const float*slot=m.slots+4*k;int src=int(slot[0]),age=int(slot[1]),delay=int(slot[2]);float w=slot[3];
        bool contextual=m.composition.enabled&&r>=2;
        if(contextual&&!delay){
          // Driver constituents are already present. Other levels are held
          // context, never counted again as newly completed constituents.
          if(r>=15||src==int(m.slots[4*m.starts[r]]))continue;
          delay=1;
        }''')
# Source age on a delayed port must match both the consumed vector and lace ID.
change('        const float*v=delay?snapshot[src]:s.hist[src][age];float*dest=delay?c:q;',
       '        const float*v=delay?snapshot[src]:s.hist[src][age];float*dest=delay?c:q;')
change('      if((r==15||r==16)&&m.cfg[11]){','      if(!m.composition.enabled&&(r==15||r==16)&&m.cfg[11]){')
change('      if(r==17){for(int j=0;j<D;j++)q[j]+=.25f*m.emb[x[t]*D+j];qw+=.0625f;}',
       '      if(r==17&&!m.composition.stream.draining){for(int j=0;j<D;j++)q[j]+=.25f*m.emb[x[t]*D+j];qw+=.0625f;}')
change('float qa=1.f/std::sqrt(std::max(1.f,qw)),ca=m.probe.config[0]/std::sqrt(std::max(1.f,cw));',
       'float qa=1.f/std::sqrt(std::max(1.f,qw)),ca=(m.composition.enabled?m.composition.context_gain:m.probe.config[0])/std::sqrt(std::max(1.f,cw));')
change('      if(tracing&&(r==15||r==16)&&m.cfg[11]){',
       '      if(tracing&&!m.composition.enabled&&(r==15||r==16)&&m.cfg[11]){')
# Pure event targets eliminate feedback leakage through the child teacher.
change('      if(r>=2&&m.probe.config[1]>0){\n        if(m.probe.config[1]>=2){',
       '      if(!m.composition.enabled&&r>=2&&m.probe.config[1]>0){\n        if(m.probe.config[1]>=2){')
change('      int best=0,runner=0;float score=-1e30f,second=-1e30f;',
       '      if(m.composition.enabled&&r>=2)memcpy(routing,q,D*sizeof(float));\n      int best=0,runner=0;float score=-1e30f,second=-1e30f;')
change('        auto&v=m.laces.records[r][cursor];\n        for(int d=0;',
'''        auto&v=m.laces.records[r][cursor];
        if(m.composition.enabled){auto&state=m.composition.stream.input[r][cursor];
          for(int d=0;d<LACE_DEPTH;d++)m.composition.stream.metrics[r][11]+=state.uses[d];
          state=InputCredit{};state.tick=s.tick;
          if(r>=2){auto&z=m.composition.stream;state.previous_tick=z.memory_tick[r];state.previous_index=z.memory_index[r];state.retention=1-alpha;z.memory_tick[r]=s.tick;z.memory_index[r]=cursor;}
        }
        for(int d=0;''')
change('      memcpy(s.hist[r][1],s.hist[r][0],D*sizeof(float));memcpy(s.hist[r][0],s.h[r],D*sizeof(float));s.valid[r]++;',
'''      memcpy(s.hist[r][1],s.hist[r][0],D*sizeof(float));memcpy(s.hist[r][0],s.h[r],D*sizeof(float));s.valid[r]++;
      if(m.composition.enabled){if(r<2)content_atom(m,r,x[t]);else if(nparts&&(r!=17||composed.ref.seq==0))content_publish(m,r,composed);}''')
# s.exec counts actual neural events, so the existing one-in-four local
# teaching schedule already operates on event time and remains unchanged.
change('    float features[2*D],logits[V],prob[V],maxlog=-1e30f;int selected[V],nout=V;',
       '    if(m.composition.stream.draining)continue;\n    float features[2*D],logits[V],prob[V],maxlog=-1e30f;int selected[V],nout=V;')
# Exact spatial legacy mode lacks the event/state Jacobians; reject in Python.
s+='\n#include "composition_api.inc"\n'
(root/'composition.cpp').write_text(s)
