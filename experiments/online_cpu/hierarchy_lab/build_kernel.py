from pathlib import Path
import difflib
ROOT=Path(__file__).parent
base=(ROOT.parent/'persistent_dual_native.cpp').read_text();s=base
def replace(old,new):
 global s
 assert s.count(old)==1,(s.count(old),old[:80])
 s=s.replace(old,new)
replace('struct State {',(ROOT/'lace_types.inc').read_text()+'\nstruct State {')
replace('  State s;','  State s;\n  LaceState laces;\n  // depth, credit gain, SDR k, task router gain, gate temperature, reuse exponent, DFA gain, task sampling stride, first sparse region, credit feature budget, third-hop stride, gated credit flush\n  float lcfg[12]{0,1,64,0,.25f,.5f,1,1,0,64,1,0};')
replace('inline float norm(const float*x)', '''inline LaceRecord* current_lace(Model&,int);
inline void route_task_update(Model&,int,LaceRecord&,const float*,float);
inline void sparse_weight_update(float*,const float*,float,uint64_t);
inline float norm(const float*x)''')
replace('inline void expert_update(Model&m,int r,const float*g,float rate){\n  State&s=m.s;int e=s.chosen[r];float a=rate/norm(s.u[r]);', '''inline void expert_update(Model&m,int r,const float*g,float rate,LaceRecord*record=nullptr){
  State&s=m.s;
  LaceRecord*v=record?record:current_lace(m,r);
  int e=record?record->expert:s.chosen[r];
  const float*u=record?record->u:s.u[r],*h=record?record->h:s.h[r];
  const float*returned=record?record->returned:s.returned[std::clamp(r-15,0,1)];
  float gate=v?v->gate:1.f,a=rate/norm(u);''')
start=s.index('inline void expert_update');end=s.index('extern "C"',start)
part=s[start:end].replace('s.h[r][i]','h[i]').replace('s.u[r][j]','u[j]').replace('a*d*u[j]','a*d*gate*u[j]').replace('s.returned[r-15][i]','returned[i]').replace('s.returned[r-15][j]','returned[j]')
part=part.replace('  if((r==15||r==16)&&m.cfg[10]&&s.recall_slot', '  if(!record&&(r==15||r==16)&&m.cfg[10]&&s.recall_slot')
part=part.replace('for(int j=0;j<D;j++)w[i*D+j]-=a*d*gate*u[j];','''if(record&&m.lcfg[9]<D){
      if(d!=0)sparse_weight_update(w+i*D,u,a*d*gate,record->input_mask);
    }else for(int j=0;j<D;j++)w[i*D+j]-=a*d*gate*u[j];''')
part=part.rstrip()[:-1]+'''  if(record&&(r==15||r==16)&&m.cfg[10]&&record->recall_slot>=0&&record->recall_birth==s.born[r-15][record->recall_slot]){
    int a=r-15,k=record->recall_slot;float influence=0;
    for(int i=0;i<D;i++){
      float contribution=int(m.cfg[10])==2?m.P[(r*D+i)*D+i]*record->returned[i]:dot64(m.P+(r*D+i)*D,record->returned);
      influence-=g[i]*(1-h[i]*h[i])*contribution;
    }
    s.utility[a][k]+=.01f*(std::clamp(influence,-1.f,1.f)-s.utility[a][k]);
  }
  if(v)route_task_update(m,r,*v,g,rate);
}
'''
s=s[:start]+part+'\n'+(ROOT/'lace_functions.inc').read_text()+'\n'+s[end:]
replace('void reset(void*p){auto&s=static_cast<Model*>(p)->s;', 'void reset(void*p){auto&m=*static_cast<Model*>(p);m.laces=LaceState{};auto&s=m.s;')
replace('void reset_context(void*p){auto&s=static_cast<Model*>(p)->s;', 'void reset_context(void*p){auto&m=*static_cast<Model*>(p);m.laces=LaceState{};auto&s=m.s;')
replace('void query_vectors(void*p,float*out)', '''size_t lace_state_size(){return sizeof(LaceState);}
void export_laces(void*p,void*out){memcpy(out,&static_cast<Model*>(p)->laces,sizeof(LaceState));}
void import_laces(void*p,void*in){memcpy(&static_cast<Model*>(p)->laces,in,sizeof(LaceState));}
void configure_laces(void*p,float*cfg){memcpy(static_cast<Model*>(p)->lcfg,cfg,12*sizeof(float));}
void lace_metrics(void*p,double*out){memcpy(out,static_cast<Model*>(p)->laces.metrics,N*12*sizeof(double));}
void lace_codes(void*p,uint64_t*out){auto&m=*static_cast<Model*>(p);for(int r=0;r<N;r++){auto*v=current_lace(m,r);out[3*r]=v?v->input_mask:0;out[3*r+1]=v?v->feature_mask:0;out[3*r+2]=v?v->expert:0;}}
void query_vectors(void*p,float*out)''')
replace('s.tick++;float snapshot[N][D];int valid[N];','s.tick++;float snapshot[N][D];int valid[N];int64_t snapshot_tick[N]{};int snapshot_index[N]{};\n    bool tracing=m.lcfg[0]>0||m.lcfg[2]<D||m.lcfg[3]>0;')
replace('valid[r]=s.valid[r];}', 'valid[r]=s.valid[r];auto*v=current_lace(m,r);snapshot_tick[r]=v?v->tick:0;snapshot_index[r]=m.laces.cursor[r];}')
replace('active[r]=true;float q[D]{},c[D]{};float qw=0,cw=0;', '''active[r]=true;
      if(mode&&!m.lcfg[11])flush_credit(m,r,lr);
      LaceParent parents[LACE_LINKS];int nparents=0;
      float q[D]{},c[D]{};float qw=0,cw=0;''')
replace('if(delay)cw+=w*w;else qw+=w*w;', '''if(delay)cw+=w*w;else qw+=w*w;
        if(tracing){auto&v=m.laces.records[src][(m.laces.cursor[src]-age)&(LACE_RING-1)];
          parents[nparents++]={src,k,delay?snapshot_tick[src]:v.tick,w*(delay?-1.f:1.f),delay?snapshot_index[src]:(m.laces.cursor[src]-age)&(LACE_RING-1)};
        }''')
replace('qgain[r]=qa;', '''qgain[r]=qa;
      for(int k=0;k<nparents;k++)parents[k].coefficient*=parents[k].coefficient<0?-ca:qa;
      if(tracing&&(r==15||r==16)&&m.cfg[11]){auto*v=current_lace(m,0);parents[nparents++]={0,-1,v?v->tick:0,qa*m.cfg[11],m.laces.cursor[0]};}''')
replace('int best=0;float score=-1e30f;', '''int best=0,runner=0;float score=-1e30f,second=-1e30f;
      int sparse_k=r>=int(m.lcfg[8])?int(m.lcfg[2]):D;
      bool task_route=m.lcfg[3]>0&&r>=int(m.lcfg[8]);
      uint64_t routing_mask=feature_mask(s.u[r],sparse_k);
      int indices[D],nindices=0;uint64_t bits=routing_mask;
      while(bits){indices[nindices++]=__builtin_ctzll(bits);bits&=bits-1;}''')
replace('float v=0;for(int j=0;j<D;j++)v+=s.u[r][j]*m.keys[(r*E+e)*D+j];', '''float v=0;
        if(sparse_k>=D)for(int j=0;j<D;j++)v+=s.u[r][j]*m.keys[(r*E+e)*D+j];
        else for(int a=0;a<nindices;a++){int j=indices[a];v+=s.u[r][j]*m.keys[(r*E+e)*D+j];}''')
replace('if(v>score){score=v;best=e;}', 'if(v>score){second=score;runner=best;score=v;best=e;}else if(v>second){second=v;runner=e;}')
replace('s.chosen[r]=best;s.exec[r]++;s.counts[r][best]++;', '''s.chosen[r]=best;s.exec[r]++;s.counts[r][best]++;
      float gate=1,gate_derivative=0;
      if(task_route){float p=1.f/(1.f+std::exp(-(score-second)/m.lcfg[4]));gate=2*p;gate_derivative=2*p*(1-p)/m.lcfg[4];}
      float residual[D];''')
replace('float z=s.u[r][i]+b[i];for(int j=0;j<D;j++)z+=w[i*D+j]*s.u[r][j];', '''float z=s.u[r][i]+b[i];
        if(!task_route){for(int j=0;j<D;j++)z+=w[i*D+j]*s.u[r][j];residual[i]=0;}
        else {residual[i]=dot64(w+i*D,s.u[r]);z+=gate*residual[i];}''')
replace('memcpy(s.hist[r][1],s.hist[r][0],D*sizeof(float));', '''if(tracing){
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
      memcpy(s.hist[r][1],s.hist[r][0],D*sizeof(float));''')
replace('for(int j=0;j<D;j++){k[j]+=(s.u[r][j]/n-k[j])*route;kn+=k[j]*k[j];}', '''if(sparse_k<D){n=0;for(int a=0;a<nindices;a++){float z=s.u[r][indices[a]];n+=z*z;}n=std::sqrt(n+1e-6f);}
        for(int j=0;j<D;j++){float cue=(routing_mask&(uint64_t(1)<<j))?s.u[r][j]/n:0;k[j]+=(cue-k[j])*route;kn+=k[j]*k[j];}''')
replace('if(m.cfg[6]&&s.valid[15]', 'if(!m.lcfg[0]&&m.cfg[6]&&s.valid[15]')
replace('      // A future observed bottom-up bundle teaches the previous publication.', '      if(mode&&m.lcfg[11])flush_credit(m,r,lr);\n      // A future observed bottom-up bundle teaches the previous publication.')
replace('// Exact local task derivative at A1 and Y;', '''if(m.lcfg[0]&&s.tick%int(m.lcfg[7])==0){auto*v=current_lace(m,17);if(v)distribute_credit(m,17,*v,direct+D,0);}
      // Exact local task derivative at A1 and Y;''')
replace('if(active[r]&&!(r==15&&m.cfg[6]))', 'if(active[r]&&!(r==15&&m.cfg[6])&&(!m.lcfg[0]||m.lcfg[6]>0))')
replace('g[j]=m.fsign[r*D+j]*broadcast[m.fperm[r*D+j]];', 'g[j]=m.fsign[r*D+j]*broadcast[m.fperm[r*D+j]]*(m.lcfg[0]?m.lcfg[6]:1.f);')
(ROOT/'lace_native.cpp').write_text(s)
(ROOT/'lace_changes.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='persistent_dual_native.cpp',tofile='lace_native.cpp')))
print('Generated',ROOT/'lace_native.cpp')
