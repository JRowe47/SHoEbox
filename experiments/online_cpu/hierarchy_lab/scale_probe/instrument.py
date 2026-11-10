from pathlib import Path
ROOT=Path(__file__).resolve().parent
text=(ROOT.parent/'learned_state/native.cpp').read_text()
text=text.replace('struct Model {','''struct ProbeRuntime {
  float config[8]{.2f,0,0,0,0,0,0,0};
  float means[N][D]{},portmem[N][D]{},cues[N][LACE_RING][D]{};
  int64_t ticks[N][LACE_RING]{};
};
struct Diagnostics {
  bool enabled=false;
  double n[N]{}, sums[N][4][D]{}, products[N][4][D][D]{}, extra[N][8]{};
};
struct Model {''',1)
text=text.replace('  State s;','  State s;\n  ProbeRuntime probe;\n  Diagnostics diag;',1)
text=text.replace('ca=.2f/std::sqrt','ca=m.probe.config[0]/std::sqrt',1)
text=text.replace('float q[D]{},c[D]{};float qw=0,cw=0;', 'float q[D]{},c[D]{},portcue[D]{};float qw=0,cw=0;',1)
text=text.replace('if(delay)cw+=w*w;else qw+=w*w;', '''if(delay)cw+=w*w;else qw+=w*w;
        if(m.probe.config[1]>=2){int slot=k-m.starts[r];
          for(int j=0;j<D;j++)portcue[slot*8+j%8]+=w*m.signs[k*D+j]*v[m.perm[k*D+j]]*.3535533905932738f;
        }''',1)
text=text.replace('      int best=0,runner=0;float score=-1e30f,second=-1e30f;', '''      float routing[D];memcpy(routing,s.u[r],D*sizeof(float));
      if(r>=2&&m.probe.config[1]>0){
        if(m.probe.config[1]>=2){
          int base=(m.starts[r+1]-m.starts[r])*8;
          if(r==17)for(int j=0;j<D;j++)portcue[base+j%8]+=.25f*m.emb[x[t]*D+j]*.3535533905932738f;
          if(r==15||r==16)for(int j=0;j<D;j++)portcue[base+j%8]+=m.cfg[11]*s.h[0][j]*.3535533905932738f;
          for(int j=0;j<D;j++)m.probe.portmem[r][j]=(1-alpha)*m.probe.portmem[r][j]+alpha*portcue[j];
        }
        for(int j=0;j<D;j++){
          float cue=m.probe.config[1]>=2?m.probe.portmem[r][j]:s.u[r][j];
          routing[j]=cue-m.probe.means[r][j];
          if(mode)m.probe.means[r][j]+=m.probe.config[2]*(cue-m.probe.means[r][j]);
        }
      }
      int best=0,runner=0;float score=-1e30f,second=-1e30f;''',1)
text=text.replace('feature_mask(s.u[r],sparse_k)','feature_mask(routing,sparse_k)',1)
text=text.replace('v+=s.u[r][j]*m.keys[(r*E+e)*D+j]','v+=routing[j]*m.keys[(r*E+e)*D+j]')
text=text.replace('v.residual,residual,D*sizeof(float));','v.residual,residual,D*sizeof(float));\n        memcpy(m.probe.cues[r][cursor],routing,D*sizeof(float));m.probe.ticks[r][cursor]=v.tick;',1)
text=text.replace('uint64_t mask=v.input_mask;\n  while(mask){int j=__builtin_ctzll(mask);mask&=mask-1;float d=step*v.u[j];', '''uint64_t mask=v.input_mask;
  const float*cue=v.u;
  if(r>=2&&m.probe.config[1]>0){int id=int(&v-&m.laces.records[r][0]);
    if(id<0||id>=LACE_RING||m.probe.ticks[r][id]!=v.tick)return;
    cue=m.probe.cues[r][id];
  }
  while(mask){int j=__builtin_ctzll(mask);mask&=mask-1;float d=step*cue[j];''',1)
text=text.replace('float n=std::sqrt(norm(s.u[r])-1.f+1e-6f),kn=0;', 'float n=std::sqrt(norm(routing)-1.f+1e-6f),kn=0;',1)
text=text.replace('float z=s.u[r][indices[a]];n+=z*z;', 'float z=routing[indices[a]];n+=z*z;',1)
text=text.replace('?s.u[r][j]/n:0;k[j]', '?routing[j]/n:0;k[j]',1)
marker='      publish_state(m,r,gate,runner);'
text=text.replace(marker,'''      if(m.diag.enabled&&s.exec[r]%16==0){
        auto &d=m.diag;d.n[r]++;
        const float* vectors[4]{q,c,s.u[r],s.h[r]};
        for(int k=0;k<4;k++)for(int i=0;i<D;i++){
          d.sums[r][k][i]+=vectors[k][i];
          for(int j=0;j<D;j++)d.products[r][k][i][j]+=double(vectors[k][i])*vectors[k][j];
        }
        d.extra[r][0]+=score-second;
        for(int j=0;j<D;j++){
          d.extra[r][1]+=s.h[r][j]*s.h[r][j]>.9025f;
          d.extra[r][2]+=q[j]*q[j];d.extra[r][3]+=c[j]*c[j];
          if(routing_mask&(uint64_t(1)<<j))d.extra[r][4]+=c[j]*c[j];
        }
      }
'''+marker,1)
text+='''
extern "C" {
void probe_configure(void*p,float*cfg){memcpy(static_cast<Model*>(p)->probe.config,cfg,8*sizeof(float));}
void diagnostic_enable(void*p,int enabled){auto &d=static_cast<Model*>(p)->diag;d=Diagnostics{};d.enabled=enabled;}
void diagnostic_export(void*p,double*out){auto&d=static_cast<Model*>(p)->diag;memcpy(out,d.n,sizeof(d.n));out+=N;memcpy(out,d.sums,sizeof(d.sums));out+=N*4*D;memcpy(out,d.products,sizeof(d.products));out+=N*4*D*D;memcpy(out,d.extra,sizeof(d.extra));}
size_t probe_state_size(){return sizeof(ProbeRuntime);}
void probe_export(void*p,void*out){memcpy(out,&static_cast<Model*>(p)->probe,sizeof(ProbeRuntime));}
void probe_import(void*p,void*in){memcpy(&static_cast<Model*>(p)->probe,in,sizeof(ProbeRuntime));}
void probe_initial_mean(void*p){auto&m=*static_cast<Model*>(p);if(m.probe.config[1]==1)memcpy(m.probe.means,m.s.u,N*D*sizeof(float));}
void probe_get_means(void*p,float*out){memcpy(out,static_cast<Model*>(p)->probe.means,N*D*sizeof(float));}
void probe_set_means(void*p,float*in){memcpy(static_cast<Model*>(p)->probe.means,in,N*D*sizeof(float));}
}
'''
(ROOT/'native.cpp').write_text(text)
