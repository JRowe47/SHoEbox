"""Controlled changes to decoder teaching and regional auxiliary targets."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent
s=(ROOT.parent/'scale_probe/native.cpp').read_text()
def replace(old,new,n=1):
    global s
    assert s.count(old)==n,(old[:100],s.count(old),n)
    s=s.replace(old,new)
replace('struct Model {', '''struct UtilityRuntime {
  // center_Y, split_head_norm, residual_A1_loss, upper_local_gain,
  // local_innovation_target, mean_rate, disable_Y_decoder, reserved
  float config[8]{0,0,0,1,0,.001f,0,0};
  float ymean[D]{},previous_q[N][D]{};
  int have_q[N]{};
};
struct Model {''')
replace('  Diagnostics diag;', '''  Diagnostics diag;
  UtilityRuntime utility;
  int utility_ablation=0; // 1: mean-Y at decoder only; never a training mode
  bool utility_observe=false;
  double utility_metrics[N][8]{}; // task norm/calls, local norm/calls, reserved
''')
replace('    W=p;p+=N*E*D*D;', '''    // Canonicalize native padding for deterministic checkpoint fingerprints.
    memset(&s,0,sizeof(s));s.rng=7131587;s.recall_slot[0]=s.recall_slot[1]=-1;
    W=p;p+=N*E*D*D;''')
replace('  learn_bottom_state(m,r,v,g,rate);', '''  if(m.utility_observe){double z=0;for(int j=0;j<D;j++)z+=double(g[j])*g[j];
    m.utility_metrics[r][0]+=std::sqrt(z);m.utility_metrics[r][1]++;}
  learn_bottom_state(m,r,v,g,rate);''')
replace('if((mode==1||mode==3||mode==6)&&s.pending[r]&&!(m.cfg[10]&&(r==15||r==16))){',
'''if((mode==1||mode==3||mode==6)&&s.pending[r]&&!(m.cfg[10]&&(r==15||r==16))&&
         (r<2||m.utility.config[3]>0)){''')
replace('err[i]=std::clamp(s.pred[r][i]-q[i],-1.f,1.f);',
'''float target=q[i];if(r>=2&&m.utility.config[4]>0&&m.utility.have_q[r])target-=m.utility.previous_q[r][i];
          err[i]=std::clamp(s.pred[r][i]-target,-1.f,1.f);''')
replace('        expert_update(m,r,g,lp);\n        float a=lp/norm(s.h[r]);',
'''        float local_rate=lp*(r>=2?m.utility.config[3]:1.f);
        if(m.utility_observe){double z=0;for(int j=0;j<D;j++)z+=double(g[j])*g[j];
          m.utility_metrics[r][2]+=std::sqrt(z);m.utility_metrics[r][3]++;}
        expert_update(m,r,g,local_rate);
        float a=local_rate/norm(s.h[r]);''')
replace('      float alpha=r==1?.125f:.5f;\n      for(int j=0;j<D;j++){s.u[r]',
'''      if(mode&&m.utility.config[4]>0){memcpy(m.utility.previous_q[r],q,D*sizeof(float));m.utility.have_q[r]=1;}
      float alpha=r==1?.125f:.5f;
      for(int j=0;j<D;j++){s.u[r]''')
replace('features[D+j]=s.h[17][j];',
'''features[D+j]=m.utility.config[6]>0?0.f:(m.utility_ablation?m.utility.ymean[j]:s.h[17][j])-m.utility.config[0]*m.utility.ymean[j];''')
replace('m.readout[node*2*D+D+j]*m.s.h[17][j]',
'''(m.utility.config[6]>0?0.f:m.readout[node*2*D+D+j]*((m.utility_ablation?m.utility.ymean[j]:m.s.h[17][j])-m.utility.config[0]*m.utility.ymean[j]))''')
replace('float direct[2*D]{},broadcast[D]{};',
'''float base_error[V]{};
      if(m.utility.config[2]>0){
        // An independently taught A1 branch supplies base logits; the Y
        // branch minimizes the combined cross entropy with that base detached.
        // Only observed target-path bits are taught; no second regional pass.
        for(int k=0;k<nout;k++){int v=selected[k];float z=m.rbias[v];
          for(int j=0;j<D;j++)z+=m.readout[v*2*D+j]*features[j];
          int bit=(y[t]>>(7-k))&1;base_error[v]=1.f/(1.f+std::exp(-z))-bit;
        }
      }
      float direct[2*D]{},broadcast[D]{};''')
replace('direct[j]+=m.readout[v*2*D+j]*prob[v];',
'''direct[j]+=m.readout[v*2*D+j]*((j<D&&m.utility.config[2]>0)?base_error[v]:prob[v]);''')
replace('      float gec[D]{};', '''      if(m.utility.config[6]>0)memset(direct+D,0,D*sizeof(float));
      float gec[D]{};''')
replace('float fn=1.f;for(float v:features)fn+=v*v;float a=lr/fn;\n      for(int k=0;k<nout;k++){int v=selected[k];m.rbias[v]-=lr*.05f*prob[v];for(int j=0;j<2*D;j++)m.readout[v*2*D+j]-=a*prob[v]*features[j];}',
'''float fn=1.f;for(float v:features)fn+=v*v;
      float norms[2]{fn,fn};if(m.utility.config[1]>0){norms[0]=norm(features);norms[1]=norm(features+D);}
      for(int k=0;k<nout;k++){int v=selected[k];
        m.rbias[v]-=lr*.05f*(m.utility.config[2]>0?base_error[v]:prob[v]);
        for(int j=0;j<2*D;j++)m.readout[v*2*D+j]-=lr/norms[j/D]*((j<D&&m.utility.config[2]>0)?base_error[v]:prob[v])*features[j];
      }
      // Estimate preprocessing means only after this prediction and teaching.
      // Frozen evaluation never updates these fitted statistics.
      for(int j=0;j<D;j++)m.utility.ymean[j]+=m.utility.config[5]*(s.h[17][j]-m.utility.ymean[j]);''')
s+='''
extern "C" {
void utility_configure(void*p,float*config){memcpy(static_cast<Model*>(p)->utility.config,config,8*sizeof(float));}
size_t utility_state_size(){return sizeof(UtilityRuntime);}
void utility_export(void*p,void*out){memcpy(out,&static_cast<Model*>(p)->utility,sizeof(UtilityRuntime));}
void utility_import(void*p,void*in){memcpy(&static_cast<Model*>(p)->utility,in,sizeof(UtilityRuntime));}
void utility_mean(void*p,float*out){memcpy(out,static_cast<Model*>(p)->utility.ymean,D*sizeof(float));}
void utility_set_mean(void*p,float*in){memcpy(static_cast<Model*>(p)->utility.ymean,in,D*sizeof(float));}
void utility_ablate(void*p,int value){static_cast<Model*>(p)->utility_ablation=value;}
void utility_observe(void*p,int value){auto&m=*static_cast<Model*>(p);m.utility_observe=value;memset(m.utility_metrics,0,sizeof(m.utility_metrics));}
void utility_metrics(void*p,double*out){memcpy(out,static_cast<Model*>(p)->utility_metrics,sizeof(static_cast<Model*>(p)->utility_metrics));}
}
'''
(ROOT/'native.cpp').write_text(s)
print('generated',ROOT/'native.cpp')
