extern "C" {
void configure_state(void*p,float*parameters,float*config){auto&m=*static_cast<Model*>(p);m.state_parameters=parameters;memcpy(m.scfg,config,STATE_CONFIG*sizeof(float));initialize_learned(m);}
size_t learned_state_size(){return sizeof(LearnedRuntime);}
void export_learned(void*p,void*out){memcpy(out,&static_cast<Model*>(p)->learned,sizeof(LearnedRuntime));}
void import_learned(void*p,void*in){memcpy(&static_cast<Model*>(p)->learned,in,sizeof(LearnedRuntime));}
void learned_metrics(void*p,double*out){memcpy(out,static_cast<Model*>(p)->learned.metrics,24*sizeof(double));}
void learned_snapshot(void*p,float*out){auto&m=*static_cast<Model*>(p);memcpy(out,m.learned.memory,2*D*sizeof(float));memcpy(out+2*D,m.s.u,2*D*sizeof(float));}
int force_revival(void*p,int r,int dead,int donor){if(r<0||r>=2||dead<0||dead>=E||donor<0||donor>=E)return 0;return revive_expert(*static_cast<Model*>(p),r,dead,donor);}
}
