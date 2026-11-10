from pathlib import Path
ROOT=Path(__file__).resolve().parent
s=(ROOT.parent/'lace_native.cpp').read_text()
def replace(a,b):
    global s
    assert s.count(a)==1,(s.count(a),a[:100])
    s=s.replace(a,b)
replace('struct Model {',(ROOT/'state_types.inc').read_text()+'\nstruct Model {')
replace('  State s;', '  State s;\n  LearnedRuntime learned;\n  float *state_parameters=nullptr;\n  float scfg[STATE_CONFIG]{};')
replace('inline float norm(const float*x)', 'inline uint64_t feature_mask(const float*,int);\ninline bool valid_bottom_publication(Model&,int,LaceRecord*);\ninline void learn_bottom_state(Model&,int,LaceRecord*,const float*,float);\ninline void bottom_task_update(Model&,int,LaceRecord&,const float*,float);\ninline float norm(const float*x)')
replace('  int e=record?record->expert:s.chosen[r];','  if(!valid_bottom_publication(m,r,v)){m.learned.metrics[r][2]++;return;}\n  learn_bottom_state(m,r,v,g,rate);\n  int e=record?record->expert:s.chosen[r];')
replace('inline void route_task_update(Model&m,int r,LaceRecord&v,const float*g,float rate){','inline void route_task_update(Model&m,int r,LaceRecord&v,const float*g,float rate){\n  if(r<2&&m.scfg[8]>0){bottom_task_update(m,r,v,g,rate);return;}')
replace('extern "C" {',(ROOT/'state_functions.inc').read_text()+'\nextern "C" {')
replace('s.rng=7131587;s.recall_slot[0]=s.recall_slot[1]=-1;}','s.rng=7131587;s.recall_slot[0]=s.recall_slot[1]=-1;initialize_learned(m);}')
replace('void reset_context(void*p){auto&m=*static_cast<Model*>(p);m.laces=LaceState{};auto&s=m.s;', 'void reset_context(void*p){auto&m=*static_cast<Model*>(p);m.laces=LaceState{};auto&s=m.s;\n memset(m.learned.memory,0,sizeof(m.learned.memory));memset(m.learned.eligibility_a,0,sizeof(m.learned.eligibility_a));memset(m.learned.eligibility_b,0,sizeof(m.learned.eligibility_b));memset(m.learned.records,0,sizeof(m.learned.records));')
replace('      int best=0,runner=0;float score=-1e30f,second=-1e30f;', '''      if(r<2&&m.scfg[0]>0)for(int j=0;j<D;j++){
        float read=m.scfg[2]*std::tanh(state_parameter(m,r,0)[j]);
        if(read!=0)s.u[r][j]+=read*m.learned.memory[r][j];
      }
      int best=0,runner=0;float score=-1e30f,second=-1e30f;''')
replace('      bool task_route=m.lcfg[3]>0&&r>=int(m.lcfg[8]);','      bool task_route=(m.lcfg[3]>0&&r>=int(m.lcfg[8]))||(r<2&&m.scfg[8]>0);\n      if(r<2&&bottom_enabled(m))sparse_k=int(m.scfg[19]);')
replace('      if(task_route){float p=1.f/(1.f+std::exp(-(score-second)/m.lcfg[4]));gate=2*p;gate_derivative=2*p*(1-p)/m.lcfg[4];}', '''      if(task_route){
        if(r<2){float a=std::tanh((score-second)/m.scfg[10]);gate=1+m.scfg[9]*a;gate_derivative=m.scfg[9]*(1-a*a)/m.scfg[10];}
        else {float p=1.f/(1.f+std::exp(-(score-second)/m.lcfg[4]));gate=2*p;gate_derivative=2*p*(1-p)/m.lcfg[4];}
      }''')
replace('      if(tracing){\n        auto*previous=current_lace(m,r);','      publish_state(m,r,gate,runner);\n      if(tracing){\n        auto*previous=current_lace(m,r);')
replace('      if(mode&&mode!=4&&route>0){','      if(mode&&mode!=4&&route>0&&r<2&&(m.scfg[8]>0||m.scfg[11]>0))update_bottom_prototype(m,r,best,routing_mask,route);\n      else if(mode&&mode!=4&&route>0){')
s+='\n'+(ROOT/'state_api.inc').read_text()
(ROOT/'native.cpp').write_text(s)
print('Generated learned-state kernel')
