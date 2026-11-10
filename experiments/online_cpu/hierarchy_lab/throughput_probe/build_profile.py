from pathlib import Path
import argparse
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--source',default='baseline');a=p.parse_args()
s=(ROOT/(a.source+'.cpp')).read_text()
s=s.replace('#include <immintrin.h>','#include <immintrin.h>\n#include "profiler.inc"')
def replace(a,b):
 global s
 assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
for name,category in [('expert_update','P_EXPERT_UPDATE'),('feature_mask','P_FEATURE_MASK'),('route_task_update','P_ROUTE_UPDATE'),('distribute_credit','P_CREDIT'),('flush_credit','P_FLUSH'),('materialize_key','P_MATERIALIZE'),('publish_state','P_PUBLISH_STATE'),('learn_bottom_state','P_STATE_LEARN'),('bottom_task_update','P_BOTTOM_ROUTE'),('update_bottom_prototype','P_PROTOTYPE')]:
 import re
 pat=r'(inline [^\n;]*\b'+name+r'\([^\n]*\)\{)'
 s,n=re.subn(pat,lambda m:m[0]+f'\n  ProfileScope scope({category});',s);assert n==1,(name,n)
replace('    s.tick++;float snapshot', '    profile_sample(m.s.tick+1);ProfileScope phase(P_BYTE);\n    s.tick++;float snapshot')
replace('      LaceParent parents[LACE_LINKS];int nparents=0;', '      phase.stage(P_INPUT);\n      LaceParent parents[LACE_LINKS];int nparents=0;')
replace('      // Sparse content lookup:', '      phase.stage(P_MEMORY);\n      // Sparse content lookup:')
replace('      // A future observed bottom-up bundle', '      phase.stage(P_LOCAL_TEACHER);\n      // A future observed bottom-up bundle')
replace('\n      float alpha=r==1?.125f:.5f;', '\n      phase.stage(P_STATE_INPUT);\n      float alpha=r==1?.125f:.5f;')
replace('      int best=0,runner=0;float score', '      phase.stage(P_ROUTING);\n      int best=0,runner=0;float score')
replace('      float residual[D];', '      phase.stage(P_FORWARD);\n      float residual[D];')
replace('      publish_state(m,r,gate,runner);', '      phase.stage(P_PUBLICATION);\n      publish_state(m,r,gate,runner);')
replace('      if(mode&&mode!=4&&route>0&&r<2', '      phase.stage(P_PROTOTYPE);\n      if(mode&&mode!=4&&route>0&&r<2')
replace('      if((mode==1||mode==3||mode==6)&&(mode!=6', '      phase.stage(P_PREDICTOR);\n      if((mode==1||mode==3||mode==6)&&(mode!=6')
replace('    float features[2*D],logits', '    phase.stage(P_HEAD);\n    float features[2*D],logits')
replace('    s.correct+=guess==y[t];', '    phase.stage(P_TASK);\n    s.correct+=guess==y[t];')
if '      float norms[2];' in s:replace('      float norms[2];','      phase.stage(P_READOUT);\n      float norms[2];')
else:replace('      float fn=1.f;for(float v:features)', '      phase.stage(P_READOUT);\n      float fn=1.f;for(float v:features)')
(ROOT/('profile.cpp' if a.source=='baseline' else 'profile_'+a.source+'.cpp')).write_text(s)
