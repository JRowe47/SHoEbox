from pathlib import Path
ROOT=Path(__file__).resolve().parent;s=(ROOT/'router.cpp').read_text()
def replace(a,b):
 global s
 assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
replace('  alignas(64) float transposed_keys[N][D][E];','''  alignas(64) float transposed_keys[N][D][E];
  float cached_read[2][D]{};bool cached_read_valid[2]{};
  void prepare_read(int r){
    if(cached_read_valid[r])return;
    for(int j=0;j<D;j++)cached_read[r][j]=std::tanh(state_parameters[(r*3)*D+j]);
    cached_read_valid[r]=true;
  }''')
replace('float old=z.memory[r][j],a=std::tanh(read[j]);','float old=z.memory[r][j],a=m.cached_read[r][j];')
replace('  float step=rate*m.scfg[1]/norm(v->u);','  m.cached_read_valid[r]=false;\n  float step=rate*m.scfg[1]/norm(v->u);')
replace('      if(r<2&&m.scfg[0]>0)for(int j=0;j<D;j++){','      if(r<2&&m.scfg[0]>0)m.prepare_read(r);\n      if(r<2&&m.scfg[0]>0)for(int j=0;j<D;j++){')
replace('float read=m.scfg[2]*std::tanh(state_parameter(m,r,0)[j]);','float read=m.scfg[2]*m.cached_read[r][j];')
replace('double loss=0;m.refresh_keys();','double loss=0;m.refresh_keys();m.cached_read_valid[0]=m.cached_read_valid[1]=false;')
replace('      if(mode==2||mode==3||mode==6){','      if((mode==2||mode==3||mode==6)&&(!m.lcfg[0]||m.lcfg[6]>0)){')
replace('''      float fn=1.f;for(float v:features)fn+=v*v;
      float norms[2]{fn,fn};if(m.utility.config[1]>0){norms[0]=norm(features);norms[1]=norm(features+D);}''','''      float norms[2];
      if(m.utility.config[1]>0){norms[0]=norm(features);norms[1]=norm(features+D);}
      else {float fn=1.f;for(float v:features)fn+=v*v;norms[0]=norms[1]=fn;}''')
replace('bool active[N]{};float qgain[N]{},hcderiv[N][D]{},hcvalue[N][D]{};', 'bool active[N]{};float qgain[N]{},hcderiv[N][D],hcvalue[N][D];\n    if(mode==5){memset(hcderiv,0,sizeof(hcderiv));memset(hcvalue,0,sizeof(hcvalue));}')
(ROOT/'cached.cpp').write_text(s)
