extern "C" {
int contribution_configure(void*p,int kind,int region){
  auto&m=*static_cast<Model*>(p);
  if(!m.composition.enabled||kind<CHECK_NONE||kind>CHECK_CONTENT_EXPERT||
     (kind==CHECK_NONE?region!=-1:region<0||region>=N)||
     ((kind==CHECK_SEMANTIC||kind==CHECK_CONTENT_EXPERT)&&region<2)||m.composition.stream.byte_position||
     m.s.ablation||m.utility_ablation)return 0;
#ifndef SHOEBOX_LEARNED_MESSAGES
  if(kind>CHECK_EXPERT)return 0;
#endif
#ifndef SHOEBOX_LEARNED_COMPLETION
  if(kind==CHECK_CONTENT_EXPERT)return 0;
#endif
  for(int r=0;r<N;r++)if(m.contribution_counts[r][0])return 0;
  m.contribution_kind=kind;m.contribution_region=region;return 1;
}
void contribution_status(void*p,int*out){
  auto&m=*static_cast<Model*>(p);out[0]=m.contribution_kind;out[1]=m.contribution_region;
}
void contribution_metrics(void*p,uint64_t*out,uint64_t*hashes){
  auto&m=*static_cast<Model*>(p);
  memcpy(out,m.contribution_counts,sizeof(m.contribution_counts));
  memcpy(hashes,m.contribution_hashes,sizeof(m.contribution_hashes));
}
}
