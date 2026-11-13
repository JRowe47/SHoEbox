// A full-ID anchor makes region, roles, order, presence and geometry visible.
// IDs remain exact structural keys; floating vectors are similarity features.
inline void semantic_encode(Model&m,int r,ContentEvent&e){
  uint64_t bits[2]{content_mix(e.ref.id.a^content_mix(e.ref.id.b)),content_mix(e.ref.id.b+content_mix(e.ref.id.a))};
  for(int j=0;j<D;j++){
    unsigned v=unsigned((bits[j/32]>>(2*(j%32)))&3);
    e.structure[j]+=.125f*(float(v)-1.5f)/1.5f;
  }
  for(int k=0;k<8;k++){
    float value=0;
    for(int j=0;j<D;j++)value+=((__builtin_parity(unsigned(j)&unsigned(k+1)))?-1.f:1.f)*e.structure[j];
    e.symbol_features[k]=std::tanh(value*.125f);
  }
  for(int j=0;j<D;j++){
    float value=0;for(int k=0;k<8;k++)value+=m.composition.symbols[r-2][j][k]*e.symbol_features[k];
    float residual=std::tanh(value);
#ifdef SHOEBOX_CONTRIBUTION_CHECKS
    if(contribution_selected(m,CHECK_SEMANTIC,r))residual=0;
#endif
    e.code[j]=e.structure[j]+.25f*residual;
    e.symbol_derivative[j]=.25f*(1-residual*residual);
  }
#ifdef SHOEBOX_CONTRIBUTION_CHECKS
  contribution_apply(m,CHECK_SEMANTIC,r);
#endif
}
