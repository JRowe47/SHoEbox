from pathlib import Path
ROOT=Path(__file__).resolve().parent
s=(ROOT/'mask_radix.cpp').read_text()
def replace(a,b):
 global s
 assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
replace('  Model(float *p,','''  // Derived layout only: no new model parameters or checkpoint state.
  alignas(64) float transposed_keys[N][D][E];
  void refresh_key(int r,int e){for(int j=0;j<D;j++)transposed_keys[r][j][e]=keys[(r*E+e)*D+j];}
  void refresh_keys(){for(int r=0;r<N;r++)for(int e=0;e<E;e++)refresh_key(r,e);}
  Model(float *p,''')
replace('  m.laces.metrics[r][4]++;','  m.refresh_key(r,v.expert);m.refresh_key(r,v.runner);\n  m.laces.metrics[r][4]++;')
replace('  n=1/std::sqrt(n);for(int j=0;j<D;j++)k[j]*=n;\n}', '  n=1/std::sqrt(n);for(int j=0;j<D;j++)k[j]*=n;\n  m.refresh_key(r,e);\n}')
replace('kn=1.f/std::sqrt(kn+1e-9f);for(int j=0;j<D;j++)k[j]*=kn;', 'kn=1.f/std::sqrt(kn+1e-9f);for(int j=0;j<D;j++)k[j]*=kn;m.refresh_key(r,best);')
replace('  Model&m=*static_cast<Model*>(p);State&s=m.s;double loss=0;', '  Model&m=*static_cast<Model*>(p);State&s=m.s;double loss=0;m.refresh_keys();')
old='''      for(int e=0;e<E;e++){
        float v=0;
        if(sparse_k>=D)for(int j=0;j<D;j++)v+=routing[j]*m.keys[(r*E+e)*D+j];
        else for(int a=0;a<nindices;a++){int j=indices[a];v+=routing[j]*m.keys[(r*E+e)*D+j];}'''
replace(old,'''      float scores[E];router_scores(m,r,routing,indices,nindices,scores);
      for(int e=0;e<E;e++){
        float v=scores[e];''')
helper='''
// Vectorize over independent experts, retaining coordinate accumulation order.
// The reference reduction rounds each multiply before its sequential additions.
__attribute__((optimize("fp-contract=off")))
static void router_scores(Model&m,int r,const float*input,const int*indices,int n,float*out){
#ifdef __AVX512F__
  for(int e=0;e<E;e+=16){
    __mmask16 mask=E-e>=16?0xffff:__mmask16((1u<<(E-e))-1);__m512 scores=_mm512_setzero_ps();
    for(int a=0;a<n;a++){int j=indices[a];
      __m512 key=_mm512_maskz_loadu_ps(mask,m.transposed_keys[r][j]+e);
      scores=_mm512_add_ps(scores,_mm512_mul_ps(_mm512_set1_ps(input[j]),key));
    }
    _mm512_mask_storeu_ps(out+e,mask,scores);
  }
#elif defined(__AVX2__)
  for(int e=0;e<E;e+=8){
    alignas(32) int bits[8];for(int b=0;b<8;b++)bits[b]=e+b<E?-1:0;
    __m256i mask=_mm256_load_si256(reinterpret_cast<const __m256i*>(bits));__m256 scores=_mm256_setzero_ps();
    for(int a=0;a<n;a++){int j=indices[a];__m256 key=_mm256_maskload_ps(m.transposed_keys[r][j]+e,mask);
      scores=_mm256_add_ps(scores,_mm256_mul_ps(_mm256_set1_ps(input[j]),key));
    }
    _mm256_maskstore_ps(out+e,mask,scores);
  }
#else
  for(int e=0;e<E;e++){float v=0;for(int a=0;a<n;a++){int j=indices[a];v+=input[j]*m.keys[(r*E+e)*D+j];}out[e]=v;}
#endif
}
'''
at=s.index('extern "C" {');s=s[:at]+helper+s[at:]
(ROOT/'router.cpp').write_text(s)
