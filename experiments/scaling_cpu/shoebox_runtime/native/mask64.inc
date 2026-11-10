// Exact top-k absolute features, with lower-coordinate tie breaking.
// The fallback preserves the original comparator for non-AVX512 builds and NaNs.
inline uint64_t original_feature_mask(const float*u,int k){
  if(k>=D)return ~uint64_t(0);
  int ids[D];for(int j=0;j<D;j++)ids[j]=j;
  std::nth_element(ids,ids+k,ids+D,[&](int a,int b){float x=std::abs(u[a]),y=std::abs(u[b]);return x==y?a<b:x>y;});
  uint64_t mask=0;for(int j=0;j<k;j++)mask|=uint64_t(1)<<ids[j];return mask;
}
#ifdef __AVX512F__
inline uint64_t rank_feature_mask(const float*u,int k){
  if(k>=D)return ~uint64_t(0);if(k<=0)return 0;
  __m512 a[4];for(int b=0;b<4;b++)a[b]=_mm512_castsi512_ps(_mm512_and_epi32(_mm512_loadu_si512(u+16*b),_mm512_set1_epi32(0x7fffffff)));
  uint64_t result=0;
  for(int i=0;i<64;i++){
    __m512 v=_mm512_set1_ps(std::abs(u[i]));uint64_t gt=0,eq=0;
    for(int b=0;b<4;b++){gt|=uint64_t(_mm512_cmp_ps_mask(a[b],v,_CMP_GT_OQ))<<(16*b);eq|=uint64_t(_mm512_cmp_ps_mask(a[b],v,_CMP_EQ_OQ))<<(16*b);}
    int rank=__builtin_popcountll(gt)+__builtin_popcountll(eq&((uint64_t(1)<<i)-1));
    if(rank<k)result|=uint64_t(1)<<i;
  }return result;
}
inline uint64_t radix_feature_mask(const float*u,int k){
  if(k>=D)return ~uint64_t(0);if(k<=0)return 0;
  __m512i a[4];for(int b=0;b<4;b++){
    a[b]=_mm512_and_epi32(_mm512_loadu_si512(u+16*b),_mm512_set1_epi32(0x7fffffff));
    if(_mm512_cmp_epu32_mask(a[b],_mm512_set1_epi32(0x7f800000),_MM_CMPINT_GT))return original_feature_mask(u,k);
  }
  uint64_t result=0,remaining=~uint64_t(0);int count=64,need=k;
  // Positive IEEE-754 float bit order agrees with magnitude order.
  for(int shift=30;shift>=0;shift--){
    __m512i bit=_mm512_set1_epi32(uint32_t(1)<<shift);uint64_t high=0;
    for(int b=0;b<4;b++)high|=uint64_t(_mm512_test_epi32_mask(a[b],bit))<<(16*b);
    high&=remaining;int n=__builtin_popcountll(high);
    if(n>=need){remaining=high;count=n;}
    else {result|=high;remaining^=high;count-=n;need-=n;}
    if(count==need)return result|remaining;
  }
  // Equal magnitudes at the cutoff: choose the lowest coordinate indices.
  while(need--){uint64_t bit=remaining&-remaining;result|=bit;remaining^=bit;}
  return result;
}
#elif defined(__AVX2__)
inline uint64_t rank_feature_mask(const float*u,int k){return original_feature_mask(u,k);}
inline uint64_t radix_feature_mask(const float*u,int k){
  if(k>=D)return ~uint64_t(0);if(k<=0)return 0;
  __m256i a[8];for(int b=0;b<8;b++){
    a[b]=_mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(u+8*b)),_mm256_set1_epi32(0x7fffffff));
    if(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(a[b],_mm256_set1_epi32(0x7f800000)))))return original_feature_mask(u,k);
  }
  uint64_t result=0,remaining=~uint64_t(0);int count=64,need=k;
  for(int shift=30;shift>=0;shift--){
    __m256i bit=_mm256_set1_epi32(uint32_t(1)<<shift);uint64_t high=0;
    for(int b=0;b<8;b++){
      int zero=_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(_mm256_and_si256(a[b],bit),_mm256_setzero_si256())));
      high|=uint64_t((~zero)&255)<<(8*b);
    }
    high&=remaining;int n=__builtin_popcountll(high);
    if(n>=need){remaining=high;count=n;}else{result|=high;remaining^=high;count-=n;need-=n;}
    if(count==need)return result|remaining;
  }
  while(need--){uint64_t bit=remaining&-remaining;result|=bit;remaining^=bit;}return result;
}
#else
inline uint64_t rank_feature_mask(const float*u,int k){return original_feature_mask(u,k);}
inline uint64_t radix_feature_mask(const float*u,int k){return original_feature_mask(u,k);}
#endif
