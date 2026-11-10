#if SHOEBOX_WIDTH == 64
#include "mask64.inc"
inline FeatureMask select_features(const float*u,int k){return radix_feature_mask(u,k);}
#else
inline FeatureMask select_features(const float*u,int k){
  FeatureMask out{};
  if(k>=D){for(auto &w:out.words)w=~uint64_t(0);return out;}
  if(k<=0)return out;
  int ids[D];for(int j=0;j<D;j++)ids[j]=j;
  std::nth_element(ids,ids+k,ids+D,[&](int a,int b){
    float x=std::abs(u[a]),y=std::abs(u[b]);
    // Inputs are finite in training. Keep a total order for diagnostic calls.
    if(std::isnan(x))return false;
    if(std::isnan(y))return true;
    return x==y?a<b:x>y;
  });
  for(int j=0;j<k;j++)mask_set(out,ids[j]);
  return out;
}
#endif
