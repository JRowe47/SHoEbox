from pathlib import Path
ROOT=Path(__file__).resolve().parent;s=(ROOT/'baseline.cpp').read_text()
start=s.index('inline uint64_t feature_mask(const float*u,int k){');end=s.index('inline LaceRecord* current_lace',start)
for method in ['rank','radix']:
 new=s[:start]+'#include "mask_kernels.inc"\ninline uint64_t feature_mask(const float*u,int k){return '+method+'_feature_mask(u,k);}\n'+s[end:]
 new+='\nextern "C" void check_masks(const float*input,uint64_t*out,int n,int k,int method){for(int i=0;i<n;i++)out[i]=method==0?original_feature_mask(input+i*64,k):method==1?rank_feature_mask(input+i*64,k):radix_feature_mask(input+i*64,k);}\n'
 (ROOT/f'mask_{method}.cpp').write_text(new)
