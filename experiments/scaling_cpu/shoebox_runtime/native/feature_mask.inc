// Width-aware feature membership. The 64D specialization retains its ABI and
// optimized selection kernel so existing checkpoints can be replayed exactly.
#if SHOEBOX_WIDTH == 64
using FeatureMask = uint64_t;
inline bool mask_has(FeatureMask m,int j){return (m&(uint64_t(1)<<j))!=0;}
inline bool mask_any(FeatureMask m){return m!=0;}
inline int mask_pop(FeatureMask &m){int j=__builtin_ctzll(m);m&=m-1;return j;}
inline void mask_set(FeatureMask &m,int j){m|=uint64_t(1)<<j;}
inline int mask_overlap(FeatureMask a,FeatureMask b){return __builtin_popcountll(a&b);}
inline uint16_t mask_chunk(FeatureMask m,int j){return uint16_t(m>>j);}
inline uint64_t mask_word(FeatureMask m,int){return m;}
#else
struct FeatureMask { uint64_t words[D/64]{}; };
inline bool mask_has(const FeatureMask &m,int j){return (m.words[j/64]&(uint64_t(1)<<(j%64)))!=0;}
inline bool mask_any(const FeatureMask &m){for(auto w:m.words)if(w)return true;return false;}
inline int mask_pop(FeatureMask &m){for(int b=0;b<D/64;b++)if(m.words[b]){int j=__builtin_ctzll(m.words[b]);m.words[b]&=m.words[b]-1;return 64*b+j;}return -1;}
inline void mask_set(FeatureMask &m,int j){m.words[j/64]|=uint64_t(1)<<(j%64);}
inline int mask_overlap(const FeatureMask &a,const FeatureMask &b){int n=0;for(int w=0;w<D/64;w++)n+=__builtin_popcountll(a.words[w]&b.words[w]);return n;}
inline uint16_t mask_chunk(const FeatureMask &m,int j){return uint16_t(m.words[j/64]>>(j%64));}
inline uint64_t mask_word(const FeatureMask &m,int b){return m.words[b];}
#endif
constexpr int MASK_WORDS=D/64;
