// Pending gradients are owned only while they contain work. This is transient
// runtime storage; the checkpoint adapters expand the historical dense arrays.
struct SparseCredit {
  using Rows=std::array<std::array<float,D>,LACE_DEPTH>;
  std::unique_ptr<Rows> rows;
  SparseCredit()=default;
  SparseCredit(const SparseCredit&other){if(other.rows)rows=std::make_unique<Rows>(*other.rows);}
  SparseCredit&operator=(const SparseCredit&other){if(this!=&other){
    if(other.rows){if(!rows)rows=std::make_unique<Rows>();*rows=*other.rows;}else rows.reset();
  }return *this;}
  SparseCredit(SparseCredit&&)=default;
  SparseCredit&operator=(SparseCredit&&)=default;
  float*operator[](size_t depth){if(!rows)rows=std::make_unique<Rows>();return (*rows)[depth].data();}
  const float*operator[](size_t depth)const{static const float zero[D]{};return rows?(*rows)[depth].data():zero;}
  void release_if_zero(){if(rows){for(const auto&row:*rows)for(float x:row){uint32_t bits;memcpy(&bits,&x,sizeof(bits));if(bits)return;}rows.reset();}}
};
template<class Target,class Source> inline void copy_credit(Target&to,const Source&from){
  for(int d=0;d<LACE_DEPTH;d++)memcpy(to[d],from[d],D*sizeof(float));
}
template<class Source> inline void copy_credit(SparseCredit&to,const Source&from){
  to.rows.reset();bool any=false;
  for(int d=0;d<LACE_DEPTH;d++)for(int j=0;j<D;j++){uint32_t bits;memcpy(&bits,&from[d][j],sizeof(bits));any|=bits!=0;}
  if(any)for(int d=0;d<LACE_DEPTH;d++)memcpy(to[d],from[d],D*sizeof(float));
}
