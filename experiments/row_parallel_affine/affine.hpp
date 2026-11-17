// A10 registration: reports/exp0305a9f1/A10_REGISTRATION.md.
// Four independent rows, coordinate order 0..63, bias AFTER the dot.
// Register-only 4x4 transpose; no coefficient packing buffer or persistent cache.
// Compile with -ffp-contract=off. No horizontal reduction or vector nonlinearity.
template<int Rows>
inline void a10_affine(const Affine* rows,const Vec& x,float* out){
    static_assert(Rows==8||Rows==16,"A10 only replaces existing 8/16-row projections");
    static_assert(D==64,"locked scalar coordinate count");
    for(int r=0;r<Rows;r+=4){
        __m128 sum=_mm_setzero_ps();
        for(int j=0;j<D;j+=4){
            __m128 c0=_mm_loadu_ps(rows[r].data()+j);
            __m128 c1=_mm_loadu_ps(rows[r+1].data()+j);
            __m128 c2=_mm_loadu_ps(rows[r+2].data()+j);
            __m128 c3=_mm_loadu_ps(rows[r+3].data()+j);
            _MM_TRANSPOSE4_PS(c0,c1,c2,c3);
            sum=_mm_add_ps(sum,_mm_mul_ps(c0,_mm_set1_ps(x[j])));
            sum=_mm_add_ps(sum,_mm_mul_ps(c1,_mm_set1_ps(x[j+1])));
            sum=_mm_add_ps(sum,_mm_mul_ps(c2,_mm_set1_ps(x[j+2])));
            sum=_mm_add_ps(sum,_mm_mul_ps(c3,_mm_set1_ps(x[j+3])));
        }
        const __m128 bias=_mm_set_ps(rows[r+3][D],rows[r+2][D],rows[r+1][D],rows[r][D]);
        _mm_storeu_ps(out+r,_mm_add_ps(bias,sum));
    }
}
