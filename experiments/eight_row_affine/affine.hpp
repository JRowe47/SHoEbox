// Fixed A11 registration: reports/exp0305a10/A11_REGISTRATION.md.
// Each lane is one row. Coordinates 0..63, separate multiply/add, bias last.
// Only an 8x8 register transpose; no persistent packing or vector tanh.
#if !defined(__AVX2__)
#error A11 requires AVX2; no substitute kernel is permitted
#endif
template<int Rows>
inline void a11_affine(const Affine* rows,const Vec& x,float* out){
    static_assert(Rows==8||Rows==16,"only existing projections");
    static_assert(D==64,"locked coordinate count");
    for(int r=0;r<Rows;r+=8){
        __m256 sum=_mm256_setzero_ps();
        for(int j=0;j<D;j+=8){
            __m256 c0=_mm256_loadu_ps(rows[r+0].data()+j);
            __m256 c1=_mm256_loadu_ps(rows[r+1].data()+j);
            __m256 c2=_mm256_loadu_ps(rows[r+2].data()+j);
            __m256 c3=_mm256_loadu_ps(rows[r+3].data()+j);
            __m256 c4=_mm256_loadu_ps(rows[r+4].data()+j);
            __m256 c5=_mm256_loadu_ps(rows[r+5].data()+j);
            __m256 c6=_mm256_loadu_ps(rows[r+6].data()+j);
            __m256 c7=_mm256_loadu_ps(rows[r+7].data()+j);
            __m256 t0=_mm256_unpacklo_ps(c0,c1),t1=_mm256_unpackhi_ps(c0,c1);
            __m256 t2=_mm256_unpacklo_ps(c2,c3),t3=_mm256_unpackhi_ps(c2,c3);
            __m256 t4=_mm256_unpacklo_ps(c4,c5),t5=_mm256_unpackhi_ps(c4,c5);
            __m256 t6=_mm256_unpacklo_ps(c6,c7),t7=_mm256_unpackhi_ps(c6,c7);
            c0=_mm256_shuffle_ps(t0,t2,0x44);c1=_mm256_shuffle_ps(t0,t2,0xee);
            c2=_mm256_shuffle_ps(t1,t3,0x44);c3=_mm256_shuffle_ps(t1,t3,0xee);
            c4=_mm256_shuffle_ps(t4,t6,0x44);c5=_mm256_shuffle_ps(t4,t6,0xee);
            c6=_mm256_shuffle_ps(t5,t7,0x44);c7=_mm256_shuffle_ps(t5,t7,0xee);
            t0=_mm256_permute2f128_ps(c0,c4,0x20);t1=_mm256_permute2f128_ps(c1,c5,0x20);
            t2=_mm256_permute2f128_ps(c2,c6,0x20);t3=_mm256_permute2f128_ps(c3,c7,0x20);
            t4=_mm256_permute2f128_ps(c0,c4,0x31);t5=_mm256_permute2f128_ps(c1,c5,0x31);
            t6=_mm256_permute2f128_ps(c2,c6,0x31);t7=_mm256_permute2f128_ps(c3,c7,0x31);
            sum=_mm256_add_ps(sum,_mm256_mul_ps(t0,_mm256_set1_ps(x[j+0])));
            sum=_mm256_add_ps(sum,_mm256_mul_ps(t1,_mm256_set1_ps(x[j+1])));
            sum=_mm256_add_ps(sum,_mm256_mul_ps(t2,_mm256_set1_ps(x[j+2])));
            sum=_mm256_add_ps(sum,_mm256_mul_ps(t3,_mm256_set1_ps(x[j+3])));
            sum=_mm256_add_ps(sum,_mm256_mul_ps(t4,_mm256_set1_ps(x[j+4])));
            sum=_mm256_add_ps(sum,_mm256_mul_ps(t5,_mm256_set1_ps(x[j+5])));
            sum=_mm256_add_ps(sum,_mm256_mul_ps(t6,_mm256_set1_ps(x[j+6])));
            sum=_mm256_add_ps(sum,_mm256_mul_ps(t7,_mm256_set1_ps(x[j+7])));
        }
        const __m256 bias=_mm256_set_ps(rows[r+7][D],rows[r+6][D],rows[r+5][D],rows[r+4][D],rows[r+3][D],rows[r+2][D],rows[r+1][D],rows[r][D]);
        _mm256_storeu_ps(out+r,_mm256_add_ps(bias,sum));
    }
}
