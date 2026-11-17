"""Independent bitwise scalar reference, then unchanged native A9 controls."""
import argparse,os,subprocess
from pathlib import Path
from . import backend
from experiments.selective_record_key_credit import check_native as old

AFFINE=r'''
#include <iostream>
using namespace shoe;
static uint64_t comparisons=0,rows_executed=0;
// Independent reference: volatile products/adds enforce scalar float32 order.
float reference(const Affine& row,const Vec& x){
    volatile float sum=0;
    for(int j=0;j<64;++j){volatile float product=row[j]*x[j];sum=sum+product;}
    volatile float z=row[64]+sum;return z;
}
template<int R> void check(const Affine* rows,const Vec& x){
    std::array<float,R> out;a10_affine<R>(rows,x,out.data());rows_executed+=R;
    for(int r=0;r<R;++r){float expected=reference(rows[r],x),legacy=affine(rows[r],x);
        if(std::memcmp(&expected,&out[r],4)||std::memcmp(&legacy,&out[r],4))
            throw std::runtime_error("A10 bitwise affine mismatch "+std::to_string(comparisons));
        ++comparisons;
        float y=std::tanh(out[r]),z=std::tanh(expected);
        if(std::memcmp(&y,&z,4))throw std::runtime_error("scalar tanh mismatch");++comparisons;
    }
}
int main(){
    std::mt19937 rng(10107);std::uniform_real_distribution<float> u(-2.f,2.f);
    std::array<Affine,80> rows{};Vec x{};
    // All 5 key phases and 4 value phases; zero, signed zero, bias, cancellation,
    // subnormal, finite extremes without overflow, and random finite inputs.
    for(int trial=0;trial<320;++trial){
        for(auto& row:rows)for(auto& v:row)v=u(rng);
        for(auto& v:x)v=u(rng);
        if(trial<8){
            for(auto& row:rows)for(auto& v:row)v=(trial%2)?-0.f:0.f;
            for(auto& v:x)v=(trial%2)?-0.f:0.f;
            if(trial>=2)for(int r=0;r<80;++r)rows[r][64]=(r%2)?-0.f:float(r-40);
            if(trial>=4)for(int r=0;r<80;++r)for(int j=0;j<64;++j){rows[r][j]=(j%2)?1e20f:-1e20f;x[j]=1e-20f;}
            if(trial==6){for(auto& row:rows)for(auto& v:row)v=std::numeric_limits<float>::denorm_min();x.fill(1.f);}
            if(trial==7){for(auto& row:rows){row.fill(std::numeric_limits<float>::max()/256.f);row[64]=-1.f;}x.fill(.25f);}
        }
        for(int phase=0;phase<5;++phase)check<16>(rows.data()+phase*16,x);
        for(int phase=0;phase<4;++phase)check<8>(rows.data()+phase*8,x);
    }
    std::cout<<"A10 bitwise comparisons "<<comparisons<<"; row-parallel rows "<<rows_executed
             <<"; independent scalar rows "<<rows_executed<<"; original affine rows "<<rows_executed
             <<"; model input/update bytes 0; persistent/scratch coefficient packing bytes 0\n";
}
'''

def main():
    p=argparse.ArgumentParser();p.add_argument('execution',choices=['scalar','row_parallel']);p.add_argument('work',type=Path);a=p.parse_args()
    a.work.mkdir(parents=True,exist_ok=False)
    if a.execution=='row_parallel':
        head=backend.header();(a.work/'readout.hpp').write_text(head)
        cpp=a.work/'affine.cpp';cpp.write_text(backend.source()+AFFINE)
        for tag,flags in [('optimized',['-O3','-march=native']),('sanitized',['-O1','-g','-fno-omit-frame-pointer','-fsanitize=address,undefined'])]:
            exe=a.work/tag
            subprocess.run(['g++','-std=c++17','-ffp-contract=off',*flags,'-I'+str(a.work),'-I'+str(backend.ROOT/'multiscale'),str(cpp),'-o',str(exe)],check=True)
            subprocess.run([str(exe)],check=True,env={**os.environ,'ASAN_OPTIONS':'detect_leaks=0:halt_on_error=1','UBSAN_OPTIONS':'halt_on_error=1:print_stacktrace=1'})
        old.header=lambda policy:head if policy=='a9' else backend.scalar.header(policy)
        # Keep test-only A5/A8 scalar local references and all seven inherited
        # finite-difference, lifecycle, causality, exhaustive and mask suites.
    old.main()
if __name__=='__main__':main()
