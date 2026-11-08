// Sequential MinKron final-state recurrence and its analytic backward.
// Equations/attribution: experiments/hat_minkron/kron.py and references/.
// No pairwise T x T tensors; saved state tape is linear in sequence length.
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <algorithm>
#include <cmath>
#include <vector>

using at::Tensor;

static void check_inputs(const Tensor& k, const Tensor& v, const Tensor& g,
                         const Tensor& beta, const Tensor& write, const Tensor& state) {
    TORCH_CHECK(k.device().is_cpu() && (k.scalar_type()==at::kFloat || k.scalar_type()==at::kDouble),
                "Fused MinKron requires CPU FP32/FP64");
    for (const auto& x : {k,v,g,beta,write,state})
        TORCH_CHECK(x.device().is_cpu() && x.scalar_type()==k.scalar_type() && x.is_contiguous(),
                    "Fused MinKron tensors must be contiguous with matching dtype");
    TORCH_CHECK(k.dim()==4 && v.dim()==4, "Directions must have shape [B,T,H,D]");
    const auto B=k.size(0), T=k.size(1), H=k.size(2), K=k.size(3), V=v.size(3);
    TORCH_CHECK(B>0 && T>0 && H>0 && K>0 && V>0 &&
                v.sizes()==at::IntArrayRef({B,T,H,V}) && g.sizes()==k.sizes() &&
                beta.sizes()==at::IntArrayRef({B,T,H}) && write.sizes()==beta.sizes() &&
                state.sizes()==at::IntArrayRef({B,H,K,V}), "Invalid fused MinKron dimensions");
}

std::vector<Tensor> final_forward(const Tensor& k, const Tensor& v, const Tensor& g,
        const Tensor& beta, const Tensor& write, const Tensor& previous, bool keep_tape) {
    check_inputs(k,v,g,beta,write,previous);
    const auto B=k.size(0), T=k.size(1), H=k.size(2), K=k.size(3), V=v.size(3), KV=K*V;
    auto final=at::empty_like(previous);
    // Final has its own compact storage, so detaching it cannot retain the tape.
    auto tape=keep_tape ? at::empty({B,H,T,K,V},k.options()) : at::empty({0},k.options());
    AT_DISPATCH_FLOATING_TYPES(k.scalar_type(), "minkron_final_forward", [&] {
        const auto A=k.data_ptr<scalar_t>(), W=v.data_ptr<scalar_t>(), G=g.data_ptr<scalar_t>();
        const auto E=beta.data_ptr<scalar_t>(), C=write.data_ptr<scalar_t>(), P=previous.data_ptr<scalar_t>();
        auto O=final.data_ptr<scalar_t>(), Tape=keep_tape ? tape.data_ptr<scalar_t>() : nullptr;
        at::parallel_for(0,B*H,16,[&](int64_t begin,int64_t end) {
            std::vector<scalar_t> s(KV), decay(K);
            for (int64_t n=begin;n<end;++n) {
                const auto b=n/H, h=n%H;
                std::copy_n(P+n*KV,KV,s.data());
                for (int64_t t=0;t<T;++t) {
                    const auto index=(b*T+t)*H+h;
                    const auto a=A+index*K, w=W+index*V;
                    if (keep_tape) std::copy_n(s.data(),KV,Tape+(n*T+t)*KV);
                    scalar_t norm=0;
                    for (int64_t j=0;j<V;++j) norm+=w[j]*w[j];
                    const scalar_t inv=1/std::max(norm,scalar_t(1e-18));
                    scalar_t score=0;
                    for (int64_t i=0;i<K;++i) {
                        decay[i]=std::exp(G[index*K+i]);
                        for (int64_t j=0;j<V;++j) {
                            s[i*V+j]*=decay[i];
                            score+=(a[i]*s[i*V+j])*w[j];
                        }
                    }
                    const scalar_t u=(C[index]*norm)*inv-(E[index]*inv)*score;
                    for (int64_t i=0;i<K;++i)
                        for (int64_t j=0;j<V;++j) s[i*V+j]+=(u*a[i])*w[j];
                }
                std::copy_n(s.data(),KV,O+n*KV);
            }
        });
    });
    return {final,tape};
}

std::vector<Tensor> final_backward(const Tensor& grad_final, const Tensor& k, const Tensor& v,
        const Tensor& g, const Tensor& beta, const Tensor& write, const Tensor& previous, const Tensor& tape) {
    check_inputs(k,v,g,beta,write,previous);
    const auto B=k.size(0), T=k.size(1), H=k.size(2), K=k.size(3), V=v.size(3), KV=K*V;
    for (const auto& x : {grad_final,tape})
        TORCH_CHECK(x.device().is_cpu() && x.scalar_type()==k.scalar_type() && x.is_contiguous(),
                    "Invalid fused MinKron gradient/tape storage");
    TORCH_CHECK(grad_final.sizes()==previous.sizes() && tape.sizes()==at::IntArrayRef({B,H,T,K,V}),
                "Invalid fused MinKron gradient/tape dimensions");
    auto dk=at::empty_like(k),dv=at::empty_like(v),dg=at::empty_like(g);
    auto db=at::empty_like(beta),dc=at::empty_like(write),ds=at::empty_like(previous);
    AT_DISPATCH_FLOATING_TYPES(k.scalar_type(), "minkron_final_backward", [&] {
        const auto A=k.data_ptr<scalar_t>(), W=v.data_ptr<scalar_t>(), G=g.data_ptr<scalar_t>();
        const auto E=beta.data_ptr<scalar_t>(), C=write.data_ptr<scalar_t>(), Tape=tape.data_ptr<scalar_t>();
        const auto DY=grad_final.data_ptr<scalar_t>();
        auto dA=dk.data_ptr<scalar_t>(),dW=dv.data_ptr<scalar_t>(),dG=dg.data_ptr<scalar_t>();
        auto dE=db.data_ptr<scalar_t>(),dC=dc.data_ptr<scalar_t>(),dS=ds.data_ptr<scalar_t>();
        at::parallel_for(0,B*H,16,[&](int64_t begin,int64_t end) {
            std::vector<scalar_t> delta(KV),decayed(KV),decay(K),weighted(K);
            for (int64_t n=begin;n<end;++n) {
                const auto b=n/H,h=n%H;
                std::copy_n(DY+n*KV,KV,delta.data());
                for (int64_t t=T-1;t>=0;--t) {
                    const auto index=(b*T+t)*H+h;
                    const auto a=A+index*K,w=W+index*V,p=Tape+(n*T+t)*KV;
                    scalar_t norm=0,score=0,du=0;
                    for (int64_t j=0;j<V;++j) norm+=w[j]*w[j];
                    const scalar_t inv=1/std::max(norm,scalar_t(1e-18));
                    for (int64_t i=0;i<K;++i) {
                        decay[i]=std::exp(G[index*K+i]);
                        scalar_t weighted_row=0;
                        for (int64_t j=0;j<V;++j) {
                            decayed[i*V+j]=p[i*V+j]*decay[i];
                            score+=(a[i]*decayed[i*V+j])*w[j];
                            weighted_row+=delta[i*V+j]*w[j];
                        }
                        weighted[i]=weighted_row;
                        du+=weighted_row*a[i];
                    }
                    const scalar_t u=(C[index]*norm)*inv-(E[index]*inv)*score;
                    const scalar_t dscore=-du*(E[index]*inv);
                    const scalar_t dinv=du*(C[index]*norm)+(-du*score)*E[index];
                    const scalar_t dnorm=(du*inv)*C[index]-(norm>=scalar_t(1e-18) ? dinv*(inv*inv) : scalar_t(0));
                    dE[index]=(-du*score)*inv;
                    dC[index]=(du*inv)*norm;
                    for (int64_t j=0;j<V;++j) {
                        scalar_t direct=0,projected=0;
                        for (int64_t i=0;i<K;++i) {
                            direct+=delta[i*V+j]*(u*a[i]);
                            projected+=dscore*(decayed[i*V+j]*a[i]);
                        }
                        dW[index*V+j]=direct+projected+2*w[j]*dnorm;
                    }
                    for (int64_t i=0;i<K;++i) {
                        scalar_t da=0,dd=0;
                        for (int64_t j=0;j<V;++j) {
                            const scalar_t projection=dscore*w[j];
                            da+=projection*decayed[i*V+j];
                            const scalar_t value=delta[i*V+j]+projection*a[i];
                            dd+=value*p[i*V+j];
                            delta[i*V+j]=value*decay[i];
                        }
                        dA[index*K+i]=weighted[i]*u+da;
                        dG[index*K+i]=dd*decay[i];
                    }
                }
                std::copy_n(delta.data(),KV,dS+n*KV);
            }
        });
    });
    return {dk,dv,dg,db,dc,ds};
}

TORCH_LIBRARY(shoebox_minkron_cpu, m) {
    m.def("final_forward(Tensor k, Tensor v, Tensor g, Tensor beta, Tensor write, Tensor previous, bool keep_tape) -> Tensor[]", &final_forward);
    m.def("final_backward(Tensor grad_final, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor write, Tensor previous, Tensor tape) -> Tensor[]", &final_backward);
}
