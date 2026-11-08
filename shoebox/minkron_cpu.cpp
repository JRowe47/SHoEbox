// Sequential MinKron final-state recurrence and its analytic backward.
// Equations/attribution: experiments/hat_minkron/kron.py and references/.
// No pairwise T x T tensors; saved state tape is linear in sequence length.
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <algorithm>
#include <cmath>
#include <vector>
#include <type_traits>
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

using at::Tensor;

template<typename T> static T scaled_dot(const T* a, const T* b, int64_t size, T scale) {
    int64_t i=0;
    T result=0;
#if defined(__AVX512F__)
    if constexpr (std::is_same_v<T,float>) {
        __m512 accum=_mm512_setzero_ps(), factor=_mm512_set1_ps(scale);
        for (;i+16<=size;i+=16)
            accum=_mm512_add_ps(accum,_mm512_mul_ps(_mm512_mul_ps(_mm512_loadu_ps(a+i),factor),_mm512_loadu_ps(b+i)));
        result=_mm512_reduce_add_ps(accum);
    } else {
        __m512d accum=_mm512_setzero_pd(), factor=_mm512_set1_pd(scale);
        for (;i+8<=size;i+=8)
            accum=_mm512_add_pd(accum,_mm512_mul_pd(_mm512_mul_pd(_mm512_loadu_pd(a+i),factor),_mm512_loadu_pd(b+i)));
        result=_mm512_reduce_add_pd(accum);
    }
#endif
    for (;i<size;++i) result+=(a[i]*scale)*b[i];
    return result;
}

std::string cpu_isa() {
#if defined(__AVX512F__)
    return "AVX512";
#else
    return "portable native";
#endif
}

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
        const Tensor& beta, const Tensor& write, const Tensor& previous, bool keep_tape,
        const std::optional<Tensor>& lengths) {
    check_inputs(k,v,g,beta,write,previous);
    const auto B=k.size(0), T=k.size(1), H=k.size(2), K=k.size(3), V=v.size(3), KV=K*V;
    const int64_t* lengths_ptr=nullptr;
    if (lengths.has_value()) {
        TORCH_CHECK(lengths->device().is_cpu() && lengths->scalar_type()==at::kLong &&
                    lengths->is_contiguous() && lengths->sizes()==at::IntArrayRef({B}), "Invalid sequence lengths");
        lengths_ptr=lengths->data_ptr<int64_t>();
        for (int64_t b=0;b<B;++b) TORCH_CHECK(lengths_ptr[b]>0 && lengths_ptr[b]<=T, "Sequence length outside input");
    }
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
                const auto count=lengths_ptr ? lengths_ptr[b] : T;
                for (int64_t t=0;t<count;++t) {
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
        const Tensor& g, const Tensor& beta, const Tensor& write, const Tensor& previous, const Tensor& tape,
        const std::optional<Tensor>& lengths) {
    check_inputs(k,v,g,beta,write,previous);
    const auto B=k.size(0), T=k.size(1), H=k.size(2), K=k.size(3), V=v.size(3), KV=K*V;
    for (const auto& x : {grad_final,tape})
        TORCH_CHECK(x.device().is_cpu() && x.scalar_type()==k.scalar_type() && x.is_contiguous(),
                    "Invalid fused MinKron gradient/tape storage");
    TORCH_CHECK(grad_final.sizes()==previous.sizes() && tape.sizes()==at::IntArrayRef({B,H,T,K,V}),
                "Invalid fused MinKron gradient/tape dimensions");
    const int64_t* lengths_ptr=nullptr;
    if (lengths.has_value()) {
        TORCH_CHECK(lengths->device().is_cpu() && lengths->scalar_type()==at::kLong &&
                    lengths->is_contiguous() && lengths->sizes()==at::IntArrayRef({B}), "Invalid sequence lengths");
        lengths_ptr=lengths->data_ptr<int64_t>();
        for (int64_t b=0;b<B;++b) TORCH_CHECK(lengths_ptr[b]>0 && lengths_ptr[b]<=T, "Sequence length outside input");
    }
    auto make_grad=[&](const Tensor& t) { return lengths_ptr ? at::zeros_like(t) : at::empty_like(t); };
    auto dk=make_grad(k),dv=make_grad(v),dg=make_grad(g);
    auto db=make_grad(beta),dc=make_grad(write),ds=at::empty_like(previous);
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
                const auto count=lengths_ptr ? lengths_ptr[b] : T;
                for (int64_t t=count-1;t>=0;--t) {
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

// Read expert weights in place. Compact derivatives use one row per selection;
// duplicate ids are summed once per bounded group of bank uses.
static void check_experts(const Tensor& x, const Tensor& up, const Tensor& down,
                          const Tensor& ids, const Tensor& gates) {
    TORCH_CHECK(x.device().is_cpu() && (x.scalar_type()==at::kFloat || x.scalar_type()==at::kDouble),
                "Indexed experts require CPU FP32/FP64");
    for (const auto& t : {x,up,down,gates})
        TORCH_CHECK(t.device().is_cpu() && t.scalar_type()==x.scalar_type() && t.is_contiguous(),
                    "Expert tensors must be contiguous with matching dtype");
    TORCH_CHECK(ids.device().is_cpu() && ids.scalar_type()==at::kLong && ids.is_contiguous(),
                "Expert ids must be contiguous CPU int64");
    TORCH_CHECK(x.dim()==2 && up.dim()==3 && ids.dim()==2, "Invalid expert dimensions");
    TORCH_CHECK(x.size(0)>0 && x.size(1)>0 && up.size(0)>0 && up.size(1)>0 && ids.size(1)>0 &&
                up.size(2)==x.size(1) && down.sizes()==up.sizes() && ids.size(0)==x.size(0) &&
                gates.sizes()==ids.sizes(), "Invalid expert dimensions");
    const auto I=ids.data_ptr<int64_t>();
    for (int64_t i=0;i<ids.numel();++i)
        TORCH_CHECK(I[i]>=0 && I[i]<up.size(0), "Expert row index out of range");
}

std::vector<Tensor> experts_forward(const Tensor& x, const Tensor& up, const Tensor& down,
                                   const Tensor& ids, const Tensor& gates, bool keep_tape) {
    check_experts(x,up,down,ids,gates);
    const auto N=x.size(0), D=x.size(1), H=up.size(1), K=ids.size(1);
    auto result=at::zeros_like(x);
    auto pre=keep_tape ? at::empty({N,K,H},x.options()) : at::empty({0},x.options());
    const auto I=ids.data_ptr<int64_t>();
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "indexed_experts_forward", [&] {
        const auto X=x.data_ptr<scalar_t>(), U=up.data_ptr<scalar_t>(), W=down.data_ptr<scalar_t>();
        const auto G=gates.data_ptr<scalar_t>();
        auto Y=result.data_ptr<scalar_t>(), P=keep_tape ? pre.data_ptr<scalar_t>() : nullptr;
        const scalar_t invsqrt2=1/std::sqrt(scalar_t(2)), scale=1/std::sqrt(scalar_t(K));
        at::parallel_for(0,N,8,[&](int64_t begin,int64_t end) {
            std::vector<scalar_t> hidden(H), raw(D);
            for (int64_t n=begin;n<end;++n) {
                for (int64_t k=0;k<K;++k) {
                    const auto row=n*K+k, bank=I[row]*H*D;
                    for (int64_t h=0;h<H;++h) {
                        const scalar_t z=scaled_dot(U+bank+h*D,X+n*D,D,scalar_t(1));
                        if (keep_tape) P[row*H+h]=z;
                        hidden[h]=scalar_t(.5)*z*(1+std::erf(z*invsqrt2));
                    }
                    std::fill(raw.begin(),raw.end(),scalar_t(0));
                    for (int64_t h=0;h<H;++h)
                        for (int64_t d=0;d<D;++d) raw[d]+=hidden[h]*W[bank+h*D+d];
                    for (int64_t d=0;d<D;++d) Y[n*D+d]+=raw[d]*G[row];
                }
                for (int64_t d=0;d<D;++d) Y[n*D+d]*=scale;
            }
        });
    });
    return {result,pre};
}

std::vector<Tensor> experts_backward(const Tensor& dy, const Tensor& x, const Tensor& up,
        const Tensor& down, const Tensor& ids, const Tensor& gates, const Tensor& pre) {
    check_experts(x,up,down,ids,gates);
    const auto N=x.size(0), D=x.size(1), H=up.size(1), K=ids.size(1);
    for (const auto& t : {dy,pre})
        TORCH_CHECK(t.device().is_cpu() && t.scalar_type()==x.scalar_type() && t.is_contiguous(),
                    "Invalid expert gradient/tape storage");
    TORCH_CHECK(dy.sizes()==x.sizes() && pre.sizes()==at::IntArrayRef({N,K,H}),
                "Invalid expert gradient/tape dimensions");
    auto dx=at::zeros_like(x),du=at::empty({N*K,H,D},x.options()),dd=at::empty_like(du);
    auto dg=at::empty_like(gates);
    const auto I=ids.data_ptr<int64_t>();
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "indexed_experts_backward", [&] {
        const auto X=x.data_ptr<scalar_t>(), U=up.data_ptr<scalar_t>(), W=down.data_ptr<scalar_t>();
        const auto G=gates.data_ptr<scalar_t>(), P=pre.data_ptr<scalar_t>(), Y=dy.data_ptr<scalar_t>();
        auto dX=dx.data_ptr<scalar_t>(), dU=du.data_ptr<scalar_t>(), dW=dd.data_ptr<scalar_t>(), dG=dg.data_ptr<scalar_t>();
        const scalar_t invsqrt2=1/std::sqrt(scalar_t(2)), scale=1/std::sqrt(scalar_t(K));
        const scalar_t invsqrt2pi=1/std::sqrt(scalar_t(2)*scalar_t(3.1415926535897932384626433832795));
        at::parallel_for(0,N,8,[&](int64_t begin,int64_t end) {
            for (int64_t n=begin;n<end;++n) {
                for (int64_t k=0;k<K;++k) {
                    const auto row=n*K+k, bank=I[row]*H*D;
                    scalar_t gate_grad=0;
                    for (int64_t h=0;h<H;++h) {
                        const scalar_t dh=scaled_dot(Y+n*D,W+bank+h*D,D,scale);
                        const scalar_t z=P[row*H+h], cdf=scalar_t(.5)*(1+std::erf(z*invsqrt2));
                        const scalar_t hidden=z*cdf, dz=(dh*G[row])*(cdf+z*std::exp(scalar_t(-.5)*z*z)*invsqrt2pi);
                        gate_grad+=dh*hidden;
                        for (int64_t d=0;d<D;++d) {
                            const auto offset=(row*H+h)*D+d;
                            dU[offset]=dz*X[n*D+d];
                            dW[offset]=((Y[n*D+d]*scale)*G[row])*hidden;
                            dX[n*D+d]+=dz*U[bank+h*D+d];
                        }
                    }
                    dG[row]=gate_grad;
                }
            }
        });
    });
    return {dx,du,dd,dg};
}

Tensor dense_row_grads(const std::vector<Tensor>& gradients) {
    TORCH_CHECK(!gradients.empty(), "Need at least one expert gradient");
    const auto& first=gradients[0];
    TORCH_CHECK(first.device().is_cpu() && first.dim()==3 && first.is_sparse() &&
                (first.scalar_type()==at::kFloat || first.scalar_type()==at::kDouble),
                "Expected sparse CPU expert gradients");
    auto result=at::zeros(first.sizes(),first.options().layout(at::kStrided));
    const auto rows=first.size(0), width=first.size(1)*first.size(2);
    AT_DISPATCH_FLOATING_TYPES(first.scalar_type(), "expert_dense_row_grads", [&] {
        auto O=result.data_ptr<scalar_t>();
        for (const auto& gradient : gradients) {
            TORCH_CHECK(gradient.device().is_cpu() && gradient.is_sparse() && gradient.sparse_dim()==1 &&
                        gradient.sizes()==first.sizes() && gradient.scalar_type()==first.scalar_type(),
                        "Incompatible sparse expert gradients");
            auto indices=gradient._indices().contiguous(), values=gradient._values().contiguous();
            const auto I=indices.data_ptr<int64_t>();
            const auto V=values.data_ptr<scalar_t>();
            for (int64_t row=0;row<values.size(0);++row) {
                TORCH_CHECK(I[row]>=0 && I[row]<rows, "Expert row index out of range");
                for (int64_t i=0;i<width;++i) O[I[row]*width+i]+=V[row*width+i];
            }
        }
    });
    return result;
}

TORCH_LIBRARY(shoebox_minkron_cpu, m) {
    m.def("cpu_isa() -> str", &cpu_isa);
    m.def("final_forward(Tensor k, Tensor v, Tensor g, Tensor beta, Tensor write, Tensor previous, bool keep_tape, Tensor? lengths=None) -> Tensor[]", &final_forward);
    m.def("final_backward(Tensor grad_final, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor write, Tensor previous, Tensor tape, Tensor? lengths=None) -> Tensor[]", &final_backward);
    m.def("experts_forward(Tensor x, Tensor up, Tensor down, Tensor ids, Tensor gates, bool keep_tape) -> Tensor[]", &experts_forward);
    m.def("experts_backward(Tensor dy, Tensor x, Tensor up, Tensor down, Tensor ids, Tensor gates, Tensor pre) -> Tensor[]", &experts_backward);
    m.def("dense_row_grads(Tensor[] gradients) -> Tensor", &dense_row_grads);
}
