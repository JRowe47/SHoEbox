// Fused CPU recurrence backward and compact expert accumulation, FP32/FP64.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <cmath>
#include <vector>

using at::Tensor;

static void check(const std::vector<Tensor>& tensors, const Tensor& q) {
    TORCH_CHECK(q.device().is_cpu() && (q.scalar_type() == at::kFloat || q.scalar_type() == at::kDouble),
                "Kron CPU kernel requires FP32/FP64 CPU tensors");
    for (const auto& t : tensors)
        TORCH_CHECK(t.device().is_cpu() && t.scalar_type() == q.scalar_type() && t.is_contiguous(),
                    "Kron CPU inputs must be contiguous with matching dtype");
}

std::vector<Tensor> kron_backward(const Tensor& dout, const Tensor& dstate, const Tensor& q,
        const Tensor& k, const Tensor& v, const Tensor& beta, const Tensor& write,
        const Tensor& final, const Tensor& decay, const Tensor& decayed, const Tensor& norm,
        const Tensor& inv, const Tensor& score, const Tensor& update, const Tensor& previous) {
    check({dout,dstate,q,k,v,beta,write,previous,final,decay,decayed,norm,inv,score,update}, q);
    TORCH_CHECK(q.dim()==3 && v.dim()==3, "Invalid Kron backward directions");
    const auto B=q.size(0), H=q.size(1), K=q.size(2), V=v.size(2), N=B*H;
    TORCH_CHECK(k.sizes()==q.sizes() && decay.sizes()==q.sizes() &&
        v.sizes()==at::IntArrayRef({B,H,V}) && dout.sizes()==v.sizes() &&
        previous.sizes()==at::IntArrayRef({B,H,K,V}) && final.sizes()==previous.sizes() &&
        dstate.sizes()==previous.sizes() && beta.sizes()==at::IntArrayRef({B,H}) &&
        write.sizes()==beta.sizes() && decayed.sizes()==previous.sizes() &&
        norm.sizes()==beta.sizes() && inv.sizes()==beta.sizes() && score.sizes()==beta.sizes() &&
        update.sizes()==beta.sizes() && K>0 && V>0, "Invalid Kron backward shapes");
    auto dq=at::empty_like(q), dk=at::empty_like(k), dv=at::empty_like(v), dg=at::empty_like(q);
    auto db=at::empty_like(beta), dc=at::empty_like(write), dp=at::empty_like(previous);
    AT_DISPATCH_FLOATING_TYPES(q.scalar_type(), "shoebox_kron_backward", [&] {
        auto Q=q.data_ptr<scalar_t>(), A=k.data_ptr<scalar_t>(), W=v.data_ptr<scalar_t>();
        auto E=beta.data_ptr<scalar_t>(), C=write.data_ptr<scalar_t>(), P=previous.data_ptr<scalar_t>();
        auto S=final.data_ptr<scalar_t>(), D=decay.data_ptr<scalar_t>(), Dec=decayed.data_ptr<scalar_t>();
        auto Norm=norm.data_ptr<scalar_t>(), Inv=inv.data_ptr<scalar_t>(), Score=score.data_ptr<scalar_t>(), U=update.data_ptr<scalar_t>();
        auto O=dout.data_ptr<scalar_t>(), Z=dstate.data_ptr<scalar_t>();
        auto dQ=dq.data_ptr<scalar_t>(), dA=dk.data_ptr<scalar_t>(), dW=dv.data_ptr<scalar_t>();
        auto dG=dg.data_ptr<scalar_t>(), dE=db.data_ptr<scalar_t>(), dC=dc.data_ptr<scalar_t>();
        auto dP=dp.data_ptr<scalar_t>();
        const scalar_t scale=std::pow(scalar_t(K),scalar_t(-.5));
        std::vector<scalar_t> weighted(K), total(K*V);
        for (int64_t n=0;n<N;++n) {
            scalar_t norm=Norm[n], inv=Inv[n], score=Score[n], u=U[n], du=0;
            for (int64_t i=0;i<K;++i) {
                scalar_t w=0,direction=0;
                for (int64_t j=0;j<V;++j) {
                    auto index=(n*K+i)*V+j;
                    scalar_t read=O[n*V+j]*scale;
                    direction+=read*S[index];
                    scalar_t value=Z[index]+Q[n*K+i]*read;
                    total[i*V+j]=value;
                    w+=value*W[n*V+j];
                }
                dQ[n*K+i]=direction;weighted[i]=w;du+=w*A[n*K+i];
            }
            scalar_t ds=-du*(E[n]*inv), di=du*(C[n]*norm)+(-du*score)*E[n];
            scalar_t dn=(du*inv)*C[n]-(norm>=scalar_t(1e-18)?di*(inv*inv):scalar_t(0));
            dE[n]=(-du*score)*inv;dC[n]=(du*inv)*norm;
            for (int64_t j=0;j<V;++j) {
                scalar_t dw=0,sw=0;
                for (int64_t i=0;i<K;++i) {
                    auto index=(n*K+i)*V+j;
                    dw+=total[i*V+j]*(u*A[n*K+i]);
                    sw+=ds*(Dec[index]*A[n*K+i]);
                }
                dW[n*V+j]=dw+sw+2*W[n*V+j]*dn;
            }
            for (int64_t i=0;i<K;++i) {
                scalar_t da=0,dd=0;
                for (int64_t j=0;j<V;++j) {
                    auto index=(n*K+i)*V+j;
                    scalar_t sd=ds*W[n*V+j];
                    da+=sd*Dec[index];
                    scalar_t delta=total[i*V+j]+sd*A[n*K+i];
                    dP[index]=delta*D[n*K+i];dd+=delta*P[index];
                }
                dA[n*K+i]=weighted[i]*u+da;dG[n*K+i]=dd*D[n*K+i];
            }
        }
    });
    return {dq,dk,dv,dg,db,dc,dp};
}

Tensor dense_rows(const Tensor& ids, const Tensor& values, int64_t rows) {
    TORCH_CHECK(ids.device().is_cpu() && ids.scalar_type()==at::kLong && ids.dim()==1 && ids.is_contiguous(),
                "Expert row ids must be contiguous CPU int64");
    check({values}, values);
    TORCH_CHECK(values.dim()>=1 && values.size(0)==ids.numel() && rows>=0, "Invalid expert row gradient shape");
    auto shape=values.sizes().vec();shape[0]=rows;
    auto result=at::zeros(shape,values.options());
    const auto count=ids.numel(), width=count ? values.numel()/count : 0;
    auto I=ids.data_ptr<int64_t>();
    AT_DISPATCH_FLOATING_TYPES(values.scalar_type(), "shoebox_dense_rows", [&] {
        auto input=values.data_ptr<scalar_t>(), output=result.data_ptr<scalar_t>();
        for (int64_t row=0;row<count;++row) {
            TORCH_CHECK(I[row]>=0 && I[row]<rows, "Expert row index out of range");
            auto from=input+row*width, to=output+I[row]*width;
            for (int64_t column=0;column<width;++column) to[column]+=from[column];
        }
    });
    return result;
}

std::vector<Tensor> expert_backward(const Tensor& dy, const Tensor& x, const Tensor& up,
        const Tensor& down, const Tensor& gates, const Tensor& pre, const Tensor& hidden, const Tensor& raw) {
    check({dy,x,up,down,gates,pre,hidden,raw},x);
    TORCH_CHECK(x.dim()==2 && up.dim()==3, "Invalid expert dimensions");
    const auto N=x.size(0), D=x.size(1), H=up.size(1);
    TORCH_CHECK(up.sizes()==at::IntArrayRef({N,H,D}) && down.sizes()==up.sizes() &&
        dy.sizes()==x.sizes() && raw.sizes()==x.sizes() && gates.sizes()==at::IntArrayRef({N}) &&
        pre.sizes()==at::IntArrayRef({N,H}) && hidden.sizes()==pre.sizes(), "Invalid expert gradient shapes");
    auto dx=at::empty_like(x), du=at::empty_like(up), dd=at::empty_like(down), dg=at::empty_like(gates);
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "shoebox_expert_backward", [&] {
        auto Y=dy.data_ptr<scalar_t>(), X=x.data_ptr<scalar_t>(), U=up.data_ptr<scalar_t>();
        auto W=down.data_ptr<scalar_t>(), G=gates.data_ptr<scalar_t>(), P=pre.data_ptr<scalar_t>();
        auto A=hidden.data_ptr<scalar_t>(), R=raw.data_ptr<scalar_t>();
        auto dX=dx.data_ptr<scalar_t>(), dU=du.data_ptr<scalar_t>(), dW=dd.data_ptr<scalar_t>(), dG=dg.data_ptr<scalar_t>();
        const scalar_t invsqrt2=1/std::sqrt(scalar_t(2)), invsqrt2pi=1/std::sqrt(scalar_t(2)*scalar_t(3.1415926535897932384626433832795));
        std::vector<scalar_t> dz(H);
        for (int64_t n=0;n<N;++n) {
            scalar_t gate_gradient=0;
            for (int64_t d=0;d<D;++d) gate_gradient+=Y[n*D+d]*R[n*D+d];
            dG[n]=gate_gradient;
            for (int64_t h=0;h<H;++h) {
                scalar_t dh=0;
                for (int64_t d=0;d<D;++d) {
                    scalar_t delta=Y[n*D+d]*G[n];
                    dh+=delta*W[(n*H+h)*D+d];
                    dW[(n*H+h)*D+d]=delta*A[n*H+h];
                }
                scalar_t z=P[n*H+h];
                scalar_t derivative=scalar_t(.5)*(1+std::erf(z*invsqrt2))+z*std::exp(scalar_t(-.5)*z*z)*invsqrt2pi;
                dz[h]=dh*derivative;
            }
            for (int64_t d=0;d<D;++d) {
                scalar_t value=0;
                for (int64_t h=0;h<H;++h) {
                    auto index=(n*H+h)*D+d;
                    dU[index]=dz[h]*X[n*D+d];
                    value+=dz[h]*U[index];
                }
                dX[n*D+d]=value;
            }
        }
    });
    return {dx,du,dd,dg};
}

TORCH_LIBRARY(shoebox_kron_cpu, m) {
    m.def("backward(Tensor dout, Tensor dstate, Tensor q, Tensor k, Tensor v, Tensor beta, Tensor write, Tensor final, Tensor decay, Tensor decayed, Tensor norm, Tensor inv, Tensor score, Tensor update, Tensor previous) -> Tensor[]", &kron_backward);
    m.def("dense_rows(Tensor ids, Tensor values, int rows) -> Tensor", &dense_rows);
    m.def("expert_backward(Tensor dy, Tensor x, Tensor up, Tensor down, Tensor gates, Tensor pre, Tensor hidden, Tensor raw) -> Tensor[]", &expert_backward);
}
