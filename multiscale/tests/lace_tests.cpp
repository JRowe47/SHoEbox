#include "../native.hpp"
#include <iostream>
#include <memory>
using namespace shoe;
int checks=0;
void check(bool b,const char* text){++checks;if(!b)throw std::runtime_error(text);}
int main(){try{
    Vec a{},z{};for(int i=0;i<D;++i)a[i]=std::sin(float(i));SparseCode code;
    for(int k=0;k<5;++k)lace_top4(a,z,&code.bits[k*2]);
    for(int k=0;k<5;++k)check(__builtin_popcountll(code.bits[k*2])+__builtin_popcountll(code.bits[k*2+1])==4,"exactly four signed features per role");
    check(std::abs(lace_similarity(code,code)-1)<1e-6,"identical sparse address");
    auto swapped=code;swapped.path={1,2,3,4};check(lace_similarity(code,swapped)<1,"ordered child identities affect matching");
    auto m=std::make_unique<Model>(4,1000000,1,false);auto& b=m->levels[3].laces;
    m->lace_training_bytes=250000;b.clock=1000;b.natural_use[0]=1000;b.expert_loss[0]=1.5;
    auto& r=b.records[0];r.identity=1;r.incumbent=0;r.code=code;r.hits=64;r.last=999;r.loss=2.5;
    b.records[1]=r;b.records[1].identity=2;b.records[1].loss=1.0;
    auto w=m->levels[3].experts[0].w;m->lace_maintain();
    check(r.status==1&&r.specialist==1,"persistent high-error collision recruits unused expert");
    check(w==m->levels[3].experts[1].w&&w==m->levels[3].experts[0].w,"copy initialization preserves incumbent");
    check(b.reserved[1]==1,"trial expert excluded from natural routes");
    r.control_n=r.trial_n=128;r.control_sum=256;r.trial_sum=240;r.control_sq=512;r.trial_sq=450;m->lace_maintain();
    check(r.status==2&&b.promotions==1,"prequential benefit retains specialist");
    auto r2=std::make_unique<Model>(4,1000000,1,false);r2->lace_training_bytes=250000;
    auto& c=r2->levels[3].laces;c.records[0]=r;c.records[0].status=1;c.records[0].control_n=c.records[0].trial_n=512;
    c.records[0].control_sum=c.records[0].trial_sum=1024;c.reserved[1]=1;
    r2->lace_maintain();check(c.rejections==1&&c.records[0].status==0,"unhelpful specialist rejected");
    c.records[0].status=1;c.records[0].specialist=1;c.records[0].last=1;c.clock=3000;
    c.records[0].control_n=c.records[0].trial_n=1;r2->lace_maintain();
    check(c.rejections==2&&c.records[0].status==0,"stale trials release bounded competition slots");
    auto protected_model=std::make_unique<Model>(4,1000000,1,false);
    auto& pb=protected_model->levels[3].laces;protected_model->lace_training_bytes=250000;pb.clock=1000;
    pb.records[0]=SemanticLace{};auto& pr=pb.records[0];pr.identity=1;pr.incumbent=0;pr.hits=64;pr.loss=2.5;pr.last=999;
    pb.records[1]=pr;pb.records[1].identity=2;pb.records[1].loss=1.;
    pb.records[2]=pr;pb.records[2].identity=3;pb.records[2].status=2;pb.records[2].incumbent=1;pb.records[2].specialist=2;
    pb.natural_use[0]=1000;pb.expert_loss[0]=1.5;pb.reserved[2]=1;
    protected_model->lace_maintain();
    check(pr.specialist==3,"another lace's incumbent and specialist cannot be overwritten");
    auto fresh=std::make_unique<Model>(4,1000000,7,false);fresh->lace_training_bytes=250000;
    fresh->lace_maintain();check(fresh->levels[3].laces.allocations==0,"unused capacity alone cannot recruit");
    // Train an allocated route with pending eligibility, then verify exact replay.
    std::vector<uint8_t>x(1031),y(1031);for(size_t i=0;i<x.size();++i){x[i]=i*13%91;y[i]=(i+1)*13%91;}
    m->run(x.data(),y.data(),1031);auto bytes=save(*m);auto n=std::make_unique<Model>(load(bytes.data(),bytes.size()));
    check(save(*n)==bytes,"lace state checkpoint round trip");
    auto loss=m->run(x.data(),y.data(),1031),loss2=n->run(x.data(),y.data(),1031);
    check(loss==loss2&&save(*m)==save(*n),"pending lace eligibility resumes exactly");
    m->finish();auto records=m->levels[3].laces.records;m->reset(true);
    check(std::memcmp(records.data(),m->levels[3].laces.records.data(),sizeof(records))==0,"context reset preserves semantic identities");
    m->inference(true);records=m->levels[3].laces.records;auto mean=m->levels[3].laces.content_mean;
    m->run(x.data(),y.data(),1031);
    check(std::memcmp(records.data(),m->levels[3].laces.records.data(),sizeof(records))==0&&mean==m->levels[3].laces.content_mean,"frozen queries never learn laces");
    std::cout<<"PASS "<<checks<<" semantic-lace checks\n";
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<"\n";return 1;}}
