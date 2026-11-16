#include "native.cpp"
#include <iostream>

int main(){
    int checks=0;
    auto require=[&](bool ok,const char* message){++checks;if(!ok)throw std::runtime_error(message);};
    auto m=std::make_unique<shoe::Model>(4,500000,7,false,false);
    m->semantic_laces=false;m->routing_arm=2;m->balance_policy=2;
    std::array<uint8_t,128> x{},y{},mask{};
    for(size_t i=0;i<x.size();++i){x[i]=uint8_t(32+i%8);y[i]=uint8_t(32+(i+1)%8);}
    mask.fill(1);
    auto direct=std::make_unique<shoe::Model>(*m);
    double loss=0;size_t targets=0;
    require(mad_train(m.get(),x.data(),y.data(),mask.data(),128,&loss,&targets)==0,"all-mask train");
    double expected=direct->run(x.data(),y.data(),128);
    require(std::abs(loss/128-expected)<1e-10&&targets==128,"masked loss equals public loss");
    require(shoe::save(*m)==shoe::save(*direct),"all-mask runtime parity");
    m->reset(true);direct->reset(true);mask.fill(0);mask.back()=1;
    auto changed=y;for(size_t i=0;i+1<y.size();++i)changed[i]^=31;
    require(mad_train(m.get(),x.data(),y.data(),mask.data(),128,&loss,&targets)==0,"sparse mask");
    require(mad_train(direct.get(),x.data(),changed.data(),mask.data(),128,&loss,&targets)==0,"changed unscored labels");
    require(targets==1&&shoe::save(*m)==shoe::save(*direct),"unscored targets cannot influence training");
    float features[64],gradient[64];std::fill_n(gradient,64,.01f);
    require(mad_features(m.get(),features)==0,"completed-state extraction");
    require(mad_credit(m.get(),gradient)==0,"live compression credit");
    require(m->levels[3].task_gradient_energy>0,"compression reaches encoder");
    require(mad_credit(m.get(),gradient)!=0,"retired credit rejected");
    auto checkpoint=shoe::save(*m);auto resumed=std::make_unique<shoe::Model>(shoe::load(checkpoint.data(),checkpoint.size()));
    require(mad_train(m.get(),x.data(),y.data(),mask.data(),128,&loss,&targets)==0,"continue original");
    require(mad_train(resumed.get(),x.data(),y.data(),mask.data(),128,&loss,&targets)==0,"continue restored");
    require(shoe::save(*m)==shoe::save(*resumed),"native continuation exact");
    m->inference(true);
    std::array<uint8_t,5> episode{{1,32,7,35,4}},score_mask{{0,0,0,1,0}};
    for(int rollout=0;rollout<3;++rollout){
        m->reset(true);double scores[7];
        require(mad_score(m.get(),episode.data(),score_mask.data(),episode.size(),32,8,rollout,scores)==0,"frozen scoring");
        require(scores[0]==1&&std::isfinite(scores[1])&&std::isfinite(scores[2]),"finite answer scores");
    }
    auto before=shoe::save(*m);
    require(mad_credit(m.get(),gradient)!=0,"frozen credit rejected");
    require(shoe::save(*m)==before,"failed credit leaves source intact");
    int count=-1;require(mad_permute(m.get(),3,47,&count)==0&&count>=0,"active expert permutation");
    double inventory[10];require(mad_inventory(m.get(),inventory)==0&&inventory[2]==64&&inventory[9]==16,"state inventory");
    std::cout<<"PASS "<<checks<<" MAD native checks\n";
}
