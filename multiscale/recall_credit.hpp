#pragma once
// Eligibility only for producers whose content weights are still frozen in
// the current constituent epoch. Older retained values remain detached.
struct RecallCreditUnit {
    Vec combined{};
    RecallKey value{},gradient{};
    int current=-1,previous=-1;
};
struct RecallCreditState {
    std::array<RecallCreditUnit,RECALL_SLOTS> units{};
    int previous=-1;
};
inline void recall_value_vjp(const RecallCreditUnit& unit,Vec& current,Vec& previous){
    for(int j=0;j<RECALL_K;++j){float dp=unit.gradient[j]*(1-unit.value[j]*unit.value[j]);
        for(int k=0;k<D/RECALL_K;++k){int i=j+RECALL_K*k;
            float d=.5f*(k%2?-.5f:.5f)*dp*(1-unit.combined[i]*unit.combined[i]);
            current[i]+=d;previous[(3*i+13)%D]+=d;
        }
    }
}
