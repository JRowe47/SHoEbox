#pragma once
// Optimizer state, independent of routing coverage and recurrent context.
// Multipliers preserve the existing fast/slow/query/head rate ratios.
struct RouteLearning {
    bool configured=false,enabled=false;
    float lower=.5f,upper=1.5f,start=1.f,target=1.f,amplitude=1.f;
    uint64_t period=256,targets=0,ramp_elapsed=0,ramp_targets=0;

    float anchor()const{
        if(!ramp_targets||ramp_elapsed>=ramp_targets)return target;
        return float(double(start)+(double(target)-start)*(double(ramp_elapsed)/ramp_targets));
    }
    float shared()const{return enabled?anchor():1.f;}
    float multiplier(uint64_t visits)const{
        if(!enabled)return 1.f;
        const double a=anchor();
        const double radius=amplitude*std::min(a-lower,double(upper)-a);
        const double phase=6.2831853071795864769*double(visits%period)/double(period);
        return float(std::clamp(a+radius*std::sin(phase),double(lower),double(upper)));
    }
    void advance(){
        if(!enabled)return;
        if(targets!=UINT64_MAX)++targets;
        if(ramp_elapsed<ramp_targets)++ramp_elapsed;
    }
    void validate()const{
        for(float x:{lower,upper,start,target,amplitude})
            if(!std::isfinite(x))throw std::invalid_argument("route learning values must be finite");
        if(lower<0||upper<lower||start<lower||start>upper||target<lower||target>upper||
           amplitude<0||amplitude>1||!period||ramp_elapsed>ramp_targets)
            throw std::invalid_argument("invalid route learning bounds, anchor, amplitude or clock");
    }
    void validate_rate(float rate)const{
        if(!std::isfinite(rate)||rate<0||
           (enabled&&double(rate)*upper>std::numeric_limits<float>::max()))
            throw std::invalid_argument("learning rate exceeds finite route learning bounds");
    }
};
