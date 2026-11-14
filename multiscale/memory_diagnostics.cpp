// Standalone inspection; no diagnostic work is added to the model's hot loop.
#ifdef SHOE_ADDRESSED_V2
#include "addressed_v2/native.hpp"
#elif defined(SHOE_ADDRESSED_V3)
#include "addressed_v3/native.hpp"
#else
#include "native.hpp"
#endif
#include <fstream>
#include <iostream>
#include <memory>
using namespace shoe;
int main(int argc,char** argv){
    AddressStream s;AddressParameters p{};AddressEvent e;Vec c{},h{};
    for(int i=0;i<D;++i)c[i]=.3f*std::sin(float(i));
    for(int i=0;i<TIME_D;++i)p.query_clock[i*TIME_D+i]=.25f;
    double max_memory=0,max_read=0;
    for(int t=0;t<10000;++t){address_forward(e,c,h,stable_bytes()[120],s,p);
        double m=0,r=0;for(float x:s.memory)m+=double(x)*x;for(float x:e.read)r+=double(x)*x;
        max_memory=std::max(max_memory,std::sqrt(m));max_read=std::max(max_read,std::sqrt(r));}
    auto model=std::make_unique<Model>();
    uint8_t key[]{1,2},value[]{4,5,6};auto handle=model->store.bind(key,2,value,3);
    auto copy=std::make_unique<Model>(*model);copy->inference(true);
    std::cout<<"{\"max_memory_norm\":"<<max_memory<<",\"max_read_norm\":"<<max_read
             <<",\"native_cold_preserves_binding\":"<<(copy->store.resolve(handle)?"true":"false");
    if(argc>1){std::ifstream file(argv[1],std::ios::binary);if(!file)return 2;
        std::string data((std::istreambuf_iterator<char>(file)),{});if(data.size()<2)return 2;
        model->run(reinterpret_cast<const uint8_t*>(data.data()),reinterpret_cast<const uint8_t*>(data.data()+1),data.size()-1);
        model->finish();model->inference(false);
        for(uint8_t b:std::string("AB")+std::string(192,'x')+"?")model->observe(b);
        std::cout<<",\"trained_slow_scales\":[";
        for(size_t i=1;i<model->levels.size();++i){const auto& stream=model->levels[i].s;
            int saturated=0;double read=0,memory=0;for(float x:stream.state)saturated+=std::abs(x)>.95f;
            for(float x:stream.address_events[0].read)read+=double(x)*x;
            for(float x:stream.addressed.memory)memory+=double(x)*x;
            if(i>1)std::cout<<',';
            std::cout<<"{\"level\":"<<i<<",\"saturated_coordinates\":"<<saturated<<",\"read_norm\":"<<std::sqrt(read)<<",\"memory_norm\":"<<std::sqrt(memory)<<'}';}
        std::cout<<']';}
    std::cout<<"}\n";
}
