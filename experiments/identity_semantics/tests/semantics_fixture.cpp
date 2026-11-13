#define SHOEBOX_IDENTITY_SEMANTICS 1
#include "binding_fixture.cpp"
extern "C" {
void semantic_fixture_structure(void*p,int src,int age){
 auto&e=static_cast<Model*>(p)->composition.stream.history[src][age];memcpy(e.structure,e.code,sizeof(e.structure));
}
void semantic_fixture_encode(void*p,int r,int kind,uint64_t*meta,float*code){
 auto&m=*static_cast<Model*>(p);ContentEvent parts[2]{m.composition.stream.history[0][1],m.composition.stream.history[0][0]};
 auto e=content_compose(m,r,parts,2,kind,false);meta[0]=e.ref.id.a;meta[1]=e.ref.id.b;memcpy(code,e.code,sizeof(e.code));
}
void semantic_fixture_change(void*p,int src,int age,int field,int64_t delta){
 auto&e=static_cast<Model*>(p)->composition.stream.history[src][age];
 if(field==0)e.ref.end+=delta;
 if(field==1)e.ref.region+=int(delta);
 if(field==2)for(int j=0;j<D;j++)e.code[j]+=float(delta);
 if(field==3){e.ref.tick+=delta;e.ref.seq+=delta;e.ref.index+=int(delta);}
}
void semantic_fixture_weight(void*p,int r,int j,int k,float delta){static_cast<Model*>(p)->composition.symbols[r-2][j][k]+=delta;}
void semantic_fixture_gradient(void*p,int r,const float*g,float rate){
 auto&m=*static_cast<Model*>(p);ContentEvent parts[2]{m.composition.stream.history[0][1],m.composition.stream.history[0][0]};
 auto e=content_compose(m,r,parts,2,1,false);auto&v=m.laces.records[r][0];v=LaceRecord{};v.tick=713;
 auto&s=m.composition.stream.input[r][0];s=InputCredit{};s.tick=v.tick;s.symbol_scale=1;
#ifdef SHOEBOX_LEARNED_COMPLETION
 s.semantic=e.semantic;
#else
 memcpy(s.symbol_features,e.symbol_features,sizeof(s.symbol_features));memcpy(s.symbol_derivative,e.symbol_derivative,sizeof(s.symbol_derivative));
#endif
 semantic_input_update(m,r,v,g,rate);
}
void semantic_fixture_saved_gradient(void*p,int r,const float*g,float rate){
 auto&m=*static_cast<Model*>(p);semantic_input_update(m,r,m.laces.records[r][0],g,rate);
}
void semantic_fixture_clear_dictionary(void*p){
 auto&m=*static_cast<Model*>(p);memset(m.composition.patterns,0,sizeof(m.composition.patterns));
}
void semantic_fixture_q(void*p,int r,float*out){
 auto&m=*static_cast<Model*>(p);memcpy(out,m.utility.previous_q[r],D*sizeof(float));
}
}
extern "C" void semantic_fixture_ablate(void*p){
 auto&m=*static_cast<Model*>(p);memset(m.composition.symbols,0,sizeof(m.composition.symbols));
}
