#include "../runtime/hierarchy_lab/throughput_probe/completion_checks.cpp"
#include <cassert>

extern "C" {
// Compare the removed preliminary evaluation against descriptor-only setup,
// including intervention accounting and the final saved content Jacobian.
void storage_fixture_encode(void*p,int old,int muted,int experts,float*out){
  auto&m=*static_cast<Model*>(p);m.content_experts=experts;
  m.contribution_kind=muted?CHECK_SEMANTIC:CHECK_NONE;m.contribution_region=6;
  ContentEvent child{};child.ref.region=2;child.ref.id={123,456};
  for(int j=0;j<D;j++){
    child.edit_values().structure[j]=.1f*std::sin(float(j));child.edit_values().code[j]=child.values().structure[j]+.02f;
    for(int k=0;k<8;k++)m.composition.symbols[4][j][k]=.1f*std::cos(float(j+k));
  }
  ContentEvent event{};event.ref.region=6;event.ref.id={543,876};
  for(int j=0;j<D;j++)event.edit_values().structure[j]=.03f*(j%7-3);
  if(old)semantic_encode(m,6,event);else semantic_prepare_constituent(m,6,event);
  constituent_encode(m,6,event,&child,1,false);
  memcpy(out,event.values().structure,sizeof(event.values().structure));
  memcpy(out+D,event.values().code,sizeof(event.values().code));
  memcpy(out+2*D,event.values().symbol_derivative,sizeof(event.values().symbol_derivative));
  out[3*D]=float(m.contribution_counts[6][4]);
}

// Nonzero legacy-only bytes must not become live runtime state. Every consumed
// scalar, feedback edge, Jacobian and pending gradient must survive conversion.
int storage_fixture_legacy_input(){
  LegacyInputCredit wire{};wire.tick=123;wire.previous_tick=99;wire.previous_index=3;
  wire.retention=.5f;wire.semantic={57,7,9};wire.symbol_scale=.25f;
  wire.message.feedback_count=2;wire.message.qa=.7f;wire.message.alpha=.5f;
  wire.message.semantic_count=3;
  for(int k=0;k<COMP_MAX;k++){
    wire.message.semantic[k].slot=k;wire.message.semantic[k].features[0]=9;
    wire.message.feedback[k]={k,k+2,100+k,.125f,k+3};
  }
  for(int j=0;j<D;j++){
    wire.symbol_derivative[j]=9;wire.message.payload_derivative[j]=8;
    wire.message.feedback_derivative[j]=.01f*j;wire.message.route_derivative[j]=-.02f*j;
    for(int d=0;d<LACE_DEPTH;d++)wire.gradient[d][j]=float(d*D+j);
  }
  for(int k=0;k<8;k++)wire.symbol_features[k]=7;
  for(int d=0;d<LACE_DEPTH;d++)wire.uses[d]=d+1;
  InputCredit compact{};completion_copy_input(compact,wire);
  LegacyInputCredit restored{};memset(&restored,0,sizeof(restored));completion_copy_input(restored,compact);
  memset(wire.symbol_features,0,sizeof(wire.symbol_features));
  memset(wire.symbol_derivative,0,sizeof(wire.symbol_derivative));
  wire.message.semantic_count=0;memset(wire.message.semantic,0,sizeof(wire.message.semantic));
  memset(wire.message.payload_derivative,0,sizeof(wire.message.payload_derivative));
  // Compare actual fields; only memcmp arrays copied byte-for-byte above.
  return restored.tick==123&&restored.previous_tick==99&&restored.symbol_scale==.25f&&
    restored.message.semantic_count==0&&restored.symbol_features[0]==0&&
    restored.symbol_derivative[0]==0&&restored.message.payload_derivative[0]==0&&
    memcmp(restored.gradient,wire.gradient,sizeof(wire.gradient))==0&&
    memcmp(restored.uses,wire.uses,sizeof(wire.uses))==0&&
    restored.previous_index==wire.previous_index&&restored.retention==wire.retention&&
    restored.semantic.serial==wire.semantic.serial&&restored.semantic.region==wire.semantic.region&&
    restored.semantic.index==wire.semantic.index&&restored.message.feedback_count==wire.message.feedback_count&&
    restored.message.qa==wire.message.qa&&restored.message.alpha==wire.message.alpha&&
    memcmp(restored.message.feedback,wire.message.feedback,sizeof(wire.message.feedback))==0&&
    memcmp(restored.message.feedback_derivative,wire.message.feedback_derivative,sizeof(wire.message.feedback_derivative))==0&&
    memcmp(restored.message.route_derivative,wire.message.route_derivative,sizeof(wire.message.route_derivative))==0;
}
}
