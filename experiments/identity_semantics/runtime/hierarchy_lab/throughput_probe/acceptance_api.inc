extern "C" {
void acceptance_status(void*p,uint64_t*events,uint64_t*primary,uint64_t*coverage,
                       uint64_t*lengths,uint64_t*recognized,uint64_t*updates){
  const auto&z=static_cast<Model*>(p)->acceptance;
  memcpy(events,z.events,sizeof(z.events));memcpy(primary,z.primary_hist,sizeof(z.primary_hist));
  memcpy(coverage,z.coverage_hist,sizeof(z.coverage_hist));memcpy(lengths,z.lengths,sizeof(z.lengths));
  memcpy(recognized,z.recognized_lengths,sizeof(z.recognized_lengths));
  memcpy(updates,z.expert_update_calls,sizeof(z.expert_update_calls));
}
void acceptance_reset(void*p){static_cast<Model*>(p)->acceptance=AcceptanceObservations{};}
}
