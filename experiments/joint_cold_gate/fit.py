import time
from .stream import advance
from experiments.query_access.run import evaluate
def fit(m,cases,dev,budget,interval,cursor):
 cpu=wall=0.;curve=[];total_loss=0.;initial=m.inventory();start_bytes=cursor['bytes'];start_targets=cursor['targets']
 while cursor['bytes']-start_bytes<budget:
  count=min(interval,budget-(cursor['bytes']-start_bytes));c=time.process_time();w=time.perf_counter()
  cursor,loss=advance(m,cursor,count,cases);m.finish();cpu+=time.process_time()-c;wall+=time.perf_counter()-w;total_loss+=loss
  curve.append(dict(bytes=cursor['bytes']-start_bytes,cpu_seconds=cpu,train_nll=total_loss/max(1,cursor['targets']-start_targets),dev=evaluate(m,dev,all_modes=False)))
 return dict(bytes=budget,targets=cursor['targets']-start_targets,padding_bytes=cursor['padding_bytes'],answer_targets=cursor['answer_targets'],cpu_seconds=cpu,wall_seconds=wall,bytes_per_cpu_second=budget/cpu,curve=curve,inventory=m.inventory(),initial_inventory=initial,order_sha256=cursor['order_sha256']),cursor

