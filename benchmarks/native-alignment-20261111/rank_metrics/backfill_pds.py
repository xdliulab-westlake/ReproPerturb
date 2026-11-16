from pathlib import Path
import json,csv,hashlib
import numpy as np
B=Path('/liaozizhuo/hypoblast/benchmarks');OLD=B/'hypo-forward-embedding-combo-20261029';O=B/'native-alignment-20261111/rank_metrics'
t=np.load(OLD/'data/truth.npz');bench=json.loads((OLD/'benchmark.json').read_text());ids=bench['splits']['test'];ii=[t['sample_ids'].tolist().index(x) for x in ids];truth=t['expression'][ii];control=t['control_expression'][ii]
def score(p,r):
 pn=np.linalg.norm(p,axis=1);rn=np.linalg.norm(r,axis=1);den=pn[:,None]*rn[None,:]
 sim=np.divide(p@r.T,den,out=np.zeros((len(p),len(r))),where=den>0);dist=1-np.clip(sim,-1,1);correct=np.diag(dist)[:,None]
 return 1-((dist<correct).sum(1)+.5*((dist==correct).sum(1)-1))/(len(r)-1)
assert np.allclose(score(np.eye(14),np.eye(14)),1)
assert np.allclose(score(np.zeros((14,14)),np.eye(14)),.5)
rows=[];audit=[]
paths=list(B.rglob('predictions.npz'))
for p in paths:
 with np.load(p) as z:
  if not {'expression','sample_ids','genes'}.issubset(z.files):continue
  if not set(ids).issubset(z['sample_ids'].tolist()):continue
  assert z['genes'].tolist()==t['genes'].tolist(),str(p)
  pred=z['expression'][[z['sample_ids'].tolist().index(x) for x in ids]]
  assert pred.shape==truth.shape and np.isfinite(pred).all(),str(p)
  values=score(pred-control,truth-control)
  digest=hashlib.sha256(p.read_bytes()).hexdigest()
  for sid,v in zip(ids,values):rows.append(dict(source=str(p),sample_id=sid,pds_cosine_stored_delta=float(v),prediction_sha256=digest))
  audit.append(dict(source=str(p),mean=float(values.mean()),conditions=len(ids),direction_reach_status='requires per-cell predictions or full predicted DE table; not inferable from means'))
for name,data in [('pds_per_condition.csv',rows),('pds_inventory.csv',audit)]:
 with (O/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
(O/'PROVENANCE.json').write_text(json.dumps(dict(official_commit='551849dc25354dff724edceb39a6108d997bf876',rank='1-(strictly_better+0.5*(tied-1))/(n-1)',candidate_ids=ids,space='existing stored benchmark expression mean minus matched real control; NOT raw-count bulk_lognorm VCC preset',chemical_target_gene_exclusion=False,tests=['identity=1','zero effect ties=0.5'],runs=len(audit),all_models_complete=False),indent=2))
print('PDS_BACKFILLED_RUNS',len(audit),'ROWS',len(rows))
