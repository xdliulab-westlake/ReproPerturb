from pathlib import Path
import numpy as np,pandas as pd,json,hashlib
B=Path('/liaozizhuo/hypoblast/benchmarks');N=B/'native-alignment-20261111';E=B/'hypo-benchmark-extension-20261105/results';O=N/'composition_extra_20261114'
t=np.load(B/'hypo-forward-embedding-combo-20261029/data/truth.npz');cfg=json.loads((B/'hypo-forward-embedding-combo-20261029/benchmark.json').read_text());ids=cfg['splits']['test'];classes=cfg['classes'];q=t['proportions'][[t['sample_ids'].tolist().index(i) for i in ids]].astype(float)
old=pd.read_csv(N/'latest_overall_benchmark_20261113/composition_per_seed_all_metrics.csv');rows=[];sources=[]
for r in old.to_dict('records'):
 m=r['method'];s=int(r['seed'])
 if m=='retrieval_only':continue
 p=E/f'locked_checkpoint/{m}_seed{s}/predictions.npz'
 if m.startswith('ridge'):p=E/f'val_selected/{m}_seed{s}/predictions.npz'
 if m=='state':p=B/f'hypo-independent-intervention-20261112/evaluation/state_seed{s}/predictions.npz'
 if m=='cellflow':p=N/f'evaluation/cellflow_seed{s}/predictions.npz'
 if m=='chemcpa':p=N/f'evaluation/chemcpa_seed{s}_chemvocabfix_20261112/predictions.npz'
 if m=='g2cp':p=B/f'hypo-native-protein-20261110/evaluation/g2cp_seed{s}/predictions.npz'
 if m=='stack_traincontext':p=B/f'stack-unified-20261113/evaluation/stack_traincontext_seed{s}/predictions.npz'
 a=np.load(p);assert set(a['sample_ids'])==set(ids);assert list(a['classes'])==classes
 x=a['proportions'][[a['sample_ids'].tolist().index(i) for i in ids]].astype(float)
 assert np.isfinite(x).all() and x.min()>=-1e-12 and np.allclose(x.sum(1),1,atol=1e-6)
 assert np.isclose(100*np.abs(x-q).mean(),r['mae_pp'],atol=1e-6),(m,s,100*np.abs(x-q).mean(),r['mae_pp'])
 x=np.clip(x,0,1);x/=x.sum(1,keepdims=True)
 h=np.sqrt(.5*((np.sqrt(x)-np.sqrt(q))**2).sum(1))
 for j,sid in enumerate(ids):
  # Tie-aware: uniform choice among exactly tied predicted maxima.
  pred=np.flatnonzero(np.isclose(x[j],x[j].max(),rtol=0,atol=1e-12));true=np.flatnonzero(np.isclose(q[j],q[j].max(),rtol=0,atol=1e-12));hit=len(set(pred)&set(true))/len(pred)
  rows.append(dict(method=m,seed=s,sample_id=sid,hellinger=h[j],dominant_accuracy_pct=100*hit,true_dominant='|'.join(classes[k] for k in true),predicted_dominant='|'.join(classes[k] for k in pred),true_top_gap=np.sort(q[j])[-1]-np.sort(q[j])[-2],version=r['version']))
 sources.append(dict(method=m,seed=s,path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),mae_reproduced=True))
d=pd.DataFrame(rows);d.to_csv(O/'per_condition.csv',index=False);ps=d.groupby(['method','seed'])[['hellinger','dominant_accuracy_pct']].mean().reset_index();ps.to_csv(O/'per_seed.csv',index=False)
su=ps.groupby('method').agg(n=('seed','size'),hellinger_mean=('hellinger','mean'),hellinger_sd=('hellinger','std'),dominant_mean=('dominant_accuracy_pct','mean'),dominant_sd=('dominant_accuracy_pct','std')).sort_values('hellinger_mean');su.to_csv(O/'summary.csv')
(O/'PROVENANCE.json').write_text(json.dumps(sources,indent=2));print(su.to_string());print('ROWS',len(d),'TRUE_DOMINANT',d[d.method=='hypo'].query('seed==43').true_dominant.value_counts().to_dict())
assert np.isclose(np.sqrt(.5*((np.sqrt([1.,0])-np.sqrt([0.,1]))**2).sum()),1)
text='''# Composition supplementary metrics\n\nFrozen predictions; 14 test conditions; 10 V7 types. No training or inference. Each seed scored separately, then equal condition averaging and seed mean/sample SD. Every source reproduces existing composition MAE within 1e-6 pp.\n\nHellinger = sqrt(0.5 sum((sqrt(pred)-sqrt(truth))^2)), range 0-1, lower better. Standard established distance, not a newly invented metric.\n\nDominant-type accuracy: whether the predicted largest-proportion cell type matches the true largest-proportion type. Exact ties (absolute tolerance 1e-12) use uniform predicted tie selection and accept any true tied maximum. Report percentage; 14 conditions means 7.14 percentage-point granularity per seed. This does not measure minority-type accuracy; near ties can be unstable. true_top_gap is retained.\n\nBoth are exploratory post-hoc additions; no checkpoint selection or claims of statistical significance. Hellinger overlaps conceptually with JSD. PerturbNet and PerturbGen remain legacy versions. Retrieval-only excluded because its separate frozen proportion array has not been verified for this supplement. Single-run baseline SD undefined.\n\n'''
(O/'README.md').write_text(text+su.to_csv())
