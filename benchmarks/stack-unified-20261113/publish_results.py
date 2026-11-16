from pathlib import Path
import json,hashlib,fcntl
import pandas as pd,numpy as np
BASE=Path('/liaozizhuo/hypoblast/benchmarks');U=BASE/'stack-unified-20261113';N=BASE/'native-alignment-20261111';lock=(N/'.latest_overall_publish.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX);O=sorted(N.glob('latest_overall_benchmark_*'))[-1];e=pd.read_csv(O/'expression_per_seed_all_metrics.csv');c=pd.read_csv(O/'composition_per_seed_all_metrics.csv');er=[];cr=[];sources=[]
for seed in [43,44,45]:
 r=U/f'runs/seed{seed}';train=json.loads((r/'TRAIN_COMPLETE.json').read_text());assert train['training_conditions']==146 and train['validation_conditions']==21
 h=json.loads((r/'history.json').read_text());assert all(x['validation_conditions']==21 for x in h)
 d=U/f'evaluation/stack_traincontext_seed{seed}';done=json.loads((d/'COMPLETE.json').read_text());assert done['evaluation_split']=='test' and len(done['evaluation_ids'])==14;assert hashlib.sha256((d/'predictions.npz').read_bytes()).hexdigest()==done['predictions_sha256']
 for method,prefix in [('stack_traincontext','stack_'),('retrieval_only','retrieval_only_')]:
  em=json.loads((d/f'{prefix}expression_metrics.json').read_text())['summary'];cm=json.loads((d/f'{prefix}composition_metrics.json').read_text());version='STACK fresh native model; fixed146/21/14, full21 generation PCC selection, train-only top3 context adaptation' if method=='stack_traincontext' else 'matched context retrieval-only baseline; sampling replicates'
  er.append(dict(model=method,seed=seed,version=version,**em));cr.append(dict(method=method,seed=seed,version=version,track='validation_gene_PCC',**{k:cm[k] for k in ['mae_pp','rmse_pp','jsd_bits']}))
 sources.append(dict(seed=seed,train=train,evaluation=done))
e=pd.concat([e[~e.model.isin(['stack_traincontext','retrieval_only'])],pd.DataFrame(er)],ignore_index=True);c=pd.concat([c[~c.method.isin(['stack_traincontext','retrieval_only'])],pd.DataFrame(cr)],ignore_index=True)
e.to_csv(O/'expression_per_seed_all_metrics.csv',index=False);c.to_csv(O/'composition_per_seed_all_metrics.csv',index=False);es=e.groupby('model').agg(runs=('model','size'),MSE=('MSE_raw','mean'),gene_PCC=('PCC_gene_raw','mean'));cs=c.groupby('method').agg(runs=('method','size'),MAE_pp=('mae_pp','mean'),RMSE_pp=('rmse_pp','mean'),JSD_bits=('jsd_bits','mean'));es.to_csv(O/'expression_summary.csv');cs.to_csv(O/'composition_summary.csv')
def md(d):
 d=d.reset_index();rows=['| '+' | '.join(d.columns)+' |','| '+' | '.join(['---']*len(d.columns))+' |']
 for r in d.itertuples(index=False,name=None):rows.append('| '+' | '.join(f'{x:.6f}' if isinstance(x,float) else str(x) for x in r)+' |')
 return '\n'.join(rows)
(O/'REPORT.md').write_text('# Latest available benchmark; STACK unified split updated\n\nSTACK3 seeds:146 training conditions,21 full-generation validation conditions for checkpoint selection,14 final test conditions. Native masked reconstruction+SW objective; fresh model weights; fixed train-context retrieval adaptation. This does not claim identical native tasks or losses across all methods. Other methods retain versions in per-seed tables; any pending PerturbGen/PerturbNet rows remain historical. PDS/Direction Reach completeness is not claimed.\n\n## Expression\n\n'+md(es)+'\n\n## Composition\n\n'+md(cs)+'\n')
v=json.loads((O/'PROVENANCE.json').read_text());v['stack_unified_update']=sources;(O/'PROVENANCE.json').write_text(json.dumps(v,indent=2));(U/'PUBLISHED.json').write_text(json.dumps(dict(output=str(O),seeds=[43,44,45],training_conditions=146,validation_conditions=21,test_conditions=14),indent=2));print('STACK_UNIFIED_ALL_THREE_TEST_RESULTS_VERIFIED_AND_PUBLISHED')
