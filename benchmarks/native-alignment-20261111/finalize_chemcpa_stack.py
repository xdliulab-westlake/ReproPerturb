from pathlib import Path
import json,csv,hashlib
import numpy as np
B=Path('/liaozizhuo/hypoblast/benchmarks/native-alignment-20261111');rows=[];per=[];sources=[]
for method in ['chemcpa','stack_traincontext']:
 for seed in [43,44,45]:
  d=B/f'evaluation/{method}_seed{seed}';done=json.loads((d/'COMPLETE.json').read_text());p=d/'predictions.npz';h=hashlib.sha256(p.read_bytes()).hexdigest();expected=done.get('prediction_sha256',done.get('predictions_sha256'));assert h==expected
  a=np.load(p);assert a['expression'].shape==(14,2000) and a['proportions'].shape==(14,10);assert np.isfinite(a['expression']).all() and np.allclose(a['proportions'].sum(1),1)
  prefix='' if method=='chemcpa' else 'stack_';e=json.loads((d/f'{prefix}expression_metrics.json').read_text());c=json.loads((d/f'{prefix}composition_metrics.json').read_text())
  rows.append(dict(method=method,seed=seed,MSE_raw=e['summary']['MSE_raw'],PCC_gene_raw=e['summary']['PCC_gene_raw'],composition_mae_pp=c['mae_pp'],composition_rmse_pp=c['rmse_pp'],jsd_bits=c['jsd_bits']))
  for sid,ee,mae in zip(a['sample_ids'].tolist(),e['per_condition'],c['per_condition_mae_pp']):per.append(dict(method=method,seed=seed,sample_id=sid,expression_mse=ee['MSE_raw'],composition_mae_pp=mae))
  sources.append(done)
  if method=='stack_traincontext':
   e=json.loads((d/'retrieval_only_expression_metrics.json').read_text())['summary'];c=json.loads((d/'retrieval_only_composition_metrics.json').read_text());rows.append(dict(method='retrieval_only',seed=seed,MSE_raw=e['MSE_raw'],PCC_gene_raw=e['PCC_gene_raw'],composition_mae_pp=c['mae_pp'],composition_rmse_pp=c['rmse_pp'],jsd_bits=c['jsd_bits']))
O=B/'chemcpa_stack_final';O.mkdir(exist_ok=True)
for name,data in [('test_metrics.csv',rows),('per_condition_metrics.csv',per)]:
 with (O/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
report='# chemCPA and STACK fixed-test evaluation\n\n14 fixed test conditions, 2000 genes, 10 V7 cell types.\n\n| Method | Seed | Expression MSE | Gene PCC | Composition MAE (pp) |\n|---|---|---|---|---|\n'
for r in rows:report+=f"| {r['method']} | {r['seed']} | {r['MSE_raw']:.6f} | {r['PCC_gene_raw']:.6f} | {r['composition_mae_pp']:.6f} |\n"
report+='\nchemCPA seed43 anomaly reproduced exactly on a fresh rerun. Train-only BN recalibration lowered validation gene PCC from 0.07437 to 0.03021 and was rejected under the fixed selection criterion. Original three seeds retained; no test-based re-selection. This is an unstable adaptation result, not evidence of inherent method inferiority.\n\nSTACK: from-scratch official self-supervised model, with top3 Jaccard training-condition prompts, native five-step generation. No target-condition measured expression used for prompting. This is an explicit unseen-combination retrieval adaptation, not the official same-perturbation cross-cell-type task or the public aligned checkpoint. Non-modelled genes held at source-control counts solely for conversion to fixed benchmark expression scale; source-control roundtrip checked per condition. The retrieval-only comparator uses those same sampled training prompt cells.\n\nExpression: all matched control cells. Composition: shared frozen readout, 5000 sampled cells times three sampling seeds 101/102/103. Full per-cell STACK outputs are retained remotely.\n'
report+='\nSTACK internal self-supervised file splits (subsets of the outer training pool, NOT the benchmark14 test): '+json.dumps(json.loads((B/'stack_task/INTERNAL_SPLIT_COUNTS.json').read_text()))+'\nThe outer21 validation conditions were not used for this STACK checkpoint selection. This limitation prevents claiming a uniformly trained native-method comparison.\n'
(O/'REPORT.md').write_text(report);(O/'VERIFICATION.json').write_text(json.dumps(dict(runs=6,conditions_per_run=14,condition_seed_rows=len(per),all_predictions_finite=True,provenance=sources),indent=2))
print(json.dumps(rows,indent=2))
