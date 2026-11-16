from pathlib import Path
import json,csv,hashlib
import numpy as np
B=Path('/liaozizhuo/hypoblast/benchmarks/native-alignment-20261111');rows=[];per=[];sources=[]
for version,suffix in [('original',''),('repaired','_chemvocabfix_20261112')]:
 for seed in [43,44,45]:
  d=B/f'evaluation/chemcpa_seed{seed}{suffix}';done=json.loads((d/'COMPLETE.json').read_text());assert hashlib.sha256((d/'predictions.npz').read_bytes()).hexdigest()==done['prediction_sha256'];a=np.load(d/'predictions.npz');assert a['expression'].shape==(14,2000) and a['proportions'].shape==(14,10);assert np.isfinite(a['expression']).all() and np.allclose(a['proportions'].sum(1),1,atol=1e-12)
  e=json.loads((d/'expression_metrics.json').read_text());c=json.loads((d/'composition_metrics.json').read_text());rows.append(dict(version=version,seed=seed,MSE_raw=e['summary']['MSE_raw'],PCC_gene_raw=e['summary']['PCC_gene_raw'],mae_pp=c['mae_pp'],rmse_pp=c['rmse_pp'],jsd_bits=c['jsd_bits']))
  for sid,ee,mae in zip(a['sample_ids'].tolist(),e['per_condition'],c['per_condition_mae_pp']):per.append(dict(version=version,seed=seed,sample_id=sid,MSE_raw=ee['MSE_raw'],mae_pp=mae))
  sources.append(done)
O=B/'chemcpa_repaired_final';O.mkdir(exist_ok=True)
for name,data in [('before_after.csv',rows),('per_condition.csv',per)]:
 with (O/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
s='# chemCPA chemical-vocabulary normalization repair\n\nRoot causes and causal validation evidence: ../CHEMCPA_ROOT_CAUSE_AND_FIX.md.\n\n| Version | Seed | MSE | Gene PCC | Composition MAE (pp) |\n|---|---|---|---|---|\n'
for r in rows:s+=f"| {r['version']} | {r['seed']} | {r['MSE_raw']:.8f} | {r['PCC_gene_raw']:.6f} | {r['mae_pp']:.6f} |\n"
s+='\nUnchanged146/21/14 split, 2000 fixed genes, frozen10-class V7 readout, identical training seeds. Selection uses full21-condition validation gene PCC; test results are evaluated only after training. All matched control cells used for expression;5000 cells x3 sampling repeats for composition. No seed deletion or test-based checkpoint selection. Both old and new outputs retained.\n'
(O/'REPORT.md').write_text(s);(O/'VERIFICATION.json').write_text(json.dumps(dict(status='all_three_repaired_seeds_test_evaluated',runs=6,condition_seed_rows=len(per),provenance=sources),indent=2));print(json.dumps(rows,indent=2))
