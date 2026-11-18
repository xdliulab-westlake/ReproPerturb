from pathlib import Path
import sys,json,csv,hashlib
import torch,numpy as np
B=Path('/liaozizhuo/hypoblast/benchmarks/hypo-forward-embedding-combo-20261029');D=B.parent/'hypo-expression-pcc-20261117';seed=int(sys.argv[1]);out=D/f'seed{seed}/test'
bench=json.loads((B/'benchmark.json').read_text());truth=dict(np.load(B/'data/truth.npz'));pred=torch.load(out/'expression_prediction.pt',map_location='cpu',weights_only=True).numpy()
with (out/'celltype_proportion_predicted.csv').open() as f:
 rows=list(csv.DictReader(f));ids=[r['sample_id'] for r in rows]
assert len(ids)==14 and set(ids)==set(bench['splits']['test']);idx=[truth['sample_ids'].tolist().index(s) for s in ids];assert pred.shape==(14,2000) and np.isfinite(pred).all()
from metrics import expression_metrics
metrics=expression_metrics(pred,truth['expression'][idx],truth['control_expression'][idx]);(out/'benchmark_expression_metrics.json').write_text(json.dumps(metrics,indent=2))
np.savez(out/'benchmark_predictions.npz',sample_ids=ids,genes=truth['genes'],expression=pred,expression_scale=truth['expression_scale'])
(out/'BENCHMARK_COMPLETE.json').write_text(json.dumps(dict(seed=seed,n_test_conditions=14,n_genes=2000,selection='validation_gene_pcc',checkpoint=str(out.parent/'best_prototype_mass_flow.pt')),indent=2));print('EXPRESSION_TEST_COMPLETE',seed,flush=True)
