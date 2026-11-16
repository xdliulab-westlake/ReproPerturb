"""Incremental frozen test evaluation of native-input G2CP and chemCPA."""
from pathlib import Path
import argparse,os,sys,json,hashlib
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--model',choices=['cellflow'],required=True);p.add_argument('--seed',type=int,choices=[43,44,45],required=True);a=p.parse_args();name=a.model;seed=a.seed
B=Path('/liaozizhuo/hypoblast/benchmarks/native-alignment-20261111');OLD=B.parent/'hypo-forward-embedding-combo-20261029';EXT=B.parent/'hypo-benchmark-extension-20261105';run=B/f'{name}_task/runs/{name}_seed{seed}';dest=B/f'evaluation/{name}_seed{seed}';dest.mkdir(parents=True,exist_ok=True)
assert not (dest/'COMPLETE.json').exists(),'Already complete'
os.environ['IPSTEM_RUN_ROOT']=str(run);os.environ['IPSTEM_TRAIN_SEED']=str(seed);sys.path.insert(0,str(run/'code'));sys.path.append(str(OLD));sys.path.append(str(EXT/'code'));sys.path.append(str(EXT))
from common import read,write,sha

import importlib.util
original_contract=sys.modules.get('contract')
spec=importlib.util.spec_from_file_location('contract',OLD/'contract.py');metric_contract=importlib.util.module_from_spec(spec);spec.loader.exec_module(metric_contract);sys.modules['contract']=metric_contract
from metrics import expression_metrics
if original_contract is not None:sys.modules['contract']=original_contract
else:sys.modules.pop('contract',None)
from composition_metrics import metrics as composition_metrics
import jax
import jax.numpy as jnp
from scipy.special import softmax
from cellflow_native import make_solver,condition,predictor
from train_cellflow import restore
cfg=read(OLD/'config/stage2.json');benchmark=read(OLD/'benchmark.json');training=read(run/'training.json');genes=read(OLD/'config/genes.json');classes=benchmark['classes'];ids=benchmark['splits']['test'];assert len(ids)==14
weight=run/'checkpoints/cellflow/best.msgpack';done=read(weight.parent/'COMPLETE.json');assert sha(weight)==done['best_sha256'] and done['train_seed']==seed
prov=read(run/'PROVENANCE.json')
for filename,h in prov['code_sha256'].items():assert sha(run/'code'/filename)==h,filename
solver=make_solver();restore(weight,solver,jax.random.split(jax.random.PRNGKey(seed),4));predict=predictor(solver)
def infer(rows,cid):
    batch=np.zeros((1024,2000),dtype=np.float32);batch[:len(rows)]=x[rows]
    out=np.asarray(predict(solver.vf_state.params,jnp.asarray(batch),condition(features[cid])))[:len(rows)]
    assert np.isfinite(out).all()
    return out
indices=dict(np.load(OLD/'data/sample_rows_all.npz'));features=dict(np.load(training['event_features_all_file']));controls={r['sample_id']:r['control_sample_id'] for r in benchmark['cohort']};x=np.load(cfg['data']['cell_hvg_npy'],mmap_mode='r');classifier=dict(np.load(EXT/'results/readout.npz'));assert list(classifier['genes'])==genes and list(classifier['classes'])==classes
coef=classifier['coef']/classifier['scale'];bias=classifier['intercept']-classifier['mean']@coef.T
means=[];soft=np.zeros((3,14,10));hard=np.zeros_like(soft);sampling=[101,102,103]
for j,cid in enumerate(ids):
 ix=indices[controls[cid]];total=np.zeros(2000)
 for start in range(0,len(ix),1024):
  pred=infer(ix[start:start+1024],cid);total+=pred.sum(0,dtype=np.float64)
 means.append(total/len(ix))
 for rr,rs in enumerate(sampling):
  rng=np.random.default_rng(rs+int(hashlib.sha256(cid.encode()).hexdigest()[:8],16));selected=rng.choice(ix,5000,replace=True)
  for start in range(0,5000,1024):
   pred=infer(selected[start:start+1024],cid);probs=softmax(pred@coef.T+bias,axis=-1)
   soft[rr,j]+=probs.sum(0,dtype=np.float64)/5000;hard[rr,j]+=np.bincount(probs.argmax(-1),minlength=10)/5000
 print('EVALUATED',name,seed,cid,flush=True)
means=np.stack(means);prop=soft.mean(0);truth=dict(np.load(OLD/'data/truth.npz'));assert truth['genes'].tolist()==genes;ix=[truth['sample_ids'].tolist().index(i) for i in ids];assert np.isfinite(means).all() and means.shape==(14,2000)
em=expression_metrics(means,truth['expression'][ix],truth['control_expression'][ix]);cm=composition_metrics(prop,truth['proportions'][ix]);np.savez_compressed(dest/'predictions.npz',sample_ids=np.array(ids),genes=np.array(genes),classes=np.array(classes),expression=means,proportions=prop,soft_sampling=soft,hard_sampling=hard)
write(dest/'expression_metrics.json',em);write(dest/'composition_metrics.json',cm)
assert sha(weight)==done['best_sha256'];assert abs(cm['mae_pp']-100*np.abs(prop-truth['proportions'][ix]).mean())<1e-12
write(dest/'COMPLETE.json',dict(status='complete',method=name,seed=seed,test_ids=ids,checkpoint=str(weight),checkpoint_sha256=sha(weight),prediction_sha256=sha(dest/'predictions.npz'),truth_sha256=sha(OLD/'data/truth.npz'),readout_sha256=sha(EXT/'results/readout.npz'),feature_sha256=sha(training['event_features_all_file']),script_sha256=sha(__file__),expression_all_control_cells=True,composition_sampling_seeds=sampling,cells_per_condition_per_repeat=5000,training_updates=0,expression_scale=benchmark['expression_scale'],model_provenance=prov))
print('COMPLETE',name,seed,em['summary']['MSE_raw'],cm['mae_pp'],flush=True)
