"""Incremental frozen test evaluation of native-input G2CP and chemCPA."""
from pathlib import Path
import argparse,os,sys,json,hashlib
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--model',choices=['state'],required=True);p.add_argument('--seed',type=int,choices=[43,44,45],required=True);a=p.parse_args();name=a.model;seed=a.seed
B=Path('/liaozizhuo/hypoblast/benchmarks/hypo-independent-intervention-20261112');OLD=B.parent/'hypo-forward-embedding-combo-20261029';EXT=B.parent/'hypo-benchmark-extension-20261105';run=B/f'{name}_task/runs/{name}_seed{seed}';dest=B/f'evaluation/{name}_seed{seed}';dest.mkdir(parents=True,exist_ok=True)
assert not (dest/'COMPLETE.json').exists(),'Already complete'
os.environ['IPSTEM_RUN_ROOT']=str(run);os.environ['IPSTEM_TRAIN_SEED']=str(seed);sys.path.insert(0,str(run/'code'));sys.path.append(str(OLD));sys.path.append(str(EXT/'code'));sys.path.append(str(EXT))
from common import read,write,sha
from train_torch import pad
import importlib.util
original_contract=sys.modules.get('contract')
spec=importlib.util.spec_from_file_location('contract',OLD/'contract.py');metric_contract=importlib.util.module_from_spec(spec);spec.loader.exec_module(metric_contract);sys.modules['contract']=metric_contract
from metrics import expression_metrics
if original_contract is not None:sys.modules['contract']=original_contract
else:sys.modules.pop('contract',None)
from composition_metrics import metrics as composition_metrics
import torch
torch.set_num_threads(2);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
device=torch.device('cuda');cfg=read(OLD/'config/stage2.json');benchmark=read(OLD/'benchmark.json');training=read(run/'training.json');genes=read(OLD/'config/genes.json');classes=benchmark['classes'];ids=benchmark['splits']['test'];assert len(ids)==14
weight=run/f'checkpoints/{name}/best.pt';done=read(weight.parent/'COMPLETE.json');assert sha(weight)==done['best_sha256'] and done['train_seed']==seed
prov=read(run/'PROVENANCE.json')
for filename,h in prov['code_sha256'].items():assert sha(run/'code'/filename)==h,filename
if name=='chemcpa':
 from adversarial_models import ChemAdversarial
 model=ChemAdversarial(genes).to(device)
else:
 from torch_models import build
 model=build(name,genes).to(device)
model.load_state_dict(torch.load(weight,map_location=device,weights_only=True)['model'],strict=True);model.eval()
indices=dict(np.load(OLD/'data/sample_rows_all.npz'));features=dict(np.load(training['event_features_all_file']));controls={r['sample_id']:r['control_sample_id'] for r in benchmark['cohort']};x=np.load(cfg['data']['cell_hvg_npy'],mmap_mode='r');classifier=dict(np.load(EXT/'results/readout.npz'));assert list(classifier['genes'])==genes and list(classifier['classes'])==classes
coef=classifier['coef']/classifier['scale'];bias=classifier['intercept']-classifier['mean']@coef.T;ct=torch.tensor(coef,dtype=torch.float64,device=device);bt=torch.tensor(bias,dtype=torch.float64,device=device)
means=[];soft=np.zeros((3,14,10));hard=np.zeros_like(soft);sampling=[101,102,103]
with torch.inference_mode():
 for j,cid in enumerate(ids):
  ix=indices[controls[cid]];events,mask=pad(features,[cid],device);total=np.zeros(2000)
  for start in range(0,len(ix),1024):
   inp=torch.tensor(np.asarray(x[ix[start:start+1024]],dtype=np.float32),device=device)[None];pred=model(inp,events,mask)[0];assert torch.isfinite(pred).all();total+=pred.cpu().numpy().sum(0,dtype=np.float64)
  means.append(total/len(ix))
  for rr,rs in enumerate(sampling):
   rng=np.random.default_rng(rs+int(hashlib.sha256(cid.encode()).hexdigest()[:8],16));selected=rng.choice(ix,5000,replace=True)
   for start in range(0,5000,512):
    inp=torch.tensor(np.asarray(x[selected[start:start+512]],dtype=np.float32),device=device)[None];pred=model(inp,events,mask)[0];assert torch.isfinite(pred).all();probs=torch.softmax(pred.double()@ct.T+bt,dim=-1);soft[rr,j]+=probs.double().sum(0).cpu().numpy()/5000;hard[rr,j]+=torch.bincount(probs.argmax(-1),minlength=10).cpu().numpy()/5000
  print('EVALUATED',name,seed,cid,flush=True)
means=np.stack(means);prop=soft.mean(0);truth=dict(np.load(OLD/'data/truth.npz'));assert truth['genes'].tolist()==genes;ix=[truth['sample_ids'].tolist().index(i) for i in ids];assert np.isfinite(means).all() and means.shape==(14,2000)
print('PROPORTION_RANGE',float(prop.min()),float(prop.max()),'FINITE',bool(np.isfinite(prop).all()),'SUM_ERROR',float(np.abs(prop.sum(1)-1).max()),flush=True)
np.savez_compressed(dest/'readout_diagnostic.npz',proportions=prop,expression=means,sample_ids=np.array(ids))
assert np.isfinite(prop).all() and prop.min()>=-1e-12 and prop.max()<=1+1e-12 and np.max(np.abs(prop.sum(1)-1))<1e-12
# Repair only verified floating-point simplex roundoff; raw values are retained above.
prop=np.clip(prop,0.,1.);prop=prop/prop.sum(1,keepdims=True)
em=expression_metrics(means,truth['expression'][ix],truth['control_expression'][ix]);cm=composition_metrics(prop,truth['proportions'][ix]);np.savez_compressed(dest/'predictions.npz',sample_ids=np.array(ids),genes=np.array(genes),classes=np.array(classes),expression=means,proportions=prop,soft_sampling=soft,hard_sampling=hard)
write(dest/'expression_metrics.json',em);write(dest/'composition_metrics.json',cm)
assert sha(weight)==done['best_sha256'];assert abs(cm['mae_pp']-100*np.abs(prop-truth['proportions'][ix]).mean())<1e-12
write(dest/'COMPLETE.json',dict(status='complete',method=name,seed=seed,test_ids=ids,checkpoint=str(weight),checkpoint_sha256=sha(weight),prediction_sha256=sha(dest/'predictions.npz'),truth_sha256=sha(OLD/'data/truth.npz'),readout_sha256=sha(EXT/'results/readout.npz'),feature_sha256=sha(training['event_features_all_file']),script_sha256=sha(__file__),expression_all_control_cells=True,composition_sampling_seeds=sampling,cells_per_condition_per_repeat=5000,training_updates=0,expression_scale=benchmark['expression_scale'],model_provenance=prov))
print('COMPLETE',name,seed,em['summary']['MSE_raw'],cm['mae_pp'],flush=True)
