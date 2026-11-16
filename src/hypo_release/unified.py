"""Shared-configuration relation training and evaluation across Flow seeds."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import argparse,ast,importlib.util,json,itertools,math,hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.linalg import solve
from hypo_release.paths import ROOT as R
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32=False

def read(p):return json.loads(Path(p).read_text())
def save(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2))
def helpers():
 p=R/'src/hypo_release/support/pathway_celltype_aux_experiment.py'
 spec=importlib.util.spec_from_file_location('v741_context_aux',p);aux=importlib.util.module_from_spec(spec);__import__('sys').modules[spec.name]=aux;spec.loader.exec_module(aux)
 return aux
def functions():
 from hypo_release import shared_math
 return {name:getattr(shared_math,name) for name in ['kernel','fitpredict','combine','evaluate']}
def examples(seed):
 helpers();return torch.load(R/f'inputs/examples_seed{seed}.pt',weights_only=False)
def extract(seed):
 from stage2.runners.evaluate_prototype_mass_flow import load_evaluation_module,select_split_records
 aux=helpers();tr,te,paths,groups=examples(seed)
 dest=R/f'features/seed{seed}';dest.mkdir(parents=True,exist_ok=True)
 assert not (dest/'COMPLETE.json').exists()
 payload=torch.load(R/'assets/pair_records.pt',weights_only=True);lookup={str(r['sample_id']):r for r in payload['records']}
 ck=torch.load(R/f'weights/seed{seed}.pt',map_location='cpu',weights_only=True)
 module=load_evaluation_module(ck['config'],ck,payload,R/'assets/pathway_base.pt',torch.device('cuda'));module.eval();module.adapter.endpoint_rollout_steps=20
 classes=ck['celltype_classes'];mapping=module.adapter.prototype_to_celltype.cuda().float()
 contexts={}
 for r in select_split_records(payload['records'],R/'assets/context_split.csv',split='train'):
  ids=tuple(sorted(map(int,r.get('original_drug_ids',r['drug_ids']))))
  if len(ids)==2 and len(set(ids))==2:contexts.setdefault((str(r['control_sample_id']),ids),dict(r,drug_ids=list(ids)))
 contexts=list(contexts.values());assert len(contexts)==64
 with torch.inference_mode():
  for strength in [.25,.75,1.5]:
   fs=[]
   for i,e in enumerate(tr+te):
    f=aux.make_feature(module,lookup[e.sample_id],classes.index(e.target_celltype),paths,mapping,strength);f[34:51]=0;fs.append(f)
   np.save(dest/f'panel_{strength}.npy',torch.stack(fs).numpy())
   fs=[]
   for r in contexts:
    f=aux.make_feature(module,r,classes.index('Hypoblast-like cell'),paths,mapping,strength);f[34:51]=0;fs.append(f)
   np.save(dest/f'six_{strength}.npy',torch.stack(fs).numpy())
   print(seed,'FEATURES',strength,flush=True)
 save(dest/'COMPLETE.json',{'seed':seed,'train_ids':[e.sample_id for e in tr],'holdout_ids':[e.sample_id for e in te],'contexts':64,'fresh_forward':True})
def features(seed,kind='panel'):
 d=R/f'features/seed{seed}';a={k:np.load(d/f'{kind}_{v}.npy').astype(float) for k,v in [('low',.25),('original',.75),('high',1.5)]};a['average']=(a['low']+a['original']+a['high'])/3;return a
def training_utility(y,logits):
 a=torch.tensor(logits).float().reshape(-1,17,2);s,d=a.max(2);order=s.argsort(dim=1,descending=True)
 yy=torch.tensor(y).reshape(-1,17,2);counts=yy.sum((1,2)).long()
 ranked_truth=yy.gather(2,d[:,:,None]).squeeze(2).gather(1,order)
 chosen=torch.arange(17)[None,:]<counts[:,None]
 hits=(ranked_truth*chosen).sum(1)
 return float(hits.sum())/int(counts.sum())+.2*int((hits==counts).sum())/len(counts)
def select():
 assert not (R/'configs/SHARED_CONFIG.json').exists(),'locked already'
 fn=functions();data={};common=None
 for seed in [43,44,45]:
  tr,te,paths,groups=examples(seed);signature=([e.sample_id for e in tr],groups)
  if common is None:common=signature
  assert signature==common
  unique=sorted(set(groups));np.random.default_rng(904).shuffle(unique);gf={g:i%5 for i,g in enumerate(unique)};fold=np.array([gf[g] for g in groups])
  data[seed]=(tr,paths,fold,features(seed),np.stack([e.label.numpy() for e in tr]).astype(float))
 save(R/'SHARED_FOLDS.json',{'train_ids':common[0],'groups':common[1],'folds':fold.tolist()})
 configs=[dict(length=l,state=s,sharing=b,alpha=a) for l in [.2,1.,5.] for s in [0.,1.] for b in [.05,.25,1.] for a in [.01,.1,1.]]
 grid=[];best=None
 for variant in ['original','low','high','average']:
  for cfg in configs:
   oofs={}
   for seed,(tr,paths,fold,fs,y) in data.items():
    x=fs[variant][:len(tr)];pred=np.zeros_like(y)
    for f in range(5):
     ti=fold!=f;vi=fold==f;pred[vi],_=fn['fitpredict'](x[ti],y[ti],x[vi],cfg)
    oofs[seed]=pred
   for ia,da in itertools.product([0,.25,.5,.75,1.],repeat=2):
    utilities=[]
    for seed,(tr,paths,fold,fs,y) in data.items():
     pred=fn['combine'](oofs[seed],fs[variant][:len(tr)],ia,da)
     utilities.append(training_utility(y,pred))
    c={'variant':variant,'kernel':cfg,'identity_alpha':ia,'direction_alpha':da,'training_oof_utility':utilities,'mean_training_oof_utility':float(np.mean(utilities))}
    grid.append(c)
    if best is None or c['mean_training_oof_utility']>best['mean_training_oof_utility']:
     best=c;best_oof={s:fn['combine'](oofs[s],data[s][3][variant][:len(data[s][0])],ia,da) for s in data}
  print('SELECTED GRID',variant,flush=True)
 save(R/'training_selection/RELATION_GRID.json',grid)
 # Lock shared relation settings BEFORE evaluating any held-out labels.
 save(R/'configs/SHARED_CONFIG.json',best)
 for seed in data:
  dest=R/f'models/seed{seed}';dest.mkdir(parents=True,exist_ok=True);np.save(dest/'oof_logits.npy',best_oof[seed])
 print('LOCKED',best,flush=True)
def metric(items,logits,mask,paths):
 from hypo_release.shared_metrics import metric as score
 return score(items,np.asarray(logits).reshape(-1,17,2),mask,paths)
def fit_evaluate():
 c=read(R/'configs/SHARED_CONFIG.json');fn=functions()
 for seed in [43,44,45]:
  tr,te,paths,groups=examples(seed);x=features(seed)[c['variant']];n=len(tr);y=np.stack([e.label.numpy() for e in tr])
  dest=R/f'models/seed{seed}';dest.mkdir(parents=True,exist_ok=True)
  assert not (dest/'kernel_model.npz').exists(),'Refuse to overwrite fitted model'
  kp,state=fn['fitpredict'](x[:n],y,x[n:],c['kernel'])
  np.savez(dest/'kernel_model.npz',**state);save(dest/'CONFIG.json',{'relation':c})
  logits=fn['combine'](kp,x[n:],c['identity_alpha'],c['direction_alpha']);np.save(dest/'holdout_logits.npy',logits)
  save(dest/'holdout_rankings.json',[{'sample_id':e.sample_id,'rankings':[{'pathway':paths[int(j)//2],'direction':['+','-'][int(j)%2],'score':float(logits[i,j])} for j in np.argsort(-logits[i],kind='stable')]} for i,e in enumerate(te)])
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['extract','select','fit-evaluate','train','forward']);p.add_argument('--seed',type=int,choices=[43,44,45]);p.add_argument('--output',type=Path);p.add_argument('--epochs',type=int,default=150);a=p.parse_args()
 if a.command=='extract':assert a.seed is not None;extract(a.seed)
 elif a.command=='select':select()
 elif a.command=='fit-evaluate':fit_evaluate()
 else:
  import shutil,sys,subprocess
  env=dict(os.environ,PYTHONPATH=str(R/'src'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
  if a.command=='train':
   assert a.output is not None and not a.output.exists() and a.seed is not None
   subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=1',str(R/'src/stage2/runners/train_prototype_mass_flow.py'),'--config',str(R/'configs/train.py'),'--pair-records',str(R/'assets/pair_records.pt'),'--pathway-base',str(R/'assets/pathway_base.pt'),'--output-dir',str(a.output),'--epochs',str(a.epochs),'--seed',str(a.seed),'--num-workers','0'],env=env,check=True)
  elif a.command=='forward':
   for seed in ([a.seed] if a.seed is not None else [43,44,45]):
    assert not (R/f'forward/seed{seed}').exists(),'Forward outputs already exist'
    common=['--flow-checkpoint',str(R/f'weights/seed{seed}.pt'),'--pair-records',str(R/'assets/pair_records.pt'),'--pathway-base',str(R/'assets/pathway_base.pt')]
    subprocess.run([sys.executable,str(R/'src/hypo_release/pipeline/evaluate_contextual_pathways_v741.py'),*common,'--split-manifest',str(R/'assets/context_split.csv'),'--output-dir',str(R/f'forward/seed{seed}/contextual'),'--context-drug-count','2','--tail-fraction','.10','--activity-strength','.75','--batch-size','24','--n-ode-steps','20','--device','cuda'],env=env,check=True)
    subprocess.run([sys.executable,str(R/'src/stage2/runners/evaluate_prototype_mass_flow.py'),*common,'--config',str(R/'configs/train.py'),'--split-manifest',str(R/'assets/stage2_split.csv'),'--output-dir',str(R/f'forward/seed{seed}/test'),'--strict-test-only','--evaluation-split','test','--device','cuda'],env=env,check=True)
if __name__=='__main__':main()
