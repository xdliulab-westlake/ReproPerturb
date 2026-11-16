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
 from experiments.perturbation.evaluate_prototype_mass_flow import load_evaluation_module,select_split_records
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
 assert not (R/'SHARED_CONFIG.json').exists(),'locked already'
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
 save(R/'SHARED_CONFIG.json',best)
 candidates=[]
 for alpha,t in itertools.product([0,.25,.5,.75,1.,1.5,2.],[.1,.2,.3,.4,.5]):
  us=[]
  for seed,(tr,paths,fold,fs,y) in data.items():
   logits=best_oof[seed].reshape(-1,17,2);score=logits.max(2);z=(score-score.mean(1,keepdims=True))/np.maximum(score.std(1,ddof=1,keepdims=True),1e-5)
   mask=np.zeros_like(z,dtype=bool)
   for f in range(5):
    ti=np.flatnonzero(fold!=f);vi=np.flatnonzero(fold==f)
    for i in vi:
     pool=[j for j in ti if tr[j].target_celltype==tr[i].target_celltype] or list(ti)
     freq=np.mean([[p in tr[j].truth for p in paths] for j in pool],0)
     mask[i]=z[i]+alpha*np.log((freq+.05)/(1-freq+.05))>=t
   mm=metric(tr,logits,mask,paths);us.append(mm['signed_f1']+.15*mm['macro_exact'])
  candidates.append({'alpha':alpha,'threshold':t,'training_oof_utility':us,'mean_training_oof_utility':float(np.mean(us))})
 prior=max(candidates,key=lambda c:c['mean_training_oof_utility']);save(R/'SHARED_UNKNOWN_K.json',prior);save(R/'training_selection/PRIOR_GRID.json',candidates)
 for seed in data:
  dest=R/f'models/seed{seed}';dest.mkdir(parents=True,exist_ok=True);np.save(dest/'oof_logits.npy',best_oof[seed])
 print('LOCKED',best,prior,flush=True)
def metric(items,logits,mask,paths):
 from hypo_release.shared_metrics import metric as score
 return score(items,np.asarray(logits).reshape(-1,17,2),mask,paths)
def fit_evaluate():
 assert not (R/'RESULTS.json').exists(),'Refuse to overwrite completed results; prepare a fresh directory'
 c=read(R/'SHARED_CONFIG.json');u=read(R/'SHARED_UNKNOWN_K.json');fn=functions();results={}
 for seed in [43,44,45]:
  tr,te,paths,groups=examples(seed);x=features(seed)[c['variant']];n=len(tr);y=np.stack([e.label.numpy() for e in tr])
  kp,state=fn['fitpredict'](x[:n],y,x[n:],c['kernel']);dest=R/f'models/seed{seed}'
  np.savez(dest/'kernel_model.npz',**state);save(dest/'CONFIG.json',{'relation':c,'unknown_k':u})
  logits=fn['combine'](kp,x[n:],c['identity_alpha'],c['direction_alpha']);np.save(dest/'holdout_logits.npy',logits)
  known=fn['evaluate'](te,torch.tensor(logits).float(),paths)
  a=logits.reshape(-1,17,2);s=a.max(2);z=(s-s.mean(1,keepdims=True))/np.maximum(s.std(1,ddof=1,keepdims=True),1e-5)
  for i,e in enumerate(te):
   pool=[t for t in tr if t.target_celltype==e.target_celltype] or tr
   freq=np.mean([[p in t.truth for p in paths] for t in pool],0);z[i]+=u['alpha']*np.log((freq+.05)/(1-freq+.05))
  unknown=metric(te,a,z>=u['threshold'],paths)
  save(R/f'artifacts/unknown_k/seed{seed}/RESULTS.json',{'alpha':u['alpha'],'threshold':u['threshold'],'holdout':unknown})
  raw=features(seed,'six')[c['variant']];xx=(raw-state['mean'])/state['std'];xx[:,61:]=raw[:,61:]
  kp=fn['kernel'](xx,state['train'],c['kernel'])@state['weight'];ll=fn['combine'](kp,raw,c['identity_alpha'],c['direction_alpha'])
  score=np.sort(ll,axis=0)[-math.ceil(.1*len(ll)):].mean(0).reshape(17,2)
  # Same direct-Flow direction route for all seeds; inherited task-aware rule.
  import csv
  dirs={r['pathway']:r['predicted_direction'] for r in csv.DictReader((R/f'forward/seed{seed}/contextual/pathway_ranking.csv').open())}
  order=sorted(range(17),key=lambda j:(-float(score[j].max()),paths[j]))
  ranking=[{'pathway':paths[j],'direction':dirs[paths[j]],'score':float(score[j].max())} for j in order]
  save(dest/'SIX_PREDICTION_BEFORE_GT.json',{'selected':ranking[:6],'rankings':ranking})
  gt=read(R/'experiments/combination_growth/data/hypoblast_evaluation_only.json');gt={p:'+' if s>0 else '-' for p,s in gt.items()}
  six={'selected':ranking[:6],'rankings':ranking,'signed_hits_at_6':sum(gt.get(r['pathway'])==r['direction'] for r in ranking[:6])}
  results[seed]={'known_k':known,'unknown_k':unknown,'hypoblast':six};print(seed,'RESULT',known['celltypes_passed'],unknown['any_protocol_celltypes'],six['signed_hits_at_6'],flush=True)
 save(R/'RESULTS.json',results)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['extract','select','fit-evaluate','prepare','train','forward','report','verify']);p.add_argument('--seed',type=int,choices=[43,44,45]);p.add_argument('--output',type=Path);p.add_argument('--epochs',type=int,default=150);a=p.parse_args()
 if a.command=='extract':assert a.seed is not None;extract(a.seed)
 elif a.command=='select':select()
 elif a.command=='fit-evaluate':fit_evaluate()
 elif a.command=='report':
  from hypo_release.report import main as report;report()
 elif a.command=='verify':
  from hypo_release.verify import main as verify;verify()
 else:
  import shutil,sys,subprocess
  env=dict(os.environ,PYTHONPATH=str(R/'src'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
  if a.command=='prepare':
   assert a.output is not None and not a.output.exists(),'Use a new --output directory'
   a.output.mkdir(parents=True)
   for name in ['src','assets','weights','configs','inputs']:shutil.copytree(R/name,a.output/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
   for name in ['code','data']:shutil.copytree(R/'experiments/combination_growth'/name,a.output/'experiments/combination_growth'/name)
   for name in ['hypo.py','PROTOCOL.json','README.rst','MODEL_CARD.txt','LICENSE_NOTICE.txt','requirements.txt']:shutil.copy2(R/name,a.output/name)
   shutil.copy2(R/'experiments/combination_growth/run.py',a.output/'experiments/combination_growth/run.py')
   print('Prepared fresh rerun directory',a.output)
  elif a.command=='train':
   assert a.output is not None and not a.output.exists() and a.seed is not None
   subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=1',str(R/'src/experiments/perturbation/train_prototype_mass_flow.py'),'--config',str(R/'configs/train.py'),'--pair-records',str(R/'assets/pair_records.pt'),'--pathway-base',str(R/'assets/pathway_base.pt'),'--output-dir',str(a.output),'--epochs',str(a.epochs),'--seed',str(a.seed),'--num-workers','0'],env=env,check=True)
  elif a.command=='forward':
   for seed in ([a.seed] if a.seed is not None else [43,44,45]):
    assert not (R/f'forward/seed{seed}').exists(),'Forward outputs already exist'
    common=['--flow-checkpoint',str(R/f'weights/seed{seed}.pt'),'--pair-records',str(R/'assets/pair_records.pt'),'--pathway-base',str(R/'assets/pathway_base.pt')]
    subprocess.run([sys.executable,str(R/'src/hypo_release/pipeline/evaluate_contextual_pathways_v741.py'),*common,'--split-manifest',str(R/'assets/context_split.csv'),'--output-dir',str(R/f'forward/seed{seed}/contextual'),'--context-drug-count','2','--tail-fraction','.10','--activity-strength','.75','--batch-size','24','--n-ode-steps','20','--device','cuda'],env=env,check=True)
    subprocess.run([sys.executable,str(R/'src/experiments/perturbation/evaluate_prototype_mass_flow.py'),*common,'--config',str(R/'configs/train.py'),'--split-manifest',str(R/'assets/stage2_split.csv'),'--output-dir',str(R/f'forward/seed{seed}/test'),'--strict-test-only','--evaluation-split','test','--device','cuda'],env=env,check=True)
if __name__=='__main__':main()
