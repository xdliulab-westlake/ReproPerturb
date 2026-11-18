from pathlib import Path
import sys,os,json,hashlib,time,fcntl
import numpy as np,torch
from torch import nn
B=Path('/liaozizhuo/hypoblast/benchmarks/hypo-forward-embedding-combo-20261029');D=B.parent/'direct-composition-20261117';seed=int(sys.argv[1]);os.environ['IPSTEM_TRAIN_SEED']=str(seed)
I=B.parent/'native-alignment-20261111';run=I/f'perturbnet_task/runs/perturbnet_seed{seed}/conditional_flow_official_adam'
sys.path.insert(0,'/liaozizhuo/iPStem/benchmark/models/perturbnet/repo')
from perturbnet.net2net.modules.flow.flatflow import ConditionalFlatCouplingFlow
import csv
out=D/(f'perturbnet_seed{seed}_smoke' if '--smoke' in sys.argv else f'perturbnet_seed{seed}');out.mkdir(exist_ok=True);lock=(out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
if (out/'COMPLETE.json').exists():sys.exit(0)
torch.set_num_threads(8);torch.manual_seed(seed);np.random.seed(seed)
read=lambda p:json.loads(Path(p).read_text());bench=read(B/'benchmark.json');genes=read(B/'config/genes.json');classes=bench['classes'];truth=dict(np.load(B/'data/truth.npz'));targets={s:p for s,p in zip(truth['sample_ids'].tolist(),truth['proportions'])};assert all(s in targets for part in ['train','val','test'] for s in bench['splits'][part])
ix=dict(np.load(B/'data/sample_rows_all.npz'));ctrl={r['sample_id']:r['control_sample_id'] for r in bench['cohort']};cohort={r['sample_id']:r for r in bench['cohort']}
routes=list(csv.DictReader((B.parent/'hypo-native-benchmark-20261110/native_inputs/drug_routes.csv').open()));chem=[r['name'] for r in routes if r['type']!='protein'];proteins=sorted(r['name'] for r in routes if r['type']=='protein')
cp=np.load(run/'chemical_posterior.npz');mu,sd,sm,ss=[torch.tensor(cp[k],device='cuda') for k in ['mu','sd','standardization_mean','standardization_sd']]
base=ConditionalFlatCouplingFlow(in_channels=10,conditioning_dim=204,embedding_dim=10,hidden_dim=1024,hidden_depth=2,n_flows=20,activation='none',conditioner_use_bn=True).cuda()
weight=run/'best.pt';assert (run/'FLOW_COMPLETE.json').exists();base.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True));head=nn.Sequential(nn.Linear(10,64),nn.GELU(),nn.Linear(64,10)).cuda()
params=list(base.parameters())+list(head.parameters());opt=torch.optim.AdamW([{'params':base.parameters(),'lr':4.5e-6},{'params':head.parameters(),'lr':5e-5}],weight_decay=.01);scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=10)
def latent(s,rows):
 n=len(rows);drugs=set(cohort[s]['drugs'].split(' + '));mask=torch.tensor([float(v in drugs) for v in chem],device='cuda')
 c=torch.zeros(n,204,device='cuda');c[:,:196]=(((mu[None]+sd[None]*torch.randn(n,len(chem),196,device='cuda')-sm)/ss)*mask[None,:,None]).sum(1)
 c[:,196:]=torch.tensor([float(v in drugs) for v in proteins],device='cuda')
 return base.sample(c).reshape(n,10)
def predict(s):
 rows=ix[ctrl[s]];total=torch.zeros(10,device='cuda')
 # Common random draws across epochs; seed derived from condition, never target values.
 with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
  torch.manual_seed(seed+int(hashlib.sha256(s.encode()).hexdigest()[:8],16))
  for k in range(0,len(rows),1024):total+=latent(s,rows[k:k+1024]).sum(0)
 return head(total/len(rows)).softmax(-1)
def evaluate(ids):
 base.eval();head.eval()
 with torch.no_grad():p=np.stack([predict(s).cpu().numpy() for s in ids])
 assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-6)
 t=np.stack([targets[s] for s in ids]);return 100*np.abs(p-t).mean(),p
best=float('inf');stale=0;start=0
if (out/'last.pt').exists():
 q=torch.load(out/'last.pt',map_location='cpu',weights_only=False);base.load_state_dict(q['base']);head.load_state_dict(q['head']);opt.load_state_dict(q['optimizer']);scheduler.load_state_dict(q['scheduler']);start=q['epoch'];best=q['best'];stale=q['stale'];torch.set_rng_state(q['rng']);torch.cuda.set_rng_state(q['cuda_rng'])
(out/'CONFIG.json').write_text(json.dumps(dict(method='PerturbNet direct proportion supervised adaptation',seed=seed,source_checkpoint=str(weight),source_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),chemical_posterior_sha256=hashlib.sha256((run/'chemical_posterior.npz').read_bytes()).hexdigest(),head='mean generated native 10D latent -> 10/64/10 MLP -> softmax',loss='condition-balanced proportion MAE',validation='21 conditions; fixed random draws; number of samples equals all matched control cells; MAE',train_samples_per_condition=1024,trainable='native conditional Flow including conditioner + independent proportion head',scvi='frozen; encoder and expression decoder unused by direct conditional generation',control_usage='sample count only; native unconditional-noise to intervention-conditioned latent route',max_epochs=3000,patience=20,backbone_lr=4.5e-6,head_lr=5e-5,batchnorm_running_stats="frozen for single-condition adaptation; affine parameters trainable",weight_decay=.01,shared_readout=False,expression_decoder_used=False,train_conditions=146,val_conditions=21,test_conditions=14),indent=2))
assert len(bench['splits']['train'])==146 and len(bench['splits']['val'])==21 and len(bench['splits']['test'])==14
assert not set(bench['splits']['train'])&set(bench['splits']['val']+bench['splits']['test'])
for epoch in range(start+1,3001) if stale<20 else []:
 ts=time.time();base.train();head.train()
 for mod in base.modules():
  if isinstance(mod,nn.BatchNorm1d):mod.eval()
 rng=np.random.default_rng(seed+epoch);ids=list(bench['splits']['train']);rng.shuffle(ids);total=0
 for j,s in enumerate(ids):
  rows=ix[ctrl[s]];rows=rng.choice(rows,1024,replace=len(rows)<1024);opt.zero_grad();p=head(latent(s,rows).mean(0)).softmax(-1);t=torch.tensor(targets[s],device='cuda',dtype=torch.float32);loss=(p-t).abs().mean();assert torch.isfinite(loss);loss.backward()
  if epoch==1 and j==0:
   grads={n:float(v.grad.abs().sum()) for n,v in base.named_parameters() if v.grad is not None}
   assert all(np.isfinite(v) for v in grads.values()) and sum(grads.values())>0
   assert sum(p.grad.abs().sum().item() for p in head.parameters() if p.grad is not None)>0
   print('GRADIENT_CHECK_PASS native_flow_and_head',list(grads)[:6],flush=True)
   if '--smoke' in sys.argv:
    base.eval();head.eval()
    with torch.no_grad():q=head(latent(s,rows).mean(0)).softmax(-1)
    assert torch.isfinite(q).all() and abs(q.sum().item()-1)<1e-6
    (out/'SMOKE_PASS.json').write_text(json.dumps(dict(shared_readout=False,gradient_check=True,finite_simplex=True)))
    print('SMOKE_PASS',flush=True);sys.exit(0)
  opt.step();total+=loss.item()
 score,p=evaluate(bench['splits']['val']);scheduler.step(score)
 if score<best:best=score;stale=0;torch.save(dict(base=base.state_dict(),head=head.state_dict(),epoch=epoch),out/'best.pt')
 else:stale+=1
 torch.save(dict(base=base.state_dict(),head=head.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,best=best,stale=stale,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),out/'last.partial.pt');os.replace(out/'last.partial.pt',out/'last.pt')
 print(json.dumps(dict(epoch=epoch,train_mae_pp=total/len(ids)*100,val_mae_pp=float(score),best=float(best),stale=stale,seconds=time.time()-ts)),flush=True)
 if stale>=20:break
q=torch.load(out/'best.pt',weights_only=False);base.load_state_dict(q['base']);head.load_state_dict(q['head']);ids=bench['splits']['test'];mae,p=evaluate(ids);t=np.stack([targets[s] for s in ids]);assert p.shape==(14,10) and np.allclose(p.sum(1),1,atol=1e-6)
np.savez(out/'predictions.npz',sample_ids=ids,classes=classes,proportions=p,truth=t)
from direct_metrics import score as metrics
m=metrics(p,t);m['dominant_accuracy_pct']=float((p.argmax(1)==t.argmax(1)).mean()*100);(out/'metrics.json').write_text(json.dumps(m,indent=2));(out/'COMPLETE.json').write_text(json.dumps(dict(seed=seed,best_epoch=q['epoch'],test_conditions=14,shared_readout=False,mae_pp=float(mae))))
print('TEST_COMPLETE',seed,mae,flush=True)
