from pathlib import Path
import sys,os,json,hashlib,time,fcntl
import numpy as np,torch
from torch import nn
B=Path('/liaozizhuo/hypoblast/benchmarks/hypo-forward-embedding-combo-20261029');D=B.parent/'direct-composition-20261117';seed=int(sys.argv[1]);os.environ['IPSTEM_TRAIN_SEED']=str(seed)
I=B.parent/'native-alignment-20261111';run=I/f'chemcpa_task/runs/chemcpa_seed{seed}_chemvocabfix_20261112';os.environ['IPSTEM_RUN_ROOT']=str(run)
sys.path.insert(0,str(run/'code'));from adversarial_models import ChemAdversarial
out=D/(f'chemcpa_seed{seed}_smoke' if '--smoke' in sys.argv else f'chemcpa_seed{seed}');out.mkdir(exist_ok=True);lock=(out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
if (out/'COMPLETE.json').exists():sys.exit(0)
torch.set_num_threads(8);torch.manual_seed(seed);np.random.seed(seed)
read=lambda p:json.loads(Path(p).read_text());bench=read(B/'benchmark.json');genes=read(B/'config/genes.json');classes=bench['classes'];truth=dict(np.load(B/'data/truth.npz'));targets={s:p for s,p in zip(truth['sample_ids'].tolist(),truth['proportions'])};assert all(s in targets for part in ['train','val','test'] for s in bench['splits'][part])
ix=dict(np.load(B/'data/sample_rows_all.npz'));ctrl={r['sample_id']:r['control_sample_id'] for r in bench['cohort']};x=np.load(read(B/'config/stage2.json')['data']['cell_hvg_npy'],mmap_mode='r');features=dict(np.load(I/'chemcpa_features_all.npz'))
base=ChemAdversarial(genes).cuda();weight=run/'checkpoints/chemcpa/best.pt';ck=torch.load(weight,map_location='cpu',weights_only=False);base.load_state_dict(ck['model']);head=nn.Sequential(nn.Linear(base.protein_embedding.embedding_dim,64),nn.GELU(),nn.Linear(64,10)).cuda()
# Only components on the direct-proportion path are updated; no expression classifier.
params=list(base.encoder.parameters())+list(base.chemical_encoder.parameters())+list(base.protein_embedding.parameters())+list(head.parameters());opt=torch.optim.AdamW(params,lr=5e-5,weight_decay=.01);scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=10)
def latent(s,rows):
 a=torch.tensor(np.array(x[rows]),device='cuda');e=torch.tensor(features[s],device='cuda')[None];mask=torch.ones(e.shape[:2],device='cuda');return base.encoder(a)+base.encode_condition(e,mask)
def predict(s):
 rows=ix[ctrl[s]];total=torch.zeros(base.protein_embedding.embedding_dim,device='cuda')
 for k in range(0,len(rows),1024):total+=latent(s,rows[k:k+1024]).sum(0)
 return head(total/len(rows)).softmax(-1)
def evaluate(ids):
 base.eval();head.eval()
 with torch.no_grad():p=np.stack([predict(s).cpu().numpy() for s in ids])
 t=np.stack([targets[s] for s in ids]);return 100*np.abs(p-t).mean(),p
best=float('inf');stale=0;start=0
if (out/'last.pt').exists():
 q=torch.load(out/'last.pt',map_location='cpu',weights_only=False);base.load_state_dict(q['base']);head.load_state_dict(q['head']);opt.load_state_dict(q['optimizer']);scheduler.load_state_dict(q['scheduler']);start=q['epoch'];best=q['best'];stale=q['stale'];torch.set_rng_state(q['rng']);torch.cuda.set_rng_state(q['cuda_rng'])
(out/'CONFIG.json').write_text(json.dumps(dict(method='chemCPA direct proportion adaptation',seed=seed,source_checkpoint=str(weight),source_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),head='mean native perturbed latent -> 128/64/10 MLP -> softmax',loss='condition-balanced proportion MAE',validation='full controls; 21 conditions; MAE',train_samples_per_condition=1024,trainable='chemCPA encoder + native chemical encoder + protein embedding + independent proportion head',max_epochs=3000,patience=20,shared_readout=False,expression_decoder_used=False),indent=2))
for epoch in range(start+1,3001) if stale<20 else []:
 ts=time.time();base.train();head.train();rng=np.random.default_rng(seed+epoch);ids=list(bench['splits']['train']);rng.shuffle(ids);total=0
 if epoch==1:ids.sort(key=lambda s:not ((features[s][:,194]==0).any() and (features[s][:,194]>0).any()))
 for j,s in enumerate(ids):
  rows=ix[ctrl[s]];rows=rng.choice(rows,1024,replace=len(rows)<1024);opt.zero_grad();p=head(latent(s,rows).mean(0)).softmax(-1);t=torch.tensor(targets[s],device='cuda',dtype=torch.float32);loss=(p-t).abs().mean();loss.backward()
  if epoch==1 and j==0:
   assert sum(p.grad.abs().sum().item() for p in base.encoder.parameters() if p.grad is not None)>0
   assert sum(p.grad.abs().sum().item() for p in base.chemical_encoder.parameters() if p.grad is not None)>0
   assert sum(p.grad.abs().sum().item() for p in head.parameters() if p.grad is not None)>0
   assert base.protein_embedding.weight.grad.abs().sum()>0
   print('GRADIENT_CHECK_PASS encoder chemical_encoder protein_embedding head',flush=True)
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
