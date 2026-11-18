from pathlib import Path
import sys,os,json,hashlib,time,fcntl
import numpy as np,torch
from torch import nn
B=Path('/liaozizhuo/hypoblast/benchmarks/hypo-forward-embedding-combo-20261029');D=B.parent/'direct-composition-20261117';seed=int(sys.argv[1]);os.environ['IPSTEM_TRAIN_SEED']=str(seed)
I=B.parent/'stack-unified-20261113';run=I/f'runs/seed{seed}';os.environ['IPSTEM_RUN_ROOT']=str(run)
sys.path.insert(0,'/liaozizhuo/iPStem/benchmark/models/stack/repo/src');from stack.models.core import StateICLModel
import anndata as ad
from torch.utils.checkpoint import checkpoint
out=D/(f'stack_seed{seed}_smoke' if '--smoke' in sys.argv else f'stack_seed{seed}');out.mkdir(exist_ok=True);lock=(out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
if (out/'COMPLETE.json').exists():sys.exit(0)
torch.set_num_threads(8);torch.manual_seed(seed);np.random.seed(seed)
read=lambda p:json.loads(Path(p).read_text());bench=read(B/'benchmark.json');genes=read(B/'config/genes.json');classes=bench['classes'];truth=dict(np.load(B/'data/truth.npz'));targets={s:p for s,p in zip(truth['sample_ids'].tolist(),truth['proportions'])};assert all(s in targets for part in ['train','val','test'] for s in bench['splits'][part])
ix=dict(np.load(B/'data/sample_rows_all.npz'));ctrl={r['sample_id']:r['control_sample_id'] for r in bench['cohort']};x=np.load(read(B/'config/stage2.json')['data']['cell_hvg_npy'],mmap_mode='r')
weight=run/'best.ckpt';ck=torch.load(weight,map_location='cpu',weights_only=False);base=StateICLModel(**ck['hyper_parameters']['model_config']).cuda();base.load_state_dict({k.removeprefix('model.'):v for k,v in ck['state_dict'].items()});dim=base.n_hidden*base.token_dim;head=nn.Sequential(nn.Linear(dim,64),nn.GELU(),nn.Linear(64,10)).cuda()
raw=ad.read_h5ad(B.parent/'native-alignment-20261111/counts_full/recovered_counts.h5ad',backed='r');gi=raw.var_names.get_indexer(genes);assert (gi>=0).all()
# Only training target cells may serve as prompts. Exclude the queried condition even in training.
agents={r['sample_id']:set(r['drugs'].split(' + ')) for r in bench['cohort']}
prompts={s:sorted([t for t in bench['splits']['train'] if t!=s],key=lambda t:(-len(agents[s]&agents[t])/max(1,len(agents[s]|agents[t])),t))[:3] for split in ['train','val','test'] for s in bench['splits'][split]}
assert all(s not in pp and set(pp)<=set(bench['splits']['train']) for s,pp in prompts.items())
cache={}
def counts(rows):
 # h5py requires sorted unique indexing; restore repetitions and order afterwards.
 u,rev=np.unique(rows,return_inverse=True);return raw[u,:].to_memory().X[:,gi].toarray().astype('float32')[rev]
def sample_counts(s,selected):
 f=I/'data'/f'{s}.h5ad'
 if not f.exists():return counts(selected)
 data=ad.read_h5ad(f);wanted=raw.obs_names[selected];order=data.obs_names.get_indexer(wanted);assert (order>=0).all(),s
 gg=data.var_names.get_indexer(genes);assert (gg>=0).all();arr=data.X[order,:][:,gg].toarray().astype('float32')
 if s==bench['splits']['train'][0]:assert np.array_equal(arr[:3],counts(selected[:3])), 'COUNTS_CACHE_MISMATCH'
 return arr
for j,s in enumerate(bench['splits']['train']):
 rr=np.random.default_rng(seed+int(hashlib.sha256(s.encode()).hexdigest()[:8],16));cache[s]=sample_counts(s,rr.choice(ix[s],512,replace=len(ix[s])<512))
 if j%25==0:print('PROMPT_COUNTS_LOADED',j+1,flush=True)
controls={s:sample_counts(s,ix[s]) for s in set(ctrl.values())};rowpos={s:{int(r):i for i,r in enumerate(ix[s])} for s in controls}
raw.file.close();print('INPUT_READY train-only prompts and control counts',flush=True)
# Only components on the direct-proportion path are updated; no expression classifier.
params=list(base.parameters())+list(head.parameters());opt=torch.optim.AdamW(params,lr=5e-5,weight_decay=.01);scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=10)
def latent(s,rows):
 query=controls[ctrl[s]][[rowpos[ctrl[s]][int(r)] for r in rows]];pool=np.concatenate([cache[t] for t in prompts[s]])
 nq=max(1,int(base.n_cells*(1-.25-.4)));nb=base.n_cells-nq;parts=[]
 for start in range(0,len(rows),nq*2):
  blocks=[];valid=[]
  for k in range(start,min(start+nq*2,len(rows)),nq):
   q=query[k:k+nq];n=len(q);q=np.concatenate([q,np.resize(query,(nq-n,2000))]) if n<nq else q
   pi=(np.arange(nb)+(k//nq)*nb)%len(pool);blocks.append(np.concatenate([pool[pi],q]));valid.append(n)
  inp=torch.tensor(np.stack(blocks),device='cuda')
  def native(a):
   return base._run_attention_layers(base._reduce_and_tokenize(torch.log1p(a))).reshape(a.shape[0],base.n_cells,dim)
  h=checkpoint(native,inp,use_reentrant=False) if base.training else native(inp)
  parts.extend(h[i,nb:nb+n] for i,n in enumerate(valid))
 return torch.cat(parts)
def predict(s):
 rows=ix[ctrl[s]];total=torch.zeros(dim,device='cuda')
 for k in range(0,len(rows),1024):total+=latent(s,rows[k:k+1024]).sum(0)
 return head(total/len(rows)).softmax(-1)
def evaluate(ids):
 base.eval();head.eval()
 with torch.no_grad():p=np.stack([predict(s).cpu().numpy() for s in ids])
 t=np.stack([targets[s] for s in ids]);return 100*np.abs(p-t).mean(),p
best=float('inf');stale=0;start=0
if (out/'last.pt').exists():
 q=torch.load(out/'last.pt',map_location='cpu',weights_only=False);base.load_state_dict(q['base']);head.load_state_dict(q['head']);opt.load_state_dict(q['optimizer']);scheduler.load_state_dict(q['scheduler']);start=q['epoch'];best=q['best'];stale=q['stale'];torch.set_rng_state(q['rng']);torch.cuda.set_rng_state(q['cuda_rng'])
(out/'CONFIG.json').write_text(json.dumps(dict(method='STACK direct proportion adaptation',seed=seed,source_checkpoint=str(weight),source_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),head='native in-context latent query tokens mean -> latent/64/10 -> softmax',prompt_rule='top3 train-only Jaccard excluding self, 512 counts each, native latent mode, prompt+context ratio .65',prompt_mapping=prompts,loss='condition-balanced proportion MAE',validation='full controls; 21 conditions; MAE',train_samples_per_condition=1024,trainable='STACK gene_reduction + tabular attention + gene position embeddings + proportion head',max_epochs=3000,patience=20,shared_readout=False,expression_decoder_used=False),indent=2))
for epoch in range(start+1,3001) if stale<20 else []:
 ts=time.time();base.train();head.train();rng=np.random.default_rng(seed+epoch);ids=list(bench['splits']['train']);rng.shuffle(ids);total=0
 for j,s in enumerate(ids):
  rows=ix[ctrl[s]];rows=rng.choice(rows,1024,replace=len(rows)<1024);opt.zero_grad();p=head(latent(s,rows).mean(0)).softmax(-1);t=torch.tensor(targets[s],device='cuda',dtype=torch.float32);loss=(p-t).abs().mean();loss.backward()
  if epoch==1 and j==0:
   assert sum(p.grad.abs().sum().item() for p in base.gene_reduction.parameters() if p.grad is not None)>0
   assert base.gene_pos_embedding.grad.abs().sum()>0
   assert sum(p.grad.abs().sum().item() for p in head.parameters() if p.grad is not None)>0
   assert sum(p.grad.abs().sum().item() for p in base.layers.parameters() if p.grad is not None)>0
   print('GRADIENT_CHECK_PASS gene_reduction attention gene_position proportion_head',flush=True)
   if '--smoke' in sys.argv:
    base.eval();head.eval()
    with torch.no_grad():q=head(latent(s,rows).mean(0)).softmax(-1)
    assert q.shape==(10,) and torch.isfinite(q).all() and abs(q.sum().item()-1)<1e-6
    (out/'SMOKE_PASS.json').write_text(json.dumps(dict(gradient_modules=['gene_reduction','attention','gene_position','proportion_head'],finite_simplex=True,shared_readout=False)))
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
