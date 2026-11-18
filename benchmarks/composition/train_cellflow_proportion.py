from pathlib import Path
import os,sys,json,hashlib,time,fcntl
import numpy as np
seed=int(sys.argv[1]);os.environ['IPSTEM_TRAIN_SEED']=str(seed)
B=Path('/liaozizhuo/hypoblast/benchmarks/hypo-forward-embedding-combo-20261029');D=B.parent/'direct-composition-20261117';run=B.parent/f'native-alignment-20261111/cellflow_task/runs/cellflow_seed{seed}';os.environ['IPSTEM_RUN_ROOT']=str(run)
sys.path.insert(0,str(run/'code'))
import jax,jax.numpy as jnp,optax
from flax import serialization
from cellflow_native import make_solver,condition,rk4_integrate,predictor
out=D/(f'cellflow_seed{seed}_smoke' if '--smoke' in sys.argv else f'cellflow_seed{seed}');out.mkdir(exist_ok=True);lock=(out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
if (out/'COMPLETE.json').exists():sys.exit(0)
assert jax.default_backend()=='gpu'
read=lambda p:json.loads(Path(p).read_text());bench=read(B/'benchmark.json');classes=bench['classes'];truth=dict(np.load(B/'data/truth.npz'));targets={s:p for s,p in zip(truth['sample_ids'].tolist(),truth['proportions'])};ix=dict(np.load(B/'data/sample_rows_all.npz'));ctrl={r['sample_id']:r['control_sample_id'] for r in bench['cohort']};x=np.load(read(B/'config/stage2.json')['data']['cell_hvg_npy'],mmap_mode='r');features=dict(np.load(read(run/'training.json')['event_features_all_file']))
assert [len(bench['splits'][k]) for k in ['train','val','test']]==[146,21,14]
solver=make_solver();weight=run/'checkpoints/cellflow/best.msgpack';native=serialization.msgpack_restore(weight.read_bytes())['params'];native=jax.tree_util.tree_map(jnp.asarray,native)
key=jax.random.PRNGKey(seed);k1,k2=jax.random.split(key)
params=dict(base=native,head=dict(w1=jax.random.normal(k1,(2000,64))*np.sqrt(2/2064),b1=jnp.zeros(64),w2=jax.random.normal(k2,(64,10))*np.sqrt(2/74),b2=jnp.zeros(10)))
optimizer=optax.inject_hyperparams(optax.adamw)(learning_rate=5e-5,weight_decay=.01);state=optimizer.init(params)
# Differentiate through the same native deterministic RK4/20 trajectory; rematerialize velocity activations.
def forward(p,a,cond):
 vf=jax.checkpoint(lambda t,z:solver.vf.apply({'params':p},t,z,cond,jnp.zeros((1,128)),train=False)[0])
 return rk4_integrate(vf,a,20)
def head(h,z):return jax.nn.softmax(jax.nn.gelu(z@h['w1']+h['b1'])@h['w2']+h['b2'])
def objective(p,a,c,t):return jnp.abs(head(p['head'],forward(p['base'],a,c).mean(0))-t).mean()
lossgrad=jax.jit(jax.value_and_grad(objective))
@jax.jit
def step(p,st,a,c,t):
 loss,g=lossgrad(p,a,c,t);u,st=optimizer.update(g,st,p);return optax.apply_updates(p,u),st,loss
infer=jax.jit(forward)
def evaluate(ids):
 pred=[]
 for s in ids:
  rows=ix[ctrl[s]];total=np.zeros(2000,dtype='float64');cond=condition(features[s])
  for k in range(0,len(rows),1024):
   rr=rows[k:k+1024];a=np.zeros((1024,2000),dtype='float32');a[:len(rr)]=x[rr];total+=np.asarray(infer(params['base'],a,cond))[:len(rr)].sum(0,dtype='float64')
  pred.append(np.asarray(head(params['head'],jnp.asarray(total/len(rows),dtype=jnp.float32))))
 pred=np.stack(pred);return float(100*np.abs(pred-np.stack([targets[s] for s in ids])).mean()),pred
best=float('inf');stale=0;start=0;lr=5e-5;scheduler_best=float('inf');scheduler_bad=0
if (out/'last.msgpack').exists():
 q=serialization.from_bytes(dict(params=params,state=state),(out/'last.msgpack').read_bytes());params=q['params'];state=q['state'];meta=read(out/'last.json');start=meta['epoch'];best=meta['best'];stale=meta['stale'];lr=meta['lr'];scheduler_best=meta['scheduler_best'];scheduler_bad=meta['scheduler_bad']
config=dict(method='CellFlow direct proportion adaptation',seed=seed,source_checkpoint=str(weight),source_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),head='mean native RK4/20 expression-space output -> 2000/64/10 MLP -> softmax',trainable='native velocity field and condition encoder plus proportion head',loss='condition-balanced proportion MAE',validation='full controls; 21 conditions; MAE',train_samples_per_condition=1024,lr=5e-5,weight_decay=.01,max_epochs=3000,patience=20,shared_readout=False,hypo_embedding_used=False,ode_steps=20)
(out/'CONFIG.json').write_text(json.dumps(config,indent=2));print('MODEL_READY',flush=True)
if start==0:
 s=bench['splits']['train'][0];rr=np.random.default_rng(seed).choice(ix[ctrl[s]],1024,replace=len(ix[ctrl[s]])<1024);a=np.array(x[rr]);c=condition(features[s]);loss,g=lossgrad(params,a,c,jnp.asarray(targets[s],dtype=jnp.float32))
 norms={k:float(optax.global_norm(v)) for k,v in g.items()};assert all(np.isfinite(v) and v>0 for v in norms.values()),norms
 original=np.asarray(predictor(solver)(native,a,c));adapted=np.asarray(infer(native,a,c));assert np.max(np.abs(original-adapted))<1e-4
 prob=np.asarray(head(params['head'],jnp.asarray(adapted).mean(0)));assert prob.shape==(10,) and np.isfinite(prob).all() and abs(prob.sum()-1)<1e-6
 print('GRADIENT_CHECK_PASS',norms,flush=True)
 if '--smoke' in sys.argv:
  (out/'SMOKE_PASS.json').write_text(json.dumps(dict(gradient_norms=norms,native_forward_max_error=float(np.max(np.abs(original-adapted))),finite_simplex=True)));print('SMOKE_PASS',flush=True);sys.exit(0)
for epoch in range(start+1,3001) if stale<20 else []:
 ts=time.time();rng=np.random.default_rng(seed+epoch);ids=list(bench['splits']['train']);rng.shuffle(ids);total=0
 for j,s in enumerate(ids):
  rows=ix[ctrl[s]];rr=rng.choice(rows,1024,replace=len(rows)<1024);params,state,loss=step(params,state,np.array(x[rr]),condition(features[s]),jnp.asarray(targets[s],dtype=jnp.float32));loss=float(loss);assert np.isfinite(loss);total+=loss
  if (j+1)%25==0:print('TRAIN',epoch,j+1,loss*100,flush=True)
 score,p=evaluate(bench['splits']['val'])
 if score<best:
  best=score;stale=0;(out/'best.partial').write_bytes(serialization.to_bytes(params));os.replace(out/'best.partial',out/'best.msgpack');(out/'best.json').write_text(json.dumps(dict(epoch=epoch,best=best)))
 else:stale+=1
 if score<scheduler_best*(1-1e-4):scheduler_best=score;scheduler_bad=0
 else:scheduler_bad+=1
 if scheduler_bad>10:
  if lr-lr*.5>1e-8:lr*=.5
  scheduler_bad=0;state=state._replace(hyperparams={**state.hyperparams,'learning_rate':jnp.asarray(lr)})
 meta=dict(epoch=epoch,best=best,stale=stale,lr=lr,scheduler_best=scheduler_best,scheduler_bad=scheduler_bad)
 (out/'last.partial').write_bytes(serialization.to_bytes(dict(params=params,state=state)));os.replace(out/'last.partial',out/'last.msgpack');(out/'last.json').write_text(json.dumps(meta));print(json.dumps(dict(**meta,train_mae_pp=total/len(ids)*100,val_mae_pp=score,seconds=time.time()-ts)),flush=True)
 if stale>=20:break
params=serialization.from_bytes(params,(out/'best.msgpack').read_bytes());ids=bench['splits']['test'];mae,p=evaluate(ids);t=np.stack([targets[s] for s in ids]);assert p.shape==(14,10) and np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-6)
np.savez(out/'predictions.npz',sample_ids=ids,classes=classes,proportions=p,truth=t);from direct_metrics import score as metrics
m=metrics(p,t);m['dominant_accuracy_pct']=float((p.argmax(1)==t.argmax(1)).mean()*100);(out/'metrics.json').write_text(json.dumps(m,indent=2));(out/'COMPLETE.json').write_text(json.dumps(dict(seed=seed,best_epoch=read(out/'best.json')['epoch'],test_conditions=14,shared_readout=False,mae_pp=mae)));print('TEST_COMPLETE',seed,mae,flush=True)
