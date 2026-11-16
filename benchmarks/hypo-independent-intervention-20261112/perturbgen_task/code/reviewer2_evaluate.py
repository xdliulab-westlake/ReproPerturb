"""New-split Reviewer 2 evaluation; immutable previous inputs, standalone outputs."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from prepare_composition import ROOT,OLD
from composition_metrics import metrics as comp_metrics
sys.path.insert(0,str(OLD))
from metrics import expression_metrics,correlation

OUT=ROOT/'results/reviewer2_new_split_20261107';OUT.mkdir(exist_ok=True)
paths=[]
def read(p):paths.append(p);return json.loads(p.read_text())
def archive(p):paths.append(p);return dict(np.load(p,allow_pickle=False))
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
 return h.hexdigest()
def write(name,x):(OUT/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def cds(a,b):return 1-np.sum(a*b,axis=-1)/np.maximum(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1),1e-30)
cfg=read(OLD/'benchmark.json');contract=read(ROOT/'config/contract.json');t=archive(OLD/'data/truth.npz');classes=t['classes'].tolist();genes=t['genes'].tolist();allids=t['sample_ids'].tolist();ids=cfg['splits']['test'];ix={s:i for i,s in enumerate(allids)}
assert sha(OLD/'data/truth.npz')==contract['truth_sha256']
rows={r['sample_id']:r for r in cfg['cohort']};splits=cfg['splits'];train,val,test=[np.array([ix[s] for s in splits[k]]) for k in ['train','val','test']]
assert all(not(set(splits[a])&set(splits[b])) for a,b in [('train','val'),('train','test'),('val','test')])
def drugs(s):return frozenset(x.strip() for x in rows[s]['drugs'].split('+'))
assert not({drugs(s) for s in splits['test']}&{drugs(s) for s in splits['train']})
assert not({drugs(s) for s in splits['test']}&{drugs(s) for s in splits['val']})
orders=np.array([len(drugs(s)) for s in ids]);support=read(ROOT/'audit/support_manifest.json');eligible=[r for r in support if r['eligible']];supported=[r['sample_id'] for r in eligible]
subsets={'all14':np.arange(14),'low_order9':np.flatnonzero(orders<4),'high_order5':np.flatnonzero(orders>=4),'supported5':np.array([ids.index(s) for s in supported])}
pd.DataFrame([dict(sample_id=s,split=k,order=len(drugs(s)),drugs=rows[s]['drugs'],control=rows[s]['control_sample_id']) for k in splits for s in splits[k]]).to_csv(OUT/'split_audit.csv',index=False)
# Unconstrained main-effects prediction on the locked expression scale.
idcache=archive(OLD/'data/drug_ids_all.npz');D=len(read(OLD/'config/adversarial_vocabulary.json')['agents']);onehot=np.zeros((len(allids),D))
for i,s in enumerate(allids):onehot[i,np.asarray(idcache[s]).astype(int).reshape(-1)-1]=1
x=t['control_expression'].astype(float);y=t['expression'].astype(float);delta=y-x
features=np.column_stack([onehot,x]);sc=StandardScaler().fit(features[train]);xx=sc.transform(features);best=None;candidates=[]
for alpha in np.logspace(-4,4,9):
 model=Ridge(alpha=alpha).fit(xx[train],delta[train]);score=float(np.mean((x[val]+model.predict(xx[val])-y[val])**2));candidates.append(dict(alpha=float(alpha),val_mse=score))
 if best is None or score<best[0]-1e-12 or (abs(score-best[0])<=1e-12 and alpha>best[1]):best=(score,float(alpha),model)
ridge=x[test]+best[2].predict(xx[test]);np.savez(OUT/'expression_ridge_identity.npz',sample_ids=ids,genes=t['genes'],expression=ridge,expression_scale=t['expression_scale'])
write('ridge_identity_provenance.json',dict(alpha=best[1],candidates=candidates,fit_ids=splits['train'],selection_ids=splits['val'],test_ids=ids,features='multi-hot drug identity and control expression; no interactions',target='expression minus matched control',clipping=False))
# Verify Ridge affine additivity by inspecting a four-condition feature contrast.
a=features[train[0]].copy();v=np.zeros_like(a);v[0]=1;w=np.zeros_like(a);w[1]=1
z=best[2].predict(sc.transform(np.stack([a,a+v,a+w,a+v+w])));assert np.max(np.abs(z[3]-z[1]-z[2]+z[0]))<1e-8
pred={'expression':{'Ridge identity':{-1:ridge}},'composition':{}}
def align(p,key,axis):
 a=archive(p);assert a[axis].tolist()==(genes if axis=='genes' else classes)
 if key=='expression':assert str(a['expression_scale'])==str(t['expression_scale'])
 ss=a['sample_ids'].tolist();assert len(ss)==len(set(ss));z=a[key][[ss.index(s) for s in ids]].astype(float);assert np.isfinite(z).all()
 if key=='proportions':assert z.min()>=0 and np.allclose(z.sum(1),1,atol=1e-7)
 return z
pred['expression']['Ridge embedding']={-1:align(OLD/'expression_baselines/ridge.npz','expression','genes')}
pred['expression']['Hypo']={r['seed']:align((OLD/r['prediction']).resolve(),'expression','genes') for r in cfg['existing_reference_runs']}
pred['composition']['Ridge identity']={-1:align(ROOT/'results/val_selected/ridge_identity_seed-1/predictions.npz','proportions','classes')}
pred['composition']['Hypo']={s:align(ROOT/f'results/locked_checkpoint/hypo_seed{s}/predictions.npz','proportions','classes') for s in [43,44,45]}
truth={'expression':y[test],'composition':t['proportions'][test]};control=x[test]
# Supplementary direct references have five rows, never extrapolated to the other nine.
direct={'expression':{},'composition':{}}
for layer,key in [('expression','expression'),('composition','proportions')]:
 singles=[t[key][[ix[s] for s in r['singletons']]].astype(float) for r in eligible]
 direct[layer]['Single-profile mean']={-1:np.stack([v.mean(0) for v in singles])}
 if layer=='expression':direct[layer]['Control-corrected additive']={-1:np.stack([x[ix[r['sample_id']]]+(v-x[ix[r['sample_id']]]).sum(0) for r,v in zip(eligible,singles)])}
for file,label in [('additive_clr','Additive CLR'),('additive_clr_eps1e5','Additive CLR eps1e-5'),('additive_clr_eps1e3','Additive CLR eps1e-3'),('additive_projected','Additive projected')]:
 a=archive(ROOT/f'results/supported_subset/{file}_seed-1/predictions.npz');assert a['sample_ids'].tolist()==supported and a['classes'].tolist()==classes;direct['composition'][label]={-1:a['proportions']}
summary=[];conditions=[]
def evaluate(layer,name,seed,p,q,subset,indices):
 if layer=='expression':
  m=expression_metrics(p,q,control[indices]);vals=m['summary'];per=m['per_condition'];vals['MAE_all']=float(np.abs(p-q).mean());vals['CDS']=float(cds(p,q).mean())
  for i,row in enumerate(per):row.update(MAE_all=float(np.abs(p[i]-q[i]).mean()),CDS=float(cds(p[i],q[i])))
 else:
  vals=comp_metrics(p,q);per=[comp_metrics(p[i:i+1],q[i:i+1]) for i in range(len(p))]
 summary.append(dict(layer=layer,method=name,seed=seed,subset=subset,**{k:v for k,v in vals.items() if isinstance(v,(float,int,np.floating)) or v is None}))
 for i,row in enumerate(per):conditions.append(dict(layer=layer,method=name,seed=seed,subset=subset,sample_id=ids[indices[i]],order=int(orders[indices[i]]),**{k:v for k,v in row.items() if isinstance(v,(float,int,np.floating)) or v is None}))
for layer,methods in pred.items():
 for name,seeds in methods.items():
  for seed,p in seeds.items():
   for sub,indices in subsets.items():evaluate(layer,name,seed,p[indices],truth[layer][indices],sub,indices)
for layer,methods in direct.items():
 for name,seeds in methods.items():
  for seed,p in seeds.items():evaluate(layer,name,seed,p,truth[layer][subsets['supported5']],'supported5',subsets['supported5'])
summ=pd.DataFrame(summary);summ.to_csv(OUT/'metrics_by_seed_subset.csv',index=False);pd.DataFrame(conditions).to_csv(OUT/'metrics_per_condition.csv',index=False)
res=[];coords=[]
for layer,methods in pred.items():
 base=methods['Ridge identity'][-1];actual=truth[layer]-base
 for seed,p in methods['Hypo'].items():
  estimated=p-base
  assert np.allclose(estimated-actual,p-truth[layer])
  for i,s in enumerate(ids):
   scale=100 if layer=='composition' else 1;mask=np.abs(actual[i])>=.01 if layer=='composition' else np.abs(actual[i])>1e-8
   res.append(dict(layer=layer,seed=seed,sample_id=s,order=int(orders[i]),residual_r=float(correlation(actual[i:i+1],estimated[i:i+1],1)[0]),residual_mae=float(scale*np.abs(actual[i]-estimated[i]).mean()),observed_residual_rms=float(scale*np.sqrt(np.mean(actual[i]**2))),sign_agreement=float(np.mean(np.sign(actual[i,mask])==np.sign(estimated[i,mask]))) if mask.any() else None,n_sign=int(mask.sum())))
  for i,s in enumerate(ids):
   for j,feature in enumerate(classes if layer=='composition' else genes):coords.append(dict(layer=layer,seed=seed,sample_id=s,feature=feature,observed_residual=float(actual[i,j]),predicted_residual=float(estimated[i,j])))
pd.DataFrame(res).to_csv(OUT/'residual_metrics.csv',index=False);pd.DataFrame(coords).to_csv(OUT/'residual_coordinates.csv',index=False)
np.savez(OUT/'analysis_arrays.npz',sample_ids=ids,orders=orders,expression_truth=truth['expression'],expression_ridge=ridge,expression_hypo=np.stack(list(pred['expression']['Hypo'].values())),composition_truth=truth['composition'],composition_ridge=pred['composition']['Ridge identity'][-1],composition_hypo=np.stack(list(pred['composition']['Hypo'].values())))
write('verification.json',dict(status='evaluation_complete',test_ids=ids,high_order_ids=[ids[i] for i in subsets['high_order5']],supported_ids=supported,checks=['split ID disjoint','test drug sets unseen in train and val','truth SHA256','gene and class order','prediction scale and finite values','composition simplex','Ridge affine identity','residual error identity'],expression_additive_negative_coordinates=int((direct['expression']['Control-corrected additive'][-1]<0).sum()),input_sha256={str(p):sha(p) for p in sorted(set(paths))},script_sha256=sha(Path(__file__))))
print('RIDGE_ALPHA',best[1],flush=True)
for layer,key in [('expression','MSE_raw'),('composition','mae_pp')]:
 print(layer,summ[summ.layer==layer].groupby(['subset','method'])[key].agg(['mean','std']).to_string(),flush=True)
print('EVALUATION_COMPLETE',flush=True)
