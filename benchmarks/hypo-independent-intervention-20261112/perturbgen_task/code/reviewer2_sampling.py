"""Descriptive random-half stability; no biological-replicate significance claims."""
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from prepare_composition import ROOT,OLD
from composition_metrics import clr_additive
OUT=ROOT/'results/reviewer2_new_split_20261107'
cfg=json.loads((OLD/'benchmark.json').read_text());data=json.loads((OLD/'config/stage2.json').read_text())['data'];support=[r for r in json.loads((ROOT/'audit/support_manifest.json').read_text()) if r['eligible']]
t=dict(np.load(OLD/'data/truth.npz'));idx={s:i for i,s in enumerate(t['sample_ids'])};classes=t['classes'].tolist();test=cfg['splits']['test']
needed=sorted(set(test+[r['control'] for r in support]+[s for r in support for s in r['singletons']]))
m=pd.read_csv(data['cell_metadata_csv'],usecols=['cell_id','condition','celltype']);assert m.cell_id.is_unique
x=np.load(data['cell_hvg_npy'],mmap_mode='r');assert x.shape==(len(m),2000)
selected=m.condition.isin(needed);m.loc[selected].assign(matrix_row=np.flatnonzero(selected)).to_csv(OUT/'sampling_cell_manifest.csv',index=False)
table=pd.read_csv(ROOT/'results/sample_proportions_v7.csv').set_index('sample_id');rng=np.random.default_rng(20261107)
def cds(a,b):return float(1-np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-30))
noise=[];means={};halves={};checks=[]
for s in needed:
 positions=np.flatnonzero(m.condition.to_numpy()==s);labels=pd.Categorical(m.iloc[positions].celltype,categories=classes).codes;assert labels.min()>=0
 a=np.asarray(x[positions],dtype=np.float64);total=a.sum(0);n=len(a);mu=total/n;counts=np.bincount(labels,minlength=10);p=counts/n
 assert np.allclose(p,table.loc[s,['prop::'+c for c in classes]].to_numpy(float),atol=1e-10)
 expected=t['expression'][idx[s]] if s in idx else t['control_expression'][next(i for i,c in enumerate(t['control_ids']) if c==s)]
 err=float(np.max(np.abs(mu-expected)));assert np.allclose(mu,expected,atol=2e-5,rtol=2e-5),(s,err)
 means[s]={'expression':mu,'composition':p};halves[s]={'expression':[],'composition':[]};checks.append(dict(sample_id=s,n_cells=n,max_mean_expression_difference=err))
 for rep in range(20):
  take=rng.permutation(n)[:n//2];e1=a[take].mean(0);e2=(total-e1*len(take))/(n-len(take));c1=np.bincount(labels[take],minlength=10);p1=c1/len(take);p2=(counts-c1)/(n-len(take))
  for layer,u,v in [('expression',e1,e2),('composition',p1,p2)]:
   halves[s][layer].append((u,v));noise.append(dict(sample_id=s,repeat=rep,layer=layer,n_cells=n,cds=cds(u,v)))
 print('SAMPLING',s,n,'verified max difference',err,flush=True)
 del a
pd.DataFrame(noise).to_csv(OUT/'sampling_half_cds.csv',index=False);pd.DataFrame(checks).to_csv(OUT/'sampling_mean_verification.csv',index=False)
# Rebuild direct references from each random half, including each component's sampling variability.
refnoise=[];deviations=[]
def reference(layer,method,ctrl,singles):
 if method=='Single-profile mean':return singles.mean(0)
 if layer=='expression':return ctrl+(singles-ctrl).sum(0)
 return clr_additive(ctrl,singles)
for r in support:
 s=r['sample_id'];c=r['control'];ss=r['singletons']
 for layer in ['expression','composition']:
  methods=['Single-profile mean','Control-corrected additive' if layer=='expression' else 'Additive CLR']
  for method in methods:
   base=reference(layer,method,means[c][layer],np.stack([means[v][layer] for v in ss]));obs=cds(means[s][layer],base)
   deviations.append(dict(sample_id=s,layer=layer,method=method,observed_cds=obs))
   for rep in range(20):
    refs=[reference(layer,method,halves[c][layer][rep][h],np.stack([halves[v][layer][rep][h] for v in ss])) for h in [0,1]]
    refnoise.append(dict(sample_id=s,layer=layer,method=method,repeat=rep,reference_half_cds=cds(refs[0],refs[1]),observed_half0_to_reference_half0_cds=cds(halves[s][layer][rep][0],refs[0]),observed_half1_to_reference_half1_cds=cds(halves[s][layer][rep][1],refs[1])))
pd.DataFrame(refnoise).to_csv(OUT/'reference_sampling_cds.csv',index=False);pd.DataFrame(deviations).to_csv(OUT/'direct_reference_cds.csv',index=False)
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
(OUT/'sampling_verification.json').write_text(json.dumps(dict(status='sampling_complete',seed=20261107,repeats=20,conditions=needed,n_selected_cells=int(selected.sum()),input_sha256={p:sha(p) for p in [data['cell_metadata_csv'],data['cell_hvg_npy']]},script_sha256=sha(__file__),inference='Random-half sampling stability only; no biological replicate p-values or significance threshold.'),indent=2)+'\n')
print('SAMPLING_COMPLETE',flush=True)
