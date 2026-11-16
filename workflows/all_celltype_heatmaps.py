from pathlib import Path
import sys,json,math
from collections import Counter
import numpy as np,pandas as pd,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
B=Path(__file__).resolve().parents[1];R=B;O=B/'all_celltype_heatmaps';O.mkdir(exist_ok=True)
sys.path.insert(0,str(R/'src'))
from hypo_release.unified import examples,features
from hypo_release.shared_math import kernel,combine
from experiments.perturbation.evaluate_prototype_mass_flow import load_evaluation_module,select_split_records
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
cfg=json.loads((R/'SHARED_CONFIG.json').read_text());assert cfg['variant']=='high'
payload=torch.load(R/'assets/pair_records.pt',map_location='cpu',weights_only=True);contexts={}
for r in select_split_records(payload['records'],R/'assets/context_split.csv',split='train'):
 ids=tuple(sorted(map(int,r.get('original_drug_ids',r['drug_ids']))))
 if len(ids)==2 and len(set(ids))==2:contexts.setdefault((str(r['control_sample_id']),ids),dict(r,drug_ids=list(ids)))
contexts=list(contexts.values());assert len(contexts)==64
tr,_,paths,_=examples(43);support=Counter(x.target_celltype for x in tr);all_scores=[];checks=[];rows=[]
for seed in [43,44,45]:
 ck=torch.load(R/f'weights/seed{seed}.pt',map_location='cpu',weights_only=True);classes=ck['celltype_classes'];mod=load_evaluation_module(ck['config'],ck,payload,R/'assets/pathway_base.pt',torch.device('cuda'));mod.eval();mod.adapter.endpoint_rollout_steps=20;mapping=mod.adapter.prototype_to_celltype.cuda().float();fs=[]
 with torch.inference_mode():
  for ri,r in enumerate(contexts):
   ids=torch.tensor(r['drug_ids'],device='cuda').view(1,-1);enc=mod.condition_encoder(ids,torch.ones_like(ids,dtype=torch.bool));act=enc.activities['selected'];z=torch.as_tensor(r['initial_mass_logit']).float().cuda().view(1,-1);p=torch.as_tensor(r['initial_presence']).float().cuda().view(1,-1)
   def predict(en,zz,pp):
    mass,end=mod.adapter.rollout_endpoint(zz,pp,en.condition_tokens,en.pathway_context);return mod.adapter.velocity_model.presence_gated_proportion(mass,torch.sigmoid(end.presence_logits))@mapping
   baseline=predict(enc,z,p)[0];gains=torch.empty(17,2,10,device='cuda')
   for j,d in enumerate([1.,-1.]):
    x=act.repeat(17,1);ix=torch.arange(17,device='cuda');x[ix,ix]+=d*1.5;en=mod.condition_encoder.encode_activities({'selected':x});gains[:,j,:]=predict(en,z.repeat(17,1),p.repeat(17,1))-baseline
   # Normalize each target exactly as the original scalar-target make_feature.
   sg=gains.reshape(34,10).T;sg=(sg-sg.mean(1,keepdim=True))/sg.std(1,keepdim=True).clamp_min(1e-6)
   initial=torch.as_tensor(r['initial_celltype_proportion']).float().cuda().repeat(10,1)
   fs.append(torch.cat([sg,torch.zeros(10,17,device='cuda'),initial,torch.eye(10,device='cuda')],dim=1).cpu().numpy())
   if ri%16==0:print('CONTEXT',seed,ri,flush=True)
 fs=np.stack(fs).astype(float);np.save(O/f'features_seed{seed}.npy',fs)
 reference=features(seed,'six')['high'];err=np.max(abs(fs[:,classes.index('Hypoblast-like cell')]-reference));assert np.allclose(fs[:,classes.index('Hypoblast-like cell')],reference,atol=2e-5,rtol=2e-5),err
 state=dict(np.load(R/f'models/seed{seed}/kernel_model.npz'));scores=[]
 for ci,cell in enumerate(classes):
  raw=fs[:,ci];x=(raw-state['mean'])/state['std'];x[:,61:]=raw[:,61:];ll=combine(kernel(x,state['train'],cfg['kernel'])@state['weight'],raw,cfg['identity_alpha'],cfg['direction_alpha']);s=np.sort(ll,axis=0)[-7:].mean(0).reshape(17,2);scores.append(s)
  for pi,path in enumerate(paths):
   for di,sign in enumerate(['+','-']):rows.append(dict(seed=seed,celltype=cell,pathway=path,direction=sign,score=s[pi,di],training_examples=support[cell]))
 scores=np.stack(scores);official=json.loads((R/'RESULTS.json').read_text())[str(seed)]['hypoblast']['rankings'];got=scores[classes.index('Hypoblast-like cell')].max(1)
 assert all(np.isclose(got[paths.index(v['pathway'])],v['score'],atol=2e-4,rtol=2e-4) for v in official)
 all_scores.append(scores);checks.append(dict(seed=seed,hypoblast_feature_max_error=float(err),hypoblast_identity_scores_reproduced=True));print('SCORED',seed,flush=True);del mod;torch.cuda.empty_cache()
a=np.stack(all_scores);pd.DataFrame(rows).to_csv(O/'scores_all_celltypes_34.csv',index=False)
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none'})
for n in [17,34]:
 matrices=a.max(3) if n==17 else a.reshape(3,10,34)
 matrices=list(matrices)+[matrices.mean(0)];vmin=min(-.01,min(x.min() for x in matrices));vmax=max(.01,max(x.max() for x in matrices));norm=TwoSlopeNorm(vmin=vmin,vcenter=0,vmax=vmax)
 fig,axs=plt.subplots(4,1,figsize=(20 if n==34 else 14,19));labels=paths if n==17 else [p+d for p in paths for d in ['+','-']]
 for ax,x,title in zip(axs,matrices,['Seed 43','Seed 44','Seed 45','Mean across seeds 43/44/45']):
  im=ax.imshow(x,aspect='auto',cmap='RdBu_r',norm=norm);ax.set(yticks=range(10),yticklabels=[c+(' *' if not support[c] else '') for c in classes],xticks=range(n),xticklabels=labels,title=title);ax.tick_params(axis='x',rotation=65,labelsize=8);ax.tick_params(axis='y',labelsize=9)
  for i in range(10):
   for j in range(n):ax.text(j,i,f'{x[i,j]:.1f}',ha='center',va='center',fontsize=6 if n==34 else 7,color='white' if abs(norm(x[i,j])-.5)>.34 else '#222222')
 fig.suptitle('Cell-type × pathway '+('identity scores (max over directions)' if n==17 else 'signed relation scores (both directions)'),fontsize=16,y=.995)
 fig.text(.5,.008,'Same 64 contexts; strength 1.5; upper-tail mean over 7/64 contexts per signed candidate. Fixed model and shared parameters.\n* No target-cell-type examples in relation training: extrapolative scores. Colors show relative scores, not activation or proportion changes.\n17-column mean averages per-seed identity scores; no direct-Flow direction is attached to these cells.',ha='center',fontsize=9)
 fig.tight_layout(rect=[0,.055,.94,.98]);cax=fig.add_axes([.95,.2,.012,.6]);fig.colorbar(im,cax=cax,label='Relative score')
 for ext in ['pdf','svg','png']:fig.savefig(O/f'celltype_pathway_{n}_heatmap.{ext}',dpi=180)
 plt.close(fig)
(O/'VERIFICATION.json').write_text(json.dumps(dict(status='complete',shape=list(a.shape),support=dict(support),checks=checks),indent=2));print('COMPLETE',flush=True)
