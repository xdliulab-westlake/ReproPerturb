"""Frozen-checkpoint nonadditivity evaluation; all outputs in a new remote directory."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from composition_metrics import clr,clr_additive,simplex_projection
from prepare_composition import ROOT,OLD

OUT=ROOT/'results/nonadditivity_20261107';OUT.mkdir(exist_ok=True)
FIG=ROOT/'figures/nonadditivity_20261107';FIG.mkdir(exist_ok=True)
inputs=[]
def read(p):
    inputs.append(p);return json.loads(p.read_text())
cfg=read(ROOT/'config/contract.json');support=read(ROOT/'audit/support_manifest.json');audit=read(ROOT/'audit/additivity_complete.json')
classes=cfg['classes']; eligible=[r for r in support if r['eligible']];ids=[r['sample_id'] for r in eligible]
assert ids==['H31','H46','H71','H85','H132']
inputs.append(ROOT/'results/sample_proportions_v7.csv')
table=pd.read_csv(inputs[-1]).set_index('sample_id')
def vec(s):return table.loc[s,['prop::'+c for c in classes]].to_numpy(float)
y=np.stack([vec(s) for s in ids])
inputs.append(OLD/'data/truth.npz');truth=np.load(inputs[-1],allow_pickle=False)
assert hashlib.sha256(inputs[-1].read_bytes()).hexdigest()==cfg['truth_sha256']
assert np.allclose(y,truth['proportions'][[list(truth['sample_ids']).index(s) for s in ids]],atol=1e-8)
metadata=[]
for r in eligible:
    assert r['sample_id'] in cfg['splits']['test']
    assert set(r['singletons'])<=set(cfg['splits']['train'])
    members=[r['sample_id'],r['control']]+r['singletons']
    for col in ['batch','day','h5_time']:
        assert table.loc[members,col].nunique()==1,(members,col)
    for s in members:
        metadata.append(dict(combination=r['sample_id'],sample_id=s,role='combination' if s==r['sample_id'] else 'control' if s==r['control'] else 'singleton',**table.loc[s,['batch','day','h5_time','sequencing_batch','drug_combination']].to_dict(),dose='not_available',treatment_schedule='not_available',donor='not_available',initial_state='not_available'))
pd.DataFrame(metadata).to_csv(OUT/'matching_audit.csv',index=False)
def load(track,name,seed):
    p=ROOT/'results'/track/f'{name}_seed{seed}/predictions.npz';inputs.append(p)
    a=np.load(p,allow_pickle=False);ss=a['sample_ids'].tolist()
    assert a['classes'].tolist()==classes and len(ss)==len(set(ss))
    z=a['proportions'][[ss.index(s) for s in ids]]
    assert np.isfinite(z).all() and z.min()>=0 and np.allclose(z.sum(1),1,atol=1e-7)
    return z
pred={}
for name in ['additive_clr','additive_clr_eps1e5','additive_clr_eps1e3','additive_projected']:
    pred[name]={-1:load('supported_subset',name,-1)}
pred['ridge_identity']={-1:load('val_selected','ridge_identity',-1)}
pred['hypo']={s:load('locked_checkpoint','hypo',s) for s in [43,44,45]}
for i,r in enumerate(eligible):
    c=vec(r['control']);sing=np.stack([vec(s) for s in r['singletons']])
    for name,eps in [('additive_clr',1e-4),('additive_clr_eps1e5',1e-5),('additive_clr_eps1e3',1e-3)]:
        assert np.allclose(pred[name][-1][i],clr_additive(c,sing,eps))
    assert np.allclose(pred['additive_projected'][-1][i],simplex_projection(c+(sing-c).sum(0)))
base=pred['additive_clr'][-1];rows=[];residual=[]
for method,seeds in pred.items():
    for seed,p in seeds.items():
        for i,s in enumerate(ids):
            a=100*(y[i]-base[i]);b=100*(p[i]-base[i]);mask=np.abs(a)>=1
            rows.append(dict(method=method,seed=seed,sample_id=s,mae_pp=np.abs(100*(p[i]-y[i])).mean(),rmse_pp=np.sqrt(np.mean((100*(p[i]-y[i]))**2)),residual_r=float(np.corrcoef(a,b)[0,1]) if np.std(b)>1e-12 else np.nan,sign_agreement_gt1pp=np.mean(np.sign(a[mask])==np.sign(b[mask])) if mask.any() else np.nan,n_gt1pp=int(mask.sum())))
            for j,c in enumerate(classes):residual.append(dict(method=method,seed=seed,sample_id=s,celltype=c,observed=y[i,j],predicted=p[i,j],additive=base[i,j],true_residual_pp=a[j],predicted_residual_pp=b[j]))
metrics=pd.DataFrame(rows);metrics.to_csv(OUT/'condition_metrics.csv',index=False)
pd.DataFrame(residual).to_csv(OUT/'residuals.csv',index=False)
seedmeans=metrics.groupby(['method','seed'])[['mae_pp','rmse_pp']].mean();seedmeans.to_csv(OUT/'seed_metrics.csv')
summary=seedmeans.groupby('method').agg(['mean','std']);summary.to_csv(OUT/'summary.csv')
# Paired comparisons average errors across seeds, rather than evaluating an ensemble.
paired=metrics.groupby(['sample_id','method']).mae_pp.mean().unstack().loc[ids]
paired['hypo_gain_vs_clr_pp']=paired.additive_clr-paired.hypo
paired['hypo_gain_vs_ridge_pp']=paired.ridge_identity-paired.hypo
paired.to_csv(OUT/'paired_comparison.csv')
third=[];hp=np.stack(list(pred['hypo'].values()))
for r in audit['three_way']:
    if not r['three_way_eligible']:continue
    s=r['sample_id'];i=ids.index(s);assert set(r['pairs']+r['singletons'])<=set(cfg['splits']['train'])
    for space,transform in [('proportion_pp',lambda x:100*x),('clr',clr)]:
        lower=-sum(transform(vec(x)) for x in r['pairs'])+sum(transform(vec(x)) for x in r['singletons'])-transform(vec('H4'))
        for k,seed in enumerate([43,44,45]):
            actual=transform(y[i])+lower;estimated=transform(hp[k,i])+lower
            for j,c in enumerate(classes):third.append(dict(sample_id=s,seed=seed,space=space,celltype=c,observed=actual[j],predicted=estimated[j]))
pd.DataFrame(third).to_csv(OUT/'third_order_conditional.csv',index=False)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
colors=['#4477AA','#228833','#CCBB44','#AA3377','#66CCEE']
fig,axs=plt.subplots(1,2,figsize=(11.4,4.8));pmean=hp.mean(0)
for i,s in enumerate(ids):axs[0].scatter(100*(y[i]-base[i]),100*(pmean[i]-base[i]),s=34,color=colors[i],label=s,alpha=.85)
lim=max(np.abs(100*(y-base)).max(),np.abs(100*(pmean-base)).max())*1.08
axs[0].plot([-lim,lim],[-lim,lim],color='.4',lw=1);axs[0].axhline(0,color='.8',lw=.7);axs[0].axvline(0,color='.8',lw=.7)
axs[0].set(xlim=(-lim,lim),ylim=(-lim,lim),xlabel='Observed - additive reference (pp)',ylabel='Hypo - additive reference (pp)',title='a  Residual recovery (mean prediction, 3 seeds)');axs[0].legend(fontsize=8,ncol=3)
names=['additive_clr','ridge_identity','hypo']
for i,s in enumerate(ids):axs[1].plot(range(3),paired.loc[s,names],'-o',color=colors[i],alpha=.85,label=s)
axs[1].plot(range(3),paired[names].mean(),'D',color='black',ms=7,label='Condition mean')
axs[1].set(xticks=range(3),xticklabels=['Additive CLR','Ridge identity','Hypo'],ylabel='Composition MAE (percentage points)',title='b  Same 5 conditions; lower is better');axs[1].legend(fontsize=8,ncol=2)
fig.suptitle('Prediction beyond an additive reference',fontsize=15)
fig.text(.5,.025,'Frozen checkpoints; development-exposed. Dose/schedule not verified; no biological-replicate inference.\nPanel b averages Hypo errors over seeds. Residual correlation alone is not evidence of interaction learning.',ha='center',fontsize=8)
fig.tight_layout(rect=[0,.10,1,.94])
for ext in ['png','pdf','svg']:fig.savefig(FIG/f'fig1f_nonadditivity.{ext}',dpi=220)
plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(12,5.5))
td=pd.DataFrame(third);labels=[c.replace(' cell','').replace('Reprogramming intermediate','Reprog. intermediate') for c in classes]
for ax,s in zip(axs,['H85','H132']):
    d=td[(td.sample_id==s)&(td.space=='proportion_pp')];obs=d.groupby('celltype')['observed'].first().reindex(classes);g=d.groupby('celltype').predicted
    ax.axvline(0,c='.7',lw=.8);ax.scatter(obs,np.arange(10)-.12,label='Observed',color='.25');ax.errorbar(g.mean().reindex(classes),np.arange(10)+.12,xerr=g.std().reindex(classes),fmt='o',color='#E68532',label='Hypo mean ± seed SD')
    ax.set(yticks=np.arange(10),yticklabels=labels,xlabel='Conditional third-order contrast (pp)',title=s);ax.invert_yaxis();ax.legend(fontsize=8)
fig.suptitle('Third-order contrast with observed lower-order terms',fontsize=14)
fig.text(.5,.025,'Only the held-out triple is replaced by its prediction; lower-order terms are observed.\nThis is a conditional diagnostic, not independent proof of learned three-way interactions or biological synergy.',ha='center',fontsize=8)
fig.tight_layout(rect=[0,.10,1,.94])
for ext in ['png','pdf','svg']:fig.savefig(FIG/f'third_order_diagnostic.{ext}',dpi=220)
plt.close(fig)
evidence={'status':'computational_evaluation_complete_matching_partial','ids':ids,'classes':classes,'checks':['truth hash and V7 values','sample and class alignment','train-only singletons and pairs','batch and endpoint day matching','additive references independently recomputed','finite simplex predictions'],'metadata_missing':['dose','treatment_schedule','donor','initial_state'],'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(inputs))},'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'hypo_better_than_clr_conditions':int((paired.hypo_gain_vs_clr_pp>0).sum()),'hypo_better_than_ridge_conditions':int((paired.hypo_gain_vs_ridge_pp>0).sum())}
(OUT/'verification.json').write_text(json.dumps(evidence,indent=2)+'\n')
print(summary.to_string());print(paired.to_string());print('VERIFIED',evidence['checks'])
