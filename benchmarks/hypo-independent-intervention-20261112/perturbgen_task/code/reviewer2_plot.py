"""Plot all prespecified cohorts and export complete metric tables."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare_composition import ROOT
O=ROOT/'results/reviewer2_new_split_20261107';F=ROOT/'figures/reviewer2_new_split_20261107';F.mkdir(exist_ok=True)
m=pd.read_csv(O/'metrics_by_seed_subset.csv');c=pd.read_csv(O/'metrics_per_condition.csv');split=pd.read_csv(O/'split_audit.csv');r=pd.read_csv(O/'residual_metrics.csv');a=np.load(O/'analysis_arrays.npz');ids=a['sample_ids'].tolist();high=a['orders']>=4
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
colors={'Hypo':'#E68532','Ridge identity':'#238B6E','Ridge embedding':'#4477AA'}
def save(fig,stem):
 for ext in ['png','pdf','svg']:fig.savefig(F/f'{stem}.{ext}',dpi=220)
 plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(13,10.5));ax=axs[0,0]
for i,(s,color) in enumerate([('train','#4477AA'),('val','#CCBB44'),('test','#E68532')]):
 counts=split[split.split==s].groupby('order').size().reindex(range(1,8),fill_value=0)
 bars=ax.bar(np.arange(1,8)+(i-1)*.25,counts,width=.23,label=s,color=color)
 for bar,n in zip(bars,counts):
  if n:ax.text(bar.get_x()+bar.get_width()/2,n+.7,str(n),ha='center',fontsize=8)
ax.set(xticks=range(1,8),xlabel='Number of drugs',ylabel='Number of conditions',title='a  Revised fixed split');ax.legend()
for ax,layer,key,title,ylabel in [(axs[0,1],'expression','MSE_raw','b  Expression error on all 14 test conditions','MSE (benchmark expression scale)'),(axs[1,0],'composition','mae_pp','c  Composition error on all 14 test conditions','MAE (percentage points)')]:
 d=c[(c.layer==layer)&(c.subset=='all14')]
 for j,v in enumerate(high):
  if v:ax.axvspan(j-.48,j+.48,color='#F8EDDD',zorder=0)
 for method in ['Ridge identity','Ridge embedding','Hypo']:
  t=d[d.method==method]
  if not len(t):continue
  g=t.groupby('sample_id')[key];mean=g.mean().reindex(ids);sd=g.std().reindex(ids).fillna(0)
  ax.errorbar(np.arange(14),mean,yerr=sd,fmt='o-',lw=1,ms=4,c=colors[method],label=method,capsize=2)
 ax.set(xticks=range(14),xticklabels=ids,ylabel=ylabel,title=title);ax.tick_params(axis='x',rotation=60);ax.legend(fontsize=8)
ax=axs[1,1];hids=[s for s,flag in zip(ids,high) if flag]
for offset,layer,col,label in [(-.12,'expression','#4477AA','Expression'),(.12,'composition','#E68532','Composition')]:
 g=r[(r.layer==layer)&r.sample_id.isin(hids)].groupby('sample_id').residual_r
 ax.plot(np.arange(5)+offset,g.mean().reindex(hids),'D',c=col,label=label,ms=6)
 for j,sid in enumerate(hids):
  values=r[(r.layer==layer)&(r.sample_id==sid)].residual_r.to_numpy()
  ax.scatter(j+offset+np.linspace(-.04,.04,len(values)),values,facecolors='none',edgecolors=col,s=30)
ax.axhline(0,color='.7',lw=.8);ax.set(xticks=range(5),xticklabels=hids,ylim=(-1.05,1.05),ylabel='Observed vs predicted residual Pearson r',title='d  High-order residuals relative to identity Ridge');ax.legend(fontsize=9)
fig.suptitle('Testing combinatorial prediction against main effects',fontsize=16)
fig.text(.5,.025,'Shaded: >=4 drugs. b,c: Hypo mean ± seed SD; Ridge deterministic. d: open circles = seeds, diamonds = means.\nHigh-order validation used for model selection; development-exposed. Residual correlations are descriptive, not independent interaction tests.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.075,1,.95]);save(fig,'reviewer2_main')
fig,axs=plt.subplots(1,2,figsize=(12.5,5.7))
for ax,layer,key,ylabel in [(axs[0],'expression','MSE_raw','Expression MSE'),(axs[1],'composition','mae_pp','Composition MAE (pp)')]:
 d=m[(m.layer==layer)&(m.subset=='supported5')];g=d.groupby('method')[key];mean=g.mean();sd=g.std().fillna(0)
 names=([ 'Single-profile mean','Control-corrected additive','Ridge identity','Ridge embedding','Hypo'] if layer=='expression' else ['Single-profile mean','Additive CLR','Additive CLR eps1e-5','Additive CLR eps1e-3','Additive projected','Ridge identity','Hypo'])
 for j,name in enumerate(names):ax.barh(j,mean[name],color=colors.get(name,'#999999'));ax.errorbar(mean[name],j,xerr=sd[name],fmt='none',ecolor='black',capsize=3)
 ax.set(yticks=range(len(names)),yticklabels=names,xlabel=ylabel,title=layer.capitalize());ax.invert_yaxis()
fig.suptitle('Direct single-agent references: same 5 supported low-order conditions',fontsize=14)
fig.text(.5,.025,'H31, H46, H71, H85, H132. No high-order test condition is included here.\nAll 10 cell types / 2,000 HVGs retained. Negative expression additive predictions retained; no silent clipping.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.10,1,.93]);save(fig,'reviewer2_direct_references')
noise=pd.read_csv(O/'sampling_half_cds.csv');ref=pd.read_csv(O/'reference_sampling_cds.csv');dev=pd.read_csv(O/'direct_reference_cds.csv');sids=['H31','H46','H71','H85','H132']
fig,axs=plt.subplots(1,2,figsize=(11.5,5.2))
for ax,layer,method in [(axs[0],'expression','Control-corrected additive'),(axs[1],'composition','Additive CLR')]:
 obs=dev[(dev.layer==layer)&(dev.method==method)].set_index('sample_id').observed_cds.reindex(sids)
 ax.plot(range(5),obs,'o-',c='#E68532',label='Observed vs direct reference')
 for df,col,value,label in [(noise[noise.layer==layer],'#4477AA','cds','Observed random halves'),(ref[(ref.layer==layer)&(ref.method==method)],'#238B6E','reference_half_cds','Reference random halves')]:
  g=df.groupby('sample_id')[value];med=g.median().reindex(sids);lo=g.quantile(.05).reindex(sids);hi=g.quantile(.95).reindex(sids)
  ax.errorbar(range(5),med,yerr=np.stack([med-lo,hi-med]),fmt='o--',c=col,capsize=3,label=label)
 ax.set(xticks=range(5),xticklabels=sids,yscale='log',ylabel='Cosine dissimilarity (CDS)',title=layer.capitalize());ax.legend(fontsize=8)
fig.suptitle('Observed deviations and random-half sampling stability',fontsize=14)
fig.text(.5,.025,'20 random half-splits per condition; points: median, bars: empirical 5th–95th percentiles.\nThese are sampling diagnostics, not biological-replicate intervals or significance thresholds.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.11,1,.93]);save(fig,'reviewer2_sampling_stability')
# Tables are a mechanical export of metrics, not generated scientific conclusions.
keys={'expression':['PCC_sample_raw','PCC_gene_raw','R2_raw','PCC_sample_delta','PCC_gene_delta','R2_delta','Spearman_delta','DEG_Jaccard100','DEG_overlap20','MAE_top100','MSE_raw','MAE_all','CDS'],'composition':['mae_pp','rmse_pp','jsd_bits']}
text=['# Complete metric tables','All available metrics; Hypo values are mean ± sample SD over three seeds. Deterministic baselines have no seed SD. No biological-replicate uncertainty is implied.']
for layer in keys:
 for sub in ['all14','low_order9','high_order5','supported5']:
  d=m[(m.layer==layer)&(m.subset==sub)];text+=['',f'## {layer}: {sub}','', '| Method | '+' | '.join(keys[layer])+' |','|---|'+'---:|'*len(keys[layer])]
  for name,g in d.groupby('method'):
   vals=[]
   for key in keys[layer]:
    v=g[key].dropna();vals.append('NA' if not len(v) else f'{v.mean():.6g}'+(f' ± {v.std():.3g}' if len(v)>1 else ''))
   text.append('| '+name+' | '+' | '.join(vals)+' |')
(O/'complete_metric_tables.md').write_text('\n'.join(text)+'\n')
print('PLOTS_AND_TABLES_COMPLETE',flush=True)
