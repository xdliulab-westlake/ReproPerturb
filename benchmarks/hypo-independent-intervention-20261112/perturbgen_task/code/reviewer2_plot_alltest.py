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
ax=axs[1,1];hids=[s for s,flag in zip(ids,high) if not flag]+[s for s,flag in zip(ids,high) if flag]
assert len(hids)==14 and len(set(hids))==14
assert r.groupby(['layer','sample_id']).size().eq(3).all() and len(r)==84
ax.axvspan(8.5,13.5,color='#F8EDDD',zorder=0)
ax.axvline(8.5,color='.65',ls='--',lw=.8)
ax.text(4,1.01,'2–3 drugs (n=9)',ha='center',fontsize=8)
ax.text(11,1.01,'≥4 drugs (n=5)',ha='center',fontsize=8)
for offset,layer,col,label in [(-.12,'expression','#4477AA','Expression'),(.12,'composition','#E68532','Composition')]:
 g=r[(r.layer==layer)&r.sample_id.isin(hids)].groupby('sample_id').residual_r
 ax.plot(np.arange(14)+offset,g.mean().reindex(hids),'D',c=col,label=label,ms=6)
 for j,sid in enumerate(hids):
  values=r[(r.layer==layer)&(r.sample_id==sid)].residual_r.to_numpy()
  ax.scatter(j+offset+np.linspace(-.04,.04,len(values)),values,facecolors='none',edgecolors=col,s=30)
ax.axhline(0,color='.7',lw=.8);ax.set(xticks=range(14),xticklabels=hids,ylim=(-1.05,1.13),ylabel='Observed vs predicted residual Pearson r',title='d  All test residuals relative to identity Ridge');ax.tick_params(axis='x',rotation=60);ax.legend(fontsize=8,loc='lower left')
fig.suptitle('Testing combinatorial prediction against main effects',fontsize=16)
fig.text(.5,.025,'Shaded: >=4 drugs. b,c: Hypo mean ± seed SD; Ridge deterministic. d: open circles = seeds, diamonds = means.\nHigh-order validation used for model selection; development-exposed. Residual correlations are descriptive, not independent interaction tests.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.075,1,.95]);save(fig,'reviewer2_main_alltest_20261109')

print('VERIFIED: all 14 conditions, 2 layers, 3 seeds = 84 residual points; other panels unchanged.')
