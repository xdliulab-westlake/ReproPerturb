from pathlib import Path
import hashlib,json
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
N=Path('/liaozizhuo/hypoblast/benchmarks/native-alignment-20261111')
O=N/'ed3_update_20261116';O.mkdir(exist_ok=True)
D=N/'latest_overall_benchmark_20261113'
e=pd.read_csv(N/'figures_latest_20261114/expression_plot_data.csv');c=pd.read_csv(D/'composition_per_seed_all_metrics.csv')
latest=pd.read_csv(D/'expression_per_seed_all_metrics.csv')
assert np.allclose(e.sort_values(['model','seed']).MSE_raw,latest.sort_values(['model','seed']).MSE_raw,equal_nan=True)
font_manager.fontManager.addfont('/tmp/hypo_Arial.ttf')
plt.rcParams.update({'font.family':'Arial','pdf.fonttype':42,'font.size':3.1,'axes.linewidth':.3,'xtick.major.width':.3,'xtick.major.size':1.4,'axes.titlesize':3.4})
ms=['hypo','state','cellflow','stack_traincontext','perturbnet','cpa','chemcpa','g2cp','perturbgen','ridge','perturbmean','control']
labels=dict(zip(ms,['ReproPerturb','STATE','CellFlow','STACK','PerturbNet','CPA','chemCPA','G2CP','PerturbGen','Ridge (emb.)','Perturbation mean','Control']))
labels.update(ridge_embedding='Ridge (emb.)',ridge_identity='Ridge (identity)')
for m in ms[:9]:
 assert set(e.loc[e.model==m,'seed'])=={43,44,45}
 assert set(c.loc[c.method==m,'seed'])=={43,44,45}
def panel(ax,t,key,col,methods,title,show):
 ax.axhspan(-.45,.45,color='#FCF3ED',zorder=0)
 for j,m in enumerate(methods):
  v=pd.to_numeric(t.loc[t[col]==m,key],errors='coerce').dropna().to_numpy()
  color='#D55E00' if m=='hypo' else '#26745F' if m in ['ridge','ridge_embedding','ridge_identity','perturbmean','control'] else '#326A9E'
  if len(v):
   ax.errorbar(v.mean(),j,xerr=v.std(ddof=1) if len(v)>1 else 0,fmt='D',ms=1.5,color=color,lw=.4,capsize=.8)
   if len(v)>1:ax.scatter(v,j+np.linspace(-.17,.17,len(v)),s=2.5,facecolors='white',edgecolors=color,linewidths=.3,zorder=4)
  else:ax.text(.92,j,'NA',transform=ax.get_yaxis_transform(),ha='right',va='center',fontsize=2.6,color='#777777')
 ax.set(ylim=(len(methods)-.5,-.7),yticks=range(len(methods)),yticklabels=[labels[m] for m in methods] if show else [],title=title)
 ax.tick_params(axis='y',length=0,pad=2);ax.grid(axis='x',lw=.2,alpha=.3);ax.locator_params(axis='x',nbins=3)
 for side in ['top','right','left']:ax.spines[side].set_visible(False)
keys=[('PCC_gene_raw','Gene PCC (raw) ↑'),('PCC_gene_delta','Gene PCC (Δ) ↑'),('PCC_sample_raw','Sample PCC (raw) ↑'),('PCC_sample_delta','Sample PCC (Δ) ↑'),('R2_raw','R² (raw) ↑'),('R2_delta','R² (Δ) ↑'),('MSE_raw','Expression MSE ↓'),('MAE_top100','Top-100 MAE ↓'),('DEG_overlap20','Top-20 overlap ↑'),('DEG_Jaccard100','Top-100 Jaccard ↑')]
f,axs=plt.subplots(2,5,figsize=(315/72,190/72));f.subplots_adjust(left=.18,right=.995,bottom=.075,top=.87,wspace=.24,hspace=.35)
for i,(ax,(key,title)) in enumerate(zip(axs.flat,keys)):panel(ax,e,key,'model',ms,title,i%5==0)
f.suptitle('Forward prediction of gene expression and perturbation responses',fontsize=5,y=.98)
f.savefig(O/'panel_c.pdf');plt.close(f)
f,ax=plt.subplots(figsize=(158/72,100/72));f.subplots_adjust(left=.43,right=.95,bottom=.15,top=.74)
mc=ms[:9]+['ridge_embedding','ridge_identity','perturbmean','control']
panel(ax,c,'rmse_pp','method',mc,'Composition RMSE (pp) ↓',True)
f.suptitle('Forward prediction of cell-type composition',fontsize=5,y=.97)
f.savefig(O/'panel_b.pdf');plt.close(f)
(O/'provenance.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [D/'expression_per_seed_all_metrics.csv',D/'composition_per_seed_all_metrics.csv',N/'figures_latest_20261114/expression_plot_data.csv']},indent=2))
print('Verified 3 seeds per trained model; STACK:',e.loc[e.model=='stack_traincontext',['seed','MSE_raw','PCC_gene_raw','PDS']].to_string(index=False))
print('FINAL_HASHES',[(p.name,hashlib.sha256(p.read_bytes()).hexdigest()) for p in [O/'panel_b.pdf',O/'panel_c.pdf']])

