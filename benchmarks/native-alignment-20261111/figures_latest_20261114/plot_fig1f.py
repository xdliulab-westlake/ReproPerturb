from pathlib import Path
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
O=Path('/liaozizhuo/hypoblast/benchmarks/native-alignment-20261111/figures_latest_20261114')
t=pd.read_csv(O/'composition_dominant_plot_data.csv');font_manager.fontManager.addfont('/tmp/hypo_Arial.ttf');plt.rcParams.update({'font.family':'Arial','pdf.fonttype':42,'font.size':3.3,'axes.linewidth':.35,'xtick.major.width':.35,'xtick.major.size':1.5})
ms=['hypo','state','cellflow','stack_traincontext','perturbnet','cpa','chemcpa','g2cp','perturbgen','retrieval_only','ridge_embedding','ridge_identity','perturbmean','control'];names=['ReproPerturb','STATE','CellFlow','STACK','PerturbNet','CPA','chemCPA','G2CP','PerturbGen','Retrieval only','Ridge (emb.)','Ridge (identity)','Perturbation mean','Control']
f,axs=plt.subplots(1,3,figsize=(215/72,116/72));f.subplots_adjust(left=.22,right=.96,bottom=.14,top=.80,wspace=.20)
for z,(ax,key,title) in enumerate(zip(axs,['mae_pp','jsd_bits','dominant_accuracy_pct'],['MAE (pp) ↓','JSD (bits) ↓','Accuracy (%) ↑'])):
 ax.axhspan(-.43,.43,facecolor='#FCF3ED',edgecolor='#F7E2D4',linewidth=.3,zorder=0)
 for j,m in enumerate(ms):
  v=t.loc[t.method==m,key].to_numpy();color='#D55E00' if m=='hypo' else '#26745F' if j>=9 else '#326A9E'
  ax.errorbar(v.mean(),j,xerr=v.std(ddof=1) if len(v)>1 else 0,fmt='D',ms=1.6,color=color,lw=.5,capsize=1)
  if len(v)>1:ax.scatter(v,j+np.linspace(-.16,.16,len(v)),s=3,facecolors='white',edgecolors=color,linewidths=.35,zorder=4)
 ax.set(ylim=(13.7,-.7),yticks=range(14),yticklabels=names if z==0 else [],title=title);ax.tick_params(axis='y',length=0,pad=2);ax.grid(axis='x',lw=.25,alpha=.3)
 for side in ['top','right','left']:ax.spines[side].set_visible(False)
 if z==2:ax.set_xlim(0,100);ax.set_xticks([0,50,100])
f.suptitle('Forward prediction of cell-type composition',fontsize=5,y=.96)
f.savefig(O/'Fig1f_latest_seeds.pdf');plt.close(f)
