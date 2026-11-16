from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
O=Path('/liaozizhuo/hypoblast/reproductions/hypo17_unified_release_20261027_retrain_20261109/all_celltype_heatmaps')
d=pd.read_csv(O/'scores_all_celltypes_34.csv');classes=d.celltype.drop_duplicates().tolist();paths=d.pathway.drop_duplicates().tolist();support=json.loads((O/'VERIFICATION.json').read_text())['support'];a=np.empty((3,10,17,2))
for si,s in enumerate([43,44,45]):
 for ci,c in enumerate(classes):
  for pi,p in enumerate(paths):
   for di,direction in enumerate(['+','-']):a[si,ci,pi,di]=d.loc[(d.seed==s)&(d.celltype==c)&(d.pathway==p)&(d.direction==direction),'score'].item()
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none'})
for n in [17,34]:
 per=a.max(3) if n==17 else a.reshape(3,10,34);mean=per.mean(0);vmin=min(-.01,per.min(),mean.min());vmax=max(.01,per.max(),mean.max());norm=TwoSlopeNorm(vmin=vmin,vcenter=0,vmax=vmax)
 # Physical axes width / height = number of columns / rows.
 cell=.38;left=2.35;bottom=2.5;top=.9;right=1.0;w=left+10*cell+right;h=bottom+n*cell+top
 fig=plt.figure(figsize=(w,h));ax=fig.add_axes([left/w,bottom/h,10*cell/w,n*cell/h]);im=ax.imshow(mean.T,aspect='equal',cmap='RdBu_r',norm=norm)
 labels=paths if n==17 else [p+sign for p in paths for sign in ['+','-']]
 ax.set(yticks=range(n),yticklabels=labels,xticks=range(10),xticklabels=[c+(' *' if not support.get(c,0) else '') for c in classes],title='Three-seed mean: '+('pathway identity scores' if n==17 else 'signed relation scores'))
 ax.tick_params(axis='x',rotation=65,labelsize=8);ax.tick_params(axis='y',labelsize=9)
 cax=fig.add_axes([(left+10*cell+.25)/w,bottom/h,.15/w,n*cell/h]);fig.colorbar(im,cax=cax,label='Relative score')
 fig.text(.5,.045,'Mean of seeds 43/44/45; same 64 contexts. Colors indicate scores, not activation or proportion changes.\n* No target-cell-type labels in relation training; extrapolative scores.\n'+('Identity score = max over directions within each seed, followed by seed averaging.' if n==17 else 'Positive and negative directions are retained as separate rows.'),ha='center',fontsize=8)
 assert not ax.texts;assert np.array_equal(np.asarray(im.get_array()),mean.T)
 fig.canvas.draw();box=ax.get_window_extent();assert abs(box.width/10-box.height/n)<1e-6
 for ext in ['pdf','svg','png']:fig.savefig(O/f'celltype_pathway_{n}_heatmap_mean_square.{ext}',dpi=220)
 plt.close(fig)
print('VERIFIED: square cells; original mean values and color limits preserved')

