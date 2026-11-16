from pathlib import Path
import json,hashlib
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
N=Path('/liaozizhuo/hypoblast/benchmarks/native-alignment-20261111');O=N/'figures_latest_20261114';D=N/'latest_overall_benchmark_20261113'
e=pd.read_csv(D/'expression_per_seed_all_metrics.csv');c=pd.read_csv(D/'composition_per_seed_all_metrics.csv')
inv=pd.read_csv(N/'rank_metrics/pds_inventory.csv');src=pd.read_csv(N/'rank_metrics/latest_models_pds_sources_20261113.csv')
e['PDS']=np.nan
for _,r in src.iterrows():
 name=r['method'].lower();m=next((k for k in ['state','cellflow','hypo','perturbnet','g2cp','stack','chemcpa','perturbgen','cpa'] if name.startswith(k)),None)
 if not m:continue
 m='stack_traincontext' if m=='stack' else m
 path=r.source
 if m=='stack_traincontext':path=f'/liaozizhuo/hypoblast/benchmarks/stack-unified-20261113/evaluation/stack_traincontext_seed{int(r.seed)}/predictions.npz'
 v=inv.loc[inv.source==path,'mean'];assert len(v)==1,(m,path)
 e.loc[(e.model==m)&(e.seed==r.seed),'PDS']=v.iloc[0]
order=['hypo','state','cellflow','stack_traincontext','perturbnet','cpa','chemcpa','g2cp','perturbgen','retrieval_only','ridge','perturbmean','control']
labels={'hypo':'Hypo (frozen)','state':'STATE (independent embedding)','cellflow':'CellFlow (updated)','stack_traincontext':'STACK (unified training)','perturbnet':'PerturbNet (legacy)*','cpa':'CPA','chemcpa':'chemCPA (repaired)','g2cp':'G2CP (native UniPert)','perturbgen':'PerturbGen (legacy)*','retrieval_only':'Retrieval-only baseline','ridge':'Ridge: Hypo embeddings','ridge_embedding':'Ridge: embeddings','ridge_identity':'Ridge: identity','perturbmean':'PerturbMean','control':'Control'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
def panel(ax,t,key,mc,ms,title):
 for j,m in enumerate(ms):
  v=pd.to_numeric(t.loc[t[mc]==m,key],errors='coerce').dropna().to_numpy();color='#D55E00' if m=='hypo' else '#8D8D8D' if m in ['perturbnet','perturbgen'] else '#237E69' if m in ['retrieval_only','ridge','ridge_embedding','ridge_identity','control','perturbmean'] else '#326A9E'
  if len(v):ax.errorbar(v.mean(),j,xerr=v.std(ddof=1) if len(v)>1 else 0,fmt='D',ms=4,color=color,capsize=3)
  if len(v)>1:
   for off,val in zip(np.linspace(-.15,.15,len(v)),v):ax.plot(val,j+off,'o',ms=3,mfc='white',mec=color,mew=.8,zorder=4)
  if not len(v):ax.text(.96,j,'NA',ha='right',va='center',transform=ax.get_yaxis_transform(),color='#777777')
 ax.set(yticks=range(len(ms)),yticklabels=[labels[m] for m in ms],ylim=(len(ms)-.5,-.5),title=title);ax.grid(axis='x',alpha=.2);ax.tick_params(axis='y',length=0);ax.spines['left'].set_visible(False)
keys=[('PCC_sample_raw','Sample PCC ↑'),('PCC_gene_raw','Gene PCC ↑'),('R2_raw','R² ↑'),('PCC_sample_delta','Sample PCC (delta) ↑'),('PCC_gene_delta','Gene PCC (delta) ↑'),('R2_delta','R² (delta) ↑'),('Spearman_delta','Spearman (delta) ↑'),('DEG_Jaccard100','Top-100 Jaccard ↑'),('DEG_overlap20','Top-20 overlap ↑'),('MAE_top100','Top-100 MAE ↓'),('MSE_raw','Expression MSE ↓'),('PDS','PDS: cosine of stored delta ↑')]
foot='Diamonds: mean; error bars: ±1 SD across training seeds 43/44/45. Open circles: individual runs.\nRetrieval-only: sampling replicates, not trained seeds. Fixed baselines have no seed SD.\n* PerturbNet/PerturbGen: legacy adapted results; replacement pipelines are incomplete. Checkpoint selection differs across methods.\nRetrospective, development-exposed evaluation. PDS uses stored expression deltas (not the VCC counts preset); Direction Reach pending.'
fig,axs=plt.subplots(4,3,figsize=(21,20));fig.subplots_adjust(left=.14,right=.98,top=.94,bottom=.095,wspace=1.05,hspace=.34)
for ax,(k,title) in zip(axs.flat,keys):panel(ax,e,k,'model',order,title)
fig.suptitle('Expression prediction benchmark | updated 2026-11-14\n14 test conditions · 2,000 genes',fontsize=19,y=.985);fig.text(.14,.018,foot,fontsize=10)
for ext in ['pdf','png']:fig.savefig(O/f'expression_benchmark_latest.{ext}',dpi=150)
plt.close(fig)
ms=order[:10]+['ridge_embedding','ridge_identity','perturbmean','control']
fig,axs=plt.subplots(1,3,figsize=(21,9));fig.subplots_adjust(left=.15,right=.98,top=.82,bottom=.25,wspace=1.02)
for ax,k,title in zip(axs,['mae_pp','rmse_pp','jsd_bits'],['Composition MAE (pp) ↓','Composition RMSE (pp) ↓','Jensen–Shannon divergence (bits) ↓']):panel(ax,c,k,'method',ms,title)
fig.suptitle('Cell-type composition benchmark | updated 2026-11-14\n14 test conditions · 10 V7 cell types',fontsize=20,y=.95)
fig.text(.15,.045,'Hypo: native proportion output. External models: generated expression → shared frozen cell-type readout.\n'+ '\n'.join(foot.splitlines()[:3])+'\nRetrospective, development-exposed evaluation.',fontsize=10)
for ext in ['pdf','png']:fig.savefig(O/f'composition_benchmark_latest.{ext}',dpi=150)
plt.close(fig)
e.to_csv(O/'expression_plot_data.csv',index=False);c.to_csv(O/'composition_plot_data.csv',index=False)
(O/'PROVENANCE.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [D/'expression_per_seed_all_metrics.csv',D/'composition_per_seed_all_metrics.csv',N/'rank_metrics/pds_inventory.csv']},indent=2))
assert e.loc[e.model.isin(order[:9])].groupby('model').size().eq(3).all()
print('PLOT_VERIFIED',len(e),len(c),e.PDS.notna().sum())
