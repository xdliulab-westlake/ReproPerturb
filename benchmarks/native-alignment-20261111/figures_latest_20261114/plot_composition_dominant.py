from pathlib import Path
p=Path('/liaozizhuo/hypoblast/benchmarks/native-alignment-20261111/figures_latest_20261114/plot.py')
exec(p.read_text().split('keys=[')[0])
extra=pd.read_csv(N/'composition_extra_20261114/per_seed.csv')
c=c.merge(extra[['method','seed','dominant_accuracy_pct']],on=['method','seed'],how='left',validate='one_to_one')
B=N.parent;truth=np.load(B/'hypo-forward-embedding-combo-20261029/data/truth.npz');cfg=json.loads((B/'hypo-forward-embedding-combo-20261029/benchmark.json').read_text());ids=cfg['splits']['test'];q=truth['proportions'][[truth['sample_ids'].tolist().index(i) for i in ids]]
retr=[]
for s in [43,44,45]:
 root=B/f'stack-unified-20261113/evaluation/stack_traincontext_seed{s}'
 x=np.stack([np.load(root/f'{i}.npz')['baseline_proportions'] for i in ids]);mask=(c.method=='retrieval_only')&(c.seed==s)
 assert np.isclose(100*np.abs(x-q).mean(),c.loc[mask,'mae_pp'].iloc[0],atol=1e-6)
 hit=[]
 for sid,a,b in zip(ids,x,q):
  pred=np.flatnonzero(np.isclose(a,a.max(),rtol=0,atol=1e-12));true=np.flatnonzero(np.isclose(b,b.max(),rtol=0,atol=1e-12));v=100*len(set(pred)&set(true))/len(pred);hit.append(v);retr.append(dict(seed=s,sample_id=sid,dominant_accuracy_pct=v))
 c.loc[mask,'dominant_accuracy_pct']=np.mean(hit)
assert c.dominant_accuracy_pct.notna().all()
ms=order[:10]+['ridge_embedding','ridge_identity','perturbmean','control']
fig,axs=plt.subplots(1,3,figsize=(21,9));fig.subplots_adjust(left=.15,right=.98,top=.82,bottom=.25,wspace=1.02)
for ax,k,title in zip(axs,['mae_pp','jsd_bits','dominant_accuracy_pct'],['Composition MAE (pp) ↓','Jensen–Shannon divergence (bits) ↓','Dominant cell-type accuracy (%) ↑']):panel(ax,c,k,'method',ms,title)
axs[2].set_xlim(0,100)
fig.suptitle('Cell-type composition benchmark | updated 2026-11-14\n14 test conditions · 10 V7 cell types',fontsize=20,y=.95)
fig.text(.15,.035,'MAE: mean absolute proportion error over 10 cell types, then equally over 14 conditions (percentage points).\nDominant accuracy: agreement of the largest-proportion cell type; exact ties are handled fractionally.\nOpen circles: seeds 43/44/45; diamonds: mean; error bars: ±1 SD. Retrieval-only: sampling replicates. Fixed baselines: no seed SD.\nHypo: native proportions. External models: generated expression → shared frozen readout.\n* PerturbNet/PerturbGen: legacy results; new pipelines pending. Checkpoint selection differs; retrospective evaluation.',fontsize=10)
for ext in ['pdf','png']:fig.savefig(O/f'composition_benchmark_dominant_accuracy.{ext}',dpi=150)
c.to_csv(O/'composition_dominant_plot_data.csv',index=False);pd.DataFrame(retr).to_csv(O/'retrieval_dominant_per_condition.csv',index=False)
print('VERIFIED',len(c),'rows; all dominant scores present; retrieval MAE matched')
