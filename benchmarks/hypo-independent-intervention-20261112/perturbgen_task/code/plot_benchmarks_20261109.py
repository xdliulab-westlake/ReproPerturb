from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
B=Path('/liaozizhuo/hypoblast/benchmarks/hypo-benchmark-extension-20261105')
OLD=B.parent/'hypo-forward-embedding-combo-20261029'
OUT=B/'figures/20261109_two_benchmarks_with_perturbgen';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
order=['hypo','state','cellflow','perturbnet','cpa','chemcpa','g2cp','perturbgen','ridge','perturbmean','control']
labels={'hypo':'Hypo (ours)','state':'STATE','cellflow':'CellFlow','perturbnet':'PerturbNet','cpa':'CPA','chemcpa':'chemCPA','g2cp':'G2CP','perturbgen':'PerturbGen','ridge':'Ridge','ridge_embedding':'Ridge: embeddings','ridge_identity':'Ridge: identity','perturbmean':'PerturbMean','control':'Control'}
def color(m):return '#D55E00' if m=='hypo' else '#21856B' if m.startswith('ridge') else '#777777' if m in ['control','perturbmean'] else '#326A9E'
def panel(ax,methods,values,title,letter):
 for y,m in enumerate(methods):
  v=np.array([x for x in values[m] if x is not None],float);v=v[np.isfinite(v)]
  if len(v):
   avg=v.mean();sd=v.std(ddof=1) if len(v)>1 else 0
   ax.errorbar(avg,y,xerr=sd,fmt='D',ms=5.5,color=color(m),elinewidth=1.8,capsize=3,zorder=4)
   if len(v)>1:ax.scatter(v,y+np.linspace(-.13,.13,len(v)),s=18,facecolors='white',edgecolors=color(m),linewidths=.9,zorder=5)
  else:ax.text(.98,y,'NA',transform=ax.get_yaxis_transform(),ha='right',va='center',color='#888888')
 ax.set_yticks(range(len(methods)),[labels[m] for m in methods]);ax.set_ylim(len(methods)-.5,-.5)
 ax.axhspan(-.42,.42,color='#D55E00',alpha=.08,zorder=0)
 ax.grid(axis='x',color='#D9DEE3',lw=.6);ax.set_axisbelow(True);ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
 for tick,m in zip(ax.get_yticklabels(),methods):
  if m=='hypo':tick.set_color(color(m));tick.set_fontweight('bold')
 ax.set_title(title,loc='left',pad=10);ax.text(-.22,1.06,letter,transform=ax.transAxes,fontweight='bold',fontsize=14)
 ax.margins(x=.12)
rows=list(csv.DictReader((OLD/'output/pdf/all_seed_metrics.csv').open()))
assert len(rows)==24
original_rows=[dict(r) for r in rows]
import sys,importlib.util,hashlib
sys.path.insert(0,str(OLD))
spec=importlib.util.spec_from_file_location('original_metrics',OLD/'metrics.py');em=importlib.util.module_from_spec(spec);spec.loader.exec_module(em)
truth=dict(np.load(OLD/'data/truth.npz',allow_pickle=False));ti={str(s):i for i,s in enumerate(truth['sample_ids'])}
cfg=json.loads((OLD/'benchmark.json').read_text())
verification=[]
for seed in [43,44,45]:
 folder=B/f'results/expression/perturbgen_seed{seed}'
 assert json.loads((B/f'runs/perturbgen_seed{seed}/COMPLETE.json').read_text())['status']=='complete'
 pred=dict(np.load(folder/'predictions.npz',allow_pickle=False))
 assert pred['sample_ids'].tolist()==cfg['splits']['test'] and np.array_equal(pred['genes'],truth['genes'])
 ix=[ti[str(s)] for s in pred['sample_ids']]
 got=em.expression_metrics(pred['expression'],truth['expression'][ix],truth['control_expression'][ix])['summary']
 old=json.loads((folder/'expression_metrics.json').read_text())['summary']
 for k,v in got.items():
  assert (v is None and old[k] is None) or (v is not None and np.isclose(v,old[k],atol=1e-10)),k
 rows.append(dict(model='perturbgen',seed=seed,**got))
 verification.append(dict(seed=seed,status='expression_recomputed_matches',prediction_path=str(folder/'predictions.npz'),prediction_sha256=hashlib.sha256((folder/'predictions.npz').read_bytes()).hexdigest(),provenance=json.loads((folder/'provenance.json').read_text())))
assert rows[:24]==original_rows and len(rows)==27
fields=list(dict.fromkeys(k for r in rows for k in r))
with (OUT/'expression_all_seed_metrics.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
(OUT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
keys=[('PCC_sample_raw','Sample-wise PCC (raw) ↑'),('PCC_gene_raw','Gene-wise PCC (raw) ↑'),('R2_raw','R² (raw) ↑'),('PCC_sample_delta','Sample-wise PCC (Δ) ↑'),('PCC_gene_delta','Gene-wise PCC (Δ) ↑'),('R2_delta','R² (Δ) ↑'),('Spearman_delta','Spearman correlation (Δ) ↑'),('DEG_Jaccard100','Top-100 Jaccard ↑'),('DEG_overlap20','Top-20 overlap ↑'),('MAE_top100','Top-100 MAE ↓'),('MSE_raw','Expression MSE ↓')]
fig,axes=plt.subplots(4,3,figsize=(16,18));fig.subplots_adjust(left=.12,right=.98,top=.92,bottom=.145,wspace=.62,hspace=.46)
for i,(k,title) in enumerate(keys):
 values={m:[float(r[k]) if r[k] else None for r in rows if r['model']==m] for m in order}
 panel(axes.flat[i],order,values,title,chr(97+i))
axes.flat[-1].axis('off')
axes.flat[-1].text(0,.95,'HOW TO READ\n\nDiamond: mean across training seeds\nOpen circles: seeds 43 / 44 / 45\nWhiskers: ±1 sample SD\nFixed baselines: one diamond\nNA: mathematically undefined\n\n↑ Higher is better   ↓ Lower is better\nEach panel uses its own native scale.',va='top',linespacing=1.6)
fig.suptitle('Expression prediction benchmark',x=.12,ha='left',y=.976,fontsize=23,fontweight='bold')
fig.text(.12,.947,'14 test conditions · 2,000 genes · 8 models × 3 seeds + 3 fixed baselines',fontsize=12,color='#4A5360')
fig.text(.12,.055,'Δ = expression minus matched endpoint control. Top-gene sets are ranked by absolute Δ, not statistical DEG calls.\nHypo: frozen epoch 150; external models: validation gene-PCC selection; Ridge: validation MSE selection.\nDevelopment-exposed retrospective evaluation. Error bars describe training-seed variability, not confidence intervals.\nPerturbGen: frozen MaskGit + expression-PCC-selected CountHead; all matched control cells.',fontsize=10,linespacing=1.5,color='#4A5360')
for ext in ['png','pdf','svg']:fig.savefig(OUT/f'expression_benchmark.{ext}',dpi=220,facecolor='white')
plt.close(fig)
methods=order[:8]+['ridge_embedding','ridge_identity','perturbmean','control'];values={k:{} for k in ['mae_pp','rmse_pp','jsd_bits']}
for m in methods:
 track='val_selected' if m.startswith('ridge') else 'locked_checkpoint'
 seeds=[43,44,45] if m in order[:8] else [-1]
 if m=='perturbgen':
  sys.path.insert(0,str(B/'code'))
  from composition_metrics import metrics as cm
  records=[]
  for seed in seeds:
   folder=B/f'results/locked_checkpoint/perturbgen_seed{seed}';pp=dict(np.load(folder/'predictions.npz',allow_pickle=False));prov=json.loads((folder/'provenance.json').read_text())
   assert hashlib.sha256((folder/'predictions.npz').read_bytes()).hexdigest()==prov['prediction_sha256']
   assert pp['classes'].tolist()==cfg['classes'] and pp['sample_ids'].tolist()==cfg['splits']['test']
   qq=truth['proportions'][[ti[str(x)] for x in pp['sample_ids']]];mm=cm(pp['proportions'],qq)
   assert np.isclose(mm['mae_pp'],100*np.abs(pp['proportions']-qq).mean())
   assert np.isclose(mm['rmse_pp'],100*np.sqrt(((pp['proportions']-qq)**2).mean()))
   mm.update(n_conditions=14,coverage=1);records.append(mm)
   (OUT/f'perturbgen_seed{seed}_composition_metrics.json').write_text(json.dumps(dict(metrics=mm,provenance=prov),indent=2)+'\n')
 else:
  records=[json.loads((B/'results'/track/f'{m}_seed{s}'/'metrics.json').read_text()) for s in seeds]
 assert all(r['n_conditions']==14 and r['coverage']==1 for r in records)
 for k in values:values[k][m]=[r[k] for r in records]
fig,axes=plt.subplots(1,3,figsize=(17,8));fig.subplots_adjust(left=.13,right=.98,top=.76,bottom=.29,wspace=.65)
for i,(k,title) in enumerate([('mae_pp','Composition MAE (pp) ↓'),('rmse_pp','Composition RMSE (pp) ↓'),('jsd_bits','Jensen–Shannon divergence (bits) ↓')]):panel(axes[i],methods,values[k],title,chr(97+i))
fig.suptitle('Cell-type composition prediction benchmark',x=.13,ha='left',y=.96,fontsize=22,fontweight='bold')
fig.text(.13,.89,'14 test conditions · 10 cell types · frozen-checkpoint comparison',fontsize=12,color='#4A5360')
fig.text(.13,.835,'Diamond: seed mean    Open circles: individual seeds    Whiskers: ±1 SD    pp: percentage points',fontsize=10)
fig.text(.13,.105,'Hypo: native composition output. External models: generated expression → one shared, frozen cell-type readout.\nSTATE/CellFlow/other prior external models: 5,000 cells × 3 repeats; PerturbGen: all matched control cells.\nRidge models are fitted for composition and selected on validation MAE; fixed baselines have no seed SD.\nCheckpoint selection differs across methods. PerturbGen: expression-selected frozen CountHead. Development-exposed evaluation.',fontsize=10,linespacing=1.5,color='#4A5360')
for ext in ['png','pdf','svg']:fig.savefig(OUT/f'composition_benchmark.{ext}',dpi=220,facecolor='white')
plt.close(fig)
with (OUT/'composition_all_seed_metrics.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['method','seed','track','mae_pp','rmse_pp','jsd_bits']);w.writeheader()
 for m in methods:
  seeds=[43,44,45] if m in order[:8] else [-1]
  for i,seed in enumerate(seeds):w.writerow(dict(method=m,seed=seed,track='val_selected' if m.startswith('ridge') else 'locked_checkpoint',**{k:values[k][m][i] for k in values}))
(OUT/'README.md').write_text('# Benchmark update with PerturbGen — 2026-11-09\nBoth expression and composition figures now include seeds 43/44/45 for PerturbGen. Prior methods and their original metric values are retained. Expression: 14 test conditions, 2000 genes, 27 rows (8 models × 3 seeds + 3 fixed baselines). Composition: the same 14 conditions and 10 V7 classes, 28 rows (8 models × 3 seeds + 4 baselines).\nEach seed is evaluated before taking mean and sample SD. PerturbGen uses the locked_checkpoint composition track: frozen MaskGit plus expression-PCC-selected CountHead, generated from all matched controls. Other previous external models retain their frozen shared-readout predictions with 5000 cells × 3 sampling repeats. The val_selected PerturbGen composition checkpoint is not substituted. Model selection and sampling are not unified across methods.\nExpression predictions for all three new seeds were re-evaluated with the original evaluator and matched their stored metrics. All original 24 expression metric rows were preserved exactly. See verification.json for expression provenance and the existing results/locked_checkpoint/perturbgen_seed*/provenance.json for composition provenance. Source tables and records remain on the server; PDF/PNG/SVG are delivered locally.\n')
print('PERTURBGEN_COMPOSITION', {k:values[k]['perturbgen'] for k in values})
print('PERTURBGEN_EXPRESSION',[(r['seed'],r['PCC_gene_raw'],r['MSE_raw']) for r in rows if r['model']=='perturbgen'])
print('FIGURES_COMPLETE',OUT)
