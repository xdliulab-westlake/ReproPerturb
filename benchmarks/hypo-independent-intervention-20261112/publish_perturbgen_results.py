from pathlib import Path
import json,hashlib,sys,shutil
import numpy as np,pandas as pd
BASE=Path('/liaozizhuo/hypoblast/benchmarks');B=BASE/'hypo-independent-intervention-20261112';P=B/'perturbgen_task';OLD=BASE/'hypo-forward-embedding-combo-20261029';O=sorted((BASE/'native-alignment-20261111').glob('latest_overall_benchmark_*'))[-1];sys.path.insert(0,str(P/'code'))
from composition_metrics import metrics
cfg=json.loads((OLD/'benchmark.json').read_text());truth=np.load(OLD/'data/truth.npz');ids=cfg['splits']['test'];ix=[truth['sample_ids'].tolist().index(s) for s in ids];e=pd.read_csv(O/'expression_per_seed_all_metrics.csv');c=pd.read_csv(O/'composition_per_seed_all_metrics.csv');ee=[];cc=[];sources=[]
version='independent trainable intervention embeddings; native PerturbGen backbone + endpoint combination adaptation; no Hypo drug embeddings'
for seed in [43,44,45]:
 done=P/f'runs/perturbgen_seed{seed}/COMPLETE.json';assert done.exists(),str(done)
 d=P/f'results/expression/perturbgen_seed{seed}';a=np.load(d/'predictions.npz');assert a['sample_ids'].tolist()==ids and a['genes'].tolist()==truth['genes'].tolist();assert a['expression'].shape==(14,2000) and np.isfinite(a['expression']).all()
 em=json.loads((d/'expression_metrics.json').read_text())['summary'];ee.append(dict(model='perturbgen',seed=seed,version=version,**em))
 old=c[(c.method=='perturbgen')&(c.seed==seed)];assert len(old)==1
 track=old.iloc[0]['track'];assert track in ['locked_checkpoint','val_selected'],track
 p=P/f'results/{track}/perturbgen_seed{seed}/predictions.npz';a=np.load(p);assert a['sample_ids'].tolist()==ids and a['classes'].tolist()==cfg['classes'];cm=metrics(a['proportions'],truth['proportions'][ix]);cc.append(dict(method='perturbgen',seed=seed,track=track,version=version,**{k:cm[k] for k in ['mae_pp','rmse_pp','jsd_bits']}));sources.append(dict(seed=seed,expression_source=str(d),composition_source=str(p),composition_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),completion=json.loads(done.read_text())))
prov=json.loads((P/'PROVENANCE.json').read_text());assert prov['hypo_drug_embedding_used'] is False
backup=O/'before_independent_perturbgen';backup.mkdir(exist_ok=True)
for n in ['expression_per_seed_all_metrics.csv','composition_per_seed_all_metrics.csv','REPORT.md','PROVENANCE.json']:shutil.copy2(O/n,backup/n)
e=pd.concat([e[e.model!='perturbgen'],pd.DataFrame(ee)],ignore_index=True);c=pd.concat([c[c.method!='perturbgen'],pd.DataFrame(cc)],ignore_index=True);e.to_csv(O/'expression_per_seed_all_metrics.csv',index=False);c.to_csv(O/'composition_per_seed_all_metrics.csv',index=False)
es=e.groupby('model').agg(runs=('model','size'),MSE=('MSE_raw','mean'),MSE_SD=('MSE_raw','std'),gene_PCC=('PCC_gene_raw','mean'),gene_PCC_SD=('PCC_gene_raw','std'));cs=c.groupby('method').agg(runs=('method','size'),MAE_pp=('mae_pp','mean'),MAE_pp_SD=('mae_pp','std'),RMSE_pp=('rmse_pp','mean'),JSD_bits=('jsd_bits','mean'));es.to_csv(O/'expression_summary.csv');cs.to_csv(O/'composition_summary.csv')
def md(d):
 d=d.reset_index();out=['| '+' | '.join(d.columns)+' |','| '+' | '.join(['---']*len(d.columns))+' |']
 for r in d.itertuples(index=False,name=None):out.append('| '+' | '.join('NA' if pd.isna(x) else f'{x:.6f}' if isinstance(x,float) else str(x) for x in r)+' |')
 return '\n'.join(out)
report='# Latest available benchmark; PerturbGen independent embeddings updated\n\n14 fixed test conditions,2000 genes,10 V7 classes. Independent per-seed metrics before averaging. PerturbGen old rows replaced only after all3 new test runs completed; historical source files and pre-update tables preserved. Other model versions unchanged by this update.\n\n## Expression\n\n'+md(es)+'\n\n## Composition\n\n'+md(cs)+'\n\nSee version column in per-seed tables for each method. Training/checkpoint protocols are not uniform; STACK remains train-context adaptation. PDS/Direction Reach completeness is not claimed. PerturbGen retains native token generation and CountHead with endpoint-combination conditioning adaptation; teacher/generated head distribution difference remains disclosed.\n';(O/'REPORT.md').write_text(report)
v=json.loads((O/'PROVENANCE.json').read_text());v['perturbgen_independent_update']=dict(version=version,sources=sources,provenance=prov);(O/'PROVENANCE.json').write_text(json.dumps(v,indent=2));(P/'PUBLISHED.json').write_text(json.dumps(dict(output=str(O),seeds=[43,44,45],hypo_drug_embedding_used=False),indent=2));print('PERTURBGEN_NEW_THREE_SEED_RESULTS_PUBLISHED')
