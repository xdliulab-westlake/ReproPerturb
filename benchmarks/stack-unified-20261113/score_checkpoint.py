from pathlib import Path
import sys,os,json,hashlib,warnings
import numpy as np,torch,anndata as ad
from scipy.special import softmax
warnings.filterwarnings('ignore',message='Observation names are not unique')
B=Path('/liaozizhuo/hypoblast/benchmarks/stack-unified-20261113');OLD=B.parent/'hypo-forward-embedding-combo-20261029';EXT=B.parent/'hypo-benchmark-extension-20261105';S=B;seed=int(sys.argv[1]);split=sys.argv[2];epoch=int(sys.argv[3]);dest=Path(os.environ.get('STACK_EVAL_DEST',str(B/(f'validation/seed{seed}/epoch{epoch}' if split=='val' else f'evaluation/stack_traincontext_seed{seed}'))));dest.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,'/liaozizhuo/iPStem/benchmark/models/stack/repo/src')
from stack.models.core import StateICLModel
sys.path.append(str(OLD));sys.path.append(str(EXT/'code'));sys.path.append(str(EXT))
from metrics import expression_metrics
from composition_metrics import metrics as composition_metrics
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
v=json.loads((OLD/'benchmark.json').read_text());genes=json.loads((OLD/'config/genes.json').read_text());co={r['sample_id']:r for r in v['cohort']};ids=v['splits'][split];classes=v['classes'];assert len(ids)==(21 if split=='val' else 14)
if os.environ.get('STACK_SMOKE'):ids=ids[:1]
best=os.environ.get('STACK_EVAL_CHECKPOINT',str(S/f'runs/seed{seed}/best.ckpt'));ck=torch.load(best,map_location='cpu',weights_only=False);model=StateICLModel(**ck['hyper_parameters']['model_config']);model.load_state_dict({k.removeprefix('model.'):val for k,val in ck['state_dict'].items()});model.cuda().eval();torch.set_num_threads(4)
indices=np.load(OLD/'data/sample_rows_all.npz');raw=ad.read_h5ad(Path('/liaozizhuo/hypoblast/benchmarks/native-alignment-20261111/counts_full/recovered_counts.h5ad'),backed='r');g=raw.var_names.get_indexer(genes);assert (g>=0).all()
conf=json.loads((OLD/'config/stage2.json').read_text());stored=np.load(conf['data']['cell_hvg_npy'],mmap_mode='r')
c=np.load(EXT/'results/readout.npz');assert c['genes'].tolist()==genes and c['classes'].tolist()==classes;coef=c['coef']/c['scale'];bias=c['intercept']-c['mean']@coef.T

def convert(pred,full,original):
 # Genes outside the 2000 modelled coordinates are held at source-control counts.
 lib=np.asarray(full.sum(1)).ravel()-original.sum(1)+pred.sum(1);assert (lib>0).all()
 f=full.astype(np.float64).multiply((1e4/lib)[:,None]).tocsr();f.data=np.log1p(f.data)
 lp=np.log1p(pred*(1e4/lib)[:,None]);old=np.log1p(original*(1e4/lib)[:,None])
 denom=np.asarray(f.sum(1)).ravel()-old.sum(1)+lp.sum(1)
 return np.log1p(lp*(1e4/denom)[:,None])

def agents(s):return set(co[s]['drugs'].split(' + '))
means=[];props=[];basemeans=[];baseprops=[];mapping=[]
for cid in ids:
 outpath=dest/f'{cid}.npz'
 if outpath.exists():
  a=np.load(outpath);means.append(a['mean']);props.append(a['proportions']);basemeans.append(a['baseline_mean']);baseprops.append(a['baseline_proportions']);continue
 q=agents(cid);prompts=sorted(v['splits']['train'],key=lambda s:(-len(q&agents(s))/len(q|agents(s)),s))[:3];assert not set(prompts)&set(v['splits']['test'])
 rng=np.random.default_rng(seed+int(hashlib.sha256(cid.encode()).hexdigest()[:8],16));pi=np.concatenate([rng.choice(indices[s],512,replace=False) for s in prompts]);pa=raw[pi,:].to_memory()[:,g].copy();pa.obs_names_make_unique();pa.var['gene_symbols']=genes
 rows=indices[co[cid]['control_sample_id']];source=raw[rows,:].to_memory();full=source.X.tocsr();query=source[:,g].copy();query.obs_names_make_unique();query.var['gene_symbols']=genes;orig=query.X.toarray().astype(np.float64)
 check=convert(orig[:32],full[:32],orig[:32]);err=float(np.max(np.abs(check-stored[rows[:32]])));assert err<2e-5,('SCALE_ROUNDTRIP',err)
 with torch.inference_mode():
  generated=model.get_incontext_generation(pa,query,str(S/'genes.pkl'),num_steps=5,prompt_ratio=.25,context_ratio=.4,mode='vanilla',batch_size=8,num_workers=0,show_progress=False,random_seed=seed,gene_name_col='gene_symbols',filter_organism=False)
 pred=generated.toarray() if hasattr(generated,'toarray') else np.asarray(generated);assert pred.shape==orig.shape and np.isfinite(pred).all() and pred.min()>=0
 expr=np.empty(pred.shape,dtype=np.float32)
 for start in range(0,len(rows),256):expr[start:start+256]=convert(pred[start:start+256].astype(np.float64),full[start:start+256],orig[start:start+256])
 probability=softmax(expr.astype(np.float64)@coef.T+bias,axis=1);ps=[]
 for rs in [101,102,103]:
  rr=np.random.default_rng(rs+int(hashlib.sha256(cid.encode()).hexdigest()[:8],16));ps.append(probability[rr.choice(len(rows),5000,replace=True)].mean(0))
 prop=np.mean(ps,axis=0);prop/=prop.sum();mean=expr.mean(0,dtype=np.float64);base=np.asarray(stored[pi]);bp=softmax(base.astype(np.float64)@coef.T+bias,axis=1).mean(0);bp/=bp.sum()
 np.savez_compressed(outpath,mean=mean,proportions=prop,baseline_mean=base.mean(0,dtype=np.float64),baseline_proportions=bp,source_rows=rows,prompt_rows=pi,prompt_sample_ids=np.array(prompts),scale_roundtrip_max_error=err,**({'expression':expr,'generated_counts':pred} if split=='test' else {}))
 means.append(mean);props.append(prop);basemeans.append(base.mean(0,dtype=np.float64));baseprops.append(bp);mapping.append(dict(sample_id=cid,prompts=prompts));print('EVALUATED',seed,cid,'rows',len(rows),flush=True)
raw.file.close();t=np.load(B/'validation_truth.npz' if split=='val' else OLD/'data/truth.npz');ix=[t['sample_ids'].tolist().index(i) for i in ids]
for label,em,pr in [('stack',means,props),('retrieval_only',basemeans,baseprops)]:
 e=expression_metrics(np.array(em),t['expression'][ix],t['control_expression'][ix]);c=composition_metrics(np.array(pr),t['proportions'][ix]);(dest/f'{label}_expression_metrics.json').write_text(json.dumps(e));(dest/f'{label}_composition_metrics.json').write_text(json.dumps(c))
np.savez_compressed(dest/'predictions.npz',sample_ids=np.array(ids),genes=np.array(genes),classes=np.array(classes),expression=np.array(means),proportions=np.array(props))
(dest/'COMPLETE.json').write_text(json.dumps(dict(seed=seed,checkpoint=best,checkpoint_sha256=sha(best),evaluation_split=split,evaluation_ids=ids,script_sha256=sha(__file__),predictions_sha256=sha(dest/'predictions.npz'),truth_sha256=sha(OLD/'data/truth.npz'),adaptation='Top3 train-only Jaccard prompt retrieval, native generator, non-modelled genes held at source-control counts for scale conversion',sampling_seeds=[101,102,103],training_updates=0,test_truth_in_generation=False),indent=2));print('COMPLETE',seed,flush=True)
