"""Prepare incremental composition references and train a shared readout."""
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
import pandas as pd
from composition_metrics import clr,softmax,metrics,clr_additive,simplex_projection,validate,rank_metrics,response_metrics

ROOT=Path(__file__).resolve().parents[1] if Path(__file__).parent.name=='code' else Path(__file__).parent
OLD=Path('/liaozizhuo/hypoblast/benchmarks/hypo-forward-embedding-combo-20261029')
PROJECT=Path('/liaozizhuo/hypoblast')

def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.partial');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def sample_proportions():
    destination=ROOT/'results/sample_proportions_v7.csv'
    if destination.exists():return pd.read_csv(destination).set_index('sample_id')
    cfg=read(OLD/'benchmark.json');data=read(OLD/'config/stage2.json')['data'];classes=cfg['classes']
    table=pd.read_csv(PROJECT/'results/raw_h5ad_combination_celltype_20260910/sample_celltype_proportions.csv').set_index('sample_id')
    meta=pd.read_csv(data['cell_metadata_csv'],usecols=['condition','celltype'])
    counts=pd.crosstab(meta.condition,meta.celltype).reindex(columns=classes,fill_value=0)
    original=table[['prop::'+c for c in classes]].copy()
    for sid in table.index:
        if sid not in counts.index:raise ValueError('Missing v7 metadata: '+sid)
        n=int(counts.loc[sid].sum());table.loc[sid,'n_terminal_cells']=n
        for c in classes:
            table.loc[sid,'count::'+c]=int(counts.loc[sid,c]);table.loc[sid,'prop::'+c]=counts.loc[sid,c]/n
    table.to_csv(destination)
    delta=(table[['prop::'+c for c in classes]]-original).abs()
    write(ROOT/'audit/annotation_reconciliation.json',dict(source=data['cell_metadata_csv'],source_sha256=sha(data['cell_metadata_csv']),previous_table='raw h5ad annotation',chosen='metadata_updated_v7 used by frozen benchmark',max_absolute_change=float(delta.to_numpy().max()),changed_classes=[c[6:] for c in delta if delta[c].max()>1e-8],changed_samples=int((delta.max(axis=1)>1e-8).sum())))
    return table

def save_prediction(name,pred,ids,classes,seed=-1,track='locked_checkpoint',**meta):
    pred=validate(pred);dest=ROOT/'results'/track/(name+f'_seed{seed}');dest.mkdir(parents=True,exist_ok=True)
    np.savez(dest/'predictions.npz',sample_ids=np.array(ids),classes=np.array(classes),proportions=pred,model=np.array(name),seed=np.array(seed))
    write(dest/'provenance.json',dict(method=name,seed=seed,track=track,prediction_sha256=sha(dest/'predictions.npz'),**meta))

def evaluate_predictions():
    cfg=read(OLD/'benchmark.json');ids=cfg['splits']['test'];classes=cfg['classes']
    truth=dict(np.load(OLD/'data/truth.npz',allow_pickle=False));idx={s:i for i,s in enumerate(truth['sample_ids'])}
    sample=sample_proportions()
    controls={r['sample_id']:r['control_sample_id'] for r in cfg['cohort']}
    rows=[]
    for f in sorted((ROOT/'results').glob('*/*/predictions.npz')):
        a=dict(np.load(f,allow_pickle=False));current=a['sample_ids'].astype(str).tolist();p=validate(a['proportions'])
        if a['classes'].tolist()!=classes or len(set(current))!=len(current):raise ValueError(str(f))
        q=truth['proportions'][[idx[s] for s in current]]
        control=np.stack([sample.loc[controls[s],['prop::'+c for c in classes]].to_numpy(float) for s in current])
        m=metrics(p,q);m.update(method=str(a['model']),seed=int(a['seed']),track=f.parent.parent.name,n_conditions=len(current),coverage=len(current)/14,sample_ids=current)
        m['response']={str(t):response_metrics(p,q,control,t) for t in [.01,.02,.05]}
        m['ranking']={str(k):[rank_metrics(p[:,j],q[:,j],k) for j in range(len(classes))] for k in [1,3,5] if len(current)>=k}
        write(f.parent/'metrics.json',m)
        rows.append({k:v for k,v in m.items() if k in ['method','seed','track','n_conditions','coverage','mae_pp','rmse_pp','jsd_bits']})
    pd.DataFrame(rows).to_csv(ROOT/'results/metrics_available.csv',index=False)
    write(ROOT/'audit/evaluation_status.json',dict(status='available_predictions_evaluated',rows=len(rows),complete_benchmark=False,updated=time.time()))

def prepare_baselines():
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    cfg=read(OLD/'benchmark.json');truth=dict(np.load(OLD/'data/truth.npz',allow_pickle=False))
    assert [len(cfg['splits'][s]) for s in ['train','val','test']]==[146,21,14]
    assert sha(OLD/'data/truth.npz')=='bfa806e230edb54c667ae264f8a9da84d2d4f9c39cc5b57a405632ad9f1c5f9a'
    classes=cfg['classes']; ids=truth['sample_ids'].astype(str).tolist();idx={s:i for i,s in enumerate(ids)};cohort={r['sample_id']:r for r in cfg['cohort']}
    split={s:np.array([idx[i] for i in cfg['splits'][s]]) for s in ['train','val','test']};test_ids=cfg['splits']['test']
    sample=sample_proportions()
    vec=lambda s:sample.loc[s,['prop::'+c for c in classes]].to_numpy(float)
    q=validate(truth['proportions']);controls=np.stack([vec(cohort[s]['control_sample_id']) for s in ids]);validate(controls)
    assert np.allclose(np.stack([vec(s) for s in ids]),q,atol=1e-8)
    write(ROOT/'config/contract.json',dict(splits=cfg['splits'],classes=classes,seeds=[43,44,45],old_benchmark=str(OLD),truth_sha256=sha(OLD/'data/truth.npz'),source='matched D20 control',primary_metric='condition_balanced_mae_pp',exposure='development_exposed',perturbgen_required=True,expression_rows_existing=24,expression_rows_new=3,expression_rows_final=27))
    for name,p in [('control',controls[split['test']]),('perturbmean',np.repeat(q[split['train']].mean(0)[None],14,axis=0))]:save_prediction(name,p,test_ids,classes)
    identity=dict(np.load(OLD/'data/drug_ids_all.npz'));embedding=dict(np.load(OLD/'data/drug_embeddings_all.npz'))
    vocabulary=read(OLD/'config/adversarial_vocabulary.json')['agents'];D=len(vocabulary)
    # Recover categorical IDs from the exact old feature archive, not display strings.
    idrows=[]
    for s in ids:
        v=np.asarray(identity[s]).astype(int).reshape(-1)-1;a=np.zeros(D);a[v]=1;idrows.append(a)
    Xs={'ridge_identity':np.column_stack([np.array(idrows),clr(controls)]),
        'ridge_embedding':np.column_stack([np.stack([embedding[s].sum(0) for s in ids]),[len(embedding[s]) for s in ids],clr(controls)])}
    for name,x in Xs.items():
        scaler=StandardScaler().fit(x[split['train']]);xx=scaler.transform(x);candidates=[];best=None
        for alpha in np.logspace(-4,4,9):
            model=Ridge(alpha=alpha).fit(xx[split['train']],clr(q[split['train']]))
            score=metrics(softmax(model.predict(xx[split['val']])),q[split['val']])['mae_pp'];candidates.append(dict(alpha=float(alpha),val_mae_pp=score))
            if best is None or score<best[0]-1e-12 or (abs(score-best[0])<=1e-12 and alpha>best[1]):best=(score,float(alpha),model)
        save_prediction(name,softmax(best[2].predict(xx[split['test']])),test_ids,classes,track='val_selected',alpha=best[1],candidates=candidates)
        np.savez(ROOT/'results'/f'{name}_fit.npz',coef=best[2].coef_,intercept=best[2].intercept_,mean=scaler.mean_,scale=scaler.scale_,alpha=best[1])
    support=[];refrows=[];refids=[];refs={key:[] for key in ['additive_clr','additive_projected','additive_clr_eps1e5','additive_clr_eps1e3']}
    for sid in test_ids:
        row=sample.loc[sid];drugs=[x.strip() for x in cohort[sid]['drugs'].split('+')];single=[];missing=[]
        for drug in drugs:
            hit=sample[(sample.batch==row.batch)&(sample.drug_combination==drug)]
            hit=hit[hit.index.isin(cfg['splits']['train'])]
            if len(hit)!=1:missing.append(drug)
            else:single.append(hit.index[0])
        support.append(dict(sample_id=sid,eligible=not missing,singletons=single,missing=missing,control=cohort[sid]['control_sample_id'],dose_timing_matching='not_yet_verified'))
        if missing:continue
        refids.append(sid);c=vec(cohort[sid]['control_sample_id']);sing=np.stack([vec(i) for i in single]);raw=c+(sing-c).sum(0)
        refs['additive_clr'].append(clr_additive(c,sing));refs['additive_projected'].append(simplex_projection(raw))
        refs['additive_clr_eps1e5'].append(clr_additive(c,sing,1e-5));refs['additive_clr_eps1e3'].append(clr_additive(c,sing,1e-3))
        for j,k in enumerate(classes):refrows.append(dict(sample_id=sid,celltype=k,observed=vec(sid)[j],raw=raw[j],clr=refs['additive_clr'][-1][j],projected=refs['additive_projected'][-1][j],raw_out_of_bounds=bool(raw[j]<0 or raw[j]>1)))
    assert refids==['H31','H46','H71','H85','H132']
    for name,values in refs.items():save_prediction(name,np.stack(values),refids,classes,track='supported_subset')
    write(ROOT/'audit/support_manifest.json',support);pd.DataFrame(refrows).to_csv(ROOT/'results/additive_references.csv',index=False)
    for seed in [43,44,45]:
        source=ROOT.parent/'hypo-forward-fig2c-20261028/references/hypo_release_20261027'/f'seed{seed}'/'celltype_proportion_predicted.csv'
        frame=pd.read_csv(source).set_index('sample_id');pred=frame.loc[test_ids,classes].to_numpy(float)
        if np.max(np.abs(pred.sum(1)-1))>1e-5:raise ValueError('Hypo rounding beyond tolerance')
        pred=pred/pred.sum(1,keepdims=True)
        save_prediction('hypo',pred,test_ids,classes,seed,source=str(source),source_sha256=sha(source),source_precision='six decimals; row normalized within 1e-5 tolerance')
    write(ROOT/'audit/baselines_complete.json',dict(status='complete',raw_out_of_bounds=int(pd.DataFrame(refrows).raw_out_of_bounds.sum()),support_count=len(refids),truth_verified=True))
    evaluate_predictions();print('BASELINES_COMPLETE',flush=True)

def fit_readout():
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    cfg=read(OLD/'benchmark.json');data=read(OLD/'config/stage2.json')['data'];classes=cfg['classes']
    rows=dict(np.load(OLD/'data/sample_rows_train_val.npz'));x=np.load(data['cell_hvg_npy'],mmap_mode='r')
    meta=pd.read_csv(data['cell_metadata_csv'],usecols=['condition','celltype']);labels=meta.celltype.map({c:i for i,c in enumerate(classes)}).to_numpy()
    selected=[];weights=[];rng=np.random.default_rng(42)
    for sid in cfg['splits']['train']:
        ix=rows[sid];ix=rng.choice(ix,min(len(ix),5000),replace=False);selected.append(ix);weights.extend([1/len(ix)]*len(ix))
    ix=np.concatenate(selected);w=np.asarray(weights);w*=len(w)/w.sum();y=labels[ix].astype(int)
    if set(y)!=set(range(len(classes))):raise ValueError('Training vocabulary incomplete')
    print('READOUT_CELLS',len(ix),'INPUT_SCALE','hypo_stage1_targets_hvg2000_as_stored',flush=True)
    xx=np.asarray(x[ix],dtype=np.float32);scaler=StandardScaler().fit(xx,sample_weight=w);xx=scaler.transform(xx)
    truth=dict(np.load(OLD/'data/truth.npz'));truthidx={s:i for i,s in enumerate(truth['sample_ids'])};best=None;records=[]
    for c in [.01,.1,1,10]:
        started=time.time();model=LogisticRegression(C=c,max_iter=500,tol=1e-4,solver='lbfgs').fit(xx,y,sample_weight=w)
        proportions=[]
        for sid in cfg['splits']['val']:
            total=np.zeros(len(classes));ri=rows[sid]
            for start in range(0,len(ri),4096):total+=model.predict_proba(scaler.transform(np.asarray(x[ri[start:start+4096]],dtype=np.float32))).sum(0)
            proportions.append(total/len(ri))
        score=metrics(np.stack(proportions),truth['proportions'][[truthidx[s] for s in cfg['splits']['val']]])['mae_pp']
        row=dict(C=c,val_mae_pp=score,iterations=int(model.n_iter_.max()),seconds=time.time()-started);records.append(row);print('READOUT_CANDIDATE',row,flush=True)
        if best is None or score<best[0]-1e-12:
            best=(score,c);np.savez(ROOT/'results/readout.npz',coef=model.coef_,intercept=model.intercept_,mean=scaler.mean_,scale=scaler.scale_,classes=np.array(classes),genes=np.array(read(OLD/'config/genes.json')),C=c)
        write(ROOT/'audit/readout_progress.json',dict(status='fitting',candidates=records,best_C=best[1]))
    write(ROOT/'audit/readout_complete.json',dict(status='complete',candidates=records,best_C=best[1],train_cells=len(ix),source_scale=cfg['expression_scale'],sha256=sha(ROOT/'results/readout.npz')))
    print('READOUT_COMPLETE',flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--readout',action='store_true');a.add_argument('--evaluate',action='store_true');args=a.parse_args()
    if args.evaluate:evaluate_predictions()
    elif args.readout:fit_readout()
    else:prepare_baselines()
