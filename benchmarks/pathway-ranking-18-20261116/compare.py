"""Fixed-cache protocol concordance; no independent biological validation claim."""
import json, sys, math
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path('/liaozizhuo/hypoblast/reproductions/hypo17_unified_release_20261027_retrain_20261109')
R=B/'run'; O=Path(__file__).resolve().parent; sys.path.insert(0,str(R/'src'))
from hypo_release.unified import examples, features
from hypo_release.shared_math import kernel, combine, fitpredict
from experiments.perturbation.evaluate_prototype_mass_flow import select_split_records

def norm(x): return ''.join(c.lower() for c in x if c.isalnum())
def metric(y,s,k=5):
    y=np.asarray(y); s=np.asarray(s); total=y.sum(); assert total>0
    ap=0.; prev=0.; rec=0.
    for v in np.unique(s)[::-1]:
        mask=s==v; n=int(mask.sum()); hit=float(y[mask].sum()); above=int((s>v).sum())
        ap+=hit/total*(prev+hit)/(above+n)
        rec+=hit*min(1,max(0,k-above)/n)/total
        prev+=hit
    return ap,rec
assert metric([1,0],[1,0])==(1.,1.)
assert metric([1,0],[0,0],1)==(.5,.5)
reg=json.loads((O/'input_registry.json').read_text())['protocols']
assert len(reg)==18 and len({p['recipe_id'] for p in reg})==15
assert len({(p['recipe_id'],p['target']) for p in reg})==16
tr,te,paths,groups=examples(43)
mp={norm(r.drug):(r.pathway,'+' if r.sign>0 else '-') for r in pd.read_csv(R/'assets/drug_pathway_sign.csv').itertuples()}
target={'8CLC':'8 cell-like cell','Naïve':'iPSC_naive','Primed':'iPSC primed','TSC':'TSC-like cell','Hypo':'Hypoblast-like cell','Amnion':'Amnion-like cell','HEP':'HEP-like cell','Meso/EXEM':'Meso/ExEM-like cell'}
for p in reg:
    p['drugset']=frozenset(norm(d) for d in p['drugs']); p['truth']=frozenset(mp[d] for d in p['drugset']); p['celltype']=target[p['target']]
    assert all(t[0] in paths for t in p['truth'])
drugsets={p['drugset'] for p in reg}; truths={p['truth'] for p in reg}
payload=torch.load(R/'assets/pair_records.pt',map_location='cpu',weights_only=True)
lookup={str(r['sample_id']):r for r in payload['records']}
def drugs(r):return frozenset(norm(d) for d in r.get('drug_names',[]))
contexts={}
for r in select_split_records(payload['records'],R/'assets/context_split.csv',split='train'):
    ids=tuple(sorted(map(int,r.get('original_drug_ids',r['drug_ids']))))
    if len(ids)==2 and len(set(ids))==2:contexts.setdefault((str(r['control_sample_id']),ids),r)
contexts=list(contexts.values()); assert len(contexts)==64
keep_context=np.array([drugs(r) not in drugsets for r in contexts]); assert keep_context.any()
keep=np.array([drugs(lookup[e.sample_id]) not in drugsets and frozenset(e.truth.items()) not in truths for e in tr]); assert keep.any()
split=pd.read_csv(R/'assets/stage2_split.csv').set_index('sample_id')['split'].to_dict()
audit=[]
for p in reg:
    matches=[sid for sid,r in lookup.items() if drugs(r)==p['drugset']]
    audit.append(dict(row=p['row'],protocol=p['protocol'],recipe_id=p['recipe_id'],target=p['target'],truth=sorted(p['truth']),stage2_matches=[(sid,split.get(sid,'unlisted')) for sid in matches],relation_recipe_matches=[e.sample_id for e in tr if drugs(lookup[e.sample_id])==p['drugset']],relation_label_matches=[e.sample_id for e in tr if frozenset(e.truth.items())==p['truth']]))
(O/'overlap_audit.json').write_text(json.dumps(dict(protocols=audit,removed_training=[e.sample_id for e,k in zip(tr,keep) if not k],remaining_training=int(keep.sum()),remaining_contexts=int(keep_context.sum()),limitations='Locked hyperparameters retain historical development exposure. Protocol labels provisional.'),indent=2))
assert not any(drugs(lookup[e.sample_id]) in drugsets or frozenset(e.truth.items()) in truths for e,k in zip(tr,keep) if k)
cfg=json.loads((R/'SHARED_CONFIG.json').read_text()); assert cfg['variant']=='high'
all_scores={}; checks=[]
for seed in [43,44,45]:
    train,_,pp,_=examples(seed); assert pp==paths and [e.sample_id for e in train]==[e.sample_id for e in tr]
    ck=torch.load(R/f'weights/seed{seed}.pt',map_location='cpu',weights_only=True); classes=ck['celltype_classes']; del ck
    fs=np.load(B/f'all_celltype_heatmaps/features_seed{seed}.npy'); assert fs.shape==(64,10,71) and np.isfinite(fs).all()
    yy=np.stack([e.label.numpy() for e in tr]); state=dict(np.load(R/f'models/seed{seed}/kernel_model.npz'))
    for mode in ['original','overlap_excluded']:
        cm=np.ones(64,dtype=bool) if mode=='original' else keep_context
        tm=np.ones(len(tr),dtype=bool) if mode=='original' else keep
        if mode=='overlap_excluded':
            _,state=fitpredict(features(seed)['high'][:len(tr)][tm],yy[tm],fs[:1,0],cfg['kernel'])
        for ci,cell in enumerate(classes):
            raw=fs[cm,ci].astype(float); x=(raw-state['mean'])/state['std']; x[:,61:]=raw[:,61:]
            pred=kernel(x,state['train'],cfg['kernel'])@state['weight']
            tail=lambda a:np.sort(a,axis=0)[-math.ceil(.1*len(a)):].mean(0)
            scores={'forward_raw':tail(raw[:,:34]),'forward_matched':tail(combine(pred,raw,0,0)), 'relation':tail(combine(pred,raw,cfg['identity_alpha'],cfg['direction_alpha']))}
            pool=np.array([e.target_celltype==cell for e in tr])&tm
            scores['frequency']=yy[pool if pool.any() else tm].mean(0)
            for method,s in scores.items():all_scores[mode,seed,cell,method]=s
            if mode=='original':
                published=pd.read_csv(B/'all_celltype_heatmaps/scores_all_celltypes_34.csv'); d=published[(published.seed==seed)&(published.celltype==cell)].set_index(['pathway','direction'])
                ref=np.array([d.loc[(p,s),'score'] for p in paths for s in ['+','-']]); err=float(abs(ref-scores['relation']).max()); assert err<2e-5; checks.append(err)
for mode in ['original','overlap_excluded']:
    for cell in classes:
        for method in ['forward_raw','forward_matched','relation','frequency']:
            all_scores[mode,'mean',cell,method]=np.mean([all_scores[mode,s,cell,method] for s in [43,44,45]],axis=0)
rows=[]; rankrows=[]
for (mode,seed,cell,method),score in all_scores.items():
    assert np.isfinite(score).all()
    for p in [p for p in reg if p['celltype']==cell]:
        y=np.array([(path,sign) in p['truth'] for path in paths for sign in ['+','-']])
        for space in ['signed34','identity17']:
            s=score if space=='signed34' else score.reshape(17,2).max(1); truth=y if space=='signed34' else y.reshape(17,2).max(1)
            ap,_=metric(truth,s)
            row=dict(mode=mode,seed=seed,row=p['row'],protocol=p['protocol'],recipe_id=p['recipe_id'],target=p['target'],method=method,space=space,AP=ap)
            for k in [3,5,10]:row[f'recall{k}']=metric(truth,s,k)[1]
            rows.append(row)
        for i,(path,sign) in enumerate((p,s) for p in paths for s in ['+','-']):rankrows.append(dict(mode=mode,seed=seed,row=p['row'],method=method,pathway=path,direction=sign,score=score[i],used=bool(y[i])))
df=pd.DataFrame(rows);df.to_csv(O/'metrics.csv',index=False);pd.DataFrame(rankrows).to_csv(O/'rankings.csv',index=False)
mean=df[df.seed=='mean']; summaries=[]
for (mode,space),d in mean.groupby(['mode','space']):
    views=d.drop_duplicates(['recipe_id','target','method'])
    wide=views.pivot(index=['recipe_id','target'],columns='method',values='AP')
    for baseline in ['forward_raw','forward_matched','frequency']:
        diff=(wide.relation-wide[baseline]).groupby(level=0).mean().to_numpy()
        # Exact paired sign-flip descriptive test over 15 recipe groups.
        signs=2*((np.arange(2**len(diff))[:,None]>>np.arange(len(diff)))&1)-1
        pv=float((abs((signs*diff).mean(1))>=abs(diff.mean())-1e-12).mean())
        rng=np.random.default_rng(20261116); boot=rng.choice(diff,(10000,len(diff)),replace=True).mean(1)
        summaries.append(dict(mode=mode,space=space,baseline=baseline,relation_mAP=float(wide.relation.groupby(level=0).mean().mean()),baseline_mAP=float(wide[baseline].groupby(level=0).mean().mean()),delta=float(diff.mean()),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),wins=int((diff>1e-10).sum()),ties=int((abs(diff)<=1e-10).sum()),n_recipes=len(diff),descriptive_signflip_p=pv))
summary=pd.DataFrame(summaries);summary.to_csv(O/'summary.csv',index=False)
mean.drop_duplicates(['mode','space','recipe_id','target','method']).groupby(['mode','space','target','method'])[['AP','recall5']].mean().to_csv(O/'target_summary.csv')
plt.rcParams.update({'pdf.fonttype':42,'font.size':9})
for mode in ['original','overlap_excluded']:
    fig,axs=plt.subplots(1,2,figsize=(14,9),sharey=True)
    for ax,space in zip(axs,['identity17','signed34']):
        d=mean[(mean['mode']==mode)&(mean.space==space)].pivot(index='row',columns='method',values='AP').reindex([p['row'] for p in reg])
        for i,(_,v) in enumerate(d.iterrows()):ax.plot([v.forward_matched,v.relation],[i,i],color='#bbbbbb',zorder=1)
        for method,color,label in [('forward_matched','#4477AA','Forward (matched pipeline)'),('relation','#CC6677','With relation'),('frequency','#228833','Frequency prior')]:ax.scatter(d[method],range(18),color=color,label=label,s=25,zorder=2)
        ax.set(xlim=(0,1.02),xlabel='Average precision',title=space,yticks=range(18),yticklabels=[p['protocol'] for p in reg]);ax.grid(axis='x',alpha=.2)
    axs[0].invert_yaxis();axs[1].legend(loc='lower right',fontsize=8)
    fig.suptitle('Protocol pathway-use concordance: '+mode)
    fig.text(.5,.025,'18 rows / 15 recipes / 16 target views; mean scores of seeds 43/44/45. Shared target-specific rankings.\nProvisional protocol labels; development-exposed evaluation, not biological validation.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.065,1,.96])
    for ext in ['pdf','png']:fig.savefig(O/f'comparison_{mode}.{ext}',dpi=200)
    plt.close(fig)
(O/'VERIFICATION.json').write_text(json.dumps(dict(status='complete',rows=18,recipes=15,views=16,maximum_published_score_error=max(checks),remaining_training=int(keep.sum()),remaining_contexts=int(keep_context.sum()),metric_tests_passed=True,disjointness_passed=True),indent=2))
print(summary.to_string(index=False));print('VERIFIED_COMPLETE')
