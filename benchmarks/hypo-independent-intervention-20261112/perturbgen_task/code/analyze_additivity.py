"""Descriptive additive and higher-order residuals on matched observed cohorts."""
import itertools
import numpy as np
import pandas as pd
from prepare_composition import ROOT,OLD,read,write,sample_proportions
from composition_metrics import clr


def main():
    cfg=read(OLD/'benchmark.json');classes=cfg['classes'];table=sample_proportions()
    support=read(ROOT/'audit/support_manifest.json');cohort={r['sample_id']:r for r in cfg['cohort']}
    vec=lambda s:table.loc[s,['prop::'+c for c in classes]].to_numpy(float)
    refs=pd.read_csv(ROOT/'results/additive_references.csv');out=[];diagnostics=[]
    for row in support:
        if not row['eligible']:continue
        sid=row['sample_id'];control=row['control'];drugs=frozenset(x.strip() for x in cohort[sid]['drugs'].split('+'))
        observed=vec(sid);base=vec(control);single=np.stack([vec(s) for s in row['singletons']]);add=refs[refs.sample_id==sid].set_index('celltype').loc[classes]
        clr_residual=clr(observed)-clr(base)-(clr(single)-clr(base)).sum(0)
        for j,c in enumerate(classes):
            out.append(dict(sample_id=sid,celltype=c,observed=observed[j],control=base[j],raw_additive=add.raw.iloc[j],clr_additive=add.clr.iloc[j],observed_minus_clr_additive_pp=100*(observed[j]-add.clr.iloc[j]),clr_nonadditivity=clr_residual[j],support='same batch training singletons; dose/timing not verified'))
        if len(drugs)!=3:continue
        pairs=[];missing=[]
        for pair in itertools.combinations(sorted(drugs),2):
            matches=[]
            for candidate in cfg['splits']['train']:
                if table.loc[candidate,'batch']!=table.loc[sid,'batch']:continue
                if frozenset(x.strip() for x in cohort[candidate]['drugs'].split('+'))==frozenset(pair):matches.append(candidate)
            if len(matches)==1:pairs.append(matches[0])
            else:missing.append('+'.join(pair))
        if missing:
            diagnostics.append(dict(sample_id=sid,three_way_eligible=False,missing_training_pairs=missing));continue
        pp=np.stack([vec(s) for s in pairs]);raw3=observed-pp.sum(0)+single.sum(0)-base
        clr3=clr(observed)-clr(pp).sum(0)+clr(single).sum(0)-clr(base)
        diagnostics.append(dict(sample_id=sid,three_way_eligible=True,pairs=pairs,singletons=row['singletons']))
        for j,c in enumerate(classes):
            hit=next(x for x in out if x['sample_id']==sid and x['celltype']==c);hit['raw_third_order_pp']=100*raw3[j];hit['clr_third_order']=clr3[j]
    pd.DataFrame(out).to_csv(ROOT/'results/additivity_descriptive.csv',index=False)
    write(ROOT/'audit/additivity_complete.json',dict(status='descriptive_complete',supported_conditions=[s['sample_id'] for s in support if s['eligible']],raw_out_of_bounds=int(refs.raw_out_of_bounds.sum()),three_way=diagnostics,
        inference='Computational deviations from a specified additive reference; not validated biological synergy.',replication='No independent biological replicate inference or cell-as-replicate p-values.',dose_timing='Not yet verified; matching established for batch and drug identities only.'))
    print('ADDITIVITY_DESCRIPTIVE_COMPLETE',flush=True)


if __name__=='__main__':main()
