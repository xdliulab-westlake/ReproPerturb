from pathlib import Path
import json
import pandas as pd
O=Path(__file__).resolve().parent
d=pd.read_csv(O/'metrics.csv'); d=d[(d.seed=='mean') & (d['mode']=='overlap_excluded')].drop_duplicates(['recipe_id','target','method','space'])
rows=[]
for space in ['identity17','signed34']:
 w=d[d.space==space].pivot(index=['recipe_id','target'],columns='method',values='AP')
 for name,z in [('all_15_recipes',w),('excluding_conflicting_Y06',w.loc[w.index.get_level_values(0)!='Y06'])]:
  for baseline in ['forward_raw','forward_matched','frequency']:
   delta=(z.relation-z[baseline]).groupby(level=0).mean()
   rows.append(dict(space=space,analysis=name,baseline=baseline,n=len(delta),delta=float(delta.mean())))
 # Shared target rankings create dependence across recipes; report target aggregates.
 for baseline in ['forward_raw','forward_matched','frequency']:
  delta=(w.relation-w[baseline]).groupby(level=1).mean()
  rows.append(dict(space=space,analysis='target_aggregate',baseline=baseline,n=len(delta),delta=float(delta.mean()),wins=int((delta>0).sum())))
pd.DataFrame(rows).to_csv(O/'sensitivity_summary.csv',index=False)
text='''# 18 protocol pathway ranking comparison

Completed using existing validated features for seeds 43/44/45, intervention strength 1.5 and the same 64 backgrounds. Top 7 backgrounds are averaged separately for each signed candidate. These are target-cell recommendations, not recipe-conditioned predictions; protocols with the same target share scores.

18 source rows represent 15 recipes and 16 recipe-target views. Full rows are displayed; mAP averages target views within each recipe and then recipes. Y06 has conflicting TSC/ExEM targets. Labels are provisional spreadsheet-derived drug/pathway usage, not quantitative activity or causal truth.

Four routes: raw standardized forward response; forward with the same combination transform but relation weights zero; full relation combination; target-specific training frequency with global fallback. Identity17 uses maximum score across signs; signed34 retains both signs. Identity ranking can therefore depend on direction scores. AP is threshold-grouped for ties; Recall@K uses expected boundary tie hits.

Overlap-excluded sensitivity removes five of 77 relation references by exact drug-set OR exact signed-pathway-label agreement with any evaluation protocol. Remaining references: 72; all 64 contexts remain. Refit uses locked historical hyperparameters, not nested re-selection. Original flow training and hyperparameter development exposure persist; do not call this blind validation. Stage2 overlaps and reference identifiers are recorded in overlap_audit.json.

Main result, overlap-excluded: identity mAP 0.723 with relation versus 0.621 matched forward, 0.632 raw forward, 0.626 frequency. Signed mAP 0.550 versus 0.526 matched forward, 0.560 raw forward, 0.547 frequency. Evidence supports an incremental benefit for pathway identity prioritization, with no robust improvement in signed ranking. It does not establish biological necessity.

Recipe-bootstrap intervals and sign-flip p values in summary.csv are descriptive only: recipes sharing a target share predictions, and sources can be related. They must not be reported as confirmatory independent-scenario tests. Target aggregates and exclusion of conflicting Y06 are provided in sensitivity_summary.csv; seed-level metrics remain in metrics.csv. Remaining timing/completeness issues were not resolved by this computational analysis.

Validation: all 18 protocols map to the released dictionary; 18/15/16 counts checked; cached dimensions and finite values checked; published relation scores reproduced; metric perfect/tie cases checked; excluded relation reference intersections verified empty. See VERIFICATION.json.
'''
(O/'REPORT.md').write_text(text)
print(pd.DataFrame(rows).to_string(index=False));print((O/'VERIFICATION.json').read_text())
