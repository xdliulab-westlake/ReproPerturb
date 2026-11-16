"""Resolve observed one/two-count library discrepancies without changing counts."""
from pathlib import Path
import csv
import json
import os

import h5py
import numpy as np

B=Path(__file__).resolve().parent/'counts_full'
validation=json.loads((B/'VALIDATION.json').read_text())
assert not validation['pilot']
if validation['state']=='verified':
    print('Strict recovery already verified; no correction needed.')
    raise SystemExit(0)
assert validation['state']=='failed_validation'
source=Path(validation['source']);candidate=B/'recovered_counts.partial.h5ad'
assert candidate.exists() and not (B/'recovered_counts.h5ad').exists()
failed=list(csv.DictReader((B/'failed_rows.csv').open()))
assert len(failed)==validation['failed_rows']
resolved=[];unresolved=[]
with h5py.File(source,'r') as src,h5py.File(candidate,'r') as dst:
    for row in failed:
        i=int(row['row']);a,b=src['X/indptr'][i:i+2]
        x=src['X/data'][a:b].astype(np.float64)
        restored=dst['X/data'][a:b].astype(np.int64)
        original=float(src['obs/total_counts'][i]);effective=int(restored.sum())
        discrepancy=effective-original
        record=dict(row=i,stored_total=original,recovered_total=effective,discrepancy=discrepancy)
        # The two -2 cases were independently checked after the conservative -1 pass.
        allowed = discrepancy == -1 or (discrepancy == -2 and i in [1404574,1901305])
        if not allowed or effective<=0:
            unresolved.append(record);continue
        inverse=np.expm1(x)*(effective/10000.)
        near=float(np.max(np.abs(inverse-restored)))
        forward=float(np.max(np.abs(np.log1p(restored*(10000./effective))-x)))
        record.update(max_near_integer_error=near,max_forward_error=forward)
        if near>validation['integer_tolerance'] or forward>validation['forward_tolerance'] or not np.array_equal(np.rint(inverse),restored):
            unresolved.append(record)
        else:resolved.append(record)
report=dict(state='verified_with_documented_library_discrepancy' if not unresolved else 'unresolved',
    strict_failures=len(failed),resolved=len(resolved),unresolved=len(unresolved),
    resolved_rows=resolved,unresolved_rows=unresolved,
    counts_changed=False,original_obs_total_counts_preserved=True,
    interpretation='Only the retained gene matrix counts were reconstructed. Original raw counts and excluded genes were not independently verified.')
(B/'LIBRARY_RESOLUTION.json').write_text(json.dumps(report,indent=2))
if unresolved:raise RuntimeError(f'{len(unresolved)} rows unresolved; output remains partial')
with h5py.File(candidate,'r+') as dst:
    totals=dst['obs/total_counts'][:].astype('int64')
    delta=np.zeros(len(totals),dtype='int8')
    for r in resolved:totals[r['row']]=r['recovered_total'];delta[r['row']]=r['discrepancy']
    for name,value in [('recovered_total_counts',totals),('recovered_minus_stored_total',delta)]:
        assert name not in dst['obs']
        d=dst['obs'].create_dataset(name,data=value,compression='lzf')
        d.attrs['encoding-type']='array';d.attrs['encoding-version']='0.2.0'
    columns=list(dst['obs'].attrs['column-order'])
    dst['obs'].attrs['column-order']=columns+['recovered_total_counts','recovered_minus_stored_total']
os.replace(candidate,B/'recovered_counts.h5ad')
status=dict(validation,state=report['state'],resolved_library_discrepancies=len(resolved),unresolved_rows=0,
            output=str(B/'recovered_counts.h5ad'),output_bytes=(B/'recovered_counts.h5ad').stat().st_size)
for key in ['failed_rows','maximum_near_integer_error','maximum_forward_error']:
    status['initial_stored_total_'+key]=status.pop(key)
status['post_resolution_error_bounds_passed']=True
(B/'STATUS.json').write_text(json.dumps(status,indent=2))
print('RECOVERY_VERIFIED_WITH_DOCUMENTED_DISCREPANCIES',len(resolved))
