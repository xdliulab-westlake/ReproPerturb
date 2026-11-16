"""Recover integer counts from log1p(CP10K); publish only after full checks."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--source', required=True)
p.add_argument('--out', required=True)
p.add_argument('--limit', type=int, default=0, help='Pilot: first N cells; zero means all cells')
a = p.parse_args()
source, out = Path(a.source), Path(a.out)
out.mkdir(parents=True, exist_ok=True)
final = out/'recovered_counts.h5ad'
partial = out/'recovered_counts.partial.h5ad'
assert not final.exists() and not partial.exists(), 'Refusing to overwrite existing output'
t0 = time.time()

def write_json(name, value):
    tmp = out/(name+'.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(out/name)

with h5py.File(source, 'r') as f, h5py.File(partial, 'w') as g, (out/'failed_rows.csv').open('w') as failure:
    x=f['X']; shape=tuple(int(v) for v in x.attrs['shape'])
    n=min(a.limit, shape[0]) if a.limit else shape[0]
    ptr=x['indptr'][:n+1]; nnz=int(ptr[-1])
    totals=f['obs/total_counts'][:n].astype(np.float64)
    assert np.all(np.isfinite(totals)) and np.all(totals>0) and np.all(totals==np.rint(totals))
    assert x.attrs['encoding-type']=='csr_matrix'
    g.attrs['encoding-type']='anndata'; g.attrs['encoding-version']='0.1.0'
    gx=g.create_group('X'); gx.attrs.update(dict(x.attrs)); gx.attrs['shape']=[n,shape[1]]
    gx.create_dataset('indptr',data=ptr,compression='lzf')
    data=gx.create_dataset('data',shape=(nnz,),dtype='int32',chunks=(min(nnz,1048576),),compression='lzf')
    indices=gx.create_dataset('indices',shape=(nnz,),dtype=x['indices'].dtype,chunks=(min(nnz,1048576),),compression='lzf')
    # Preserve annotation and exact order. Pilot obs is sliced explicitly.
    f.copy('var',g)
    if n==shape[0]: f.copy('obs',g)
    else:
        obs=g.create_group('obs');obs.attrs.update(dict(f['obs'].attrs))
        for key,node in f['obs'].items():
            if isinstance(node,h5py.Dataset):
                d=obs.create_dataset(key,data=node[:n]);d.attrs.update(dict(node.attrs))
            else:
                f.copy(node,obs,name=key)
                if 'codes' in obs[key]:
                    codes=obs[key]['codes'][:n];attrs=dict(obs[key]['codes'].attrs)
                    del obs[key]['codes'];d=obs[key].create_dataset('codes',data=codes);d.attrs.update(attrs)
    for name in ['layers','obsm','obsp','varm','varp','uns']:
        z=g.create_group(name);z.attrs['encoding-type']='dict';z.attrs['encoding-version']='0.1.0'
    writer=csv.writer(failure);writer.writerow(['row','near_integer_error','forward_error','recovered_total','stored_total'])
    count_bad=0;max_integer=0.;max_forward=0.;processed=0
    h=hashlib.sha256()
    for start in range(0,n,512):
        stop=min(start+512,n);lo,hi=int(ptr[start]),int(ptr[stop])
        values=x['data'][lo:hi].astype(np.float64)
        if not np.isfinite(values).all() or (values<0).any(): raise ValueError(f'Invalid expression at rows {start}:{stop}')
        lengths=np.diff(ptr[start:stop+1]);scale=np.repeat(totals[start:stop]/10000.,lengths)
        inverse=np.expm1(values)*scale;rounded=np.rint(inverse)
        if (rounded<0).any() or (rounded>np.iinfo(np.int32).max).any():raise ValueError('Invalid recovered counts range')
        errors=np.abs(inverse-rounded)
        forward=np.abs(np.log1p(rounded/scale)-values)
        cursor=0
        for j,length in enumerate(lengths):
            length=int(length);sl=slice(cursor,cursor+length);cursor+=length
            ei=float(errors[sl].max()) if length else 0.; ef=float(forward[sl].max()) if length else 0.
            total=float(rounded[sl].sum());max_integer=max(max_integer,ei);max_forward=max(max_forward,ef)
            if ei>0.01 or ef>2e-6 or total!=totals[start+j]:
                writer.writerow([start+j,ei,ef,total,totals[start+j]]);count_bad+=1
        counts=rounded.astype('int32');data[lo:hi]=counts;indices[lo:hi]=x['indices'][lo:hi]
        h.update(counts.tobytes());processed=stop
        if start%8192==0 or stop==n:
            status=dict(state='recovering',source=str(source),cells_done=stop,cells_total=n,nnz_total=nnz,
                        failed_rows=count_bad,max_near_integer_error=max_integer,max_forward_error=max_forward,seconds=time.time()-t0)
            write_json('STATUS.json',status);print(json.dumps(status),flush=True)
    report=dict(state='verified' if count_bad==0 else 'failed_validation',source=str(source),source_size=source.stat().st_size,
        shape=[n,shape[1]],nnz=nnz,failed_rows=count_bad,maximum_near_integer_error=max_integer,
        maximum_forward_error=max_forward,integer_tolerance=.01,forward_tolerance=2e-6,
        restored_nonzero_values_sha256=h.hexdigest(),formula='round(expm1(X) * obs.total_counts / 10000)',
        provenance='Recovered from normalized expression and stored library totals, not independently matched to an original raw-count archive.',
        pilot=bool(a.limit),seconds=time.time()-t0)
    write_json('VALIDATION.json',report)
if count_bad: raise RuntimeError(f'{count_bad} failed rows; candidate remains partial, not released for training')
with h5py.File(partial,'r') as f:
    assert tuple(f['X'].attrs['shape'])==(n,shape[1])
    assert len(f['obs'][f['obs'].attrs['_index']])==n
    assert len(f['var'][f['var'].attrs['_index']])==shape[1]
    assert f['X/data'].dtype==np.dtype('int32') and f['X/indptr'][-1]==nnz
os.replace(partial,final)
report['output']=str(final);report['output_bytes']=final.stat().st_size
write_json('STATUS.json',report);print('RECOVERED_COUNTS_VERIFIED',json.dumps(report),flush=True)
