#!/usr/bin/env bash
set -euo pipefail
B=/liaozizhuo/hypoblast/benchmarks/native-alignment-20261111
pid=$(cat "$B/counts_full.pid")
while kill -0 "$pid" 2>/dev/null; do
 state=$(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null || true)
 [ "$state" = Z ] && break
 sleep 15
done
/liaozizhuo/iPStem/benchmark/envs/torch_native/bin/python "$B/resolve_counts_library.py"
/liaozizhuo/iPStem/benchmark/envs/torch_native/bin/python - <<'PY'
from pathlib import Path
import anndata as ad,numpy as np,json
b=Path('/liaozizhuo/hypoblast/benchmarks/native-alignment-20261111/counts_full')
a=ad.read_h5ad(b/'recovered_counts.h5ad',backed='r')
assert a.shape==(2550159,37961)
key='recovered_total_counts' if 'recovered_total_counts' in a.obs else 'total_counts'
for start in np.linspace(0,a.n_obs-32,16,dtype=int):
 x=a.X[start:start+32]
 assert np.issubdtype(x.dtype,np.integer) and (x.data>=0).all()
 assert np.array_equal(np.asarray(x.sum(1)).ravel(),a.obs[key].to_numpy()[start:start+32])
a.file.close()
(b/'ANNDATA_READBACK.json').write_text(json.dumps(dict(passed=True,shape=[2550159,37961],integer_sparse=True,checked_cells=512,total_column=key),indent=2))
print('FULL_COUNTS_ANNDATA_READBACK_PASSED')
PY
