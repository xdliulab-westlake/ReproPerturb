"""Prepare full-gene recovered counts from training conditions and their controls."""
from pathlib import Path
import json
import numpy as np
import anndata as ad

B = Path('/liaozizhuo/hypoblast/benchmarks/native-alignment-20261111')
OLD = B.parent / 'hypo-forward-embedding-combo-20261029'
out = B / 'perturbnet_task'
out.mkdir(exist_ok=True)
benchmark = json.loads((OLD / 'benchmark.json').read_text())
train = set(benchmark['splits']['train'])
allowed = train | {r['control_sample_id'] for r in benchmark['cohort'] if r['sample_id'] in train}
assert not allowed.intersection(benchmark['splits']['test'])
assert not allowed.intersection(benchmark['splits']['val'])
index = np.load(OLD / 'data/sample_rows_all.npz')
rows = np.unique(np.concatenate([index[k] for k in sorted(allowed)]))
assert len(rows) == sum(len(index[k]) for k in allowed)
source = B / 'counts_full/recovered_counts.h5ad'
dest = out / 'scvi_train_counts.h5ad'
assert not dest.exists(), 'Preserve existing prepared data'
print('LOADING', len(rows), 'training/control cells; all genes', flush=True)
full = ad.read_h5ad(source, backed='r')
data = full[rows].to_memory()
full.file.close()
assert np.issubdtype(data.X.dtype, np.integer) and data.X.data.min() >= 0
data.obs['source_row'] = rows
data.write_h5ad(out / 'scvi_train_counts.partial.h5ad')
(out / 'scvi_train_counts.partial.h5ad').rename(dest)
np.save(out / 'scvi_train_source_rows.npy', rows)
manifest = dict(source=str(source), output=str(dest), shape=list(data.shape),
                samples=sorted(allowed), training_conditions=len(train),
                validation_conditions_used=False, test_conditions_used=False,
                gene_selection='all genes; preserve full-library decoding scale',
                counts_origin='recovered; see counts_full/LIBRARY_RESOLUTION.json')
(out / 'COUNTS_READY.json').write_text(json.dumps(manifest, indent=2))
print('COUNTS_READY', manifest['shape'], flush=True)
