"""Prepare chemCPA and CellFlow without Hypo feature/decoder dependencies."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import shutil
import numpy as np

B = Path(__file__).resolve().parent
OLD = B.parent / 'hypo-forward-embedding-combo-20261029'
PREV = B.parent / 'hypo-native-benchmark-20261110'
M = Path('/liaozizhuo/iPStem/benchmark/models')
bench = json.loads((OLD/'benchmark.json').read_text())
routes = list(csv.DictReader((PREV/'native_inputs/drug_routes.csv').open()))
proteins = sorted(r['name'] for r in routes if r['type'] == 'protein')
assert len(proteins) == 8
protein_ids = {n:i+1 for i,n in enumerate(proteins)}
for n in proteins:
    assert any(n in r['drugs'].split(' + ') and r['sample_id'] in bench['splits']['train'] for r in bench['cohort']), n

# Execute the exact, isolated official fingerprint function without importing JAX.
src = M/'cellflow/repo/cellflow/preprocessing/_preprocessing.py'
tree = ast.parse(src.read_text())
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_get_fingerprint')
module = ast.Module(body=[fn], type_ignores=[])
scope = {'np':np, 'ArrayLike':np.ndarray}
exec(compile(module, str(src), 'exec'), scope)
fingerprint = scope['_get_fingerprint']
descriptors = dict(np.load(PREV/'native_inputs/chemcpa_native_descriptors.npz'))
vectors = {'chemcpa':{}, 'cellflow':{}}
for r in routes:
    n = r['name']; pid = protein_ids.get(n, 0)
    vectors['chemcpa'][n] = np.r_[descriptors.get(n, np.zeros(194)), pid].astype('float32')
    identity = np.zeros(8, dtype='float32')
    if pid: identity[pid-1] = 1
    chemical = np.zeros(1024) if pid else fingerprint(r['smiles'], radius=4, n_bits=1024)
    assert chemical is not None and len(chemical) == 1024
    vectors['cellflow'][n] = np.r_[chemical, identity].astype('float32')
(B/'protein_ids.json').write_text(json.dumps(protein_ids, indent=2))

for method, dim in [('chemcpa',195), ('cellflow',1032)]:
    features = {r['sample_id']:np.stack([vectors[method][n] for n in sorted(set(r['drugs'].split(' + ')))]) for r in bench['cohort']}
    assert all(v.shape[1]==dim and np.isfinite(v).all() for v in features.values())
    keep = set(bench['splits']['train']+bench['splits']['val'])
    np.savez(B/f'{method}_features_all.npz', **features)
    np.savez(B/f'{method}_features_train_val.npz', **{k:v for k,v in features.items() if k in keep})
    task = B/f'{method}_task'; task.mkdir(exist_ok=True)
    source = PREV/'chemcpa_task' if method=='chemcpa' else OLD
    for d in ['data','splits']:
        if not (task/d).exists(): (task/d).symlink_to((source/d).resolve(), target_is_directory=True)
    shutil.copytree(source/'config', task/'config', dirs_exist_ok=True)
    cfg = json.loads((source/'config/training.json').read_text())
    cfg.update(feature_dim=dim, event_features_file=str(B/f'{method}_features_train_val.npz'), event_features_all_file=str(B/f'{method}_features_all.npz'))
    (task/'config/training.json').write_text(json.dumps(cfg, indent=2))
    for seed in [43,44,45]:
        run = task/f'runs/{method}_seed{seed}'
        assert not (run/'code').exists(), f'Refusing overwrite: {run}'
        shutil.copytree(source/f'runs/{method}_seed{seed}/code', run/'code')
        for p in (run/'code').glob('*.py'):
            p.write_text(p.read_text().replace(str(source), str(task)))
        if method=='chemcpa':
            p=run/'code/torch_models.py';s=p.read_text()
            start=s.index('        self.event_bridge=');end=s.index('\n',start)
            s=s[:start]+"        self.protein_embedding=nn.Embedding(9,args['hparams']['dim'],padding_idx=0)"+s[end:]
            start=s.index('    def encode_condition(',s.index('class ChemCombination'))
            end=s.index('    def forward(',start)
            s=s[:start]+'''    def encode_condition(self,events,mask):
        b,l,_=events.shape
        flat=events.reshape(b*l,-1); keep=mask.reshape(-1).bool()
        ids=flat[:,194].long()
        values=self.protein_embedding(ids)
        chem=keep & (ids==0)
        if chem.any():
            encoded=self.chemical_encoder(flat[chem,:194])
            values=values.index_copy(0,chem.nonzero().flatten(),encoded)
        return (values.reshape(b,l,-1)*mask.unsqueeze(-1)).sum(1)

'''+s[end:]
            s=s.replace('Hypo per-drug embedding to 194 native descriptor coordinates; sum composition, no doser or protocol metadata.', 'Native RDKit2D chemical encoder; learned protein identity embeddings in latent space; mixed-input and presence-only combination adaptation.')
            p.write_text(s)
            p=run/'code/train_four_gpu.py';s=p.read_text().replace('changed[:,0,0]+=1;changed[:,0,2048]+=1','changed[:,0,194]=changed[:,0,194]%8+1')
            s=s.replace('nonzero_adapter_gradient=True','nonzero_trainable_gradient=True');p.write_text(s)
            p=run/'code/adversarial_models.py';s=p.read_text().replace('per-drug embedding descriptor bridge without a dose gate','native chemical encoder and trainable protein identity embedding without a dose gate');p.write_text(s)
        else:
            p=run/'code/common.py';s=p.read_text()
            line=next(x for x in s.splitlines() if 'self.z=np.load' in x)
            s=s.replace(line,'        self.z=self.x  # Direct expression-space flow; no Hypo Stage1.');p.write_text(s)
            p=run/'code/cellflow_native.py';s=p.read_text().replace('FEATURE_DIM=2048','FEATURE_DIM=1032').replace('output_dim=128','output_dim=2000');p.write_text(s)
            p=run/'code/train_cellflow_four_gpu.py';s=p.read_text()
            s=s.replace('from decoder_client import DecoderClient','''class DecoderClient:
    def sum_expression(self,x):
        return np.asarray(x,dtype=np.float64).sum(0)
    def close(self):
        pass''')
            s=s.replace('4096,128','4096,2000').replace('4,1024,128','4,1024,2000')
            s=s.replace('random velocity field; shared frozen Stage1','random official velocity field in 2000-gene expression space; native fingerprints + protein identity covariates')
            s=s.replace('frozen_decoder_gpu=0','hypo_decoder_used=False');p.write_text(s)
        for p in (run/'code').glob('*.py'): compile(p.read_text(),str(p),'exec')
        (run/'training.json').write_text(json.dumps(dict(cfg,seed=seed),indent=2))
        (run/'PROVENANCE.json').write_text(json.dumps(dict(method=method,seed=seed,hypo_embedding_used=False,hypo_decoder_used=False,
            chemical_route='RDKit2D pretrained chemical encoder' if method=='chemcpa' else 'Official Morgan fingerprint radius4 bits1024',
            protein_route='learned identity in latent space' if method=='chemcpa' else 'one-hot identity covariates',
            protein_support='explicit minimal mixed-input adaptation',
            code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (run/'code').glob('*.py')}),indent=2))
    (task/'STATUS').write_text('PREPARED_SMOKE_PENDING\n')
print('PREPARED: chemCPA 3 seeds; CellFlow 3 seeds. PerturbNet awaiting counts/asset verification.')
