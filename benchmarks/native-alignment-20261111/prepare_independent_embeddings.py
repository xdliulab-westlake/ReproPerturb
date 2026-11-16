from pathlib import Path
import json,shutil,hashlib
import numpy as np
BASE=Path('/liaozizhuo/hypoblast/benchmarks');OLD=BASE/'hypo-forward-embedding-combo-20261029';EXT=BASE/'hypo-benchmark-extension-20261105';B=BASE/'hypo-independent-intervention-20261112';B.mkdir(exist_ok=True)
bench=json.loads((OLD/'benchmark.json').read_text());co={r['sample_id']:r for r in bench['cohort']}
def agents(s):return co[s]['drugs'].split(' + ')
vocab=sorted({a for s in bench['splits']['train'] for a in agents(s)});lookup={a:i+1 for i,a in enumerate(vocab)};assert len(vocab)==24
(B/'agent_vocabulary.json').write_text(json.dumps(dict(agents=vocab,ids=lookup,fit_split='train',seed_initialization='independent per-model learned parameters; no Hypo embeddings'),indent=2))
features={s:np.array([[lookup.get(a,0)] for a in agents(s)],dtype=np.float32) for s in co};np.savez(B/'agent_ids_all.npz',**features);tv=set(bench['splits']['train'])|set(bench['splits']['val']);np.savez(B/'agent_ids_train_val.npz',**{s:features[s] for s in tv});unknown={s:[a for a in agents(s) if a not in lookup] for s in co};assert not any(unknown.values()),unknown
# STATE: preserve native ST and its gene remap, independently learn per-reagent identities.
T=B/'state_task';T.mkdir(exist_ok=True)
for n in ['data','splits']:(T/n).symlink_to(OLD/n,target_is_directory=True)
shutil.copytree(OLD/'config',T/'config');cfg=json.loads((T/'config/training.json').read_text());cfg.update(feature_dim=1,event_features_file=str(B/'agent_ids_train_val.npz'),event_features_all_file=str(B/'agent_ids_all.npz'));(T/'config/training.json').write_text(json.dumps(cfg,indent=2))
for seed in [43,44,45]:
 src=OLD/f'runs/state_seed{seed}';r=T/f'runs/state_seed{seed}';r.mkdir(parents=True);shutil.copytree(src/'code',r/'code')
 for p in (r/'code').glob('*.py'):
  s=p.read_text().replace(str(OLD),str(T));p.write_text(s)
 p=r/'code/common.py';s=p.read_text();s=s.replace("self.z=np.load(read(B/'config/stage2.json')['data']['cell_embedding_npy'],mmap_mode='r')","self.z=self.x");p.write_text(s)
 p=r/'code/torch_models.py';s=p.read_text();start=s.index('class DrugSetEncoder');end=s.index('\nclass ChemCombination',start);s=s[:start]+'''class DrugSetEncoder(nn.Module):
    def __init__(self,out_dim):
        super().__init__();self.embedding=nn.Embedding(25,out_dim,padding_idx=0)
    def forward(self,events,mask):
        return (self.embedding(events[:,:,0].long())*mask.unsqueeze(-1)).sum(1)
''' +s[end:];p.write_text(s)
 p=r/'code/train_four_gpu.py';s=p.read_text().replace("else:changed[:,0,-1]+=1","else:changed[:,0,0]=changed[:,0,0]%24+1");p.write_text(s)
 config=dict(cfg,seed=seed);(r/'training.json').write_text(json.dumps(config,indent=2));(r/'PROVENANCE.json').write_text(json.dumps(dict(method='state',seed=seed,hypo_drug_embedding_used=False,hypo_cell_embedding_used=False,intervention_encoding='trainable independent per-reagent embedding summed over combination',vocabulary=str(B/'agent_vocabulary.json'),code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (r/'code').glob('*.py')}),indent=2))
# PerturbGen: same native token model and CountHead, new independent reagent vectors.
P=B/'perturbgen_task';P.mkdir();shutil.copytree(EXT/'code',P/'code')
for n in ['data','vendor','env']:(P/n).symlink_to(EXT/n,target_is_directory=True)
for n in ['runs','logs','audit','results']:(P/n).mkdir()
(P/'results/readout.npz').symlink_to(EXT/'results/readout.npz')
for n in ['tokens_complete.json','readout_complete.json']:
 if (EXT/'audit'/n).exists():shutil.copy2(EXT/'audit'/n,P/'audit'/n)
p=P/'code/prepare_composition.py';s=p.read_text().replace("OLD=ROOT.parent/'hypo-forward-embedding-combo-20261029'",f"OLD=Path({str(OLD)!r})").replace('PROJECT=ROOT.parents[1]',"PROJECT=Path('/liaozizhuo/hypoblast')");p.write_text(s)
p=P/'code/train_perturbgen.py';s=p.read_text().replace('self.combination_projection=nn.Linear(2048,768,bias=False)','self.combination_projection=nn.Embedding(25,768,padding_idx=0)').replace('self.combination_projection(events.sum(0,keepdim=True))','self.combination_projection(events[:,0].long()).sum(0,keepdim=True)').replace("OLD/'data/drug_embeddings_train_val.npz'",f"Path({str(B/'agent_ids_train_val.npz')!r})").replace('torch.cat([events,events*.5],dim=0)','torch.cat([events,events[:1]%24+1],dim=0)');p.write_text(s)
p=P/'code/finish_perturbgen.py';s=p.read_text().replace("OLD/'data/drug_embeddings_all.npz'",f"Path({str(B/'agent_ids_all.npz')!r})");p.write_text(s)
(P/'PROVENANCE.json').write_text(json.dumps(dict(method='perturbgen',hypo_drug_embedding_used=False,hypo_cell_embedding_used=False,intervention_encoding='independently initialized learned reagent Embedding(25,768) plus native adaptation count token',vocabulary=str(B/'agent_vocabulary.json'),code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'code').glob('*.py')}),indent=2))
(T/'STATUS').write_text('PREPARED_SMOKE_PENDING\n');(P/'STATUS').write_text('PREPARED_SMOKE_PENDING\n');print('PREPARED',B,'24 train-only reagent IDs; no unknown test reagents')
