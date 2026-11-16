from pathlib import Path
import json,pickle
import numpy as np
BASE=Path('/liaozizhuo/hypoblast/benchmarks');N=BASE/'native-alignment-20261111';OLD=BASE/'hypo-forward-embedding-combo-20261029';PREV=N/'stack_task';U=BASE/'stack-unified-20261113';U.mkdir(exist_ok=True)
v=json.loads((OLD/'benchmark.json').read_text());tr=v['splits']['train'];va=v['splits']['val'];te=v['splits']['test'];assert (len(tr),len(va),len(te))==(146,21,14) and not (set(tr)&set(va) or set(tr)&set(te) or set(va)&set(te))
(U/'data').mkdir(exist_ok=True)
for sid in tr:
 p=U/'data'/f'{sid}.h5ad';src=PREV/'data'/f'{sid}.h5ad';assert src.exists()
 if not p.exists():p.symlink_to(src)
if not (U/'genes.pkl').exists():(U/'genes.pkl').symlink_to(PREV/'genes.pkl')
for name in ['runs','validation','evaluation','logs','smoke']:(U/name).mkdir(exist_ok=True)
(U/'SPLIT.json').write_text(json.dumps(dict(train=tr,val=va,test=te),indent=2))
t=np.load(OLD/'data/truth.npz');idx=[t['sample_ids'].tolist().index(s) for s in va];np.savez_compressed(U/'validation_truth.npz',sample_ids=np.array(va),genes=t['genes'],expression=t['expression'][idx],control_expression=t['control_expression'][idx],proportions=t['proportions'][idx])
s=(PREV/'evaluate_stack.py').read_text().replace(str(N),str(U)).replace("S=B/'stack_task'","S=B")
s=s.replace("seed=int(sys.argv[1]);dest=B/f'evaluation/stack_traincontext_seed{seed}';dest.mkdir(exist_ok=True)","seed=int(sys.argv[1]);split=sys.argv[2];epoch=int(sys.argv[3]);dest=Path(os.environ.get('STACK_EVAL_DEST',str(B/(f'validation/seed{seed}/epoch{epoch}' if split=='val' else f'evaluation/stack_traincontext_seed{seed}'))));dest.mkdir(parents=True,exist_ok=True)")
s=s.replace("ids=v['splits']['test'];classes=v['classes'];assert len(ids)==14","ids=v['splits'][split];classes=v['classes'];assert len(ids)==(21 if split=='val' else 14)\nif os.environ.get('STACK_SMOKE'):ids=ids[:1]")
a=s.index('last=torch.load');z=s.index('ck=torch.load',a);s=s[:a]+"best=os.environ.get('STACK_EVAL_CHECKPOINT',str(S/f'runs/seed{seed}/best.ckpt'));"+s[z:]
s=s.replace("B/'counts_full/recovered_counts.h5ad'",f"Path({str(N/'counts_full/recovered_counts.h5ad')!r})")
s=s.replace("t=np.load(OLD/'data/truth.npz')","t=np.load(B/'validation_truth.npz' if split=='val' else OLD/'data/truth.npz')")
s=s.replace('expression=expr,generated_counts=pred,scale_roundtrip_max_error=err',"scale_roundtrip_max_error=err,**({'expression':expr,'generated_counts':pred} if split=='test' else {}))FIX")
s=s.replace('))FIX)', '))')
# Replace a potential extra closing token explicitly and validate generated source.
s=s.replace('test_ids=ids','evaluation_split=split,evaluation_ids=ids')
compile(s,'score_checkpoint.py','exec');(U/'score_checkpoint.py').write_text(s)
(U/'CONFIG.json').write_text(json.dumps(dict(seeds=[43,44,45],architecture='same official STACK model config; fresh weights',training_conditions=146,validation_conditions=21,test_conditions=14,training_objective='native STACK masked reconstruction + SW',checkpoint_selection='full21 condition generated-expression gene PCC; constant genes contribute zero',max_epochs=3000,early_stop_patience=20,lr=5e-5,weight_decay=.01,batch_size=8,sample_size=256,prompt_rule='fixed top3 Jaccard train-only conditions;512 prompt cells per condition',generation_steps=5,validation_sampling_seed='fixed per training seed and condition',test_truth_used_in_training=False,counts_to_benchmark_scale='non-modelled genes held at source control counts; numerical roundtrip verified',gene_count=2000),indent=2));print('STACK_UNIFIED_DATA_AND_EVALUATOR_PREPARED',U)
