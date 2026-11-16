from pathlib import Path
import json,os
import numpy as np,anndata as ad,scvi
B=Path('/liaozizhuo/hypoblast/benchmarks/native-alignment-20261111/perturbnet_task')
x=ad.read_h5ad(B/'scvi_train_counts.h5ad');rows=np.load(B/'scvi_train_source_rows.npy');assert len(rows)==x.n_obs
for s in [43,44,45]:
 d=B/f'runs/perturbnet_seed{s}';out=d/'train_latents.npy'
 if (d/'LATENT_COMPLETE.json').exists():continue
 assert (d/'scvi/SCVI_COMPLETE.json').exists()
 model=scvi.model.SCVI.load(d/'scvi',adata=x,accelerator='gpu',device='auto')
 z=model.get_latent_representation(adata=x,batch_size=1024,give_mean=True)
 assert z.shape==(x.n_obs,10) and np.isfinite(z).all()
 tmp=d/'train_latents.partial.npy';np.save(tmp,z);os.replace(tmp,out)
 (d/'LATENT_COMPLETE.json').write_text(json.dumps({'seed':s,'shape':list(z.shape),'source_rows':str(B/'scvi_train_source_rows.npy'),'posterior_mean':True,'training_only':True,'conditional_flow_trained':False}))
 print('LATENT_COMPLETE',s,z.shape,flush=True)
