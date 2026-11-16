"""New composition readout of old frozen expression models; no old writes."""
import argparse,hashlib,os,sys
from pathlib import Path
import numpy as np
from prepare_composition import ROOT,OLD,read,write,sha,save_prediction
from composition_metrics import softmax

SAMPLING_SEEDS=[101,102,103]

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True,choices=['cpa','chemcpa','state','g2cp','perturbnet','cellflow'])
    p.add_argument('--seed',required=True,type=int,choices=[43,44,45]);p.add_argument('--decode-cellflow',action='store_true');args=p.parse_args()
    name=args.model;run=OLD/'runs'/f'{name}_seed{args.seed}';dest=ROOT/'results/locked_checkpoint'/f'{name}_seed{args.seed}'
    if (dest/'provenance.json').exists():print('ALREADY_EXPORTED',name,args.seed);return
    os.environ['IPSTEM_RUN_ROOT']=str(run);os.environ['IPSTEM_TRAIN_SEED']=str(args.seed)
    sys.path.insert(0,str(run/'code'))
    b=read(OLD/'benchmark.json');cfg=read(OLD/'config/stage2.json');train=read(OLD/'config/training.json')
    weight=run/'checkpoints'/name/('best.msgpack' if name=='cellflow' else 'best.pt')
    done=read(weight.parent/'COMPLETE.json');assert sha(weight)==done['best_sha256'] and done['train_seed']==args.seed
    rows=dict(np.load(OLD/'data/sample_rows_all.npz'));controls={r['sample_id']:r['control_sample_id'] for r in b['cohort']}
    features=dict(np.load(OLD/train['cpa_features_all_file' if name=='cpa' else 'event_features_all_file']))
    genes=read(OLD/'config/genes.json');ids=b['splits']['test'];classes=b['classes']
    x=np.load(cfg['data']['cell_hvg_npy'],mmap_mode='r');z=np.load(cfg['data']['cell_embedding_npy'],mmap_mode='r')
    classifier=dict(np.load(ROOT/'results/readout.npz'));assert list(classifier['genes'])==genes and list(classifier['classes'])==classes
    assert (ROOT/'audit/readout_complete.json').exists()
    coef=classifier['coef']/classifier['scale'];bias=classifier['intercept']-classifier['mean']@coef.T
    latent_file=ROOT/'data'/f'cellflow_seed{args.seed}_generated_latent.npy'
    def selected(cid,seed):
        rng=np.random.default_rng(seed+int(hashlib.sha256(cid.encode()).hexdigest()[:8],16))
        return rng.choice(rows[controls[cid]],5000,replace=True)
    if name=='cellflow' and not args.decode_cellflow:
        import jax
        import jax.numpy as jnp
        from cellflow_native import make_solver,predictor,condition
        from train_cellflow import restore
        assert jax.default_backend()=='gpu'
        solver=make_solver();restore(weight,solver,jax.random.split(jax.random.PRNGKey(42),4));predict=predictor(solver)
        latent=np.lib.format.open_memmap(latent_file,mode='w+',dtype=np.float32,shape=(3,14,5000,128))
        for r,sampling in enumerate(SAMPLING_SEEDS):
            for j,cid in enumerate(ids):
                ix=selected(cid,sampling);cond=condition(features[cid])
                for start in range(0,5000,1024):
                    n=min(1024,5000-start);a=np.zeros((1024,128),np.float32);a[:n]=z[ix[start:start+n]]
                    pred=np.asarray(predict(solver.vf_state.params,jnp.asarray(a),cond))[:n]
                    if not np.isfinite(pred).all():raise ValueError('Nonfinite CellFlow latent')
                    latent[r,j,start:start+n]=pred
                print('LATENT_EXPORTED',name,args.seed,r,cid,flush=True)
        latent.flush();write(ROOT/'audit'/f'cellflow_seed{args.seed}_latent.json',dict(status='complete',checkpoint_sha256=sha(weight),shape=list(latent.shape),ode_steps=20,sampling_seeds=SAMPLING_SEEDS))
        return
    import torch
    torch.set_num_threads(4);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda');decode=None
    from train_torch import pad
    if name in ['cpa','chemcpa']:
        from adversarial_models import CPACombination,ChemAdversarial
        model={'cpa':CPACombination,'chemcpa':ChemAdversarial}[name](genes).to(device)
    elif name=='perturbnet':
        from latent_models import PerturbNetCombination,stage1_decoder
        model=PerturbNetCombination(genes).to(device);decode=stage1_decoder(device)
        means=dict(np.load(OLD/'data/latent_means_train_val.npz'))
    elif name=='cellflow':
        from latent_models import stage1_decoder
        assert (ROOT/'audit'/f'cellflow_seed{args.seed}_latent.json').exists()
        decode=stage1_decoder(device);latent=np.load(latent_file,mmap_mode='r');model=None
    else:
        from torch_models import build
        model=build(name,genes).to(device)
    if model is not None:
        model.load_state_dict(torch.load(weight,map_location=device,weights_only=True)['model'],strict=True);model.eval()
    ct=torch.as_tensor(coef,dtype=torch.float32,device=device);bt=torch.as_tensor(bias,dtype=torch.float32,device=device)
    result=np.zeros((3,14,10));hard=np.zeros_like(result)
    with torch.inference_mode():
        for r,sampling in enumerate(SAMPLING_SEEDS):
            for j,cid in enumerate(ids):
                ix=selected(cid,sampling);events,mask=pad(features,[cid],device)
                rng=np.random.default_rng(sampling+int(hashlib.sha256(cid.encode()).hexdigest()[:8],16))
                for start in range(0,5000,512):
                    n=min(512,5000-start)
                    if name=='cellflow':pred=decode(torch.as_tensor(np.array(latent[r,j,start:start+n]),device=device))
                    elif name=='perturbnet':
                        noise=torch.as_tensor(rng.standard_normal((1,n,128)).astype(np.float32),device=device)
                        source=torch.as_tensor(means[controls[cid]],device=device)[None]
                        pred=decode(model.sample(noise,source,events,mask)[0])
                    else:
                        a=torch.as_tensor(np.array(x[ix[start:start+n]],dtype=np.float32),device=device)[None]
                        pred=model(a,events,mask)[0]
                    if not torch.isfinite(pred).all():raise ValueError('Nonfinite generated expression')
                    probs=torch.softmax(pred@ct.T+bt,dim=-1)
                    result[r,j]+=probs.double().sum(0).cpu().numpy()/5000
                    hard[r,j]+=torch.bincount(probs.argmax(-1),minlength=10).cpu().numpy()/5000
                print('COMPOSITION_EXPORTED',name,args.seed,r,cid,flush=True)
    assert sha(weight)==done['best_sha256']
    save_prediction(name,result.mean(0),ids,classes,args.seed,checkpoint=str(weight),checkpoint_sha256=done['best_sha256'],readout_sha256=sha(ROOT/'results/readout.npz'),sampling_seeds=SAMPLING_SEEDS,cells_per_condition_per_seed=5000,aggregation='average per-cell probabilities; then average sampling replicates',training_updates=0)
    np.savez(dest/'sampling_diagnostics.npz',sampling_seeds=SAMPLING_SEEDS,soft_proportions=result,hard_proportions=hard,sampling_sd=result.std(0,ddof=1))
    print('EXPORT_COMPLETE',name,args.seed,flush=True)

if __name__=='__main__':main()
