"""Native CountHead fitting, target-free generation, and frozen-contract export.

Runs only after the mask stage. Expression and composition heads are selected
separately on validation data; no test result feeds selection or early stopping.
"""
import argparse,hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
import torch
sys.dont_write_bytecode=True
from prepare_composition import ROOT,OLD,read,write,sha,save_prediction
from composition_metrics import metrics
from train_perturbgen import CombinationPerturbGen,seed_all,target_tokens
from perturbgen.Modules.transformer import CountHead


def pcc_gene(pred,true):
    p=pred-pred.mean(0);t=true-true.mean(0);den=np.sqrt((p*p).sum(0)*(t*t).sum(0))
    return float(np.divide((p*t).sum(0),den,out=np.zeros_like(den),where=den>1e-12).mean())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',required=True,type=int,choices=[43,44,45]);ap.add_argument('--smoke',action='store_true');ap.add_argument('--cache-worker',type=int,default=-1);a=ap.parse_args()
    seed_all(a.seed);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=read(OLD/'benchmark.json');data=read(OLD/'config/stage2.json')['data'];run=ROOT/'runs'/f'perturbgen_seed{a.seed}'
    if not a.smoke and not (run/'MASK_COMPLETE.json').exists():raise RuntimeError('Mask stage has not completed')
    if (run/'COMPLETE.json').exists() and not a.smoke:return
    rows=dict(np.load(OLD/'data/sample_rows_train_val.npz'));controls={r['sample_id']:r['control_sample_id'] for r in cfg['cohort']}
    features=dict(np.load(Path('/liaozizhuo/hypoblast/benchmarks/hypo-independent-intervention-20261112/agent_ids_all.npz')));tokens=np.load(ROOT/'data/tokens_train_val_source.npy',mmap_mode='r')
    x=np.load(data['cell_hvg_npy'],mmap_mode='r');truth=dict(np.load(OLD/'data/truth.npz'));ti={s:i for i,s in enumerate(truth['sample_ids'])}
    classifier=dict(np.load(ROOT/'results/readout.npz'));coef=classifier['coef']/classifier['scale'];bias=classifier['intercept']-classifier['mean']@coef.T
    ct=torch.as_tensor(coef,dtype=torch.float32,device='cuda');bt=torch.as_tensor(bias,dtype=torch.float32,device='cuda')
    model=CombinationPerturbGen(a.seed).cuda()
    if not a.smoke:
        state=torch.load(run/'mask_best.pt',map_location='cpu',weights_only=True)
        missing,unexpected=model.load_state_dict(state['model'],strict=False)
        assert not unexpected and all(k.startswith('encoder_layers.') for k in missing)
        mask_sha=sha(run/'mask_best.pt')
    else:mask_sha='smoke-only'
    model.eval();model.requires_grad_(False)
    torch.manual_seed(a.seed+10000)
    head=CountHead(loss_mode='mse',n_genes=2000,d_model=768,use_size_factor=False,use_observed_size_factor=False).cuda()
    optim=torch.optim.AdamW(head.parameters(),lr=5e-5,weight_decay=.01)
    if a.smoke:
        sid=cfg['splits']['train'][0];ix=rows[sid][:2];src=torch.as_tensor(np.array(tokens[rows[controls[sid]][:2]],dtype=np.int64),device='cuda')
        target=target_tokens(tokens,ix);events=torch.as_tensor(features[sid],device='cuda')
        model.count_only=True;model.set_combination(events)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):emb=model(src_input_id=src,tgt_input_id_dict={'tgt_input_ids_t1':target},not_masked=True)[1]['mean_embedding']
        pred=head(emb.float())['count_lognorm'];true=torch.as_tensor(np.array(x[ix]),device='cuda')
        loss=torch.nn.functional.mse_loss(pred,true);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(head.parameters(),1.));assert np.isfinite(norm) and norm>0;optim.step()
        write(ROOT/'audit/perturbgen_count_smoke.json',dict(status='passed',native_head='perturbgen.Modules.transformer.CountHead',gradient_norm=norm,loss=float(loss),output_shape=list(pred.shape),test_access=False))
        print('COUNT_SMOKE_PASSED',flush=True);return
    cache=run/'cache';cache.mkdir(exist_ok=True)
    write(run/('parallel_count_manifest.json' if a.cache_worker<0 else f'cache_worker{a.cache_worker}_manifest.json'),dict(mask_checkpoint_sha256=mask_sha,native_head='CountHead',loss='MSE on stored HVG2000 scale',training_cells_per_condition_max=5000,training_sampling_seed=42,validation='all 21 conditions, all matched control source cells, generation without target information',expression_selection='PCC_gene; constant genes score zero',composition_selection='condition-balanced MAE_pp',max_epochs=3000,early_stop_patience=20,lr=5e-5,weight_decay=.01,test_only_after_selection=True,script_sha256=sha(__file__)))

    def build_cache(label,ids,teacher=False,sampling=None):
        info=cache/(label+'.json');f=cache/(label+'.npy')
        if info.exists():
            meta=read(info);assert meta['mask_checkpoint_sha256']==mask_sha
            return np.load(f,mmap_mode='r'),meta
        rng=np.random.default_rng(42);chosen={}
        for sid in ids:
            if teacher:chosen[sid]=rng.choice(rows[sid],min(5000,len(rows[sid])),replace=False)
            elif sampling is None:chosen[sid]=rows[controls[sid]]
            else:chosen[sid]=np.random.default_rng(sampling+int(hashlib.sha256(sid.encode()).hexdigest()[:8],16)).choice(rows[controls[sid]],5000,replace=True)
        if label=='val_generated' and a.cache_worker<0:
            shardroot=cache/'val_shards'
            for sid in ids:
                while not (shardroot/(sid+'.npy')).exists():time.sleep(10)
            offsets={};allix=[];offset=0;n=sum(map(len,chosen.values()))
            out=np.lib.format.open_memmap(f,mode='w+',dtype=np.float32,shape=(n,768))
            for sid in ids:
                z=np.load(shardroot/(sid+'.npy'),mmap_mode='r');ix=chosen[sid]
                assert z.shape==(len(ix),768) and np.isfinite(z).all()
                out[offset:offset+len(ix)]=z;offsets[sid]=[offset,offset+len(ix)];offset+=len(ix);allix.extend(ix.tolist())
            out.flush();meta=dict(mask_checkpoint_sha256=mask_sha,shape=list(out.shape),offsets=offsets,target_tokens_used=False,sampling_seed=None)
            np.save(cache/(label+'_rows.npy'),np.array(allix));write(info,meta)
            return out,meta
        n=sum(map(len,chosen.values()));out=np.lib.format.open_memmap(f,mode='w+',dtype=np.float32,shape=(n,768));offsets={};allix=[];offset=0
        for sid in ids:
            ix=chosen[sid];offsets[sid]=[offset,offset+len(ix)];events=torch.as_tensor(features[sid],device='cuda')
            torch.manual_seed((sampling if sampling is not None else 42)+int(hashlib.sha256(sid.encode()).hexdigest()[:8],16))
            for start in range(0,len(ix),8):
                sub=ix[start:start+8]
                source_ix=rng.choice(rows[controls[sid]],len(sub),replace=True) if teacher else sub
                src=torch.as_tensor(np.array(tokens[source_ix],dtype=np.int64),device='cuda')
                assert bool((src[:,0]==2).all())
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    if teacher:
                        model.count_only=True;model.set_combination(events)
                        emb=model(src_input_id=src,tgt_input_id_dict={'tgt_input_ids_t1':target_tokens(tokens,sub)},not_masked=True)[1]['mean_embedding']
                    else:
                        model.count_only=False;generated,_=model.generate_from_source(src,events);emb=generated[1]['mean_embedding']
                if not torch.isfinite(emb).all():raise ValueError('Nonfinite embedding')
                out[offset+start:offset+start+len(sub)]=emb.float().cpu().numpy()
            allix.extend(ix.tolist());offset+=len(ix);out.flush()
            write(run/'status.json',dict(status='running',stage='cache_'+label,condition=sid,cells_done=offset,total_cells=n))
            print('CACHE',label,sid,offset,n,flush=True)
        meta=dict(mask_checkpoint_sha256=mask_sha,shape=list(out.shape),offsets=offsets,target_tokens_used=teacher,sampling_seed=sampling)
        np.save(cache/(label+'_rows.npy'),np.array(allix));write(info,meta)
        return out,meta

    if a.cache_worker>=0:
        for j,sid in enumerate(cfg['splits']['val']):
            dest=cache/'val_shards'/(sid+'.npy')
            if j%2!=a.cache_worker or dest.exists():continue
            z,meta=build_cache('parallel_val_'+sid,[sid])
            # Atomic promotion; model seed and per-condition generation seed unchanged.
            import os
            tmp=dest.with_suffix('.partial.npy');np.save(tmp,np.asarray(z));os.replace(tmp,dest)
            print('SHARD_READY',sid,flush=True)
        print('WORKER_COMPLETE',a.cache_worker,flush=True);return

    train,trainmeta=build_cache('train_teacher',cfg['splits']['train'],teacher=True)
    trainrows=np.load(cache/'train_teacher_rows.npy');val,valmeta=build_cache('val_generated',cfg['splits']['val'])
    valtruth=truth['expression'][[ti[s] for s in cfg['splits']['val']]];valprop=truth['proportions'][[ti[s] for s in cfg['splits']['val']]]

    @torch.no_grad()
    def summaries(head,emb,meta,ids):
        ep=[];pp=[]
        head.eval()
        for sid in ids:
            start,end=meta['offsets'][sid];total=np.zeros(2000);prop=np.zeros(10)
            for i in range(start,end,1024):
                pred=head(torch.as_tensor(np.array(emb[i:min(i+1024,end)]),device='cuda'))['count_lognorm']
                if not torch.isfinite(pred).all():raise ValueError('Nonfinite generated expression')
                total+=pred.double().sum(0).cpu().numpy();prop+=torch.softmax(pred@ct.T+bt,dim=-1).double().sum(0).cpu().numpy()
            ep.append(total/(end-start));pp.append(prop/(end-start))
        return np.stack(ep),np.stack(pp)

    best_expr=-float('inf');best_comp=float('inf');stale_expr=stale_comp=0;rng=np.random.default_rng(a.seed)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optim,patience=10,factor=.5)
    for epoch in range(1,3001):
        head.train();losses=[]
        for sid in rng.permutation(cfg['splits']['train']):
            start,end=trainmeta['offsets'][sid];ix=rng.integers(start,end,size=512)
            emb=torch.as_tensor(np.array(train[ix]),device='cuda');target=torch.as_tensor(np.array(x[trainrows[ix]]),device='cuda')
            pred=head(emb)['count_lognorm'];loss=torch.nn.functional.mse_loss(pred,target)
            if not torch.isfinite(loss):raise ValueError('Nonfinite count loss')
            optim.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);optim.step();losses.append(float(loss))
        ve,vp=summaries(head,val,valmeta,cfg['splits']['val']);pe=pcc_gene(ve,valtruth);pc=metrics(vp,valprop)['mae_pp']
        scheduler.step(float(((ve-valtruth)**2).mean()))
        for key,improved in [('expression',pe>best_expr+1e-6),('composition',pc<best_comp-1e-6)]:
            if improved:
                torch.save(dict(head=head.state_dict(),epoch=epoch,val_pcc_gene=pe,val_mae_pp=pc,mask_checkpoint_sha256=mask_sha),run/(key+'_best.pt'))
                if key=='expression':best_expr=pe;stale_expr=0
                else:best_comp=pc;stale_comp=0
            elif key=='expression':stale_expr+=1
            else:stale_comp+=1
        status=dict(status='running',stage='count_head',epoch=epoch,train_mse=float(np.mean(losses)),val_pcc_gene=pe,val_mae_pp=pc,best_pcc_gene=best_expr,best_mae_pp=best_comp)
        write(run/'status.json',status);print('COUNT_EPOCH',json.dumps(status),flush=True)
        if min(stale_expr,stale_comp)>=20:break
    write(run/'COUNT_COMPLETE.json',dict(epochs=epoch,best_val_pcc_gene=best_expr,best_val_mae_pp=best_comp,mask_checkpoint_sha256=mask_sha))
    # Only now may test inference run. Source rows are still matched controls.
    ids=cfg['splits']['test'];test,testmeta=build_cache('test_generated_all_controls',ids)
    head.load_state_dict(torch.load(run/'expression_best.pt',weights_only=True)['head']);expression,proportion=summaries(head,test,testmeta,ids)
    output=ROOT/'results/expression'/f'perturbgen_seed{a.seed}';output.mkdir(parents=True,exist_ok=True)
    np.savez(output/'predictions.npz',sample_ids=np.array(ids),genes=truth['genes'],expression_scale=truth['expression_scale'],expression=expression,model=np.array('perturbgen'),backbone_seed=np.array(a.seed))
    sys.path.insert(0,str(OLD))
    spec=importlib.util.spec_from_file_location('frozen_expression_metrics',OLD/'metrics.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    selected=[ti[s] for s in ids];result=module.expression_metrics(expression,truth['expression'][selected],truth['control_expression'][selected])
    write(output/'expression_metrics.json',result);write(output/'provenance.json',dict(mask_checkpoint_sha256=mask_sha,count_checkpoint_sha256=sha(run/'expression_best.pt'),all_control_cells=True,native_maskgit_iterations=18,target_information_at_generation=False,old_evaluator_sha256=sha(OLD/'metrics.py')))
    save_prediction('perturbgen',proportion,ids,cfg['classes'],a.seed,track='locked_checkpoint',selection='expression PCC_gene',source_cells='all matched controls',mask_checkpoint_sha256=mask_sha,count_checkpoint_sha256=sha(run/'expression_best.pt'))
    head.load_state_dict(torch.load(run/'composition_best.pt',weights_only=True)['head']);replicates=[]
    for sampling in [101,102,103]:
        emb,meta=build_cache(f'test_generated_sampling{sampling}',ids,sampling=sampling);_,prop=summaries(head,emb,meta,ids);replicates.append(prop)
    save_prediction('perturbgen',np.mean(replicates,axis=0),ids,cfg['classes'],a.seed,track='val_selected',selection='composition MAE_pp',sampling_seeds=[101,102,103],cells_per_condition_per_seed=5000,mask_checkpoint_sha256=mask_sha,count_checkpoint_sha256=sha(run/'composition_best.pt'))
    np.savez(ROOT/'results/val_selected'/f'perturbgen_seed{a.seed}'/'sampling_diagnostics.npz',soft_proportions=np.stack(replicates))
    write(run/'COMPLETE.json',dict(status='complete',expression_metric_path=str(output/'expression_metrics.json'),composition_tracks=['locked_checkpoint','val_selected'],seed=a.seed))


if __name__=='__main__':main()
