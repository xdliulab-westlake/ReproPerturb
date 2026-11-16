"""Stage 1 of the native PerturbGen extension: condition-aware masked training.

This is an endpoint-control adaptation, not a temporal trajectory benchmark.
Final expression/composition evaluation requires the subsequent native CountHead.
"""
import argparse,json,os,pickle,random,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from prepare_composition import ROOT,OLD,read,write,sha
from training_state import capture_rng,restore_rng,replay_legacy_rng

VENDOR=ROOT/'vendor/Perturbgen'
sys.path.insert(0,str(VENDOR))
from perturbgen.Modules.transformer import PerturbGen


class CombinationPerturbGen(PerturbGen):
    """One permutation-invariant memory token; native source/decoder untouched."""
    def __init__(self,seed):
        mapping=pickle.loads((VENDOR/'perturbgen/pp/token_dict_gftokens_gc95M.pkl').read_bytes())
        ckpt=next((VENDOR/'pretraining_cohort').glob('*.ckpt'))
        super().__init__(tgt_vocab_size=len(mapping),d_model=768,num_heads=8,num_layers=6,
            d_ff=64,max_seq_length=2048,dropout=0.,pred_tps=[1],n_total_tps=1,
            context_mode=False,encoder_path=str(ckpt),gene_to_rowid=mapping,seed=seed)
        self.combination_projection=nn.Embedding(25,768,padding_idx=0)
        self.count_projection=nn.Linear(1,768,bias=False)
        self.combination=None
        self.count_only=False

    def set_combination(self,events):
        self.combination=self.combination_projection(events[:,0].long()).sum(0,keepdim=True)+self.count_projection(
            events.new_tensor([[len(events)]]))

    def call_decoder(self,enc_output,src_attention_mask,dec_embedding,tgt_pad,**kwargs):
        c=self.combination.to(enc_output.dtype).unsqueeze(1).expand(enc_output.shape[0],1,-1)
        memory=torch.cat([enc_output,c],dim=1)
        mask=torch.cat([src_attention_mask,torch.zeros((len(c),1),dtype=torch.bool,device=c.device)],dim=1)
        if self.count_only:kwargs['compute_logits']=False
        out=super().call_decoder(memory,mask,dec_embedding,tgt_pad,**kwargs)
        # Native MaskGIT uses a float32 masking sentinel; it overflows bf16 logits.
        if not self.training and out['dec_logits'] is not None:out['dec_logits']=out['dec_logits'].float()
        return out

    @torch.no_grad()
    def generate_from_source(self,source,events,iterations=18):
        self.eval();self.set_combination(events)
        # A constant template: no observed target tokens, lengths, labels or counts.
        template=torch.ones((len(source),2048),dtype=torch.long,device=source.device)
        template[:,0]=2
        return self.generate(source,{'tgt_input_ids_t1':template},cond_length=1,
                             sequence_length=2048,iterations=iterations)


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def target_tokens(tokens,ix):
    x=np.array(tokens[ix,:2048],dtype=np.int64)
    # Preserve EOS when truncating the source-length token archive.
    truncated=(x[:,-1]!=0)&(x[:,-1]!=3);x[truncated,-1]=3
    if (x[:,0]!=2).any():raise ValueError('Unprepared or excluded target rows')
    return torch.as_tensor(x,device='cuda')

def loss_for(model,source,target,events):
    model.set_combination(events)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        out=model(src_input_id=source,tgt_input_id_dict={'tgt_input_ids_t1':target})[1]
        return nn.functional.cross_entropy(out['dec_logits'].flatten(0,1).float(),out['labels'].flatten(),ignore_index=-100)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True,choices=[43,44,45]);ap.add_argument('--smoke',action='store_true');ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    seed_all(args.seed);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    config=read(OLD/'benchmark.json');rows=dict(np.load(OLD/'data/sample_rows_train_val.npz'))
    tokens=np.load(ROOT/'data/tokens_train_val_source.npy',mmap_mode='r')
    controls={r['sample_id']:r['control_sample_id'] for r in config['cohort']}
    features=dict(np.load(Path('/liaozizhuo/hypoblast/benchmarks/hypo-independent-intervention-20261112/agent_ids_train_val.npz')))
    run=ROOT/'runs'/f'perturbgen_seed{args.seed}';run.mkdir(parents=True,exist_ok=True)
    if (run/'MASK_COMPLETE.json').exists() and not args.smoke:raise RuntimeError('Mask stage already complete')
    model=CombinationPerturbGen(args.seed).cuda()
    optim=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=5e-5,weight_decay=.01)
    rng=np.random.default_rng(args.seed)
    def batch(sid,n,r):
        source_ix=r.choice(rows[controls[sid]],n,replace=True);target_ix=r.choice(rows[sid],n,replace=True)
        src=torch.as_tensor(np.array(tokens[source_ix],dtype=np.int64),device='cuda')
        if (src[:,0]!=2).any():raise ValueError('Unprepared source rows')
        return src,target_tokens(tokens,target_ix),torch.as_tensor(features[sid],device='cuda')
    if args.smoke:
        sid=config['splits']['train'][0];src,tgt,events=batch(sid,2,rng)
        # Two distinct events exercise permutation invariance even for singleton sid.
        both=torch.cat([events,events[:1]%24+1],dim=0);model.set_combination(both);a=model.combination.detach().clone()
        model.set_combination(both.flip(0));assert torch.allclose(a,model.combination,atol=1e-6)
        before=model.combination_projection.weight.detach().clone()
        model.train();model.encoder_layers.eval();loss=loss_for(model,src,tgt,events)
        assert torch.isfinite(loss);loss.backward()
        grad=float(model.combination_projection.weight.grad.norm());assert grad>0
        optim.step();assert not torch.equal(before,model.combination_projection.weight)
        del loss;optim.zero_grad(set_to_none=True)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            out,generated=model.generate_from_source(src,events,iterations=2)
        assert torch.isfinite(out[1]['mean_embedding']).all()
        result=dict(status='passed',condition_gradient_norm=grad,source_shape=list(src.shape),target_shape=list(tgt.shape),generated_shape=list(generated['tgt_input_ids_t1'].shape),peak_cuda_gb=torch.cuda.max_memory_allocated()/1024**3,encoder_frozen=all(not p.requires_grad for p in model.encoder_layers.parameters()),test_target_access=False)
        write(ROOT/'audit/perturbgen_smoke.json',result);print('SMOKE_PASSED',result,flush=True);return
    if not (ROOT/'audit/tokens_complete.json').exists():raise RuntimeError('Tokenization is not complete')
    manifest=dict(model='perturbgen',variant='native_endpoint_control_with_unordered_combination_memory',seed=args.seed,
        vendor_commit='2507b90f52613384da258eae45a4b01d89f2f7aa',encoder_sha256='49bfc4283ca82ba39f5082920f7f1d962cd6ae768dae586731ffac87fd75821b',
        stage='masked_decoder_training',max_epochs=3000,early_stop_patience=20,microbatch=8,accumulation=4,cells_per_condition_per_epoch=32,
        lr=5e-5,weight_decay=.01,validation_conditions=21,validation_cells_per_condition=64,intermediate_selection='condition_balanced_masked_token_cross_entropy',
        final_selection_pending={'expression':'full21 validation PCC_gene after native CountHead','composition':'full21 validation MAE_pp after native CountHead'},
        test_access=False,final_benchmark_ready=False,script_sha256=sha(__file__))
    best=float('inf');stale=0;start=time.time();first_epoch=1
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optim,patience=10,factor=.5)
    if args.resume:
        path=run/('mask_last.pt' if (run/'mask_last.pt').exists() else 'mask_best.pt')
        state=torch.load(path,map_location='cpu',weights_only=True)
        assert state['manifest']['seed']==args.seed
        missing,unexpected=model.load_state_dict(state['model'],strict=False)
        assert not unexpected and all(k.startswith('encoder_layers.') for k in missing)
        optim.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        first_epoch=state['epoch']+1;best=state['val_loss'];stale=state.get('stale',0)
        if 'rng' in state:restore_rng(state['rng'],rng);rng_mode='checkpoint'
        else:
            dummy=torch.full((8,2048),4,device='cuda',dtype=torch.long);pad=torch.zeros_like(dummy,dtype=torch.bool)
            def draw():model.generate_mask(dummy.clone(),pad,mask_mode='MASKGIT')
            replay_legacy_rng(rng,state['epoch'],config['splits']['train'],rows,controls,draw);rng_mode='replayed_fixed_legacy_draws'
        manifest=dict(state['manifest'],resume_script_sha256=sha(__file__))
        record=dict(checkpoint=str(path),checkpoint_sha256=sha(path),checkpoint_epoch=state['epoch'],first_epoch=first_epoch,random_state=rng_mode,hostname=os.uname().nodename,pid=os.getpid())
        write(run/'resume.json',record);print('RESUMED',json.dumps(record),flush=True)
        del state
    elif (run/'mask_best.pt').exists():raise RuntimeError('Existing checkpoint requires --resume')
    write(run/'manifest.json',manifest)
    for epoch in range(first_epoch,3001):
        model.train();model.encoder_layers.eval();trainloss=[]
        for sid in rng.permutation(config['splits']['train']):
            optim.zero_grad(set_to_none=True)
            for micro in range(4):
                src,tgt,events=batch(sid,8,rng);loss=loss_for(model,src,tgt,events)
                if not torch.isfinite(loss):raise ValueError('Nonfinite MLM loss')
                (loss/4).backward();trainloss.append(float(loss.detach()))
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0);optim.step()
            if len(trainloss)%80==0:
                write(run/'status.json',dict(status='running',stage='mask',epoch=epoch,conditions_completed=len(trainloss)//4,train_loss=float(np.mean(trainloss)),elapsed_seconds=time.time()-start))
        model.eval();vl=[]
        with torch.no_grad(),torch.random.fork_rng(devices=[0]):
            # Validation corruption and source/target cell sampling fixed across epochs.
            torch.manual_seed(4242);torch.cuda.manual_seed_all(4242);vr=np.random.default_rng(4242)
            for sid in config['splits']['val']:
                for j in range(8):
                    src,tgt,events=batch(sid,8,vr);vl.append(float(loss_for(model,src,tgt,events)))
        score=float(np.mean(vl));scheduler.step(score);improved=score<best-1e-6
        if improved:
            best=score;stale=0
        else:stale+=1
        # Save continuation state each epoch; best is an atomic hard link to it.
        state={k:v.detach().cpu() for k,v in model.state_dict().items() if not k.startswith('encoder_layers.')}
        torch.save(dict(model=state,optimizer=optim.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,val_loss=best,stale=stale,manifest=manifest,rng=capture_rng(rng)),run/'mask_last.pt.partial')
        (run/'mask_last.pt.partial').replace(run/'mask_last.pt')
        if improved:
            link=run/'mask_best.pt.partial'
            if link.exists():link.unlink()
            os.link(run/'mask_last.pt',link);link.replace(run/'mask_best.pt')
        status=dict(status='running',stage='mask',epoch=epoch,train_loss=float(np.mean(trainloss)),val_loss=score,best_val_loss=best,stale_epochs=stale,elapsed_seconds=time.time()-start)
        write(run/'status.json',status);print('EPOCH',json.dumps(status),flush=True)
        if stale>=20:break
    write(run/'MASK_COMPLETE.json',dict(status='mask_stage_complete',epochs=epoch,best_val_loss=best,checkpoint_sha256=sha(run/'mask_best.pt'),next_stage='native CountHead training and generation-based validation; not final benchmark result'))


if __name__=='__main__':main()
