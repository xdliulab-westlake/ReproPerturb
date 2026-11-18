"""Supervised population adapter for PerturbGen's frozen source encoder and trainable decoder."""
from pathlib import Path
import fcntl, json, os, sys
import numpy as np
import torch
from torch import nn

D = Path(os.environ.get('REPROPERTURB_COMPOSITION_DIR', '/liaozizhuo/hypoblast/benchmarks/direct-composition-20261117'))
P = D.parent/'hypo-independent-intervention-20261112/perturbgen_task'
sys.path.insert(0,str(P/'code'))
from train_perturbgen import CombinationPerturbGen, seed_all
from prepare_composition import OLD, read, sha
seed = int(sys.argv[1])
out = D/f'perturbgen_seed{seed}'
out.mkdir(exist_ok=True)
lock = (out/'run.lock').open('w')
fcntl.flock(lock,fcntl.LOCK_EX)
if (out/'COMPLETE.json').exists(): sys.exit(0)
seed_all(seed)
cfg = read(OLD/'benchmark.json')
splits = cfg['splits']
control = {r['sample_id']:r['control_sample_id'] for r in cfg['cohort']}
rows = np.load(OLD/'data/sample_rows_all.npz')
tokens = np.load(P/'data/tokens_train_val_source.npy',mmap_mode='r')
events = np.load(P.parent/'agent_ids_all.npz')
truth = np.load(OLD/'data/truth.npz')
targets = dict(zip(truth['sample_ids'],truth['proportions']))
model = CombinationPerturbGen(seed).cuda()
weight = P/f'runs/perturbgen_seed{seed}/mask_best.pt'
missing, unexpected = model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True)['model'],strict=False)
assert not unexpected and all(k.startswith('encoder_layers.') for k in missing)
assert all(not p.requires_grad for p in model.encoder_layers.parameters())
head = nn.Sequential(nn.Linear(768,64),nn.GELU(),nn.Linear(64,10)).cuda()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad]+list(head.parameters()),lr=5e-5,weight_decay=.01)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=10,factor=.5)
cache={}
@torch.no_grad()
def population(cid):
    if cid not in cache:
        model.encoder_layers.eval()
        total=torch.zeros(768,device='cuda',dtype=torch.float64)
        ix=rows[cid]
        for start in range(0,len(ix),8):
            src=torch.as_tensor(np.array(tokens[ix[start:start+8]],dtype=np.int64),device='cuda')
            assert bool((src[:,0]==2).all())
            with torch.autocast('cuda',dtype=torch.bfloat16):
                rep=model.call_encoder(src,src.eq(0))
            total+=rep[:,0].double().sum(0)
        cache[cid]=(total/len(ix)).float().view(1,1,768)
    return cache[cid]

def predict(s):
    source=population(control[s])
    model.set_combination(torch.as_tensor(events[s],device='cuda'))
    cls=torch.full((1,1),2,device='cuda',dtype=torch.long)
    query=model.pos_embedding(model.token_embedding(cls),1)
    model.count_only=True
    result=model.call_decoder(source,cls.eq(0),query,cls.eq(0))
    return head(result['dec_embedding'][0,0]).softmax(-1)

def evaluate(ids):
    model.eval();head.eval()
    with torch.no_grad(): pred=np.stack([predict(s).cpu().numpy() for s in ids])
    assert np.isfinite(pred).all() and np.allclose(pred.sum(1),1.,atol=1e-6)
    return float(100*np.abs(pred-np.stack([targets[s] for s in ids])).mean()),pred

def save(name,obj):
    tmp=out/(name+'.tmp');torch.save(obj,tmp);os.replace(tmp,out/name)

(out/'CONFIG.json').write_text(json.dumps(dict(seed=seed,mask_checkpoint=str(weight),sha256=sha(weight),source='mean frozen CLS over all matched control cells',readout='native decoder CLS query attends population and learned combination tokens -> independent 768/64/10 head',selection='condition-balanced validation MAE',patience=20,max_epochs=3000,lr=5e-5),indent=2))
trainable=dict((k,p) for k,p in model.named_parameters() if p.requires_grad)
best=float('inf');stale=0;start=0
if (out/'last.pt').exists():
    q=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(q['model'],strict=False);head.load_state_dict(q['head']);opt.load_state_dict(q['optimizer']);sched.load_state_dict(q['scheduler'])
    best,stale,start=q['best'],q['stale'],q['epoch']
for epoch in range(start+1,3001):
    if stale>=20: break
    model.train();model.encoder_layers.eval();head.train()
    for j,s in enumerate(np.random.default_rng(seed+epoch).permutation(splits['train'])):
        opt.zero_grad()
        p=predict(s);loss=(p-torch.as_tensor(targets[s],device='cuda')).abs().mean()
        assert torch.isfinite(loss)
        loss.backward()
        if epoch==1 and j==0:
            for param in (model.combination_projection.weight,head[-1].weight):
                assert param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum()>0
            print('GRADIENT_PASS',flush=True)
        torch.nn.utils.clip_grad_norm_(list(trainable.values())+list(head.parameters()),1.);opt.step()
    val,_=evaluate(splits['val']);sched.step(val)
    state=dict(model={k:p.detach() for k,p in trainable.items()},head=head.state_dict(),epoch=epoch)
    if val<best:
        best=val;stale=0;save('best.pt',state)
    else: stale+=1
    save('last.pt',dict(**state,optimizer=opt.state_dict(),scheduler=sched.state_dict(),stale=stale,best=best))
    print(epoch,val,best,stale,flush=True)
q=torch.load(out/'best.pt',weights_only=True);model.load_state_dict(q['model'],strict=False);head.load_state_dict(q['head'])
ids=splits['test'];mae,p=evaluate(ids);t=np.stack([targets[s] for s in ids]);assert p.shape==(14,10)
np.savez(out/'predictions.npz',sample_ids=ids,classes=cfg['classes'],proportions=p,truth=t)
from direct_metrics import score as metrics
m=metrics(p,t);(out/'metrics.json').write_text(json.dumps(m,indent=2))
(out/'COMPLETE.json').write_text(json.dumps(dict(seed=seed,mae_pp=mae,best_epoch=q['epoch'],test_conditions=14)))
