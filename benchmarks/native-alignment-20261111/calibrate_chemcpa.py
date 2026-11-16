from pathlib import Path
import os,sys,json,copy
import numpy as np,torch
B=Path('/liaozizhuo/hypoblast/benchmarks/native-alignment-20261111');seed=int(sys.argv[1]);r=B/f'chemcpa_task/runs/chemcpa_seed{seed}';os.environ['IPSTEM_RUN_ROOT']=str(r);os.environ['IPSTEM_TRAIN_SEED']=str(seed);sys.path.insert(0,str(r/'code'))
from common import FullData,read,gene_pcc
from adversarial_models import ChemAdversarial
from train_torch import pad
torch.set_num_threads(4);d=FullData();model=ChemAdversarial(read(B/'chemcpa_task/config/genes.json')).cuda();ck=torch.load(r/'checkpoints/chemcpa/best.pt',map_location='cuda',weights_only=True);model.load_state_dict(ck['model']);model.eval();dest=B/f'chemcpa_bn_calibration/seed{seed}';dest.mkdir(parents=True,exist_ok=True)
def activate(module):
 for v in module.modules():
  if isinstance(v,torch.nn.modules.batchnorm._BatchNorm):v.reset_running_stats();v.momentum=None;v.train()
with torch.no_grad():
 # Chemical BN: all training chemical descriptors jointly; no protein/dummy rows.
 f=np.concatenate([d.features[c] for c in d.train_ids]);f=f[f[:,194]==0,:194];activate(model.chemical_encoder);model.chemical_encoder(torch.tensor(f,device='cuda'));model.chemical_encoder.eval()
 activate(model.encoder);activate(model.decoder)
 for cid in d.train_ids:
  x,y=d.batch(cid,np.random.default_rng(seed),latent=False);x=torch.tensor(x,device='cuda');y=torch.tensor(y,device='cuda');e,m=pad(d.features,[cid],'cuda');cond=model.encode_condition(e,m).expand(len(x),-1);obs=torch.cat([x,y]);model.reconstruct(obs,torch.cat([torch.zeros_like(cond),cond]))
 model.eval();pred=[];truth=[]
 for cid in d.val_ids:
  row=d.by_id[cid];ix=d.indices[row['ctrl_sample']];e,m=pad(d.features,[cid],'cuda');total=np.zeros(2000)
  for start in range(0,len(ix),1024):
   x=torch.tensor(np.asarray(d.x[ix[start:start+1024]],dtype=np.float32),device='cuda')[None];p=model(x,e,m)[0];assert torch.isfinite(p).all();total+=p.cpu().numpy().sum(0,dtype=np.float64)
  pred.append(total/len(ix));truth.append(d.means[row['sample']])
  print('VAL',seed,cid,flush=True)
 pred=np.array(pred);truth=np.array(truth);report=dict(seed=seed,original_best_val_gene_pcc=ck['meta']['best'],calibrated_val_gene_pcc=float(gene_pcc(pred,truth)),calibrated_val_mse=float(((pred-truth)**2).mean()),calibration_conditions=len(d.train_ids),test_used=False,parameter_updates=0)
 ck['model']=model.state_dict();ck['bn_calibration']=report;torch.save(ck,dest/'calibrated.pt');(dest/'validation.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
