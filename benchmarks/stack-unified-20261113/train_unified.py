from pathlib import Path
import argparse,json,os,sys,subprocess,hashlib,time
import numpy as np,torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
sys.path.insert(0,'/liaozizhuo/iPStem/benchmark/models/stack/repo/src')
from stack.data.training.datasets import DatasetConfig,SimplifiedMultiDataset,worker_init_fn
from stack.training.lightning import LegacyLightningGeneModel
U=Path('/liaozizhuo/hypoblast/benchmarks/stack-unified-20261113');N=U.parent/'native-alignment-20261111';args=argparse.ArgumentParser();args.add_argument('--seed',type=int,choices=[43,44,45],required=True);args.add_argument('--smoke',action='store_true');a=args.parse_args();cfg=json.loads((U/'CONFIG.json').read_text());split=json.loads((U/'SPLIT.json').read_text());seed=a.seed
pl.seed_everything(seed,workers=True);torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
run=U/('smoke' if a.smoke else 'runs')/f'seed{seed}';run.mkdir(parents=True,exist_ok=True)
files=[str(U/'data'/f'{s}.h5ad') for s in split['train']];assert len(files)==146 and len(set(files))==146
# File lists are explicit: the official dataset must not create another random split.
dataset=SimplifiedMultiDataset([DatasetConfig(path=str(U/'data'),filter_organism=False,gene_name_col='gene_symbols')],genelist_path=str(U/'genes.pkl'),sample_size=256,mode='train',random_state=seed,cache_file=str(run/'train_metadata.pkl'),train_files=files,val_files=[],test_files=[])
assert set(dataset.active_files)==set(files)
loader=DataLoader(dataset,batch_size=8,shuffle=True,num_workers=4,pin_memory=True,persistent_workers=True,drop_last=True,worker_init_fn=worker_init_fn)
old=torch.load(N/'stack_task/runs/seed43/last.ckpt',map_location='cpu',weights_only=False);architecture=old['hyper_parameters']['model_config'];del old
class NativeModel(LegacyLightningGeneModel):
 def configure_optimizers(self):return torch.optim.AdamW(self.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
class GenerationSelection(pl.Callback):
 def __init__(self):self.best=-float('inf');self.stale=0;self.history=[];self.scheduler=None;self.pending_scheduler=None
 def state_dict(self):return dict(best=self.best,stale=self.stale,history=self.history,scheduler=self.scheduler.state_dict() if self.scheduler else None)
 def load_state_dict(self,s):self.best=s['best'];self.stale=s['stale'];self.history=s['history'];self.pending_scheduler=s.get('scheduler')
 def on_fit_start(self,trainer,module):
  self.scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(trainer.optimizers[0],mode='max',patience=10,factor=.5)
  if self.pending_scheduler:self.scheduler.load_state_dict(self.pending_scheduler)
 def on_train_epoch_end(self,trainer,module):
  epoch=trainer.current_epoch+1;temp=run/'validation_input.ckpt';trainer.save_checkpoint(temp)
  dest=(run/f'validation_epoch{epoch}') if a.smoke else U/f'validation/seed{seed}/epoch{epoch}'
  env=dict(os.environ,STACK_EVAL_CHECKPOINT=str(temp),STACK_EVAL_DEST=str(dest),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
  if a.smoke:env['STACK_SMOKE']='1'
  (run/'STATUS').write_text(f'VALIDATING_EPOCH{epoch}\n')
  with (run/f'validation_epoch{epoch}.log').open('w') as log:
   subprocess.run([sys.executable,str(U/'score_checkpoint.py'),str(seed),'val',str(epoch)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  pred=np.load(dest/'predictions.npz');truth=np.load(U/'validation_truth.npz');ix=[truth['sample_ids'].tolist().index(s) for s in pred['sample_ids'].tolist()];p=pred['expression'].astype(np.float64);t=truth['expression'][ix].astype(np.float64);mse=float(((p-t)**2).mean());p-=p.mean(0);t-=t.mean(0);den=np.sqrt((p*p).sum(0)*(t*t).sum(0));score=float(np.divide((p*t).sum(0),den,out=np.zeros_like(den),where=den>1e-12).mean());assert np.isfinite(score) and np.isfinite(mse)
  improved=score>self.best
  if improved:self.best=score;self.stale=0
  else:self.stale+=1
  self.scheduler.step(score);r=dict(epoch=epoch,validation_gene_pcc=score,validation_mse=mse,best_validation_gene_pcc=self.best,stale=self.stale,validation_conditions=len(ix),test_access=False);self.history.append(r);(run/'history.json').write_text(json.dumps(self.history,indent=2));print('GENERATION_VALIDATION_COMPLETE',r,flush=True)
  if improved:trainer.save_checkpoint(run/'best.ckpt')
  trainer.save_checkpoint(run/'last.ckpt');(run/'STATUS').write_text(f'TRAINING_AFTER_EPOCH{epoch}\n')
  if self.stale>=20:trainer.should_stop=True
model=NativeModel(model_config=architecture,learning_rate=cfg['lr'],weight_decay=cfg['weight_decay'],scheduler_config={});selection=GenerationSelection()
provenance=dict(seed=seed,split=split,training_file_count=146,training_loss='native STACK masked reconstruction + SW',selection='full21 generated-expression gene PCC; all matched controls',fresh_weights=True,public_model_architecture=architecture,prompt_rule=cfg['prompt_rule'],test_access=False,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());(run/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2));(run/'STATUS').write_text('TRAINING\n')
trainer=pl.Trainer(max_epochs=1 if a.smoke else 3000,accelerator='gpu',devices=1,strategy='auto',precision='bf16-mixed',gradient_clip_val=1.,logger=TensorBoardLogger(str(run/'tensorboard'),name='stack_unified'),callbacks=[selection],enable_checkpointing=False,enable_progress_bar=False,log_every_n_steps=25,limit_train_batches=2 if a.smoke else 1.0,num_sanity_val_steps=0,benchmark=True)
resume=str(run/'last.ckpt') if (run/'last.ckpt').exists() and not a.smoke else None
trainer.fit(model,train_dataloaders=loader,ckpt_path=resume)
(run/('SMOKE_COMPLETE.json' if a.smoke else 'TRAIN_COMPLETE.json')).write_text(json.dumps(dict(seed=seed,best_validation_gene_pcc=selection.best,epochs=len(selection.history),test_access=False,training_conditions=146,validation_conditions=1 if a.smoke else 21,checkpoint=str(run/'best.ckpt')),indent=2));(run/'STATUS').write_text('SMOKE_COMPLETE\n' if a.smoke else 'TRAIN_COMPLETE_TEST_PENDING\n')
