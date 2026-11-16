"""Native scVI expression representation for the PerturbNet route, train-only."""
from pathlib import Path
import argparse
import json
import anndata as ad
import scvi
from lightning.pytorch.callbacks import ModelCheckpoint
from scvi.train import Trainer, TrainRunner

p = argparse.ArgumentParser()
p.add_argument('--seed', type=int, choices=[43, 44, 45], required=True)
args = p.parse_args()
B = Path('/liaozizhuo/hypoblast/benchmarks/native-alignment-20261111/perturbnet_task')
dest = B / f'runs/perturbnet_seed{args.seed}/scvi'
assert (B / 'COUNTS_READY.json').exists()
if (dest / 'SCVI_COMPLETE.json').exists():
    print('Already complete', args.seed, flush=True)
    raise SystemExit(0)
scvi.settings.seed = args.seed
data = ad.read_h5ad(B / 'scvi_train_counts.h5ad')
scvi.model.SCVI.setup_anndata(data)
model = scvi.model.SCVI(data, n_latent=10)
checkpoint_dir = dest.parent / 'training_checkpoints'
checkpoint = ModelCheckpoint(dirpath=str(checkpoint_dir), save_last=True, save_top_k=0, every_n_epochs=1)
resume_path = checkpoint_dir / 'last.ckpt'
class ResumeTrainer(Trainer):
    def fit(self, *args, **kwargs):
        kwargs['ckpt_path'] = str(resume_path) if resume_path.exists() else None
        return super().fit(*args, **kwargs)

class ResumeRunner(TrainRunner):
    _trainer_cls = ResumeTrainer

model._train_runner_cls = ResumeRunner
# Internal early-stopping cells are drawn only from benchmark training conditions.
model.train(max_epochs=420, accelerator='gpu', devices=1, batch_size=1024,
            train_size=0.9, validation_size=0.1, early_stopping=True,
            early_stopping_patience=20, callbacks=[checkpoint], enable_checkpointing=True)
model.save(dest, overwrite=False, save_anndata=False)
for name, value in model.history.items():
    value.to_csv(dest / f'history_{name}.csv')
(dest / 'SCVI_COMPLETE.json').write_text(json.dumps(dict(
    stage='scvi_only_flow_pending', seed=args.seed, scvi_version=scvi.__version__,
    n_cells=data.n_obs, n_genes=data.n_vars, n_latent=10,
    benchmark_val_test_used=False, source=str(B / 'COUNTS_READY.json')), indent=2))
