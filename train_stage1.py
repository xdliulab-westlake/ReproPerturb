"""Stage-1 SoftVQ training; launch with torchrun."""
import sys, importlib, copy, argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from stage1.train import train_vqvae_scrna
if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config-module',default='configs.stage1.stage1_softvq_hypo_hvg2000_celltype_updated_v7')
    a=p.parse_args(); train_vqvae_scrna(copy.deepcopy(importlib.import_module(a.config_module).CONFIG))
