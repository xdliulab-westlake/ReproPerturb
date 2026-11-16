"""Portable package paths and path-only checkpoint configuration normalization."""
import copy
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def resolve(key, suffix=''):
    bases = {'root': ROOT, 'source': ROOT / 'src',
             'legacy': ROOT / 'src/hypo_release/legacy',
             'support': ROOT / 'src/hypo_release/support',
             'assets': ROOT / 'assets'}
    if key == 'work':
        if not os.environ.get('HYPO_WORK_DIR'):
            raise RuntimeError('Use the CLI to set an isolated --work directory.')
        bases['work'] = Path(os.environ['HYPO_WORK_DIR'])
    return str(bases[key] / suffix.lstrip('/'))


def normalize_config(config):
    result = copy.deepcopy(config)
    result['stage1']['checkpoint'] = str(ROOT / 'weights/stage1.pt')
    result['stage1']['config_module'] = 'configs.stage1.stage1_softvq_hypo_hvg2000_celltype_updated_v7'
    result['pathway']['sign_csv'] = str(ROOT / 'assets/drug_pathway_sign.csv')
    result['pathway']['pathways_yaml'] = str(ROOT / 'assets/pathways.yaml')
    result['training']['split_path'] = str(ROOT / 'assets/stage2_split.csv')
    return result


def training_config():
    return normalize_config(json.loads((ROOT / 'configs/model.json').read_text()))
