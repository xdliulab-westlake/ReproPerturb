"""Audit shared hyperparameters, saved refits, fresh features, and result scoring."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from hypo_release.paths import ROOT as R
from hypo_release.unified import examples,functions,training_utility,metric,read

def main():
 c=read(R/'SHARED_CONFIG.json');u=read(R/'SHARED_UNKNOWN_K.json');res=read(R/'RESULTS.json');fn=functions()
 cfgs=[];shapes=[];models=[];checks=[]
 for s in [43,44,45]:
  ck=torch.load(R/f'weights/seed{s}.pt',map_location='cpu',weights_only=True);cfgs.append(ck['config']);shapes.append({k:tuple(v.shape) for k,v in ck['model'].items()});del ck
  model=read(R/f'models/seed{s}/CONFIG.json');models.append(model);assert model=={'relation':c,'unknown_k':u}
  tr,te,paths,groups=examples(s);pred=np.load(R/f'models/seed{s}/holdout_logits.npy');m=fn['evaluate'](te,torch.tensor(pred).float(),paths)
  assert m==res[str(s)]['known_k']
  oof=np.load(R/f'models/seed{s}/oof_logits.npy');y=np.stack([e.label.numpy() for e in tr]);slow=fn['evaluate'](tr,torch.tensor(oof).float(),paths)['utility'];fast=training_utility(y,oof);assert abs(fast-slow)<1e-12
  assert (R/f'features/seed{s}/COMPLETE.json').exists()
  assert read(R/f'forward/seed{s}/test/metrics.json')['n_samples']==14
  checks.append({'seed':s,'same_shared_config':True,'known_k_rescored':m['celltypes_passed'],'training_objective_verified':True})
 assert cfgs[0]==cfgs[1]==cfgs[2] and shapes[0]==shapes[1]==shapes[2]
 grid=read(R/'training_selection/RELATION_GRID.json');assert c==max(grid,key=lambda x:x['mean_training_oof_utility'])
 prior=read(R/'training_selection/PRIOR_GRID.json');assert u==max(prior,key=lambda x:x['mean_training_oof_utility'])
 known=sum(read(R/f'experiments/combination_growth/known_k/seed{s}/COMPLETE.json')['queries'] for s in [43,44,45])
 unknown=len(read(R/'experiments/combination_growth/unknown_k/ALL_PREDICTIONS.json'))
 result={'all_flow_configs_and_state_shapes_identical':True,'all_relation_hyperparameters_identical':True,'all_unknown_k_hyperparameters_identical':True,'shared_training_only_grid_argmax_verified':True,'fresh_feature_extraction':True,'fresh_flow_validation_all_seeds':True,'fresh_relation_refits':True,'full_flow_retraining':False,'known_combination_queries':known,'unknown_combination_queries':unknown,'seed_checks':checks}
 print(json.dumps(result,indent=2))
 return result
if __name__=='__main__':main()
