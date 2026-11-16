from pathlib import Path
import json,os,sys
import numpy as np
B=Path('/liaozizhuo/hypoblast/benchmarks/hypo-independent-intervention-20261112');r=B/'state_task/runs/state_seed43';os.environ['IPSTEM_RUN_ROOT']=str(r);os.environ['IPSTEM_TRAIN_SEED']='43';sys.path.insert(0,str(r/'code'))
real=np.load;loaded=[]
def guarded(file,*a,**kw):
 p=str(file);loaded.append(p)
 assert 'drug_embeddings' not in p and 'cell_ze.npy' not in p,('FORBIDDEN_HYPO_INPUT',p)
 return real(file,*a,**kw)
np.load=guarded
from common import FullData
d=FullData();assert all(v.shape[1]==1 for v in d.features.values())
report=dict(status='passed',state_runtime_loads=loaded,training_conditions=len(d.train_ids),validation_conditions=len(d.val_ids),intervention_feature_dimension=1,forbidden_hypo_drug_or_cell_embeddings_loaded=False,perturbgen_training_feature_path=str(B/'agent_ids_train_val.npz'),perturbgen_inference_feature_path=str(B/'agent_ids_all.npz'))
(B/'NO_HYPO_INPUT_AUDIT.json').write_text(json.dumps(report,indent=2));print('RUNTIME_INPUT_AUDIT_PASSED')
