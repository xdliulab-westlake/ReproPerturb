from pathlib import Path
import os,sys,json
import torch
B=Path('/liaozizhuo/hypoblast/benchmarks/native-alignment-20261111');r=B/'chemcpa_task/runs/chemcpa_seed43_chemvocabfix_20261112';os.environ['IPSTEM_RUN_ROOT']=str(r);os.environ['IPSTEM_TRAIN_SEED']='43';sys.path.insert(0,str(r/'code'))
from common import read
from adversarial_models import ChemAdversarial
torch.set_num_threads(4);m=ChemAdversarial(read(B/'chemcpa_task/config/genes.json')).cuda();m.train();v=m.chemical_vocabulary
rows=torch.cat([v[:2],torch.zeros((2,1),device='cuda')],dim=1)
def encode(x):return m.encode_condition(x[None],torch.ones((1,len(x)),device='cuda'))
with torch.no_grad():
 a=encode(rows[:1]);b=encode(rows[1:]);ab=encode(rows);err=float((ab-a-b).abs().max());assert err<1e-5,err
 p=torch.zeros((1,195),device='cuda');p[:,194]=1;out=encode(p);assert torch.isfinite(out).all()
report=dict(composition_additivity_max_error=err,single_chemical_finite=bool(torch.isfinite(a).all()),protein_only_finite=bool(torch.isfinite(out).all()),training_only_chemical_vocabulary_size=len(v),optimizer_updates=0)
(B/'chemcpa_composition_contract_test.json').write_text(json.dumps(report,indent=2));print(report)
