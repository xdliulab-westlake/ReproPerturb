"""Launch exactly three incremental native PerturbGen runs on visible GPUs."""
import argparse,datetime,fcntl,os,subprocess,sys,time
from pathlib import Path
from prepare_composition import ROOT,OLD,read,write,evaluate_predictions


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true')
    ap.add_argument('--seeds',type=int,nargs='+',default=[43,44,45]);ap.add_argument('--gpus',nargs='+',default=['0','1','0'])
    ap.add_argument('--adopt',action='append',default=[],help='Existing local mask worker as seed:pid')
    args=ap.parse_args();seeds=args.seeds
    assert len(seeds)==len(args.gpus) and len(set(seeds))==len(seeds) and set(seeds)<=set([43,44,45])
    gpus=dict(zip(seeds,args.gpus));host=os.uname().nodename
    adopted={int(v.split(':')[0]):int(v.split(':')[1]) for v in args.adopt};assert set(adopted)<=set(seeds)
    if not args.execute:print('PLAN: 24 existing expression rows; 3 new PerturbGen rows; 27 final rows.');return
    lock=(ROOT/'audit'/f'perturbgen_runner_{host}.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for file in ['tokens_complete','readout_complete','perturbgen_smoke','perturbgen_count_smoke']:
        status=read(ROOT/'audit'/f'{file}.json')['status'];assert status in ['complete','passed'],(file,status)
    rows={};children={};logs={}
    def launch(seed,stage):
        script='train_perturbgen.py' if stage=='mask' else 'finish_perturbgen.py'
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpus[seed],OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
        log=(ROOT/'logs'/f'perturbgen_seed{seed}_{stage}.log').open('a');logs[seed]=log
        command=[str(ROOT/'env/bin/python'),'-u',str(ROOT/'code'/script),'--seed',str(seed)]
        if stage=='mask' and (ROOT/'runs'/f'perturbgen_seed{seed}'/'mask_best.pt').exists():command.append('--resume')
        p=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        children[seed]=p;rows[seed]=dict(seed=seed,gpu=gpus[seed],hostname=host,pid=p.pid,stage=stage,status='running',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    class Adopted:
        def __init__(self,seed,pid):
            self.seed=seed;self.pid=pid
            cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
            assert str(ROOT/'code/train_perturbgen.py').encode() in cmd and b'--seed' in cmd and str(seed).encode() in cmd
        def poll(self):
            stat=Path('/proc')/str(self.pid)/'stat'
            if stat.exists() and stat.read_text().rsplit(')',1)[1].strip().split()[0]!='Z':return None
            return 0 if (ROOT/'runs'/f'perturbgen_seed{self.seed}'/'MASK_COMPLETE.json').exists() else 1
    for seed in seeds:
        run=ROOT/'runs'/f'perturbgen_seed{seed}'
        if (run/'COMPLETE.json').exists():rows[seed]=dict(seed=seed,status='complete');continue
        if seed in adopted:
            children[seed]=Adopted(seed,adopted[seed]);rows[seed]=dict(seed=seed,gpu=gpus[seed],hostname=host,pid=adopted[seed],stage='mask',status='running',adopted=True);continue
        launch(seed,'finish' if (run/'MASK_COMPLETE.json').exists() else 'mask')
    def record():write(ROOT/'audit'/f'perturbgen_queue_{host}.json',dict(existing_expression_rows=24,total_new_rows=3,local_new_rows=len(seeds),final_expression_rows=27,hostname=host,runs=list(rows.values()),pid=os.getpid()))
    record()
    while children:
        for seed,p in list(children.items()):
            code=p.poll()
            if code is None:continue
            log=logs.pop(seed,None)
            if log is not None:log.close()
            del children[seed]
            if code:
                rows[seed].update(status='failed',exit_code=code)
            elif rows[seed]['stage']=='mask':launch(seed,'finish')
            else:rows[seed].update(status='complete',exit_code=0)
            record()
        if children:time.sleep(20)
    evaluation_lock=(ROOT/'audit/composition_evaluation.lock').open('w')
    fcntl.flock(evaluation_lock,fcntl.LOCK_EX);evaluate_predictions()
    if any(r['status']=='failed' for r in rows.values()):raise RuntimeError('A run failed; inspect per-seed logs')
    print('PERTURBGEN_ALL_COMPLETE',flush=True)


if __name__=='__main__':main()
