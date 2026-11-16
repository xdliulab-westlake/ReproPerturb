"""Run the authorized native-alignment jobs, each requiring all four GPUs."""
import json
import os
from pathlib import Path
import subprocess
import time

B = Path('/liaozizhuo/hypoblast/benchmarks/native-alignment-20261111')
ENV = Path('/liaozizhuo/iPStem/benchmark/envs')


def main():
    # The already-launched CellFlow smoke owns the GPUs until it exits.
    while Path('/proc/16079').exists():
        if Path('/proc/16079/stat').read_text().split()[2] == 'Z':
            break
        time.sleep(15)
    for method in ['chemcpa', 'cellflow']:
        task = B / f'{method}_task'
        smoke = task / f'runs/{method}_seed43/development/{method}/smoke.json'
        if not smoke.exists() or not json.loads(smoke.read_text()).get('passed', False):
            (task / 'STATUS').write_text('BLOCKED_SMOKE_NOT_PASSED\n')
            continue
        for seed in [43, 44, 45]:
            run = task / f'runs/{method}_seed{seed}'
            if (run / f'checkpoints/{method}/COMPLETE.json').exists():
                continue
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3',
                       OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                       XLA_PYTHON_CLIENT_PREALLOCATE='false',
                       IPSTEM_TRAIN_SEED=str(seed), IPSTEM_RUN_ROOT=str(run))
            if method == 'chemcpa':
                cmd = [str(ENV / 'cpa_native/bin/python'), '-m', 'torch.distributed.run',
                       '--standalone', '--nproc_per_node=4',
                       str(run / 'code/train_four_gpu.py'), '--model', method]
            else:
                cmd = [str(ENV / 'cellflow_native/bin/python'),
                       str(run / 'code/train_cellflow_four_gpu.py')]
            (task / 'STATUS').write_text(f'TRAINING_SEED{seed}\n')
            with (run / 'train.log').open('a') as log:
                result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode or not (run / f'checkpoints/{method}/COMPLETE.json').exists():
                (task / 'STATUS').write_text(f'FAILED_SEED{seed}_EXIT{result.returncode}\n')
                break
        else:
            (task / 'STATUS').write_text('TRAINING_COMPLETE_EVALUATION_PENDING\n')


if __name__ == '__main__':
    main()
