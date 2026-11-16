set -eu
U=/liaozizhuo/hypoblast/benchmarks/stack-unified-20261113
P=/liaozizhuo/iPStem/benchmark/envs/perturbnet_native/bin/python
seed=$1
gpu=$2
export CUDA_VISIBLE_DEVICES=$gpu PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p "$U/runs/seed$seed"
if test ! -f "$U/runs/seed$seed/TRAIN_COMPLETE.json"; then
 "$P" "$U/train_unified.py" --seed "$seed" > "$U/logs/train_seed$seed.log" 2>&1 || { echo TRAIN_FAILED > "$U/runs/seed$seed/STATUS"; exit 1; }
fi
if test ! -f "$U/evaluation/stack_traincontext_seed$seed/COMPLETE.json"; then
 echo TEST_EVALUATING > "$U/runs/seed$seed/STATUS"
 "$P" "$U/score_checkpoint.py" "$seed" test 0 > "$U/logs/test_seed$seed.log" 2>&1 || { echo TEST_FAILED > "$U/runs/seed$seed/STATUS"; exit 1; }
fi
echo TEST_COMPLETE > "$U/runs/seed$seed/STATUS"
