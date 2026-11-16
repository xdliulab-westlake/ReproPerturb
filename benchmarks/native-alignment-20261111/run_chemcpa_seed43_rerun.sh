set -eu
B=/liaozizhuo/hypoblast/benchmarks/native-alignment-20261111
export CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 IPSTEM_TRAIN_SEED=43
export IPSTEM_RUN_ROOT="$B/chemcpa_task/runs/chemcpa_seed43_rerun_20261112"
echo TRAINING > "$IPSTEM_RUN_ROOT/STATUS"
P=/liaozizhuo/iPStem/benchmark/envs/cpa_native/bin/python
"$P" -m torch.distributed.run --standalone --nproc_per_node=4 "$IPSTEM_RUN_ROOT/code/train_four_gpu.py" --model chemcpa > "$IPSTEM_RUN_ROOT/train.log" 2>&1 || { echo TRAINING_FAILED > "$IPSTEM_RUN_ROOT/STATUS"; exit 1; }
echo EVALUATING > "$IPSTEM_RUN_ROOT/STATUS"
CUDA_VISIBLE_DEVICES=0 "$P" "$B/evaluate_chemcpa_seed43_rerun.py" --model chemcpa --seed 43 > "$IPSTEM_RUN_ROOT/evaluation.log" 2>&1 || { echo EVALUATION_FAILED > "$IPSTEM_RUN_ROOT/STATUS"; exit 1; }
echo COMPLETE > "$IPSTEM_RUN_ROOT/STATUS"
