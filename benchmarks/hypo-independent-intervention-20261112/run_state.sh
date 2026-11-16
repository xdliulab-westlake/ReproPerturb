set -eu
B=/liaozizhuo/hypoblast/benchmarks/hypo-independent-intervention-20261112
P=/liaozizhuo/iPStem/benchmark/envs/torch_native/bin/python
# Respect the four-GPU resource requirement of the existing chemCPA task.
while test ! -f /liaozizhuo/hypoblast/benchmarks/native-alignment-20261111/evaluation/chemcpa_seed45_chemvocabfix_20261112/COMPLETE.json; do sleep 20; done
export CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
for s in 43 44 45; do
 export IPSTEM_TRAIN_SEED=$s IPSTEM_RUN_ROOT="$B/state_task/runs/state_seed$s"
 echo SMOKE > "$B/state_task/STATUS"
 "$P" -m torch.distributed.run --standalone --nproc_per_node=4 "$IPSTEM_RUN_ROOT/code/train_four_gpu.py" --model state --smoke > "$IPSTEM_RUN_ROOT/smoke.log" 2>&1
 echo TRAINING_SEED$s > "$B/state_task/STATUS"
 "$P" -m torch.distributed.run --standalone --nproc_per_node=4 "$IPSTEM_RUN_ROOT/code/train_four_gpu.py" --model state > "$IPSTEM_RUN_ROOT/train.log" 2>&1
 CUDA_VISIBLE_DEVICES=0 "$P" "$B/evaluate_state.py" --model state --seed "$s" > "$IPSTEM_RUN_ROOT/evaluation.log" 2>&1
done
echo TEST_COMPLETE_3_SEEDS > "$B/state_task/STATUS"
