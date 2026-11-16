set -eu
B=/liaozizhuo/hypoblast/benchmarks/native-alignment-20261111
P=/liaozizhuo/iPStem/benchmark/envs/cpa_native/bin/python
export CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
for s in 43 44 45; do
 export IPSTEM_TRAIN_SEED=$s IPSTEM_RUN_ROOT="$B/chemcpa_task/runs/chemcpa_seed${s}_chemvocabfix_20261112"
 echo TRAINING > "$IPSTEM_RUN_ROOT/STATUS"
 "$P" -m torch.distributed.run --standalone --nproc_per_node=4 "$IPSTEM_RUN_ROOT/code/train_four_gpu.py" --model chemcpa > "$IPSTEM_RUN_ROOT/train.log" 2>&1
 echo EVALUATING > "$IPSTEM_RUN_ROOT/STATUS"
 CUDA_VISIBLE_DEVICES=0 "$P" "$B/evaluate_chemcpa_chemvocabfix.py" --model chemcpa --seed "$s" > "$IPSTEM_RUN_ROOT/evaluation.log" 2>&1
 echo COMPLETE > "$IPSTEM_RUN_ROOT/STATUS"
done
/liaozizhuo/iPStem/benchmark/envs/torch_native/bin/python "$B/finalize_chemcpa_repair.py" > "$B/finalize_chemcpa_repair.log" 2>&1
/liaozizhuo/iPStem/benchmark/envs/torch_native/bin/python "$B/rank_metrics/backfill_pds.py" > "$B/rank_metrics/backfill_repaired.log" 2>&1
echo REPAIRED_THREE_SEED_TEST_EVALUATION_COMPLETE > "$B/chemcpa_task/STATUS"
