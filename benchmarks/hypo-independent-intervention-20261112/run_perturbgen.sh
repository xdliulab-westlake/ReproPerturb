set -eu
B=/liaozizhuo/hypoblast/benchmarks/hypo-independent-intervention-20261112/perturbgen_task
P="$B/env/bin/python"
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
"$P" "$B/code/train_perturbgen.py" --seed 43 --smoke > "$B/logs/independent_embedding_smoke.log" 2>&1
"$P" "$B/code/finish_perturbgen.py" --seed 43 --smoke > "$B/logs/count_head_smoke.log" 2>&1
for s in 43 44 45; do
 echo MASK_TRAINING_SEED$s > "$B/STATUS"
 "$P" "$B/code/train_perturbgen.py" --seed "$s" > "$B/logs/seed${s}_mask.log" 2>&1
 echo COUNT_HEAD_TRAINING_SEED$s > "$B/STATUS"
 "$P" "$B/code/finish_perturbgen.py" --seed "$s" > "$B/logs/seed${s}_count.log" 2>&1
done
echo TEST_COMPLETE_3_SEEDS > "$B/STATUS"
"$P" "$B/../publish_perturbgen_results.py" > "$B/logs/publish_results.log" 2>&1
