#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/liaozizhuo/hypoblast/reproductions/hypo17_unified_release_20261027_retrain_20261109
SOURCE=/liaozizhuo/hypoblast/model_packages/hypo17_unified_release_20261027
PY="$ROOT/env/bin/python"
RUN="$ROOT/run"
mkdir -p "$ROOT/logs"
trap 'echo "FAILED line=$LINENO time=$(date -Iseconds)" > "$ROOT/STATUS"' ERR
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
cd "$SOURCE"
echo SOURCE_VERIFICATION > "$ROOT/STATUS"
sha256sum -c --quiet TRANSFER_SHA256SUMS > "$ROOT/logs/source_hash_check.log"
"$PY" -m pip freeze > "$ROOT/environment_freeze.txt"
"$PY" hypo.py prepare --output "$RUN" > "$ROOT/logs/prepare.log" 2>&1
cd "$RUN"
echo TRAINING_THREE_SEEDS > "$ROOT/STATUS"
pids=()
for seed in 43 44 45; do
 gpu=$((seed-43))
 (
  export CUDA_VISIBLE_DEVICES="$gpu"
  "$PY" hypo.py train --seed "$seed" --epochs 150 --output "$ROOT/training/seed$seed" > "$ROOT/logs/train_seed$seed.log" 2>&1
  "$PY" - "$ROOT/training/seed$seed/snapshot_epoch150.pt" "$RUN/weights/seed$seed.pt" <<'PY'
import sys,torch,shutil
c=torch.load(sys.argv[1],map_location='cpu',weights_only=True)
assert c['epoch']==150
shutil.copy2(sys.argv[1],sys.argv[2])
PY
  "$PY" hypo.py extract --seed "$seed" > "$ROOT/logs/extract_seed$seed.log" 2>&1
  "$PY" hypo.py forward --seed "$seed" > "$ROOT/logs/forward_seed$seed.log" 2>&1
  echo completed > "$ROOT/logs/seed$seed.COMPLETE"
 ) &
 pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ "$failed" == 0 ]]
echo SHARED_SELECTION_AND_REFIT > "$ROOT/STATUS"
"$PY" hypo.py select > "$ROOT/logs/select.log" 2>&1
"$PY" hypo.py fit-evaluate > "$ROOT/logs/fit_evaluate.log" 2>&1
export CUDA_VISIBLE_DEVICES=0
echo COMBINATION_AUDITS > "$ROOT/STATUS"
"$PY" experiments/combination_growth/run.py infer --variant known_k --output "$RUN/experiments/combination_growth/known_k" > "$ROOT/logs/known_k.log" 2>&1
"$PY" experiments/combination_growth/run.py infer --variant unknown_k --output "$RUN/experiments/combination_growth/unknown_k" > "$ROOT/logs/unknown_k.log" 2>&1
"$PY" hypo.py report > "$ROOT/logs/report.log" 2>&1
"$PY" hypo.py verify > "$ROOT/logs/verify.log" 2>&1
# Upstream report/verify contain historical 'no retraining' text; these are kept as
# original software outputs. The reproduction record supersedes that metadata.
"$PY" "$ROOT/summarize_pathways.py" --root "$ROOT"
echo COMPLETE > "$ROOT/STATUS"
