set -eu
I=/liaozizhuo/hypoblast/benchmarks/hypo-independent-intervention-20261112
B=$I/perturbgen_task
s=$1
export CUDA_VISIBLE_DEVICES=$2 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
exec 9>"$B/runs_seed${s}.lock"
flock -n 9
P=$B/env/bin/python
if test ! -f "$B/runs/perturbgen_seed$s/MASK_COMPLETE.json"; then
 r=""
 if test -f "$B/runs/perturbgen_seed$s/mask_last.pt"; then r=--resume; fi
 "$P" "$B/code/train_perturbgen.py" --seed "$s" $r > "$B/logs/seed${s}_mask.log" 2>&1
fi
"$P" "$B/code/finish_perturbgen.py" --seed "$s" > "$B/logs/seed${s}_count.log" 2>&1
