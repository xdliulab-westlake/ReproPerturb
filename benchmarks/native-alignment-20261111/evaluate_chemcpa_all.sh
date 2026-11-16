set -e
B=/liaozizhuo/hypoblast/benchmarks/native-alignment-20261111
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
for s in 43 44 45; do
 /liaozizhuo/iPStem/benchmark/envs/cpa_native/bin/python "$B/evaluate_chemcpa.py" --model chemcpa --seed "$s" > "$B/evaluate_chemcpa_seed$s.log" 2>&1
done
