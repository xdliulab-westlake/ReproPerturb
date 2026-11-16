set -e
B=/liaozizhuo/hypoblast/benchmarks/native-alignment-20261111
export CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
for s in 43 44 45; do
 /liaozizhuo/iPStem/benchmark/envs/cellflow_native/bin/python "$B/evaluate_cellflow.py" --model cellflow --seed "$s" > "$B/evaluate_cellflow_seed$s.log" 2>&1
done
