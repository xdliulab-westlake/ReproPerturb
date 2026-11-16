set -e
B=/liaozizhuo/hypoblast/benchmarks/native-alignment-20261111
export CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
for s in 43 44 45; do
 /liaozizhuo/iPStem/benchmark/envs/perturbnet_native/bin/python "$B/train_scvi.py" --seed "$s" > "$B/perturbnet_scvi_seed$s.log" 2>&1
done
