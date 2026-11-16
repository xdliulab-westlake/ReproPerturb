set -e
B=/liaozizhuo/hypoblast/benchmarks/native-alignment-20261111
export CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
/liaozizhuo/iPStem/benchmark/envs/torch_native/bin/python "$B/prepare_scvi_counts.py"
while kill -0 24850 2>/dev/null; do sleep 15; done
P=/liaozizhuo/iPStem/benchmark/envs/perturbnet_native/bin/python
"$P" -c 'import scvi; print(scvi.__version__)'
for s in 43 44 45; do
 "$P" "$B/train_scvi.py" --seed "$s" > "$B/perturbnet_scvi_seed$s.log" 2>&1
done
