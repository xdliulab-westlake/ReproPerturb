set -eu
UV=/liaozizhuo/iPStem/benchmark/envs/bootstrap_tools/bin/uv
E=/liaozizhuo/iPStem/benchmark/envs/cell_eval2_cpu
export UV_HTTP_TIMEOUT=180
"$UV" python install 3.11
"$UV" venv --python 3.11 "$E"
"$UV" pip install --python "$E/bin/python" -e /liaozizhuo/iPStem/benchmark/models/cell-eval2
CUDA_VISIBLE_DEVICES= "$E/bin/python" -c 'import cell_eval2; print("CELL_EVAL2_CPU_IMPORT_OK")'
