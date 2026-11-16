set -eu
UV=/liaozizhuo/iPStem/benchmark/envs/bootstrap_tools/bin/uv
E=/liaozizhuo/iPStem/benchmark/envs/perturbnet_native
"$UV" venv --system-site-packages --python /liaozizhuo/iPStem/benchmark/envs/torch_native/bin/python "$E"
"$UV" pip install --python "$E/bin/python" 'scvi-tools==1.3.3'
"$E/bin/python" -c 'import scvi; print(scvi.__version__)'
