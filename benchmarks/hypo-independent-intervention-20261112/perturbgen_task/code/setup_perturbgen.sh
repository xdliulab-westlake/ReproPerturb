#!/usr/bin/env bash
set -euo pipefail
B=/liaozizhuo/hypoblast/benchmarks/hypo-benchmark-extension-20261105
BASE=/liaozizhuo/iPStem/benchmark/envs/torch_native
if [ ! -x "$B/env/bin/python" ]; then "$BASE/bin/python" -m venv "$B/env"; fi
printf '%s\n' "$BASE/lib/python3.10/site-packages" > "$B/env/lib/python3.10/site-packages/benchmark_base.pth"
/liaozizhuo/iPStem/benchmark/envs/bootstrap_tools/bin/uv pip install --python "$B/env/bin/python" 'datasets==3.2.0' > "$B/logs/datasets_install.log" 2>&1
CKPT="$B/vendor/Perturbgen/pretraining_cohort/20250709_1223_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=00.ckpt"
if [ "$(stat -c %s "$CKPT")" -lt 1000000 ]; then
 curl --fail --location --retry 2 --connect-timeout 15 'https://huggingface.co/lotfollahi-lab/PerturbGen/resolve/main/20250709_1223_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=00.ckpt' --output "$CKPT.part"
 mv "$CKPT.part" "$CKPT"
fi
sha256sum "$CKPT" > "$B/audit/encoder.sha256"
echo SETUP_COMPLETE
