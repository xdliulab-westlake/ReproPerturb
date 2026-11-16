set -eu
U=/liaozizhuo/hypoblast/benchmarks/stack-unified-20261113
exec 9>"$U/run.lock"
flock -n 9
while kill -0 54469 2>/dev/null; do sleep 15; done
if test ! -f "$U/smoke/seed43/SMOKE_COMPLETE.json"; then echo SMOKE_FAILED > "$U/STATUS"; exit 1; fi
echo TRAINING_THREE_SEEDS > "$U/STATUS"
bash "$U/run_seed.sh" 43 0 & p43=$!
bash "$U/run_seed.sh" 44 1 & p44=$!
bash "$U/run_seed.sh" 45 3 & p45=$!
failed=0
for p in "$p43" "$p44" "$p45"; do if ! wait "$p"; then failed=1; fi; done
if test "$failed" = 1; then echo FAILED_CHECK_LOGS > "$U/STATUS"; exit 1; fi
echo TEST_COMPLETE_THREE_SEEDS > "$U/STATUS"
/liaozizhuo/iPStem/benchmark/envs/torch_native/bin/python "$U/publish_results.py" > "$U/logs/publish.log" 2>&1
/liaozizhuo/iPStem/benchmark/envs/torch_native/bin/python /liaozizhuo/hypoblast/benchmarks/native-alignment-20261111/rank_metrics/backfill_pds.py > "$U/logs/pds_backfill.log" 2>&1
echo TEST_VERIFIED_AND_PUBLISHED_THREE_SEEDS > "$U/STATUS"
