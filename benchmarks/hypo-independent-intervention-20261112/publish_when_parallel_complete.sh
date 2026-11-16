set -eu
I=/liaozizhuo/hypoblast/benchmarks/hypo-independent-intervention-20261112
B=$I/perturbgen_task
for s in 43 44 45; do
 while test ! -f "$B/runs/perturbgen_seed$s/COMPLETE.json"; do sleep 60; done
done
"$B/env/bin/python" "$I/publish_perturbgen_results.py" > "$B/logs/publish_results.log" 2>&1
echo TEST_COMPLETE_3_SEEDS > "$B/STATUS"
