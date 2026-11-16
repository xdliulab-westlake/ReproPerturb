# Additional expression ranking metrics

Official implementation pinned: ArcInstitute/cell-eval2 551849dc25354dff724edceb39a6108d997bf876.

Requested scope: all model versions/seeds in expression benchmarks, including Hypo and deterministic baselines; new models automatically require both metrics before final report publication. No test-set tuning, unchanged fixed14 test conditions and2000 genes.

Current deliverables: pds_inventory.csv (46 available frozen prediction files), pds_per_condition.csv (644 rows). This first pass is explicitly pds_cosine_stored_delta: official midrank/n-1 ranking definition, applied to the existing mean expression minus condition-matched real control. It is NOT the VCC raw-count bulk_lognorm preset and must not be compared to competition scores. Versions remain separate. Missing Hypo/baseline formats still require inventory/adapter work; all-model coverage is not claimed.

Direction Reach specification: de_wilcoxon_direction_reach_raw, reference-significant gene pool, predicted significance then adjusted p then p then descending absolute LFC then gene-name ordering, deepest prefix satisfying raw sign purity>=0.90, divided by reference-significant budget. Empty real-significant pools remain undefined, not zero. Chemical combinations are not gene knockouts: no artificial target-gene removal.

Outstanding: retrieve/generate per-cell frozen predictions and aligned real/control samples, run pinned DE backend and full unthresholded tables; mean-only outputs cannot establish predicted Wilcoxon significance. Preserve all inputs, seeds, per-condition results, context mappings, count/processed-scale conversions and missing-input reasons. Do not substitute absolute-effect ranking or bootstrap copies of pseudobulk means. Aggregate per seed first, then report mean/SD across training seeds. No VCC baseline-to-replicate rescaling without independently constructed matched anchors.

Training jobs are independent and remain authorized/running. STACK CLI retry uses valid official strategy ddp (one visible GPU); previous auto argument was rejected before training.
