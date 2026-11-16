# Final-figure benchmark sources

Only final Extended Data Figure 3 plotting/aggregation, adopted CellFlow/chemCPA/STATE evaluators and unified STACK training/scoring are retained. Old plots, protocol-ranking comparisons, queues, installation/recovery scripts and incomplete replacement PerturbNet/PerturbGen analyses have been removed from this publication. Server originals are unchanged.

- `native-alignment-20261111/ed3_update_20261116/plot.py`: final panels.
- `native-alignment-20261111/composition_extra_20261114/calculate.py`: composition metrics.
- `native-alignment-20261111/finalize_chemcpa_*.py`: final metric-table updates.
- `native-alignment-20261111/evaluate_cellflow.py`, `evaluate_chemcpa*.py`: adopted evaluation routes; repaired chemCPA has distinct seed-43 and seed-44/45 routes.
- `hypo-independent-intervention-20261112/evaluate_state.py`: adopted STATE evaluation.
- `stack-unified-20261113/`: adopted unified STACK training/scoring.

External model source, data, predictions and environment paths are required. This is not a standalone implementation of every benchmark. Final figures retain legacy adapted PerturbNet/PerturbGen values; incomplete replacement training is excluded. Retrospective evaluation and differing checkpoint selection remain limitations.
