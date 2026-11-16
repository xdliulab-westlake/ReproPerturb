# ReproPerturb

Source-only snapshot of the full-model workflow used in the latest 2026-11-16 figures. The default pathway output is **direct ranking of all 34 signed relation scores** (17 pathways × activation/inhibition), not pathway selection followed by direction selection.

## Architecture

1. **Stage-1 cell representation**: precomputed scFoundation embeddings → 128-dimensional SoftVQ representation (4 codebooks, 128 entries per codebook, 32 dimensions each). Expression reconstruction and 10-class cell-type supervision train the representation; 128 population prototypes summarize the starting population.
2. **Stage-2 response model**: signed 17-pathway condition encoding and combination interactions condition a prototype mass flow model. Self-attention, condition cross-attention and time modulation predict prototype mass, presence and state, with composition/expression readouts. The full-model configuration is retained, including matched-pair supervision.
3. **Relation scoring**: probe each pathway in both directions at activity strength **1.5**, normalize responses and apply the fitted kernel relation scorer and its locked combination rule. Keep all 34 output scores. Aggregate the highest 7 of 64 background scores per signed candidate, then average seeds 43/44/45. Rank all 34 candidates together; both directions of one pathway may appear near the top. Internal identity/direction score construction remains part of the fitted scorer, but no direction is discarded before ranking.

The adopted Stage-2 checkpoints were freshly trained for 150 epochs in the 2026-11-09 reproduction; Stage-1 was reused. The latest benchmark figure retains the full model. Without-matched ablations are not the default release.

## Code organization

| Block | Architecture | Training / evaluation |
|---|---|---|
| **Stage-1** | [`model.py`](src/stage1/model.py), [`components/`](src/stage1/components/) | [`train.py`](src/stage1/train.py), [`train_stage1.py`](train_stage1.py), [`evaluation/`](src/stage1/evaluation/) |
| **Stage-2** | [`src/stage2/model/`](src/stage2/model/) | [`src/stage2/runners/`](src/stage2/runners/) |
| **Relation scorer** | [`shared_math.py`](src/hypo_release/shared_math.py) | [`unified.py`](src/hypo_release/unified.py) |
| **Signed34 output** | [`all_celltype_heatmaps.py`](workflows/all_celltype_heatmaps.py) | [`rank_signed34.py`](rank_signed34.py) |
| **Final benchmark** | [`benchmarks/README.md`](benchmarks/README.md) | Adopted evaluation/aggregation only |
| **Shared dependencies** | `src/data/`, `src/scfoundation/`, `src/configs/` | Used by Stage-1/Stage-2 |

See the [Stage-1 guide](src/stage1/README.md) and [Stage-2 guide](src/stage2/README.md). Trial analyses, protocol-ranking experiments, incomplete replacement benchmark jobs, installation/recovery scripts and old plotting workflows are excluded. Server originals are untouched. Imported registry components remain to preserve model loading.

## Environment and external assets

Use Python 3.10 and `pip install -r requirements.txt` with compatible PyTorch/CUDA. External benchmark models require their upstream repositories, pretrained backbones and separate runtimes; see their scripts. Respect `LICENSE_NOTICE.txt`; this repository does not grant new rights to third-party models or data.

No checkpoints, datasets, fitted relation parameters, predictions, figures, logs or cached features are uploaded. Supply authorized assets under `assets/`, `weights/`, `inputs/`, and `models/`: Stage-1/Stage-2 checkpoints, pair records, pathway base vectors, drug-pathway sign mapping, Stage-2/context splits, example records and fitted kernel models. Stage-1 also requires precomputed cell embeddings, expression targets, gene indices and its split. Historical absolute data paths in configs and benchmark scripts must be adapted to your environment. This is not a turnkey data-inclusive package.

## Training and evaluation

After configuring data paths and restoring required assets:

```bash
torchrun --standalone --nproc_per_node=1 train_stage1.py
python hypo.py train --seed 43 --epochs 150 --output /path/to/new_training_seed43
# Repeat Stage-2 training for seeds 44 and 45; install the epoch-150
# checkpoints as weights/seed43.pt, seed44.pt and seed45.pt.
python hypo.py extract --seed 43
python hypo.py extract --seed 44
python hypo.py extract --seed 45
# For fresh relation selection, work in a separate run directory without
# existing SHARED_CONFIG.json / fitted outputs; select refuses overwrites.
python hypo.py select
python hypo.py fit-evaluate
python workflows/all_celltype_heatmaps.py
python rank_signed34.py --scores all_celltype_heatmaps/scores_all_celltypes_34.csv --output signed34_rankings.csv
```

The shipped shared settings describe the adopted run; do not delete them merely to rerun inference. Legacy `hypo.py` report/verify and known-/unknown-K evaluations retain their historical selection semantics and are **not the default 34-candidate ranking interface**. Its `forward` command also includes a historical strength-0.75 contextual audit; use the explicit strength-1.5 workflow above for the final ranking. `prepare` expects the full historical release assets and is not a source-only bootstrap command.

## Benchmark interpretation

The latest figure uses completed seeds 43/44/45 for STATE independent embeddings, CellFlow updated pipeline, STACK unified training, repaired chemCPA and G2CP native UniPert. PerturbNet and PerturbGen figure values retain legacy adapted results while replacements remain incomplete. Included experimental scripts do not imply their runs are complete. Model-specific checkpoint selection and retrospective development exposure limit comparisons. Computational pathway scores are relative rankings, not biological validation or measured pathway activity.

## Provenance and verification

`SOURCE_MANIFEST.json` records original server source locations/hashes; `PUBLICATION_MANIFEST.json` records the final packaged files after entry-point and path adjustments. Publication checks cover Python/JSON and shell syntax, common credential patterns, artifact exclusion and a synthetic 34-way ranking case preserving opposite directions. No GPU training or numerical reproduction was rerun for this publication.
