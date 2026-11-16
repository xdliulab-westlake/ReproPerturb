# Stage-2: conditional population response

- `model/prototype_mass_flow.py`: mass flow architecture and attention blocks.
- `model/pathway_condition.py`: signed pathway/combination condition encoder.
- `model/prototype_vq.py`: population prototype representation.
- `runners/train_prototype_mass_flow.py`: training.
- `runners/evaluate_prototype_mass_flow.py`: evaluation and model loading.
- `runners/stage1_boundary.py`: frozen Stage-1 interface.

Run `python scripts/hypo.py train --seed 43 --epochs 150 --output /path/to/new_run`. Config: `configs/model.json`. Shared data helpers are in `src/data/`; encoder support is in `src/scfoundation/`. Optional conditioners remain as loader dependencies; presence does not imply enabled configuration.
