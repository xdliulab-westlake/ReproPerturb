# Stage-2: conditional population response

- `model/prototype_mass_flow.py`: mass flow architecture and attention blocks.
- `model/pathway_condition.py`: signed pathway/combination condition encoder.
- `model/prototype_vq.py`: population prototype representation.
- `runners/train_prototype_mass_flow.py`: training.
- `runners/evaluate_prototype_mass_flow.py`: evaluation and model loading.
- `runners/stage1_boundary.py`: frozen Stage-1 interface.

Run `python scripts/hypo.py train --seed 43 --output /path/to/new_run` from the repository root. Training follows the stopping criteria in `configs/model.json`. Use `best_prototype_mass_flow.pt`, selected by the lowest validation base loss, for the corresponding seed’s evaluation and pathway workflow.

Shared data helpers are in `src/data/`, and encoder support is in `src/scfoundation/`. Optional conditioners remain available to the model loader.
