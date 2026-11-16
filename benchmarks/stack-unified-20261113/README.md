# STACK unified146/21/14 retraining

Old exploratory runs remain in native-alignment-20261111/stack_task. New runs use exactly146 outer training-condition files, with no internal random split. There is no test loader during training. Architecture matches the prior official STACK config; all weights are freshly initialized for seeds43/44/45.

Native masked reconstruction+SW loss is retained. Every epoch saves a temporary model for full21-condition generation using only fixed top3 Jaccard training-context prompts. Validation gene PCC across2000 genes (zero for constant genes) selects the best checkpoint. Max3000 epochs, patience20, AdamW lr5e-5 weight decay0.01, plateau patience10 factor0.5. Final14 test conditions are generated only after training. Same2000 genes, V7 readout, all matched control source cells,5000x3 composition sampling.

Each condition's native generation counts are converted to the established benchmark scale with non-modelled genes held at source-control counts; roundtrip checks are mandatory. This remains a train-context adaptation, not the public aligned checkpoint's original same-perturbation cross-cell-type task. No Hypo learned drug/cell embeddings are used.

Verification: two native training batches followed by generation of the first validation condition, with no test target access, completed successfully. Formal runs execute complete21 validation conditions. run_all.sh launches three seeds, then verifies and publishes all3 final results and backfills PDS. Old checkpoints and results remain intact.
