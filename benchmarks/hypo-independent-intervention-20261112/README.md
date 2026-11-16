# STATE and PerturbGen independent intervention representations

User authorization: use native inputs or independently retrained intervention embeddings, without Hypo drug embeddings.

Fixed outer split:146 training,21 validation,14 test conditions. Train-only vocabulary24 intervention identities, including chemical compounds and proteins; all validation/test reagents occur in training. Source expression2000 genes and shared frozen10-class V7 readout retained.

STATE: native ST backbone and verified gene-remapped public initialization, plus fresh Embedding(25,pert_dim) per seed. Padding ID0. Sum component vectors and pass through the native perturbation input. Direct observed expression input; old unused Hypo cell-latent load removed. Four-GPU smoke, training and test evaluation are queued after chemCPA seed45's fixed-test evaluation releases the four-GPU task. See run_state.sh and state_task/STATUS.

PerturbGen: native gene-token encoder/decoder, native generation and CountHead. Fresh Embedding(25,768) replaces the Hypo2048-to768 intervention projection. A learned combination-order projection is retained. Embeddings are trained during masked-token training; CountHead is then trained separately. This remains an endpoint combination-memory adaptation, not a fully native chemical-intervention API. Teacher/generated CountHead distribution mismatch is retained from the prior route and remains disclosed.

Verified: STATE runtime np.load audit forbids drug_embeddings files and cell_ze.npy, passed. PerturbGen independent embedding gradient/permutation/parameter-update/generation checks passed; CountHead smoke passed; seed43 formal masked-training process started. Seeds44/45 and full expression/composition evaluation follow automatically.

Source and provenance: agent_vocabulary.json, NO_HYPO_INPUT_AUDIT.json, per-run PROVENANCE.json, perturbgen_task/PROVENANCE.json. Historical checkpoints/predictions are not loaded as continuation weights for the new intervention embeddings. Public ST/Geneformer pretrained backbone initialization remains allowed. No test expression or labels supervise the intervention embeddings.
