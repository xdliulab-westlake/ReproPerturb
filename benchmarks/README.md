# Response benchmarks

The composition comparison uses one direct proportion predictor per model and seed. The test set contains 14 conditions with 10 cell types; the outer training and validation sets contain 146 and 21 conditions. ReproPerturb trains a new Stage-2 model with the native loss. External models fine-tune their own prediction path and proportion head. Select the checkpoint with the lowest validation composition MAE (patience 20, maximum 3,000 epochs).

[Composition training and scoring](composition/) covers ReproPerturb, STATE, CellFlow, CPA, chemCPA, G2CP, STACK, PerturbNet and PerturbGen. The model-specific training scripts require the prepared datasets, upstream model code and checkpoint directories named in each script. PerturbGen uses frozen matched-control source representations and a separately supervised population head. All three seeds completed. Set REPROPERTURB_COMPOSITION_DIR to the prepared benchmark directory when using a different data root. The sibling PerturbGen runtime supplies its native model implementation and checkpoint.

Score each seed (43, 44, 45) independently, then report the mean and sample SD. Report MAE and RMSE in percentage points, Jensen–Shannon divergence in bits, mean per-condition Hellinger distance and dominant-type accuracy with fractional credit for ties. CSV inputs have a sample_id column and one column per class. NPZ predictions contain sample_ids, classes and proportions.

```bash
python benchmarks/composition/evaluate.py --prediction predictions.npz --truth truth.npz --split benchmark.json
```

[Expression training and scoring](expression/) uses a separate ReproPerturb checkpoint selected on validation gene PCC. The evaluator compares aligned condition-by-gene matrices over 2,000 genes, using matched control expression for delta metrics. The ten displayed metrics are sample and gene PCC and sample-wise R² for raw and delta expression, raw MSE, true top-100 MAE, top-20 overlap and top-100 Jaccard. Undefined correlations are excluded and defined counts are returned. score_expression.py reads the prepared server layout. Comparisons are computational evaluations.
