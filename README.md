# ReproPerturb

ReproPerturb predicts how combinations of small molecules and proteins change cell-type composition and gene expression in a starting population. It was developed for stem-cell differentiation experiments that include hypoblast-like populations. Given a target cell type, the model also prioritizes pathway activation and inhibition candidates.

[Source release v0.1.0](https://github.com/xdliulab-westlake/ReproPerturb/releases/tag/v0.1.0) · [Version history](CHANGELOG.md) · [Raw count matrix](https://huggingface.co/datasets/xdliulab-westlake/ReproPerturb)

## How it works

- **Stage-1** learns cell-state representations from scFoundation embeddings with soft vector quantization. The resulting representations summarize the starting population as a distribution over cell-state prototypes.
- **Stage-2** uses a conditional flow model to predict how the intervention combination changes prototype abundance and state. It produces endpoint cell-type proportions and gene expression estimates.
- **Pathway prioritization** probes the trained Stage-2 model across reference backgrounds. A trained relation scorer uses the predicted target-cell response to score and rank all 34 pathway–direction candidates together for each target cell type.

## Data and setup

The raw single-cell RNA-seq count matrix is available in AnnData format on [Hugging Face](https://huggingface.co/datasets/xdliulab-westlake/ReproPerturb). Training and model-based analysis require prepared inputs and model assets. The workflow guides describe the expected inputs and checkpoints.

Use Python 3.10 and a compatible CUDA environment for training and response extraction.

```bash
git clone https://github.com/xdliulab-westlake/ReproPerturb.git
cd ReproPerturb
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

## Training and evaluation

| Workflow | Entry point |
| --- | --- |
| Stage-1 representation model | [Architecture and training guide](src/stage1/README.md) · [Training script](scripts/train_stage1.py) |
| Stage-2 perturbation response model | [Architecture, training, and evaluation guide](src/stage2/README.md) · [Workflow CLI](scripts/hypo.py) |
| Signed pathway scoring and ranking | [Scoring workflow](workflows/all_celltype_heatmaps.py) · [Direct ranking script](scripts/rank_signed34.py) |
| Composition and expression benchmarks | [Benchmark guide](benchmarks/README.md) · [Composition](benchmarks/composition/) · [Expression](benchmarks/expression/) |

Use the best checkpoint selected on the validation split for each workflow. The composition benchmark compares models with dedicated proportion heads and reports measures including mean absolute error, Jensen–Shannon divergence, and dominant-cell-type accuracy. The expression benchmark evaluates predicted expression and changes relative to matched controls.

## Citation and contact

Cite the [source release](https://github.com/xdliulab-westlake/ReproPerturb/releases/tag/v0.1.0) or record the commit used in your analysis. Upstream model, code, and data terms are described in [LICENSE_NOTICE.txt](LICENSE_NOTICE.txt).

For implementation questions, open a [GitHub issue](https://github.com/xdliulab-westlake/ReproPerturb/issues) or contact [liaozizhuo@westlake.edu.cn](mailto:liaozizhuo@westlake.edu.cn).
