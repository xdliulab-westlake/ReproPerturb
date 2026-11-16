"""Evaluate held-out cell-type proportions from prototype-mass flow predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from experiments.perturbation.train_prototype_mass_flow import (
    DEFAULT_EXTRA_EVALUATION,
    ConditionalPrototypeMassFlow,
    _load_config,
    collate_pairs,
    split_pair_records_from_manifest,
)
from model.perturbation.condition_io import build_configured_condition_encoder
from model.perturbation.prototype_mass_flow import PrototypeMassFlow, PrototypeMassFlowAdapter
from common.evaluate_all import evaluate_all
from data.drug_combo_tokenizer import configured_combo_tokenizer, transform_combo_records


def _round(value: float) -> float:
    return round(float(value), 6)


def _js_divergence(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    midpoint = 0.5 * (prediction + truth)
    return 0.5 * (
        (prediction * torch.log(prediction.clamp_min(1e-8) / midpoint.clamp_min(1e-8))).sum(dim=-1)
        + (truth * torch.log(truth.clamp_min(1e-8) / midpoint.clamp_min(1e-8))).sum(dim=-1)
    )


def celltype_metric_tables(
    truth: torch.Tensor,
    prediction: torch.Tensor,
    sample_ids: list[str],
    classes: list[str],
    splits: list[str],
) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Build wide tables where every cell-type is a separate column."""
    absolute_error = (prediction - truth).abs()
    def rows(values: torch.Tensor) -> list[dict]:
        return [{
            "sample_id": sample_ids[index], "split": splits[index],
            **{celltype: _round(values[index, column].item()) for column, celltype in enumerate(classes)},
        } for index in range(len(sample_ids))]
    per_type_mae = absolute_error.mean(dim=0)
    per_type_rmse = (prediction - truth).square().mean(dim=0).sqrt()
    summary = {
        "n_samples": len(sample_ids),
        "mean_proportion_mae": _round(absolute_error.mean().item()),
        "mean_proportion_rmse": _round((prediction - truth).square().mean().sqrt().item()),
        "per_celltype_mae": {name: _round(per_type_mae[index].item()) for index, name in enumerate(classes)},
        "per_celltype_rmse": {name: _round(per_type_rmse[index].item()) for index, name in enumerate(classes)},
    }
    return rows(truth), rows(prediction), rows(absolute_error), summary


def celltype_expression_metric_tables(
    truth: torch.Tensor,
    prediction: torch.Tensor,
    valid_mask: torch.Tensor,
    sample_ids: list[str],
    classes: list[str],
    splits: list[str],
) -> tuple[list[dict], dict]:
    """Report expression MSE per observed sample × cell-type profile."""
    squared_error = (prediction - truth).square()
    rows = []
    for sample_index, sample_id in enumerate(sample_ids):
        for celltype_index, celltype in enumerate(classes):
            if not bool(valid_mask[sample_index, celltype_index]):
                continue
            rows.append({
                "sample_id": sample_id,
                "split": splits[sample_index],
                "celltype": celltype,
                "mse": _round(squared_error[sample_index, celltype_index].mean().item()),
            })
    per_celltype_mse = {}
    for celltype_index, celltype in enumerate(classes):
        selected = valid_mask[:, celltype_index]
        if selected.any():
            per_celltype_mse[celltype] = _round(
                squared_error[selected, celltype_index].mean().item()
            )
    selected_errors = squared_error[valid_mask]
    summary = {
        "n_samples": len(sample_ids),
        "n_valid_profiles": len(rows),
        "per_celltype_mse": per_celltype_mse,
        "mean_expression_mse": _round(selected_errors.mean().item()) if selected_errors.numel() else None,
    }
    return rows, summary


def _pearson_correlation(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left.float() - left.float().mean()
    right = right.float() - right.float().mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    return (left * right).sum() / denominator if denominator > 0 else left.new_tensor(float("nan"))


def population_expression_metric_tables(
    truth: torch.Tensor,
    prediction: torch.Tensor,
    control: torch.Tensor,
    sample_ids: list[str],
    splits: list[str],
    *,
    top_k: int = 200,
) -> tuple[list[dict], list[dict], dict]:
    """Report Pearson correlations and matched-control top-DEG overlap."""
    if truth.shape != prediction.shape or truth.shape != control.shape:
        raise ValueError("truth, prediction, and control must share [samples, genes] shape")
    if top_k < 1 or top_k > truth.shape[1]:
        raise ValueError("top_k must be in [1, n_genes]")

    gene_pcc = torch.stack([
        _pearson_correlation(truth[:, gene], prediction[:, gene])
        for gene in range(truth.shape[1])
    ])
    gene_rows = [
        {"gene_index": gene, "pcc": _round(value.item())}
        for gene, value in enumerate(gene_pcc) if torch.isfinite(value)
    ]
    sample_rows = []
    for index, sample_id in enumerate(sample_ids):
        sample_pcc = _pearson_correlation(truth[index], prediction[index])
        truth_de = (truth[index] - control[index]).abs()
        prediction_de = (prediction[index] - control[index]).abs()
        truth_top = torch.topk(truth_de, top_k).indices
        prediction_top = torch.topk(prediction_de, top_k).indices
        overlap_count = int(torch.isin(prediction_top, truth_top).sum().item())
        sample_rows.append({
            "sample_id": sample_id,
            "split": splits[index],
            "pcc": _round(sample_pcc.item()) if torch.isfinite(sample_pcc) else None,
            f"deg_overlap_at_{top_k}_count": overlap_count,
            f"deg_overlap_at_{top_k}_fraction": _round(overlap_count / top_k),
        })
    sample_pcc = [row["pcc"] for row in sample_rows if row["pcc"] is not None]
    overlap_counts = [row[f"deg_overlap_at_{top_k}_count"] for row in sample_rows]
    overlap_fractions = [row[f"deg_overlap_at_{top_k}_fraction"] for row in sample_rows]
    finite_gene_pcc = gene_pcc[torch.isfinite(gene_pcc)]
    summary = {
        "n_samples": len(sample_ids),
        "n_genes": int(truth.shape[1]),
        "mean_gene_wise_pcc": _round(finite_gene_pcc.mean().item()) if finite_gene_pcc.numel() else None,
        "n_valid_gene_wise_pcc": int(finite_gene_pcc.numel()),
        "mean_sample_wise_pcc": _round(sum(sample_pcc) / len(sample_pcc)) if sample_pcc else None,
        "n_valid_sample_wise_pcc": len(sample_pcc),
        f"mean_deg_overlap_at_{top_k}_count": _round(sum(overlap_counts) / len(overlap_counts)),
        f"mean_deg_overlap_at_{top_k}_fraction": _round(sum(overlap_fractions) / len(overlap_fractions)),
    }
    return gene_rows, sample_rows, summary


def sample_ids_from_manifest_test_split(records: list[dict], manifest_path: str | Path) -> set[str]:
    """Reuse the training split parser so CSV evaluation matches training exactly."""
    _, _, test_records = split_pair_records_from_manifest(records, manifest_path)
    return {str(record["sample_id"]) for record in test_records}


def select_split_records(
    records: list[dict], manifest_path: str | Path, *, split: str
) -> list[dict]:
    """Select one explicit manifest partition without diagnostic extras."""
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of: train, val, test")
    partitions = split_pair_records_from_manifest(records, manifest_path)
    return list(partitions[("train", "val", "test").index(split)])


def select_evaluation_records(
    records: list[dict], test_samples: set[str], extras: set[str], *, strict_test_only: bool,
) -> list[dict]:
    """Select the fixed test cohort, optionally appending diagnostic-only samples."""
    selected_samples = test_samples if strict_test_only else test_samples | extras
    return [record for record in records if str(record["sample_id"]) in selected_samples]


def load_evaluation_module(
    config: dict,
    checkpoint: dict,
    payload: dict,
    pathway_base: str | Path | None,
    device: torch.device,
) -> ConditionalPrototypeMassFlow:
    """Strictly reconstruct a baseline or pathway-cell-type evaluation module."""
    from hypo_release.paths import normalize_config
    config = normalize_config(config)
    pathway_config = config["pathway"]
    expression_dim = None
    if pathway_config.get("celltype_bottleneck", {}).get("enabled", False):
        records = payload.get("records", [])
        if not records or "terminal_celltype_expression" not in records[0]:
            raise ValueError(
                "celltype bottleneck evaluation requires terminal celltype expression targets"
            )
        expression_dim = int(records[0]["terminal_celltype_expression"].shape[-1])
    condition_encoder, _, _, _ = build_configured_condition_encoder(
        pathway_config,
        checkpoint["drug_names"],
        pathway_base,
        prototype_to_celltype=checkpoint.get("prototype_to_celltype"),
        expression_dim=expression_dim,
    )
    flow_cfg = config["flow"]
    flow = PrototypeMassFlow(
        checkpoint["prototype_codebook"],
        condition_dim=condition_encoder.condition_token_dim,
        model_dim=flow_cfg.get("token_model_dim", 256),
        time_dim=flow_cfg.get("time_freqs", 1024),
        num_layers=flow_cfg.get("num_token_layers", 3),
        ff_dim=flow_cfg.get("token_ff_dim", 4096),
        nhead=flow_cfg.get("token_nhead", 8),
        use_pathway_prototype_adaln=flow_cfg.get("pathway_prototype_adaln", False),
        pathway_prototype_injection_after_block=flow_cfg.get(
            "pathway_prototype_injection_after_block", 0
        ),
        pathway_adaln_mode=flow_cfg.get("pathway_adaln_mode"),
        pathway_adaln_blocks=tuple(flow_cfg.get("pathway_adaln_blocks", ())),
    )
    stage1_boundary = None
    if any(key.startswith("adapter.stage1_boundary.") for key in checkpoint["model"]):
        from experiments.perturbation.stage1_boundary import load_frozen_stage1_boundary

        stage1_cfg = config["stage1"]
        stage1_boundary = load_frozen_stage1_boundary(
            stage1_cfg["config_module"], stage1_cfg["checkpoint"], device
        )
    train_cfg = config["training"]
    module = ConditionalPrototypeMassFlow(
        condition_encoder,
        PrototypeMassFlowAdapter(
            flow,
            prototype_to_celltype=checkpoint["prototype_to_celltype"],
            stage1_boundary=stage1_boundary,
            endpoint_rollout_steps=flow_cfg.get("endpoint_rollout_steps", 20),
        ),
        unknown_effect_l1_weight=train_cfg.get("unknown_effect_l1_weight", 0.0),
        unknown_gate_l1_weight=train_cfg.get("unknown_gate_l1_weight", 0.0),
        auxiliary_proportion_weight=train_cfg.get("auxiliary_proportion_weight", 0.0),
        auxiliary_expression_weight=train_cfg.get("auxiliary_expression_weight", 0.0),
        pathway_shuffle_contrastive_weight=train_cfg.get(
            "pathway_shuffle_contrastive_weight", 0.0
        ),
    ).to(device)
    state = dict(checkpoint["model"])
    # Backward compatibility: checkpoints created before optional combo tokens
    # have no token-size buffer; every legacy token represents one drug.
    for key, value in module.state_dict().items():
        if key.endswith("token_sizes") and key not in state:
            state[key] = value
    module.load_state_dict(state, strict=True)
    module.eval()
    return module


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--flow-checkpoint", required=True)
    parser.add_argument("--pair-records", required=True)
    parser.add_argument("--pathway-base")
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-ode-steps", type=int, default=20)
    parser.add_argument("--population-cache", help="Population cache that supplies matched-control expression for DEG metrics.")
    parser.add_argument("--deg-top-k", type=int, default=200)
    parser.add_argument("--extra-evaluation-samples", nargs="*", default=list(DEFAULT_EXTRA_EVALUATION))
    parser.add_argument(
        "--evaluation-split", choices=("val", "test"),
        help="Evaluate one explicit split; defaults to the historical test-plus-extras behavior.",
    )
    parser.add_argument(
        "--strict-test-only", action="store_true",
        help="Evaluate exactly the split-manifest test cohort; ignore extra diagnostic samples.",
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    device = torch.device(args.device)
    config = _load_config(args.config)
    payload = torch.load(args.pair_records, map_location="cpu", weights_only=True)
    checkpoint = torch.load(args.flow_checkpoint, map_location="cpu", weights_only=True)
    # Learned BPE merges live in the checkpoint and must override the template
    # config used to start training.
    checkpoint_config = dict(checkpoint.get("config", {}))
    if configured_combo_tokenizer(checkpoint_config) is not None:
        config = checkpoint_config
    if args.strict_test_only and args.evaluation_split not in {None, "test"}:
        raise ValueError("--strict-test-only conflicts with --evaluation-split val")
    evaluation_split = "test" if args.strict_test_only else args.evaluation_split
    module = load_evaluation_module(
        config, checkpoint, payload, args.pathway_base, device
    )
    flow_cfg = config["flow"]
    extras = set(map(str, args.extra_evaluation_samples))
    if evaluation_split is not None:
        selected = select_split_records(
            payload["records"], args.split_manifest, split=evaluation_split
        )
        extras = set()
    else:
        test_samples = sample_ids_from_manifest_test_split(
            payload["records"], args.split_manifest
        )
        selected = select_evaluation_records(
            payload["records"], test_samples, extras, strict_test_only=False,
        )
    if not selected:
        raise ValueError("no pair record matched the selected evaluation cohort")
    tokenizer = configured_combo_tokenizer(config)
    if tokenizer is not None:
        selected = transform_combo_records(selected, tokenizer)
    classes = [str(value) for value in checkpoint["celltype_classes"]]
    mapping = checkpoint["prototype_to_celltype"].to(device).float()
    all_truth, all_prediction, all_prototype_rows = [], [], []
    all_expression = []
    all_expression_truth = []
    all_expression_control = []
    all_celltype_expression_prediction = []
    all_celltype_expression_truth = []
    all_celltype_expression_mask = []
    expression_flags = [
        "terminal_celltype_expression" in record and "terminal_expression" in record
        for record in selected
    ]
    control_expression = {}
    if args.population_cache:
        cache = torch.load(args.population_cache, map_location="cpu", weights_only=True)
        control_expression = {
            str(sample_id): torch.as_tensor(expression).float()
            for sample_id, expression in cache.get("sample_expression", {}).items()
        }
    # Extra evaluation records may contain proportions but no expression
    # target; keep proportion evaluation and report expression metrics only
    # when the selected records all provide that target.
    evaluate_celltype_expression = all(expression_flags) and module.adapter.stage1_boundary is not None
    sample_ids, splits = [], []
    with torch.no_grad():
        for start in range(0, len(selected), args.batch_size):
            records = selected[start:start + args.batch_size]
            batch = {key: value.to(device) for key, value in collate_pairs(records).items()}
            encoding = module.condition_encoder(batch["drug_ids"], batch["drug_mask"])
            tokens = encoding.condition_tokens
            mass = batch["initial_mass_logit"].clone()
            for step in range(args.n_ode_steps):
                time = torch.full((len(records),), (step + 0.5) / args.n_ode_steps, device=device)
                output = module.adapter.velocity_model(
                    mass, batch["initial_presence"], tokens, time, encoding.pathway_context,
                )
                mass = mass + output.mass_velocity / args.n_ode_steps
            terminal = module.adapter.velocity_model(
                mass, batch["initial_presence"], tokens, torch.ones(len(records), device=device),
                encoding.pathway_context,
            )
            predicted_presence = torch.sigmoid(terminal.presence_logits)
            prototype_proportion = module.adapter.velocity_model.presence_gated_proportion(mass, predicted_presence)
            celltype_prediction = prototype_proportion @ mapping
            if evaluate_celltype_expression and module.adapter.stage1_boundary is not None:
                all_expression.append(
                    module.adapter.decode_population_expression(
                        prototype_proportion, terminal.state_delta,
                    ).cpu()
                )
                if (
                    "terminal_expression" in batch
                    and control_expression
                    and all(
                        str(record["control_sample_id"]) in control_expression
                        for record in records
                    )
                ):
                    all_expression_truth.append(batch["terminal_expression"].cpu())
                    all_expression_control.append(torch.stack([
                        control_expression[str(record["control_sample_id"])]
                        for record in records
                    ]))
            if evaluate_celltype_expression:
                all_celltype_expression_prediction.append(
                    module.adapter.decode_celltype_expression(
                        prototype_proportion, terminal.state_delta,
                    ).cpu()
                )
                all_celltype_expression_truth.append(torch.stack([
                    torch.as_tensor(record["terminal_celltype_expression"]).float()
                    for record in records
                ]))
                all_celltype_expression_mask.append(torch.stack([
                    torch.as_tensor(record.get("terminal_celltype_expression_mask", torch.ones(len(classes), dtype=torch.bool))).bool()
                    for record in records
                ]))
            all_truth.append(batch["terminal_celltype_proportion"].cpu())
            all_prediction.append(celltype_prediction.cpu())
            for row, record in enumerate(records):
                all_prototype_rows.append({
                    "sample_id": str(record["sample_id"]),
                    "prototype_js": _round(_js_divergence(prototype_proportion[row], batch["terminal_proportion"][row]).item()),
                    "presence_precision": _round(((predicted_presence[row] >= 0.5) & batch["terminal_presence"][row].bool()).sum().item() / max((predicted_presence[row] >= 0.5).sum().item(), 1)),
                    "presence_recall": _round(((predicted_presence[row] >= 0.5) & batch["terminal_presence"][row].bool()).sum().item() / max(batch["terminal_presence"][row].sum().item(), 1)),
                })
                sample_ids.append(str(record["sample_id"]))
                splits.append(
                    "extra_evaluation"
                    if str(record["sample_id"]) in extras
                    else (evaluation_split or "test")
                )
    truth, prediction = torch.cat(all_truth), torch.cat(all_prediction)
    true_rows, predicted_rows, error_rows, metrics = celltype_metric_tables(
        truth, prediction, sample_ids, classes, splits,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(output_dir / "celltype_proportion_true.csv", true_rows)
    _write_rows(output_dir / "celltype_proportion_predicted.csv", predicted_rows)
    _write_rows(output_dir / "celltype_proportion_absolute_error.csv", error_rows)
    _write_rows(output_dir / "prototype_diagnostics.csv", all_prototype_rows)
    if all_expression:
        expression_prediction = torch.cat(all_expression)
        torch.save(expression_prediction, output_dir / "expression_prediction.pt")
        if all_expression_truth:
            expression_truth = torch.cat(all_expression_truth)
            torch.save(expression_truth, output_dir / "expression_true.pt")
            expression_mse = (expression_prediction - expression_truth).square().mean(dim=-1)
            (output_dir / "expression_metrics.json").write_text(json.dumps({
                "n_samples": int(expression_prediction.shape[0]),
                "n_genes": int(expression_prediction.shape[1]),
                "mean_pseudobulk_mse": _round(expression_mse.mean().item()),
                "per_sample_mse": {
                    sample_id: _round(value.item())
                    for sample_id, value in zip(sample_ids, expression_mse)
                },
            }, indent=2), encoding="utf-8")
            if all_expression_control:
                expression_control = torch.cat(all_expression_control)
                gene_rows, sample_rows, expression_summary = population_expression_metric_tables(
                    expression_truth, expression_prediction, expression_control,
                    sample_ids, splits, top_k=args.deg_top_k,
                )
                _write_rows(output_dir / "expression_gene_wise_pcc.csv", gene_rows)
                _write_rows(output_dir / "expression_sample_wise_metrics.csv", sample_rows)
                (output_dir / "expression_similarity_metrics.json").write_text(
                    json.dumps(expression_summary, indent=2), encoding="utf-8"
                )
                all_metrics = evaluate_all(
                    expression_truth, expression_prediction, expression_control,
                )
                (output_dir / "expression_all_metrics.json").write_text(
                    json.dumps(all_metrics, indent=2), encoding="utf-8"
                )
                metrics["expression_similarity"] = all_metrics
    if all_celltype_expression_prediction:
        expression_prediction = torch.cat(all_celltype_expression_prediction)
        expression_truth = torch.cat(all_celltype_expression_truth)
        expression_mask = torch.cat(all_celltype_expression_mask)
        expression_rows, expression_summary = celltype_expression_metric_tables(
            expression_truth, expression_prediction, expression_mask,
            sample_ids, classes, splits,
        )
        torch.save(expression_prediction, output_dir / "celltype_expression_prediction.pt")
        torch.save(expression_truth, output_dir / "celltype_expression_true.pt")
        torch.save(expression_mask, output_dir / "celltype_expression_valid_mask.pt")
        _write_rows(output_dir / "celltype_expression_mse.csv", expression_rows)
        metrics["celltype_expression"] = expression_summary
    metrics.update({"n_ode_steps": args.n_ode_steps, "classes": classes})
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
