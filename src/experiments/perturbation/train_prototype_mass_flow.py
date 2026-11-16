"""DDP training of pathway-conditioned flow over prototype mass logits."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

# Allow ``python experiments/perturbation/train_prototype_mass_flow.py`` from
# the project root without requiring callers to set PYTHONPATH manually.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from data.drug_combo_tokenizer import fit_combo_tokenizer, transform_combo_records
from model.perturbation.condition_io import build_configured_condition_encoder
from model.perturbation.prototype_mass_flow import PrototypeMassFlow, PrototypeMassFlowAdapter
from model.perturbation.prototype_vq import build_prototype_celltype_map_from_sample_counts


DEFAULT_FORCED_TEST = ("H226", "H227", "H240")
DEFAULT_EXTRA_EVALUATION = ("H163", "H164")
TRAIN_LOSS_NAMES = (
    "total", "base_total", "flow", "presence", "proportion", "expression",
    "unknown_effect", "unknown_gate", "aux_proportion", "aux_expression",
    "pathway_shuffle_contrastive",
    "matched_pair",
)


def split_pair_records(
    records: list[dict], *, val_fraction: float, test_fraction: float, seed: int,
    forced_test_samples: set[str], extra_evaluation_samples: set[str],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split by biological sample, never by cells; held-out predictions never leak."""
    forced = {str(sample) for sample in forced_test_samples | extra_evaluation_samples}
    by_sample = {str(record["sample_id"]): record for record in records}
    test = [record for sample, record in by_sample.items() if sample in forced]
    candidates = [record for sample, record in by_sample.items() if sample not in forced]
    random.Random(seed).shuffle(candidates)
    desired_test = max(1, round(test_fraction * len(records)))
    n_random_test = min(len(candidates), max(0, desired_test - len(test)))
    test.extend(candidates[:n_random_test])
    remaining = candidates[n_random_test:]
    n_val = max(1, round(val_fraction * len(remaining)))
    n_val = min(n_val, len(remaining) - 1)
    val, train = remaining[:n_val], remaining[n_val:]
    return train, val, test


def split_pair_records_from_manifest(
    records: list[dict], split_path: str | Path,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split records using an explicit sample-level CSV manifest.

    The manifest must contain ``sample_id`` and ``split`` columns, with split
    values in ``train``, ``val`` or ``test``.  Splitting is always by biological
    sample, so all pair records for one sample stay in the same partition.
    """
    manifest_path = Path(split_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"explicit stage-2 split manifest not found: {manifest_path}")
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"explicit stage-2 split manifest is empty: {manifest_path}")
    required = {"sample_id", "split"}
    missing_columns = required - set(rows[0])
    if missing_columns:
        raise ValueError(
            f"explicit stage-2 split manifest missing columns {sorted(missing_columns)}: "
            f"{manifest_path}"
        )
    assignments: dict[str, str] = {}
    for row in rows:
        sample_id = str(row["sample_id"] or "").strip()
        split = str(row["split"] or "").strip().lower()
        if not sample_id:
            raise ValueError(f"explicit stage-2 split manifest contains a blank sample_id: {manifest_path}")
        if split not in {"train", "val", "test"}:
            raise ValueError(
                f"invalid split {split!r} for sample {sample_id!r}; "
                "expected train, val or test"
            )
        previous = assignments.get(sample_id)
        if previous is not None and previous != split:
            raise ValueError(
                f"sample {sample_id!r} appears in multiple splits: {previous}, {split}"
            )
        assignments[sample_id] = split

    record_sample_ids = {str(record["sample_id"]) for record in records}
    missing = sorted(record_sample_ids - assignments.keys())
    if missing:
        raise ValueError(
            f"{len(missing)} pair-record sample(s) are absent from explicit split "
            f"manifest; first entries: {missing[:10]}"
        )
    extra = sorted(assignments.keys() - record_sample_ids)
    if extra:
        print(
            f"[stage2 split] ignoring {len(extra)} manifest sample(s) without pair records; "
            f"first entries: {extra[:10]}",
            flush=True,
        )

    partitions = {
        split: [record for record in records if assignments[str(record["sample_id"])] == split]
        for split in ("train", "val", "test")
    }
    if any(not partitions[split] for split in ("train", "val", "test")):
        raise ValueError(
            "explicit stage-2 split must produce non-empty train, val and test records: "
            + ", ".join(f"{split}={len(partitions[split])}" for split in ("train", "val", "test"))
        )
    print(
        f"[stage2 split] manifest={manifest_path} "
        + " ".join(f"{split}={len(partitions[split])}" for split in ("train", "val", "test")),
        flush=True,
    )
    return partitions["train"], partitions["val"], partitions["test"]


class PairDataset(Dataset):
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        return self.records[index]


class MatchedAddOneDataset(Dataset):
    """Real same-control perturbation pairs differing by exactly one drug."""

    def __init__(self, records: list[dict]):
        by_control: dict[str, list[dict]] = {}
        for record in records:
            by_control.setdefault(str(record["control_sample_id"]), []).append(record)
        self.pairs: list[tuple[dict, dict]] = []
        for group in by_control.values():
            drug_sets = [(
                record,
                frozenset(map(int, record.get("original_drug_ids", record["drug_ids"]))),
            ) for record in group]
            for smaller, smaller_set in drug_sets:
                for larger, larger_set in drug_sets:
                    if len(larger_set) == len(smaller_set) + 1 and smaller_set < larger_set:
                        self.pairs.append((smaller, larger))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[dict, dict]:
        return self.pairs[index]


def collate_pairs(records: list[dict]) -> dict[str, torch.Tensor]:
    max_drugs = max(len(record["drug_ids"]) for record in records)
    drug_ids = torch.zeros(len(records), max_drugs, dtype=torch.long)
    drug_mask = torch.zeros(len(records), max_drugs, dtype=torch.bool)
    for row, record in enumerate(records):
        n_drugs = len(record["drug_ids"])
        drug_ids[row, :n_drugs] = torch.as_tensor(record["drug_ids"], dtype=torch.long)
        drug_mask[row, :n_drugs] = True
    keys = (
        "initial_mass_logit", "terminal_mass_logit", "initial_presence",
        "terminal_presence", "terminal_proportion", "terminal_celltype_proportion",
    )
    batch = {key: torch.stack([torch.as_tensor(record[key]).float() for record in records]) for key in keys}
    batch["drug_ids"] = drug_ids
    batch["drug_mask"] = drug_mask
    if all("terminal_expression" in record for record in records):
        batch["terminal_expression"] = torch.stack([
            torch.as_tensor(record["terminal_expression"]).float()
            for record in records
        ])
    auxiliary_keys = (
        "initial_celltype_proportion",
        "initial_celltype_expression",
        "terminal_celltype_expression",
        "initial_celltype_expression_mask",
        "terminal_celltype_expression_mask",
    )
    for key in auxiliary_keys:
        present = [key in record for record in records]
        if any(present) and not all(present):
            raise ValueError(f"{key} must be present for every record in a batch")
        if all(present):
            values = [torch.as_tensor(record[key]) for record in records]
            batch[key] = torch.stack(values)
    return batch


def collate_matched_add_one(
    pairs: list[tuple[dict, dict]],
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        "smaller": collate_pairs([pair[0] for pair in pairs]),
        "larger": collate_pairs([pair[1] for pair in pairs]),
    }


class ConditionalPrototypeMassFlow(nn.Module):
    """The condition encoder and prototype flow are optimized as one DDP model."""

    def __init__(
        self,
        condition_encoder: nn.Module,
        adapter: PrototypeMassFlowAdapter,
        unknown_effect_l1_weight: float = 0.0,
        unknown_gate_l1_weight: float = 0.0,
        auxiliary_proportion_weight: float = 0.0,
        auxiliary_expression_weight: float = 0.0,
        pathway_shuffle_contrastive_weight: float = 0.0,
        matched_pair_weight: float = 0.0,
        matched_pair_hypoblast_weight: float = 0.0,
        hypoblast_celltype_index: int | None = None,
    ):
        super().__init__()
        self.condition_encoder = condition_encoder
        self.adapter = adapter
        self.unknown_effect_l1_weight = float(unknown_effect_l1_weight)
        self.unknown_gate_l1_weight = float(unknown_gate_l1_weight)
        self.auxiliary_proportion_weight = float(auxiliary_proportion_weight)
        self.auxiliary_expression_weight = float(auxiliary_expression_weight)
        self.pathway_shuffle_contrastive_weight = float(
            pathway_shuffle_contrastive_weight
        )
        self.matched_pair_weight = float(matched_pair_weight)
        self.matched_pair_hypoblast_weight = float(matched_pair_hypoblast_weight)
        self.hypoblast_celltype_index = hypoblast_celltype_index

    def predict_celltype_proportion(
        self, batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        encoding = self.condition_encoder(batch["drug_ids"], batch["drug_mask"])
        mass, terminal = self.adapter.rollout_endpoint(
            batch["initial_mass_logit"], batch["initial_presence"],
            encoding.condition_tokens, encoding.pathway_context,
        )
        proportion = self.adapter.velocity_model.presence_gated_proportion(
            mass, torch.sigmoid(terminal.presence_logits),
        )
        mapping = self.adapter.prototype_to_celltype
        return proportion if mapping is None else proportion @ mapping.to(proportion)

    def matched_pair_loss(
        self, matched_pair_batch: dict[str, dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        smaller, larger = matched_pair_batch["smaller"], matched_pair_batch["larger"]
        predicted_delta = (
            self.predict_celltype_proportion(larger)
            - self.predict_celltype_proportion(smaller)
        )
        observed_delta = (
            larger["terminal_celltype_proportion"]
            - smaller["terminal_celltype_proportion"]
        ).to(predicted_delta)
        per_celltype = torch.nn.functional.smooth_l1_loss(
            predicted_delta, observed_delta, reduction="none",
        )
        loss = per_celltype.mean()
        if self.hypoblast_celltype_index is not None and self.matched_pair_hypoblast_weight:
            loss = loss + self.matched_pair_hypoblast_weight * per_celltype[
                :, self.hypoblast_celltype_index
            ].mean()
        return loss

    def loss_terms(
        self, batch: dict[str, torch.Tensor],
        matched_pair_batch: dict[str, dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, torch.Tensor]:
        encoding = self.condition_encoder(batch["drug_ids"], batch["drug_mask"])
        tokens = encoding.condition_tokens
        terms = self.adapter.loss_terms(
            batch["initial_mass_logit"], batch["terminal_mass_logit"],
            batch["initial_presence"], batch["terminal_presence"],
            batch["terminal_proportion"], batch["terminal_celltype_proportion"], tokens,
            pathway_context=encoding.pathway_context,
            terminal_expression=batch.get("terminal_expression"),
        )
        regularizers = self.condition_encoder.unknown_regularization_terms()
        base_total = terms["total"]
        terms["base_total"] = base_total
        terms["unknown_effect"] = regularizers["unknown_effect"]
        terms["unknown_gate"] = regularizers["unknown_gate"]
        zero = base_total.new_zeros(())
        auxiliary = (
            self.condition_encoder.auxiliary_loss_terms(encoding, batch)
            if hasattr(self.condition_encoder, "auxiliary_loss_terms")
            else {
                "aux_proportion": zero,
                "aux_expression": zero,
                "pathway_shuffle_contrastive": zero,
            }
        )
        terms.update(auxiliary)
        terms["matched_pair"] = (
            self.matched_pair_loss(matched_pair_batch)
            if matched_pair_batch is not None else zero
        )
        terms["total"] = (
            base_total
            + self.unknown_effect_l1_weight * terms["unknown_effect"]
            + self.unknown_gate_l1_weight * terms["unknown_gate"]
            + self.auxiliary_proportion_weight * terms["aux_proportion"]
            + self.auxiliary_expression_weight * terms["aux_expression"]
            + self.pathway_shuffle_contrastive_weight
            * terms["pathway_shuffle_contrastive"]
            + self.matched_pair_weight * terms["matched_pair"]
        )
        return terms

    def forward(
        self, batch: dict[str, torch.Tensor],
        matched_pair_batch: dict[str, dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.loss_terms(batch, matched_pair_batch)


def checkpoint_score(terms: dict[str, float]) -> float:
    """Select checkpoints with the unchanged predictive validation objective."""
    return float(terms["base_total"])


def build_optimizer_parameter_groups(
    module: nn.Module, training_config: dict
) -> list[dict]:
    """Assign bottleneck parameters a dedicated LR without duplicates."""
    base_lr = float(training_config["lr"])
    pathway_lr = float(training_config.get("pathway_celltype_lr", base_lr))
    weight_decay = float(training_config.get("weight_decay", 0.0))
    encoder = getattr(module, "condition_encoder", None)
    special: list[nn.Parameter] = []
    if encoder is not None and hasattr(encoder, "pathway_celltype_weight"):
        special.append(encoder.pathway_celltype_weight)
        special.extend(encoder.expression_projection.parameters())
    special_ids = {id(parameter) for parameter in special}
    base = [
        parameter
        for parameter in module.parameters()
        if parameter.requires_grad and id(parameter) not in special_ids
    ]
    special = [parameter for parameter in special if parameter.requires_grad]
    groups = []
    if base:
        groups.append({
            "name": "base", "params": base, "lr": base_lr,
            "weight_decay": weight_decay,
        })
    if special:
        groups.append({
            "name": "pathway_celltype", "params": special, "lr": pathway_lr,
            "weight_decay": weight_decay,
        })
    parameter_ids = [id(parameter) for group in groups for parameter in group["params"]]
    if len(parameter_ids) != len(set(parameter_ids)):
        raise ValueError("optimizer parameter groups contain duplicate parameters")
    return groups


def synchronize_optimizer_learning_rates(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    """Copy rank-zero scheduler learning rates to every DDP optimizer."""
    learning_rates = torch.tensor(
        [group["lr"] for group in optimizer.param_groups],
        dtype=torch.float64,
        device=device,
    )
    dist.broadcast(learning_rates, src=0)
    for group, learning_rate in zip(optimizer.param_groups, learning_rates.tolist()):
        group["lr"] = learning_rate


def _load_config(path: str) -> dict:
    config_path = Path(path)
    spec = importlib.util.spec_from_file_location("prototype_mass_config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module.CONFIG)


def validate_prototype_selection_strategy(config: dict, payload: dict) -> str:
    """Refuse benchmark configs paired with a different prototype vocabulary."""
    expected = str(
        config.get("prototype", {}).get("selection_strategy", "global_frequency")
    )
    actual_value = payload.get("prototype_selection_strategy")
    actual = "global_frequency" if actual_value is None else str(actual_value)
    if expected != actual:
        raise ValueError(
            f"config requests prototype selection '{expected}', but pair records "
            f"contain '{actual}'; rebuild the prototype pipeline with the same config"
        )
    return actual


def _mean_terms(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    totals: dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            terms = model.loss_terms(batch)
            batch_size = len(batch["drug_ids"])
            for key, value in terms.items():
                totals[key] = totals.get(key, 0.0) + value.item() * batch_size
            count += batch_size
    return {key: value / max(count, 1) for key, value in totals.items()}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--pair-records", required=True)
    parser.add_argument("--pathway-base")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--val-fraction", type=float)
    parser.add_argument("--test-fraction", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-dir")
    parser.add_argument("--forced-test-samples", nargs="*")
    parser.add_argument("--extra-evaluation-samples", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if "RANK" not in os.environ:
        raise RuntimeError("prototype mass-flow training must be launched with torchrun DDP")
    rank, local_rank, world_size = (int(os.environ[key]) for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    config = _load_config(args.config)
    train_config, flow_config, pathway_config = config["training"], config["flow"], config["pathway"]
    epochs = args.epochs or train_config["epochs"]
    batch_size = args.batch_size or train_config["batch_size"]
    lr = args.lr or train_config["lr"]
    weight_decay = args.weight_decay if args.weight_decay is not None else train_config.get("weight_decay", 0.0)
    patience = args.patience or train_config["early_stop_patience"]
    val_fraction = args.val_fraction if args.val_fraction is not None else train_config["val_fraction"]
    test_fraction = args.test_fraction if args.test_fraction is not None else train_config["test_fraction"]

    forced_test_samples = args.forced_test_samples
    if forced_test_samples is None:
        forced_test_samples = train_config.get("forced_test_samples", DEFAULT_FORCED_TEST)
    extra_evaluation_samples = args.extra_evaluation_samples
    if extra_evaluation_samples is None:
        extra_evaluation_samples = train_config.get("extra_evaluation_samples", DEFAULT_EXTRA_EVALUATION)
    payload = torch.load(args.pair_records, map_location="cpu", weights_only=True)
    validate_prototype_selection_strategy(config, payload)
    explicit_split_path = train_config.get("split_path")
    if explicit_split_path:
        train_records, val_records, test_records = split_pair_records_from_manifest(
            payload["records"], explicit_split_path,
        )
    else:
        train_records, val_records, test_records = split_pair_records(
            payload["records"], val_fraction=val_fraction, test_fraction=test_fraction, seed=args.seed,
            forced_test_samples=set(forced_test_samples),
            extra_evaluation_samples=set(extra_evaluation_samples),
        )
    drug_vocabulary = list(payload.get("drug_names") or [])
    if not drug_vocabulary:
        raise ValueError("pair records must include drug_names for condition encoder construction")
    combo_config = config.get("drug_combo_tokenizer", {})
    combo_tokenizer = None
    if combo_config.get("enabled", False):
        combo_tokenizer = fit_combo_tokenizer(
            train_records, drug_vocabulary,
            min_frequency=int(combo_config.get("min_frequency", 2)),
            max_merges=int(combo_config.get("max_merges", 16)),
            max_token_size=int(combo_config.get("max_token_size", 4)),
        )
        combo_tokenizer["enabled"] = True
        train_records = transform_combo_records(train_records, combo_tokenizer)
        val_records = transform_combo_records(val_records, combo_tokenizer)
        test_records = transform_combo_records(test_records, combo_tokenizer)
        drug_vocabulary = (
            combo_tokenizer["base_drug_names"]
            + combo_tokenizer["combo_token_names"]
        )
        config["drug_combo_tokenizer"] = combo_tokenizer
        pathway_config["combo_token_components"] = [
            merge["components"] for merge in combo_tokenizer["merges"]
        ]
    expression_weight = float(train_config.get("expression_loss_weight", 0.0))
    sample_counts = payload.get("prototype_celltype_counts_by_sample", {})
    prototype_to_celltype = (
        build_prototype_celltype_map_from_sample_counts(
            sample_counts, [str(record["sample_id"]) for record in train_records],
        )
        if sample_counts else payload["prototype_to_celltype"]
    )
    condition_encoder, drug_names, _, pathway_names = build_configured_condition_encoder(
        pathway_config,
        list(drug_vocabulary),
        args.pathway_base,
        prototype_to_celltype=prototype_to_celltype,
        expression_dim=(
            int(train_records[0]["terminal_celltype_expression"].shape[-1])
            if pathway_config.get("celltype_bottleneck", {}).get("enabled", False)
            else None
        ),
    )
    flow = PrototypeMassFlow(
        payload["prototype_codebook"], condition_dim=condition_encoder.condition_token_dim,
        model_dim=flow_config.get("token_model_dim", 256),
        time_dim=flow_config.get("time_freqs", 1024),
        num_layers=flow_config.get("num_token_layers", 3),
        ff_dim=flow_config.get("token_ff_dim", 4096), nhead=flow_config.get("token_nhead", 8),
        use_pathway_prototype_adaln=flow_config.get("pathway_prototype_adaln", False),
        pathway_prototype_injection_after_block=flow_config.get(
            "pathway_prototype_injection_after_block", 0,
        ),
        pathway_adaln_mode=flow_config.get("pathway_adaln_mode"),
        pathway_adaln_blocks=tuple(flow_config.get("pathway_adaln_blocks", ())),
    )
    stage1_boundary = None
    if expression_weight:
        from experiments.perturbation.stage1_boundary import load_frozen_stage1_boundary
        stage1_cfg = config.get("stage1", {})
        stage1_boundary = load_frozen_stage1_boundary(
            stage1_cfg["config_module"], stage1_cfg["checkpoint"], device,
        )
    module = ConditionalPrototypeMassFlow(
        condition_encoder,
        PrototypeMassFlowAdapter(
            flow, prototype_to_celltype=prototype_to_celltype,
            stage1_boundary=stage1_boundary,
            presence_weight=train_config.get("presence_loss_weight", 1.0),
            proportion_weight=train_config.get("proportion_loss_weight", 1.0),
            expression_weight=expression_weight,
            random_flow_weight=train_config.get("random_flow_loss_weight", 0.25),
            source_flow_weight=train_config.get("source_flow_loss_weight", 1.0),
            endpoint_rollout_steps=flow_config.get("endpoint_rollout_steps", 20),
        ),
        unknown_effect_l1_weight=train_config.get("unknown_effect_l1_weight", 0.0),
        unknown_gate_l1_weight=train_config.get("unknown_gate_l1_weight", 0.0),
        auxiliary_proportion_weight=train_config.get(
            "auxiliary_proportion_weight", 0.0
        ),
        auxiliary_expression_weight=train_config.get(
            "auxiliary_expression_weight", 0.0
        ),
        pathway_shuffle_contrastive_weight=train_config.get(
            "pathway_shuffle_contrastive_weight", 0.0
        ),
        matched_pair_weight=train_config.get("matched_pair_weight", 0.0),
        matched_pair_hypoblast_weight=train_config.get(
            "matched_pair_hypoblast_weight", 0.0
        ),
        hypoblast_celltype_index=next(
            (
                index for index, name in enumerate(payload["celltype_classes"])
                if "hypoblast" in str(name).lower()
            ),
            None,
        ),
    ).to(device)
    # state_delta is reserved for the optional frozen-decoder expression loss,
    # so it is intentionally unused in the composition-only objective.
    model = DDP(module, device_ids=[local_rank], find_unused_parameters=True)
    optimizer_config = dict(train_config, lr=lr, weight_decay=weight_decay)
    optimizer = torch.optim.AdamW(
        build_optimizer_parameter_groups(module, optimizer_config)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=train_config.get("scheduler_factor", 0.5),
        patience=train_config.get("scheduler_patience", 10),
    )
    train_set, val_set = PairDataset(train_records), PairDataset(val_records)
    sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate_pairs)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, collate_fn=collate_pairs)
    matched_set = MatchedAddOneDataset(train_records)
    matched_sampler = None
    matched_loader = None
    if train_config.get("matched_pair_weight", 0.0):
        if not len(matched_set):
            raise ValueError("matched_pair_weight is nonzero but no train add-one pairs exist")
        matched_sampler = DistributedSampler(
            matched_set, num_replicas=world_size, rank=rank, shuffle=True,
            seed=args.seed + 100003,
        )
        matched_loader = DataLoader(
            matched_set,
            batch_size=int(train_config.get("matched_pair_batch_size", batch_size)),
            sampler=matched_sampler, num_workers=args.num_workers, pin_memory=True,
            collate_fn=collate_matched_add_one,
        )
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(args.log_dir or str(output_dir / "tensorboard"))
        (output_dir / "split_manifest.json").write_text(json.dumps({
            "train_samples": [record["sample_id"] for record in train_records],
            "val_samples": [record["sample_id"] for record in val_records],
            "test_samples": [record["sample_id"] for record in test_records],
            "explicit_split_path": str(explicit_split_path) if explicit_split_path else None,
            "forced_test_samples": sorted(set(forced_test_samples)),
            "extra_evaluation_samples": sorted(set(extra_evaluation_samples)),
        }, indent=2), encoding="utf-8")
        if combo_tokenizer is not None:
            (output_dir / "drug_combo_tokenizer.json").write_text(
                json.dumps(combo_tokenizer, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        print(f"DDP world_size={world_size} device=cuda:{local_rank} train={len(train_set)} val={len(val_set)} test={len(test_records)} matched_add_one={len(matched_set)}", flush=True)
    else:
        writer = None
    best, stale = float("inf"), 0
    for epoch in range(1, epochs + 1):
        sampler.set_epoch(epoch)
        if matched_sampler is not None:
            matched_sampler.set_epoch(epoch)
        model.train()
        sums = torch.zeros(len(TRAIN_LOSS_NAMES), device=device)
        matched_iterator = iter(matched_loader) if matched_loader is not None else None
        for batch in train_loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            matched_batch = None
            if matched_iterator is not None:
                try:
                    matched_batch = next(matched_iterator)
                except StopIteration:
                    matched_iterator = iter(matched_loader)
                    matched_batch = next(matched_iterator)
                matched_batch = {
                    side: {
                        key: value.to(device, non_blocking=True)
                        for key, value in side_batch.items()
                    }
                    for side, side_batch in matched_batch.items()
                }
            terms = model(batch, matched_batch)
            optimizer.zero_grad(set_to_none=True)
            terms["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            sums += torch.stack([terms[name].detach() for name in TRAIN_LOSS_NAMES])
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        denom = torch.tensor(float(len(train_loader) * world_size), device=device).clamp_min(1.0)
        train_terms = (sums / denom).tolist()
        if rank == 0:
            model.eval()
            val_terms = _mean_terms(model.module, val_loader, device)
            scheduler.step(checkpoint_score(val_terms))
            for name, value in zip(TRAIN_LOSS_NAMES, train_terms):
                writer.add_scalar(f"loss/train_{name}", value, epoch)
            for name, value in val_terms.items():
                writer.add_scalar(f"loss/val_{name}", value, epoch)
            writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)
            checkpoint = {
                "epoch": epoch, "model": model.module.state_dict(), "optimizer": optimizer.state_dict(),
                "config": config, "metrics": {"train_total": train_terms[0], **{f"val_{key}": value for key, value in val_terms.items()}},
                "prototype_codebook": payload["prototype_codebook"],
                "prototype_to_celltype": prototype_to_celltype,
                "prototype_selection_strategy": payload.get("prototype_selection_strategy"),
                "prototype_selection_metadata": payload.get("prototype_selection_metadata"),
                "prototype_selection_config": payload.get("prototype_selection_config"),
                "celltype_classes": payload["celltype_classes"], "drug_names": drug_names,
                "pathway_names": pathway_names,
                "selection_metric": "val_base_total",
            }
            torch.save(checkpoint, output_dir / "last_prototype_mass_flow.pt")
            if epoch in set(map(int, train_config.get("snapshot_epochs", ()))):
                torch.save(
                    checkpoint,
                    output_dir / f"snapshot_epoch{epoch}.pt",
                )
            score = checkpoint_score(val_terms)
            if score < best:
                best, stale = score, 0
                torch.save(checkpoint, output_dir / "best_prototype_mass_flow.pt")
            else:
                stale += 1
            print(f"epoch={epoch} train_loss={train_terms[0]:.6f} val_base_loss={score:.6f} lr={optimizer.param_groups[0]['lr']:.3e}", flush=True)
        synchronize_optimizer_learning_rates(optimizer, device)
        stop = torch.tensor([stale >= patience if rank == 0 else False], device=device)
        dist.broadcast(stop, src=0)
        if stop.item():
            break
    if writer is not None:
        writer.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
