"""Load and batch pathway-conditioned perturbation inputs."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
import yaml
from torch import Tensor

from data.perturbation_schema import read_sign_csv
from model.perturbation.pathway_condition import SignedInteractionCondition
from model.perturbation.pathway_celltype_condition import PathwayCelltypeCondition
from model.perturbation.pathway_prototype_condition import PathwayPrototypeCondition


def load_pathway_names(pathways_yaml: str | Path) -> list[str]:
    with Path(pathways_yaml).open(encoding="utf-8") as handle:
        pathways = yaml.safe_load(handle).get("pathways", {})
    if not pathways:
        raise ValueError("pathways file contains no pathway gene sets")
    return [str(name) for name in pathways]


def load_ordered_pathway_base(
    pathway_base_path: str | Path,
    pathways_yaml: str | Path,
) -> tuple[list[str], Tensor]:
    names = load_pathway_names(pathways_yaml)
    pooled = torch.load(pathway_base_path, map_location="cpu", weights_only=True)
    missing = [name for name in names if name not in pooled]
    if missing:
        raise ValueError(f"pathway base is missing: {missing}")
    base = torch.stack([torch.as_tensor(pooled[name]).float() for name in names])
    if base.ndim != 2:
        raise ValueError(f"pathway base must be [P,D], got {tuple(base.shape)}")
    return names, base


def load_drug_names(sign_csv: str | Path) -> list[str]:
    with Path(sign_csv).open(newline="", encoding="utf-8") as handle:
        return list(dict.fromkeys(str(row["drug"]) for row in csv.DictReader(handle)))


def build_condition_encoder(
    pathway_base_path: str | Path,
    pathways_yaml: str | Path,
    sign_csv: str | Path,
    drugs: list[str] | None = None,
    pathway_specific_magnitude: bool = False,
    interaction_hidden_dim: int = 32,
    interaction_scale: float = 0.5,
    use_drug_embedding: bool = False,
    constrain_sign: bool = True,
    learn_unknown_pathway_effects: bool = False,
    known_effect_initialization: float = 1.0,
    unknown_effect_mode: str = "none",
    unknown_effect_rank: int = 4,
    pair_interaction_rank: int | None = None,
    triple_interaction_rank: int | None = None,
    higher_order_scale: float = 1.25,
    combo_token_components: list[list[int]] | None = None,
) -> tuple[SignedInteractionCondition, list[str], dict[str, int], list[str]]:
    pathway_names, pathway_base = load_ordered_pathway_base(
        pathway_base_path, pathways_yaml
    )
    drug_names = drugs or load_drug_names(sign_csv)
    combo_token_components = combo_token_components or []
    base_count = len(drug_names) - len(combo_token_components)
    if base_count < 1:
        raise ValueError("combo token metadata is inconsistent with drug vocabulary")
    base_sign_matrix = read_sign_csv(sign_csv, drug_names[:base_count], pathway_names)
    combo_rows = [
        base_sign_matrix[torch.as_tensor(components, dtype=torch.long)].sum(dim=0)
        for components in combo_token_components
    ]
    sign_matrix = (
        torch.cat((base_sign_matrix, torch.stack(combo_rows)), dim=0)
        if combo_rows else base_sign_matrix
    )
    token_sizes = torch.tensor(
        [1] * base_count + [len(components) for components in combo_token_components],
        dtype=torch.float32,
    )
    encoder = SignedInteractionCondition(
        pathway_base, sign_matrix,
        pathway_specific_magnitude=pathway_specific_magnitude,
        interaction_hidden_dim=interaction_hidden_dim,
        interaction_scale=interaction_scale,
        use_drug_embedding=use_drug_embedding,
        constrain_sign=constrain_sign,
        learn_unknown_pathway_effects=learn_unknown_pathway_effects,
        known_effect_initialization=known_effect_initialization,
        unknown_effect_mode=unknown_effect_mode,
        unknown_effect_rank=unknown_effect_rank,
        pair_interaction_rank=pair_interaction_rank,
        triple_interaction_rank=triple_interaction_rank,
        higher_order_scale=higher_order_scale,
        token_sizes=token_sizes,
    )
    return encoder, drug_names, {name: i for i, name in enumerate(drug_names)}, pathway_names


def build_configured_condition_encoder(
    pathway_config: dict,
    drugs: list[str],
    pathway_base_path: str | Path | None,
    *,
    prototype_to_celltype: Tensor | None = None,
    expression_dim: int | None = None,
) -> tuple[SignedInteractionCondition | PathwayPrototypeCondition | PathwayCelltypeCondition, list[str], dict[str, int], list[str]]:
    """Build the configured legacy or pathway-to-prototype conditioner."""
    if pathway_base_path is None:
        raise ValueError("--pathway-base is required to load pathway gene embeddings")
    conditioner = pathway_config.get("conditioner", "legacy_pathway_token")
    if conditioner == "legacy_pathway_token":
        built = build_condition_encoder(
            pathway_base_path, pathway_config["pathways_yaml"], pathway_config["sign_csv"], drugs,
            pathway_specific_magnitude=pathway_config.get("pathway_specific_magnitude", False),
            interaction_hidden_dim=pathway_config.get("interaction_hidden_dim", 32),
            interaction_scale=pathway_config.get("interaction_scale", 0.5),
            use_drug_embedding=pathway_config.get("use_drug_embedding", False),
            constrain_sign=pathway_config.get("constrain_sign", True),
            learn_unknown_pathway_effects=pathway_config.get("learn_unknown_pathway_effects", False),
            known_effect_initialization=pathway_config.get("known_effect_initialization", 1.0),
            unknown_effect_mode=pathway_config.get("unknown_effect_mode", "none"),
            unknown_effect_rank=pathway_config.get("unknown_effect_rank", 4),
            pair_interaction_rank=pathway_config.get("pair_interaction_rank"),
            triple_interaction_rank=pathway_config.get("triple_interaction_rank"),
            higher_order_scale=pathway_config.get("higher_order_scale", 1.25),
            combo_token_components=pathway_config.get("combo_token_components"),
        )
        bottleneck = pathway_config.get("celltype_bottleneck", {})
        if not bottleneck.get("enabled", False):
            return built
        if prototype_to_celltype is None or expression_dim is None:
            raise ValueError(
                "celltype bottleneck requires prototype_to_celltype and expression_dim"
            )
        encoder, drug_names, drug_to_index, pathway_names = built
        wrapped = PathwayCelltypeCondition(
            encoder,
            prototype_to_celltype,
            expression_dim,
            dynamic_token_route=bottleneck.get("dynamic_token_route", "canonical"),
            route_dropout_probability=bottleneck.get(
                "route_dropout_probability", 0.0
            ),
            adaln_context_dim=bottleneck.get("adaln_context_dim", 256),
            shuffle_contrastive=bottleneck.get("shuffle_contrastive", False),
            shuffle_margin_fraction=bottleneck.get("shuffle_margin_fraction", 0.05),
            combination_rank=bottleneck.get("combination_rank"),
            combination_scale=bottleneck.get("combination_scale", 0.5),
        )
        return wrapped, drug_names, drug_to_index, pathway_names
    if conditioner != "pathway_prototype":
        raise ValueError(f"unsupported pathway conditioner: {conditioner!r}")

    pathway_names, pathway_base = load_ordered_pathway_base(
        pathway_base_path, pathway_config["pathways_yaml"],
    )
    configured_count = pathway_config.get("num_pathways")
    if configured_count is not None and int(configured_count) != len(pathway_names):
        raise ValueError("configured num_pathways does not match pathways YAML")
    mode = pathway_config["mode"]
    sign_matrix = None
    if mode != "direct":
        sign_csv = pathway_config.get("sign_csv")
        if not sign_csv:
            raise ValueError(f"{mode} requires pathway.sign_csv")
        sign_matrix = read_sign_csv(sign_csv, drugs, pathway_names)
    encoder = PathwayPrototypeCondition(
        pathway_base, len(drugs), pathway_config.get("num_prototypes", 128), mode,
        sign_matrix=sign_matrix,
        unknown_effect_rank=pathway_config.get("unknown_effect_rank", 4),
    )
    return encoder, drugs, {name: i for i, name in enumerate(drugs)}, pathway_names
