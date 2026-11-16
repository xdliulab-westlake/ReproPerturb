"""Optional train-only set-BPE tokenizer for unordered drug combinations."""

from __future__ import annotations

from collections import Counter
from itertools import combinations


def fit_combo_tokenizer(
    records: list[dict], drug_names: list[str], *, min_frequency: int = 2,
    max_merges: int = 16, max_token_size: int = 4,
) -> dict:
    """Learn deterministic pair merges from training records only.

    Tokens form a partition of each drug set.  A merge is counted at most once
    per biological record and ties are resolved lexicographically, making the
    learned vocabulary reproducible.
    """
    if min_frequency < 2:
        raise ValueError("min_frequency must be at least 2")
    if max_merges < 0 or max_token_size < 2:
        raise ValueError("max_merges must be nonnegative and max_token_size >= 2")
    base_count = len(drug_names)
    token_components = {index: (index,) for index in range(base_count)}
    encoded = [sorted(set(map(int, record["drug_ids"]))) for record in records]
    merges = []
    for _ in range(max_merges):
        support = Counter()
        for tokens in encoded:
            for left, right in combinations(sorted(tokens), 2):
                merged = set(token_components[left]) | set(token_components[right])
                if len(merged) <= max_token_size:
                    support[(left, right)] += 1
        eligible = [(count, pair) for pair, count in support.items() if count >= min_frequency]
        if not eligible:
            break
        count, (left, right) = min(eligible, key=lambda item: (-item[0], item[1]))
        new_id = base_count + len(merges)
        components = tuple(sorted(set(token_components[left]) | set(token_components[right])))
        token_components[new_id] = components
        merges.append({
            "left": left, "right": right, "new_id": new_id,
            "support": count, "components": list(components),
        })
        for index, tokens in enumerate(encoded):
            if left in tokens and right in tokens:
                encoded[index] = sorted(
                    [token for token in tokens if token not in {left, right}] + [new_id]
                )
    combo_names = [
        "<COMBO:" + "+".join(drug_names[index] for index in merge["components"]) + ">"
        for merge in merges
    ]
    return {
        "type": "set_bpe", "base_drug_count": base_count,
        "min_frequency": min_frequency, "max_merges": max_merges,
        "max_token_size": max_token_size, "merges": merges,
        "base_drug_names": list(drug_names), "combo_token_names": combo_names,
    }


def encode_drug_ids(drug_ids, tokenizer: dict) -> list[int]:
    tokens = sorted(set(map(int, drug_ids)))
    for merge in tokenizer.get("merges", []):
        left, right = int(merge["left"]), int(merge["right"])
        if left in tokens and right in tokens:
            tokens = sorted(
                [token for token in tokens if token not in {left, right}]
                + [int(merge["new_id"])]
            )
    return tokens


def transform_combo_records(records: list[dict], tokenizer: dict) -> list[dict]:
    names = list(tokenizer["base_drug_names"]) + list(tokenizer["combo_token_names"])
    transformed = []
    for record in records:
        item = dict(record)
        original = list(map(int, record.get("original_drug_ids", record["drug_ids"])))
        item["original_drug_ids"] = original
        item["original_drug_names"] = list(
            record.get("original_drug_names", record.get("drug_names", []))
        )
        item["drug_ids"] = encode_drug_ids(original, tokenizer)
        item["drug_names"] = [names[index] for index in item["drug_ids"]]
        transformed.append(item)
    return transformed


def configured_combo_tokenizer(config: dict) -> dict | None:
    tokenizer = config.get("drug_combo_tokenizer")
    return tokenizer if tokenizer and tokenizer.get("enabled") and tokenizer.get("merges") else None
