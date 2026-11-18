"""Response features and the serialized training example schema."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch.nn import functional as F

@dataclass
class Example:
    sample_id: str
    target_celltype: str
    feature: torch.Tensor
    label: torch.Tensor
    truth: dict[str, str]
    confidence: float
    curated: bool

def make_feature(module, record, target_index, pathway_names, mapping, strength):
    device = mapping.device
    ids = torch.as_tensor(record['drug_ids']).long().view(1, -1).to(device)
    mask = torch.ones_like(ids, dtype=torch.bool)
    base_encoding = module.condition_encoder(ids, mask)
    base_activity = base_encoding.activities['selected']
    source_mass = torch.as_tensor(record['initial_mass_logit']).float().view(1, -1).to(device)
    source_presence = torch.as_tensor(record['initial_presence']).float().view(1, -1).to(device)
    (base_mass, base_endpoint) = module.adapter.rollout_endpoint(source_mass, source_presence, base_encoding.condition_tokens, base_encoding.pathway_context)
    base_prop = module.adapter.velocity_model.presence_gated_proportion(base_mass, torch.sigmoid(base_endpoint.presence_logits)) @ mapping
    baseline = base_prop[0, target_index]
    gains = torch.empty(len(pathway_names), 2, device=device)
    for (direction_index, direction_value) in enumerate((1.0, -1.0)):
        activities = base_activity.repeat(len(pathway_names), 1)
        index = torch.arange(len(pathway_names), device=device)
        activities[index, index] += direction_value * strength
        encoding = module.condition_encoder.encode_activities({'selected': activities})
        (mass, endpoint) = module.adapter.rollout_endpoint(source_mass.repeat(len(pathway_names), 1), source_presence.repeat(len(pathway_names), 1), encoding.condition_tokens, encoding.pathway_context)
        prop = module.adapter.velocity_model.presence_gated_proportion(mass, torch.sigmoid(endpoint.presence_logits)) @ mapping
        gains[:, direction_index] = prop[:, target_index] - baseline
    signed_gain = gains.reshape(-1)
    signed_gain = (signed_gain - signed_gain.mean()) / signed_gain.std().clamp_min(1e-06)
    initial_prop = torch.as_tensor(record['initial_celltype_proportion']).float().to(device)
    target_onehot = F.one_hot(torch.tensor(target_index, device=device), num_classes=mapping.shape[1]).float()
    feature = torch.cat([signed_gain, base_activity[0], initial_prop, target_onehot])
    return feature.cpu()
