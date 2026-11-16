import json
import math
import os
from pathlib import Path
import tempfile

# Some server images ship TensorBoard generated ``*_pb2.py`` files that are
# incompatible with the installed protobuf runtime (the
# ``Descriptors cannot not be created directly`` failure).  Select protobuf's
# compatibility parser before importing ``torch.utils.tensorboard`` below.
# Users can override this explicitly when a newer, regenerated TensorBoard is
# installed.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import torch
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np

from stage1.registry import build_vqvae, build_prior
from stage1.evaluation.losses import reconstruction_loss, prior_nll_loss
from stage1.evaluation.validation import validate_vqvae, validate_prior
from stage1.evaluation.visualization import save_reconstruction_grid, save_generation_grid


def setup_ddp() -> int:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def teardown_ddp() -> None:
    dist.destroy_process_group()


def sync_should_stop(local_should_stop: bool, device: torch.device) -> bool:
    stop = torch.tensor([int(local_should_stop)], device=device)
    dist.broadcast(stop, src=0)
    return bool(stop.item())


def compute_celltype_class_weights(
    class_counts: torch.Tensor,
    strategy: str = "none",
    max_class_weight: float = 5.0,
) -> torch.Tensor:
    """Return mean-one CE weights derived from training-split class counts."""
    counts = torch.as_tensor(class_counts, dtype=torch.float32)
    if counts.ndim != 1 or counts.numel() == 0:
        raise ValueError("class_counts must be a non-empty one-dimensional tensor")
    if strategy not in {"none", "inverse_sqrt_frequency"}:
        raise ValueError(
            "celltype_classifier.class_weighting must be 'none' or "
            "'inverse_sqrt_frequency'"
        )
    if not math.isfinite(max_class_weight) or max_class_weight < 1.0:
        raise ValueError("celltype_classifier.max_class_weight must be finite and >= 1")
    if strategy == "none":
        return torch.ones_like(counts)
    if (counts <= 0).any():
        missing = torch.where(counts <= 0)[0].tolist()
        raise ValueError(f"cell types with zero training cells: {missing}")

    raw = counts.rsqrt()
    # Find scale s such that mean(min(s * raw, cap)) == 1.  This preserves
    # inverse-sqrt ratios for uncapped classes and keeps the final cap exact.
    low, high = 0.0, max_class_weight / float(raw.min())
    for _ in range(64):
        scale = 0.5 * (low + high)
        mean = torch.clamp(raw * scale, max=max_class_weight).mean().item()
        if mean < 1.0:
            low = scale
        else:
            high = scale
    return torch.clamp(raw * (0.5 * (low + high)), max=max_class_weight)


def _training_celltype_counts(dataset, train_idx, num_classes: int) -> torch.Tensor:
    if getattr(dataset, "celltypes", None) is None:
        raise ValueError("celltype class weights require dataset celltype labels")
    labels = torch.tensor(
        [dataset.celltype_to_index[dataset.celltypes[int(index)]] for index in train_idx],
        dtype=torch.long,
    )
    return torch.bincount(labels, minlength=num_classes)


def celltype_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    weights = class_weights.to(device=logits.device, dtype=logits.dtype) if class_weights is not None else None
    return F.cross_entropy(logits, labels, weight=weights)


def _state_dict_from_checkpoint(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError("checkpoint does not contain a non-empty state dictionary")
    if all(str(key).startswith("module.") for key in checkpoint):
        checkpoint = {str(key)[7:]: value for key, value in checkpoint.items()}
    return checkpoint


def load_vqvae_initial_checkpoint(model, checkpoint_path, device):
    checkpoint_path = str(checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Stage-1 init checkpoint not found: {checkpoint_path}")
    payload = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    state_dict = _state_dict_from_checkpoint(payload)
    model.load_state_dict(state_dict, strict=True)
    return {"checkpoint": checkpoint_path, "loaded_keys": len(state_dict)}


def build_vqvae_finetune_optimizer(model, train_cfg):
    lrs = train_cfg.get("parameter_lrs")
    if lrs is None:
        return torch.optim.Adam(
            model.parameters(),
            lr=train_cfg["lr"],
            amsgrad=True,
        )

    required = {"encoder", "pre_quantization", "quantizer", "decoder"}
    if getattr(model, "celltype_classifier", None) is not None:
        required.add("celltype_classifier")
    if getattr(model, "contrastive_projector", None) is not None:
        required.add("contrastive_projector")
    if not isinstance(lrs, dict) or set(lrs) != required:
        raise ValueError(f"parameter_lrs must contain exactly {sorted(required)}")

    projection = list(model.pre_quantization_linear.parameters())
    projection += list(model.quant_proj.parameters())
    projection += list(model.post_quant_proj.parameters())
    groups = [
        {
            "name": "encoder",
            "params": list(model.encoder.parameters()),
            "lr": float(lrs["encoder"]),
        },
        {
            "name": "pre_quantization",
            "params": projection,
            "lr": float(lrs["pre_quantization"]),
        },
        {
            "name": "quantizer",
            "params": list(model.quantizer.parameters()),
            "lr": float(lrs["quantizer"]),
        },
        {
            "name": "decoder",
            "params": list(model.decoder.parameters()),
            "lr": float(lrs["decoder"]),
        },
    ]
    if model.celltype_classifier is not None:
        groups.append(
            {
                "name": "celltype_classifier",
                "params": list(model.celltype_classifier.parameters()),
                "lr": float(lrs["celltype_classifier"]),
            }
        )
    if model.contrastive_projector is not None:
        groups.append(
            {
                "name": "contrastive_projector",
                "params": list(model.contrastive_projector.parameters()),
                "lr": float(lrs["contrastive_projector"]),
            }
        )
    parameter_ids = [
        id(parameter)
        for group in groups
        for parameter in group["params"]
    ]
    model_parameter_ids = {id(parameter) for parameter in model.parameters()}
    if (
        len(parameter_ids) != len(set(parameter_ids))
        or set(parameter_ids) != model_parameter_ids
    ):
        raise ValueError("Stage-1 optimizer groups overlap or omit parameters")
    if any(
        not math.isfinite(group["lr"]) or group["lr"] < 0
        for group in groups
    ):
        raise ValueError("Stage-1 learning rates must be finite and non-negative")
    return torch.optim.Adam(groups, amsgrad=True)


def should_run_codebook_init(cfg):
    codebook_cfg = cfg.get("codebook_init", {})
    return (
        not cfg.get("training", {}).get("init_checkpoint")
        and bool(codebook_cfg.get("npy_path"))
        and bool(codebook_cfg.get("meta_csv"))
    )


def log_vqvae_optimizer_lrs(writer, optimizer, step):
    for group in optimizer.param_groups:
        name = group.get("name", "all_parameters")
        writer.add_scalar(f"train/lr/{name}", float(group["lr"]), step)


def write_vqvae_finetune_metadata(checkpoint_path, init_info, optimizer):
    metadata_path = Path(f"{checkpoint_path}.metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    optimizer_lrs = {
        group.get("name", "all_parameters"): float(group["lr"])
        for group in optimizer.param_groups
    }
    metadata = {
        "init_checkpoint": init_info["checkpoint"] if init_info else None,
        "loaded_keys": int(init_info["loaded_keys"]) if init_info else 0,
        "optimizer_lrs": optimizer_lrs,
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return metadata_path


def save_vqvae_finetune_checkpoint(checkpoint_path, model, init_info, optimizer):
    checkpoint_path = Path(checkpoint_path)
    metadata_path = Path(f"{checkpoint_path}.metadata.json")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=checkpoint_path.parent,
        prefix=f".{checkpoint_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_checkpoint_path = Path(handle.name)
    temp_metadata_path = Path(f"{temp_checkpoint_path}.metadata.json")
    try:
        torch.save(model.state_dict(), temp_checkpoint_path)
        write_vqvae_finetune_metadata(
            temp_checkpoint_path,
            init_info,
            optimizer,
        )
        os.replace(temp_checkpoint_path, checkpoint_path)
        os.replace(temp_metadata_path, metadata_path)
    finally:
        temp_checkpoint_path.unlink(missing_ok=True)
        temp_metadata_path.unlink(missing_ok=True)


def load_mnist(batch_size: int, num_workers: int, rank: int, world_size: int):
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST("data/", train=True, download=True, transform=transform)
    val_ds = datasets.MNIST("data/", train=False, download=True, transform=transform)

    x_train_var = float(np.var(train_ds.data.numpy() / 255.0))

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, train_sampler, x_train_var


def train_vqvae(cfg: dict) -> None:
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    is_main = local_rank == 0

    train_loader, val_loader, train_sampler, x_train_var = load_mnist(
        cfg["training"]["batch_size"], cfg["training"]["num_workers"], local_rank, world_size
    )

    model = build_vqvae(cfg).to(device)
    model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"], amsgrad=True)

    writer = SummaryWriter(log_dir="logs/vqvae") if is_main else None
    global_step = 0
    best_val_recon = float("inf")
    patience_counter = 0
    patience = cfg["training"].get("early_stop_patience", None)

    for epoch in range(cfg["training"]["epochs_vqvae"]):
        model.train()
        train_sampler.set_epoch(epoch)
        should_stop = False
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            x_hat, losses = model(x)
            recon = reconstruction_loss(x_hat, x, x_train_var)
            total = recon + losses["embedding"]
            total.backward()
            optimizer.step()

            if is_main:
                writer.add_scalar("train/recon_loss", recon.item(), global_step)
                writer.add_scalar("train/embedding_loss", losses["embedding"].item(), global_step)
                writer.add_scalar("train/commitment_loss", losses["commitment"].item(), global_step)
                writer.add_scalar("train/codebook_loss", losses["codebook"].item(), global_step)
                writer.add_scalar("train/total_loss", total.item(), global_step)
            global_step += 1

        if is_main:
            val_metrics = validate_vqvae(model.module, val_loader, x_train_var, device)
            writer.add_scalar("val/recon_loss", val_metrics["val_recon_loss"], epoch)
            writer.add_scalar("val/perplexity", val_metrics["val_perplexity"], epoch)
            writer.add_scalar("val/codebook_utilization", val_metrics["val_codebook_utilization"], epoch)
            writer.add_scalar("val/active_codes", val_metrics["val_active_codes"], epoch)
            print(f"[VQVAE epoch {epoch}] recon={val_metrics['val_recon_loss']:.4f} "
                  f"perplexity={val_metrics['val_perplexity']:.1f} "
                  f"active_codes={val_metrics['val_active_codes']}")

            if epoch % 50 == 0:
                with torch.no_grad():
                    x_sample = next(iter(val_loader))[0][:16].to(device)
                    x_hat_sample, _ = model.module(x_sample)
                save_reconstruction_grid(x_sample.cpu(), x_hat_sample.cpu(),
                                         f"results/recon_epoch{epoch:03d}.png")

            # early stopping
            if patience is not None:
                if val_metrics["val_recon_loss"] < best_val_recon:
                    best_val_recon = val_metrics["val_recon_loss"]
                    patience_counter = 0
                    torch.save(model.module.state_dict(), "results/vqvae_best.pt")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stop at epoch {epoch} (no improvement for {patience} epochs)")
                        should_stop = True

        if sync_should_stop(should_stop, device):
            break

    if is_main:
        writer.close()
        os.makedirs("results", exist_ok=True)
        if patience is None:
            # no early stopping — save final as best
            torch.save(model.module.state_dict(), "results/vqvae_best.pt")
        print(f"Best val_recon_loss: {best_val_recon:.4f} -> results/vqvae_best.pt")

    teardown_ddp()


def train_prior(cfg: dict) -> None:
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    is_main = local_rank == 0

    train_loader, val_loader, train_sampler, x_train_var = load_mnist(
        cfg["training"]["batch_size"], cfg["training"]["num_workers"], local_rank, world_size
    )

    vqvae = build_vqvae(cfg).to(device)
    vqvae.load_state_dict(torch.load("results/vqvae_best.pt", map_location=device))
    for p in vqvae.parameters():
        p.requires_grad_(False)
    vqvae.eval()

    prior = build_prior(cfg).to(device)
    prior = DDP(prior, device_ids=[local_rank])
    optimizer = torch.optim.Adam(prior.parameters(), lr=cfg["training"]["lr"], amsgrad=True)

    writer = SummaryWriter(log_dir="logs/prior") if is_main else None
    global_step = 0
    best_val_nll = float("inf")
    patience_counter = 0
    patience = cfg["training"].get("early_stop_patience", None)

    for epoch in range(cfg["training"]["epochs_prior"]):
        prior.train()
        train_sampler.set_epoch(epoch)
        should_stop = False
        for x, labels in train_loader:
            x, labels = x.to(device), labels.to(device)
            with torch.no_grad():
                indices = vqvae.encode(x)
            logits = prior(indices, labels)
            loss = prior_nll_loss(logits, indices)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if is_main:
                writer.add_scalar("train/prior_nll_loss", loss.item(), global_step)
            global_step += 1

        if is_main:
            val_metrics = validate_prior(prior.module, vqvae, val_loader, device)
            writer.add_scalar("val/prior_nll", val_metrics["val_prior_nll"], epoch)
            print(f"[Prior epoch {epoch}] nll={val_metrics['val_prior_nll']:.4f}")

            if epoch % 50 == 0:
                labels_gen = torch.arange(10, device=device).repeat(10)
                with torch.no_grad():
                    indices_gen = prior.module.sample(100, device, labels=labels_gen, shape=(7, 7))
                    x_gen = vqvae.decode_from_indices(indices_gen)
                save_generation_grid(x_gen.cpu(), f"results/gen_epoch{epoch:03d}.png", nrow=10)

            # early stopping
            if patience is not None:
                if val_metrics["val_prior_nll"] < best_val_nll:
                    best_val_nll = val_metrics["val_prior_nll"]
                    patience_counter = 0
                    torch.save(prior.module.state_dict(), "results/prior_best.pt")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stop at epoch {epoch} (no improvement for {patience} epochs)")
                        should_stop = True

        if sync_should_stop(should_stop, device):
            break

    if is_main:
        writer.close()
        if patience is None:
            torch.save(prior.module.state_dict(), "results/prior_best.pt")
        print(f"Best val_prior_nll: {best_val_nll:.4f} -> results/prior_best.pt")

    teardown_ddp()


# ---------- scRNA path ----------

def init_codebook_from_embeddings(
    model: torch.nn.Module,
    npy_path: str,
    meta_csv: str,
    celltype_col: str,
    embedding_dim: int,
    seed: int,
    noise_scale: float,
    rank: int,
) -> None:
    """Initialize the VQ codebook from current encoder quantizer-space outputs.

    Steps (rank 0 only; others wait at barrier):
      1. Load all downsample embeddings and their cell-type labels.
      2. Run them through the current encoder and pre-quantization layer to get
         z_e vectors in the same space that the quantizer sees during training.
      3. Compute per-cell-type mean z_e vectors.
      4. Fill remaining codebook slots by interpolation inside the quantizer.
      5. Call quantizer.initialize_from_celltype_means with the result.

    The model must NOT be wrapped in DDP when this is called — pass model.module
    if already wrapped, or the raw model before DDP.
    """
    import pandas as pd

    if dist.is_available() and dist.is_initialized():
        dist.barrier()   # ensure all ranks reach this point before rank 0 mutates weights

    if rank == 0:
        print("[codebook init] opening embeddings as read-only memmap ...", flush=True)
        device = next(model.parameters()).device
        all_emb = np.load(npy_path, mmap_mode="r")           # (N, 3072)
        meta = pd.read_csv(meta_csv, usecols=[celltype_col])
        raw_labels = meta[celltype_col]
        if len(raw_labels) != len(all_emb):
            raise ValueError(
                f"embedding rows {len(all_emb)} != metadata rows {len(raw_labels)}"
            )

        all_rows = np.arange(len(all_emb), dtype=np.int64)
        if len(all_rows) == 0:
            raise ValueError("codebook initialization requires at least one row")

        stripped_labels = raw_labels.astype("string").str.strip()
        missing_labels = (
            raw_labels.isna()
            | stripped_labels.isna()
            | stripped_labels.eq("")
        )
        if bool(missing_labels.any()):
            bad_rows = all_rows[missing_labels.to_numpy()][:10].tolist()
            raise ValueError(
                f"cell-type labels are missing or blank for rows {bad_rows}"
            )
        labels = stripped_labels.astype(str).to_numpy()

        if not getattr(model, "_flat", False):
            raise ValueError("scRNA codebook initialization requires flat VQ-VAE path")
        if not hasattr(model, "pre_quantization_linear"):
            raise ValueError("model must expose pre_quantization_linear for codebook init")

        was_training = model.training
        model.eval()
        celltypes = sorted(set(labels))
        label_to_index = {celltype: i for i, celltype in enumerate(celltypes)}
        label_indices = np.fromiter(
            (label_to_index[label] for label in labels),
            dtype=np.int64,
            count=len(labels),
        )
        z_sums = None
        z_counts = np.zeros(len(celltypes), dtype=np.int64)
        init_batch_size = 4096
        try:
            with torch.no_grad():
                for start in range(0, len(all_emb), init_batch_size):
                    emb_batch = np.array(
                        all_emb[start:start + init_batch_size],
                        dtype=np.float32,
                        copy=True,
                    )
                    x = torch.as_tensor(
                        emb_batch,
                        dtype=torch.float32,
                        device=device,
                    )
                    h = model.encoder(x)
                    z_e = model.pre_quantization_linear(h)
                    if (
                        getattr(model.quantizer, "latent_format", "sequence")
                        == "sequence"
                        and hasattr(model, "quant_proj")
                    ):
                        z_tokens = model.quant_proj(z_e.unsqueeze(1))
                        if z_tokens.shape[1] > 1:
                            z_e = z_tokens
                        else:
                            z_e = z_tokens[:, 0, :]
                    if z_e.shape[-1] != embedding_dim:
                        raise ValueError(
                            f"encoder quantizer-space dim {z_e.shape[-1]} != "
                            f"embedding_dim {embedding_dim}"
                        )
                    if not torch.isfinite(z_e).all():
                        raise ValueError(
                            "encoder produced non-finite values during "
                            "codebook initialization"
                        )
                    z_np = z_e.detach().cpu().numpy().astype(np.float32)
                    if z_sums is None:
                        z_sums = np.zeros(
                            (len(celltypes),) + z_np.shape[1:],
                            dtype=np.float64,
                        )
                    batch_labels = label_indices[start:start + len(z_np)]
                    np.add.at(z_sums, batch_labels, z_np)
                    z_counts += np.bincount(
                        batch_labels,
                        minlength=len(celltypes),
                    )
        finally:
            if was_training:
                model.train()
        if z_sums is None or np.any(z_counts == 0):
            raise ValueError("codebook initialization encountered an empty cell type")

        # Per-cell-type mean in the current quantizer input space.  Keep only
        # one mean per label instead of retaining all N encoder outputs.
        reshape = (len(z_counts),) + (1,) * (z_sums.ndim - 1)
        ct_means = (z_sums / z_counts.reshape(reshape)).astype(np.float32)
        if not np.isfinite(ct_means).all():
            raise ValueError("cell-type means contain non-finite values")
        print(
            f"[codebook init] {len(celltypes)} cell types from "
            f"{len(all_emb)} rows: {celltypes}",
            flush=True,
        )

        ct_t = torch.tensor(ct_means, dtype=torch.float32, device=device)
        model.quantizer.initialize_from_celltype_means(
            ct_t,
            seed=seed,
            noise_scale=noise_scale,
        )
        print(
            f"[codebook init] done — {len(celltypes)} slots from cell-type means, "
            f"{model.quantizer.n_embeddings - len(celltypes)} slots from interpolation.",
            flush=True,
        )

    if dist.is_available() and dist.is_initialized():
        # broadcast the initialized codebook weights to all ranks
        if hasattr(model.quantizer, "broadcast_codebooks"):
            model.quantizer.broadcast_codebooks(src=0)
        elif hasattr(model.quantizer, "embedding"):
            dist.broadcast(model.quantizer.embedding.weight.data, src=0)
        else:
            raise AttributeError("quantizer must expose broadcast_codebooks() or embedding")
        dist.barrier()


def _fit_pca_on_train(dataset, train_idx, n_components, fit_n, seed, device):
    """Fit PCA on a sample of training-set targets (no val/test leakage).

    Returns a dict with 'mean' (n_predict,), 'components' (n_components, n_predict)
    as float32 tensors on `device`, plus 'n_components'. PCA.transform is then
    target_pca = (target - mean) @ components.T, applied per-batch on GPU.
    """
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(seed)
    sample_idx = train_idx
    if fit_n is not None and len(train_idx) > fit_n:
        sample_idx = rng.choice(train_idx, size=fit_n, replace=False)

    targets = np.stack([dataset[int(i)]["target"].numpy() for i in sample_idx])
    pca = PCA(n_components=n_components, random_state=seed)
    coeffs = pca.fit_transform(targets)
    return {
        "mean": torch.tensor(pca.mean_, dtype=torch.float32, device=device),
        "components": torch.tensor(pca.components_, dtype=torch.float32, device=device),
        "n_components": n_components,
        "coeff_var": float(np.var(coeffs)),
    }


def _rank_local_shard(indices, rank: int, world_size: int) -> np.ndarray:
    """Return the fixed, non-overlapping shard owned by one DDP rank."""
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError(f"invalid rank/world_size: rank={rank}, world_size={world_size}")
    indices = np.asarray(indices, dtype=np.int64)
    return indices[rank::world_size]


def project_pca(target: torch.Tensor, pca_state: dict) -> torch.Tensor:
    """Project gene-space targets into PCA coefficient space. (B, n_predict) -> (B, n_comp)."""
    return (target - pca_state["mean"]) @ pca_state["components"].t()


def _reconstruction_task_summary(recon_target: str, n_predict: int, out_dim: int, pca_state) -> str:
    if recon_target == "pca":
        n_components = (
            pca_state["n_components"]
            if pca_state is not None and "n_components" in pca_state
            else out_dim
        )
        return (
            "[reconstruction task] target=PCA "
            f"n_components={n_components} source_hvg_dim={n_predict} out_dim={out_dim}"
        )
    return f"[reconstruction task] target=HVG/MSE n_hvg={n_predict} out_dim={out_dim}"


def load_saved_scrna_split(split_path: str | Path, dataset_size: int):
    """Load and validate a complete saved train/validation/test partition."""
    split_path = Path(split_path)
    required = ("train_idx", "val_idx", "test_idx")
    with np.load(split_path, allow_pickle=False) as saved_split:
        missing = [name for name in required if name not in saved_split.files]
        if missing:
            raise ValueError(
                f"saved split {split_path} is missing arrays: {missing}"
            )
        partitions = tuple(
            np.asarray(saved_split[name], dtype=np.int64) for name in required
        )

    if any(indices.ndim != 1 for indices in partitions):
        raise ValueError(f"saved split {split_path} indices must be one-dimensional")
    combined = np.concatenate(partitions)
    if len(combined) != dataset_size:
        raise ValueError(
            f"saved split {split_path} has {len(combined)} rows, "
            f"expected {dataset_size}"
        )
    if len(np.unique(combined)) != dataset_size:
        raise ValueError(f"saved split {split_path} has overlapping indices")
    if len(combined) and (combined.min() < 0 or combined.max() >= dataset_size):
        raise ValueError(f"saved split {split_path} contains out-of-range indices")
    return partitions


def load_scrna(cfg: dict, rank: int, world_size: int, device=None):
    """
    Build train/val/test DataLoaders for scRNA data.

    Split: stratified 80/10/10 by cell type by default. When
    data.test_group_key and data.test_group_values are configured, every row
    matching those metadata values is forced into test. If data.test_fraction
    is configured, additional rows are sampled from the remaining cells until
    test reaches that fraction of the complete dataset. In that mode,
    data.val_fraction is also interpreted as a fraction of the complete
    dataset. Stratified splits fall back to random if labels are unavailable
    or a class is too small to stratify.

    When cfg["data"]["recon_target"] == "pca", fit PCA on the training targets
    and return its state on `device`; otherwise pca_state is None. `rank` is
    the global distributed rank. When omitted, `device` falls back to cuda:rank
    for backward compatibility.

    Returns (train_loader, val_loader, test_loader, train_sampler,
             x_train_var, dataset, pca_state).
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from torch.utils.data import Subset
    from data.scrna_dataset import ScRNADataset

    recon_target = cfg["data"].get("recon_target", "mse")
    pca_n_components = cfg["data"].get("pca_n_components", 50)

    dataset = ScRNADataset(
        h5ad_path=cfg["data"].get("h5ad_path"),
        gene_index_path=cfg["encoder"].get("gene_index_path"),
        predict_genes=cfg["data"].get("predict_genes", "hvg"),
        hvg_genes_path=cfg["data"].get("hvg_genes_path"),
        tgthighres=cfg["data"].get("tgthighres", "a5"),
        embedding_npy_path=cfg["data"].get("embedding_npy_path"),
        embedding_meta_csv=cfg["data"].get("embedding_meta_csv"),
        target_npy_path=cfg["data"].get("target_npy_path"),
        celltype_key=cfg["data"].get("celltype_key", "cell_type"),
        precomputed_only=cfg["data"].get("precomputed_only", False),
    )

    seed = cfg["data"].get("split_seed", 42)
    all_idx = np.arange(len(dataset))
    strat = np.asarray(dataset.celltypes) if dataset.celltypes is not None else None
    split_path = cfg.get("training", {}).get("split_path")
    saved_split = (
        load_saved_scrna_split(split_path, len(dataset))
        if split_path and Path(split_path).is_file()
        else None
    )

    def _split(idx, frac, stratify):
        try:
            return train_test_split(
                idx, test_size=frac, random_state=seed, stratify=stratify
            )
        except ValueError as e:
            if rank == 0:
                print(f"[load_scrna] stratified split failed ({e}); "
                      f"falling back to random split.")
            return train_test_split(idx, test_size=frac, random_state=seed)

    test_group_key = cfg["data"].get("test_group_key")
    test_group_values = cfg["data"].get("test_group_values")
    requested_test_groups = None
    if bool(test_group_key) != bool(test_group_values):
        raise ValueError(
            "data.test_group_key and data.test_group_values must be configured together"
        )

    if saved_split is not None:
        train_idx, val_idx, test_idx = saved_split
        if rank == 0:
            print(f"[load_scrna] reused split: {split_path}", flush=True)
    elif test_group_key:
        if isinstance(test_group_values, str):
            raise TypeError("data.test_group_values must be a sequence, not a string")
        requested_test_groups = list(dict.fromkeys(map(str, test_group_values)))
        meta_path = cfg["data"].get("embedding_meta_csv")
        if not meta_path:
            raise ValueError(
                "data.embedding_meta_csv is required for metadata-group holdout"
            )
        try:
            group_meta = pd.read_csv(meta_path, usecols=[test_group_key])
        except ValueError as exc:
            raise ValueError(
                f"{meta_path} does not contain test group column "
                f"{test_group_key!r}"
            ) from exc

        if dataset.embedding_rows is None:
            if len(group_meta) != len(dataset):
                raise ValueError(
                    f"metadata rows ({len(group_meta)}) do not match dataset rows "
                    f"({len(dataset)})"
                )
            groups = group_meta[test_group_key].astype(str).to_numpy()
        else:
            embedding_rows = np.asarray(dataset.embedding_rows, dtype=np.int64)
            if (
                len(embedding_rows) != len(dataset)
                or (len(embedding_rows) and embedding_rows.max() >= len(group_meta))
            ):
                raise ValueError(
                    "dataset embedding rows cannot be aligned to test-group metadata"
                )
            groups = (
                group_meta[test_group_key]
                .astype(str)
                .to_numpy()[embedding_rows]
            )

        available_groups = set(groups)
        missing_groups = [
            group for group in requested_test_groups
            if group not in available_groups
        ]
        if missing_groups:
            raise ValueError(
                f"test group values not found in {test_group_key!r}: "
                f"{missing_groups}"
            )

        test_mask = np.isin(groups, requested_test_groups)
        forced_test_idx = all_idx[test_mask]
        train_val_idx = all_idx[~test_mask]
        if not len(forced_test_idx) or not len(train_val_idx):
            raise ValueError(
                "metadata-group holdout must leave non-empty test and train/val sets"
            )

        test_fraction = cfg["data"].get("test_fraction")
        if test_fraction is not None:
            test_fraction = float(test_fraction)
            if not 0.0 < test_fraction < 1.0:
                raise ValueError("data.test_fraction must be between 0 and 1")
            target_test_n = int(np.ceil(len(all_idx) * test_fraction))
            if target_test_n < len(forced_test_idx):
                raise ValueError(
                    f"forced test groups already contain {len(forced_test_idx)} "
                    f"rows, exceeding the requested test size {target_test_n}"
                )
            extra_test_n = target_test_n - len(forced_test_idx)
            if extra_test_n:
                train_val_strat = (
                    strat[train_val_idx] if strat is not None else None
                )
                train_val_idx, extra_test_idx = _split(
                    train_val_idx, extra_test_n, train_val_strat
                )
                test_idx = np.concatenate((forced_test_idx, extra_test_idx))
            else:
                test_idx = forced_test_idx
        else:
            test_idx = forced_test_idx

        val_fraction = float(cfg["data"].get("val_fraction", 0.1))
        if not 0.0 < val_fraction < 1.0:
            raise ValueError("data.val_fraction must be between 0 and 1")
        if test_fraction is not None:
            if test_fraction + val_fraction >= 1.0:
                raise ValueError(
                    "data.test_fraction + data.val_fraction must be less than 1"
                )
            target_val_n = int(np.ceil(len(all_idx) * val_fraction))
            if target_val_n >= len(train_val_idx):
                raise ValueError(
                    f"requested validation size {target_val_n} leaves no training rows"
                )
            val_split_size = target_val_n
        else:
            val_split_size = val_fraction
        train_val_strat = (
            strat[train_val_idx] if strat is not None else None
        )
        train_idx, val_idx = _split(
            train_val_idx, val_split_size, train_val_strat
        )
        if rank == 0:
            print(
                f"[load_scrna] held-out test groups: "
                f"{test_group_key}={requested_test_groups} "
                f"forced_rows={len(forced_test_idx)}",
                flush=True,
            )
            if test_fraction is not None:
                print(
                    f"[load_scrna] test target: fraction={test_fraction} "
                    f"total_rows={len(test_idx)} "
                    f"additional_rows={len(test_idx) - len(forced_test_idx)}",
                    flush=True,
                )
    else:
        # 80% train, 20% temp -> split temp into 10% val / 10% test
        train_idx, temp_idx = _split(all_idx, 0.20, strat)
        temp_strat = strat[temp_idx] if strat is not None else None
        val_idx, test_idx = _split(temp_idx, 0.50, temp_strat)

    if rank == 0:
        print(f"[load_scrna] split: train={len(train_idx)} "
              f"val={len(val_idx)} test={len(test_idx)} "
              f"(stratified={strat is not None})")

    # Keep a fixed shard per rank so the selected rows can be materialized once
    # in rank-local RAM and reused across all epochs.  A regular
    # DistributedSampler reshuffles ownership every epoch, which would defeat
    # this preload strategy.
    train_rank_idx = _rank_local_shard(train_idx, rank, world_size)
    val_rank_idx = _rank_local_shard(val_idx, rank, world_size)
    test_rank_idx = _rank_local_shard(test_idx, rank, world_size)
    preload_rank_shards = cfg["data"].get(
        "preload_rank_shards",
        bool(cfg["data"].get("precomputed_only", False)),
    )

    # Fit PCA and estimate the MSE scale before rank-local preloading.  These
    # setup steps intentionally inspect only a small sample (or the PCA fit
    # sample) through the shared read-only memmap.
    pca_state = None
    if recon_target == "pca":
        if device is None:
            device = torch.device(f"cuda:{rank}")
        pca_state = _fit_pca_on_train(
            dataset, train_idx,
            n_components=pca_n_components,
            fit_n=cfg["data"].get("pca_fit_n"),
            seed=seed, device=device,
        )
        x_train_var = pca_state["coeff_var"]
    else:
        sample_targets = []
        for idx in train_idx[:min(2000, len(train_idx))]:
            sample_targets.append(dataset[int(idx)]["target"].numpy())
        x_train_var = float(np.var(np.stack(sample_targets)))

    if preload_rank_shards and hasattr(dataset, "preload_rows"):
        rank_rows = np.concatenate((train_rank_idx, val_rank_idx, test_rank_idx))
        dataset.preload_rows(rank_rows)

    train_ds = Subset(dataset, train_rank_idx)
    val_ds = Subset(dataset, val_rank_idx)
    test_ds = Subset(dataset, test_rank_idx)

    # Each dataset is already rank-local, so the sampler only needs to shuffle
    # that rank's own rows.  num_replicas=1 avoids a second partitioning step.
    train_sampler = DistributedSampler(
        train_ds, num_replicas=1, rank=0, shuffle=True
    )
    val_sampler = DistributedSampler(
        val_ds, num_replicas=1, rank=0, shuffle=False
    )
    test_sampler = DistributedSampler(
        test_ds, num_replicas=1, rank=0, shuffle=False
    )
    if rank == 0:
        print(
            f"[load_scrna] rank-local rows: train={len(train_rank_idx)} "
            f"val={len(val_rank_idx)} test={len(test_rank_idx)} "
            f"preloaded={bool(preload_rank_shards)}",
            flush=True,
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        sampler=train_sampler,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        sampler=val_sampler,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["training"]["batch_size"],
        sampler=test_sampler,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
    )

    embedding_rows = (
        dataset.embedding_rows
        if dataset.embedding_rows is not None
        else np.arange(len(dataset), dtype=np.int64)
    )
    split_info = {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "train_embedding_rows": embedding_rows[train_idx],
        "val_embedding_rows": embedding_rows[val_idx],
        "test_embedding_rows": embedding_rows[test_idx],
        "embedding_rows": embedding_rows,
        "cell_ids": np.asarray(dataset.cell_ids, dtype=object),
    }
    if requested_test_groups is not None:
        split_info["test_group_key"] = np.asarray(str(test_group_key))
        split_info["test_group_values"] = np.asarray(
            requested_test_groups, dtype=str
        )

    return train_loader, val_loader, test_loader, train_sampler, x_train_var, dataset, pca_state, split_info


def _distributed_contrastive_batch(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather equal-sized rank-local batches, keeping local gradients."""
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return embeddings, labels
    world_size = dist.get_world_size()
    local_size = torch.tensor([embeddings.shape[0]], device=embeddings.device)
    sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(sizes, local_size)
    if any(int(size.item()) != int(local_size.item()) for size in sizes):
        # The final batch can differ between rank-local shards. Falling back
        # to the local batch avoids padding false positives.
        return embeddings, labels
    gathered_embeddings = [torch.empty_like(embeddings) for _ in range(world_size)]
    gathered_labels = [torch.empty_like(labels) for _ in range(world_size)]
    dist.all_gather(gathered_embeddings, embeddings.detach())
    dist.all_gather(gathered_labels, labels)
    gathered_embeddings[dist.get_rank()] = embeddings
    return torch.cat(gathered_embeddings, dim=0), torch.cat(gathered_labels, dim=0)


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
    gather_distributed: bool = True,
) -> torch.Tensor:
    """Supervised contrastive loss using same-label samples as positives."""
    if embeddings.ndim != 2:
        raise ValueError(f"contrastive embeddings must be 2-D, got {tuple(embeddings.shape)}")
    if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
        raise ValueError("contrastive labels must be a 1-D vector matching the batch")
    if temperature <= 0.0 or not math.isfinite(float(temperature)):
        raise ValueError("contrastive temperature must be finite and positive")
    if gather_distributed:
        embeddings, labels = _distributed_contrastive_batch(embeddings, labels)
    n = embeddings.shape[0]
    if n < 2:
        return embeddings.sum() * 0.0

    z = F.normalize(embeddings, dim=-1)
    logits = torch.matmul(z, z.transpose(0, 1)) / float(temperature)
    not_self = ~torch.eye(n, dtype=torch.bool, device=embeddings.device)
    positive = labels[:, None].eq(labels[None, :]) & not_self
    positive_count = positive.sum(dim=1)
    valid = positive_count > 0
    if not bool(valid.any()):
        return embeddings.sum() * 0.0

    logits = logits.masked_fill(~not_self, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    mean_log_prob_pos = (
        log_prob.masked_fill(~positive, 0.0).sum(dim=1)
        / positive_count.clamp_min(1)
    )
    return -mean_log_prob_pos[valid].mean()


def _scheduled_contrastive_weight(cfg: dict, epoch: int) -> float:
    """Warm up and ramp the contrastive term without changing the config."""
    contrastive_cfg = cfg.get("contrastive") or {}
    base = float(contrastive_cfg.get("loss_weight", 0.0))
    warmup = max(int(contrastive_cfg.get("warmup_epochs", 0)), 0)
    ramp = max(int(contrastive_cfg.get("ramp_epochs", 0)), 0)
    if epoch < warmup:
        return 0.0
    if ramp == 0:
        return base
    progress = min(1.0, float(epoch - warmup + 1) / float(ramp))
    return base * progress


def _validate_vqvae_scrna(
    model,
    val_loader,
    x_train_var,
    device,
    pca_state=None,
    celltype_loss_weight: float = 0.0,
    celltype_class_weights: torch.Tensor | None = None,
    contrastive_loss_weight: float = 0.0,
    contrastive_temperature: float = 0.1,
    contrastive_gather: bool = True,
):
    """Validate one fixed rank shard and all-reduce exact global metrics."""
    from stage1.evaluation.losses import reconstruction_loss

    model.eval()
    n_e = model.quantizer.n_embeddings
    n_codebooks = int(getattr(model.quantizer, "num_codebooks", 1))
    metrics = {
        "recon_sum": 0.0,
        "total_sum": 0.0,
        "count": 0,
        "celltype_loss_sum": 0.0,
        "celltype_correct_sum": 0.0,
        "contrastive_loss_sum": 0.0,
        "code_counts": torch.zeros(
            n_codebooks,
            n_e,
            dtype=torch.long,
            device=device,
        ),
    }

    with torch.no_grad():
        for batch in val_loader:
            x      = batch["x"].to(device)
            target = batch["target"].to(device)
            if pca_state is not None:
                target = project_pca(target, pca_state)
            x_hat, losses = model(x)
            recon = reconstruction_loss(x_hat, target, x_train_var)
            total = recon + _quantizer_training_loss(losses, model._quantizer_loss_weights)
            if celltype_loss_weight > 0.0:
                labels = batch["celltype_label"].to(device)
                logits = losses["celltype_logits"]
                celltype_loss = celltype_cross_entropy(
                    logits, labels, celltype_class_weights
                )
                total = total + celltype_loss_weight * celltype_loss
            if contrastive_loss_weight > 0.0:
                labels = batch["celltype_label"].to(device)
                contrastive_loss = supervised_contrastive_loss(
                    losses["contrastive_embedding"],
                    labels,
                    temperature=contrastive_temperature,
                    gather_distributed=contrastive_gather,
                )
                total = total + contrastive_loss_weight * contrastive_loss
            indices = model.encode(x)
            batch_n = int(x.shape[0])
            metrics["recon_sum"] += recon.item() * batch_n
            metrics["total_sum"] += total.item() * batch_n
            metrics["count"] += batch_n
            if celltype_loss_weight > 0.0:
                metrics["celltype_loss_sum"] += celltype_loss.item() * batch_n
                metrics["celltype_correct_sum"] += (
                    logits.argmax(dim=-1).eq(labels).sum().item()
                )
            if contrastive_loss_weight > 0.0:
                metrics["contrastive_loss_sum"] += contrastive_loss.item() * batch_n
            _accumulate_code_counts(metrics["code_counts"], indices)

    return _reduce_scrna_eval_metrics(metrics, device, n_e)


def _accumulate_code_counts(
    code_counts: torch.Tensor,
    indices: torch.Tensor,
) -> None:
    """Accumulate hard assignments without merging independent codebooks."""
    if code_counts.dim() == 1:
        flat_indices = indices.reshape(-1)
        code_counts.scatter_add_(
            0,
            flat_indices,
            torch.ones_like(flat_indices),
        )
        return

    if code_counts.dim() != 2:
        raise ValueError(
            f"code_counts must be 1D or 2D, got {tuple(code_counts.shape)}"
        )
    if indices.dim() < 2 or indices.shape[1] != code_counts.shape[0]:
        raise ValueError(
            "indices codebook dimension does not match code_counts: "
            f"indices={tuple(indices.shape)} counts={tuple(code_counts.shape)}"
        )
    for group_idx in range(code_counts.shape[0]):
        group_indices = indices[:, group_idx].reshape(-1)
        code_counts[group_idx].scatter_add_(
            0,
            group_indices,
            torch.ones_like(group_indices),
        )


def _reduce_scrna_eval_metrics(metrics: dict, device: torch.device, n_embeddings: int) -> dict:
    """All-reduce sample-weighted losses and global code-assignment counts."""
    scalar = torch.tensor(
        [
            metrics["recon_sum"],
            metrics["total_sum"],
            float(metrics["count"]),
            metrics.get("celltype_loss_sum", 0.0),
            metrics.get("celltype_correct_sum", 0.0),
            metrics.get("contrastive_loss_sum", 0.0),
        ],
        dtype=torch.float64,
        device=device,
    )
    code_counts = metrics["code_counts"].to(device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(scalar, op=dist.ReduceOp.SUM)
        dist.all_reduce(code_counts, op=dist.ReduceOp.SUM)

    count = float(scalar[2].item())
    if count <= 0:
        raise ValueError("validation/test loaders produced no samples")
    codebook_stats = _codebook_usage_statistics(code_counts, n_embeddings)
    return {
        "val_recon_loss": float(scalar[0].item() / count),
        "val_total_loss": float(scalar[1].item() / count),
        "val_perplexity": codebook_stats["perplexity_mean"],
        "val_perplexity_min": codebook_stats["perplexity_min"],
        "val_perplexity_max": codebook_stats["perplexity_max"],
        "val_perplexity_per_codebook": codebook_stats["perplexity_per_codebook"],
        "val_codebook_utilization": codebook_stats["utilization_mean"],
        "val_codebook_utilization_min": codebook_stats["utilization_min"],
        "val_codebook_utilization_max": codebook_stats["utilization_max"],
        "val_utilization_per_codebook": codebook_stats["utilization_per_codebook"],
        "val_active_codes": codebook_stats["active_total"],
        "val_active_codes_min": codebook_stats["active_min"],
        "val_active_codes_max": codebook_stats["active_max"],
        "val_active_codes_per_codebook": codebook_stats["active_per_codebook"],
        "val_celltype_loss": float(scalar[3].item() / count),
        "val_celltype_accuracy": float(scalar[4].item() / count),
        "val_contrastive_loss": float(scalar[5].item() / count),
    }


def _quantizer_training_loss(
    losses: dict[str, torch.Tensor],
    weights: dict[str, float],
) -> torch.Tensor:
    total = None
    for name, weight in weights.items():
        term = losses[name] * weight
        total = term if total is None else total + term
    if total is None:
        raise ValueError("quantizer loss weights cannot be empty")
    return total


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _codebook_usage_statistics(
    code_counts: torch.Tensor,
    n_embeddings: int,
) -> dict:
    """Summarize each independent codebook without merging local code IDs."""
    if code_counts.dim() == 1:
        code_counts = code_counts.unsqueeze(0)
    if code_counts.dim() != 2:
        raise ValueError(
            f"code_counts must be 1D or 2D, got {tuple(code_counts.shape)}"
        )
    if code_counts.shape[1] != n_embeddings:
        raise ValueError(
            f"code_counts width {code_counts.shape[1]} != "
            f"n_embeddings {n_embeddings}"
        )

    counts_f = code_counts.float()
    totals = counts_f.sum(dim=1)
    probabilities = counts_f / totals.clamp_min(1.0).unsqueeze(1)
    entropy = -torch.sum(
        probabilities * torch.log(probabilities + 1e-10),
        dim=1,
    )
    perplexity = torch.where(
        totals > 0,
        torch.exp(entropy),
        torch.zeros_like(entropy),
    )
    active = (code_counts > 0).sum(dim=1)
    utilization = active.float() / float(n_embeddings)

    perplexity_values = [float(value) for value in perplexity.cpu().tolist()]
    active_values = [int(value) for value in active.cpu().tolist()]
    utilization_values = [float(value) for value in utilization.cpu().tolist()]
    return {
        "perplexity_mean": float(perplexity.mean().item()),
        "perplexity_min": float(perplexity.min().item()),
        "perplexity_max": float(perplexity.max().item()),
        "perplexity_per_codebook": perplexity_values,
        "active_total": int(active.sum().item()),
        "active_min": int(active.min().item()),
        "active_max": int(active.max().item()),
        "active_per_codebook": active_values,
        "utilization_mean": float(utilization.mean().item()),
        "utilization_min": float(utilization.min().item()),
        "utilization_max": float(utilization.max().item()),
        "utilization_per_codebook": utilization_values,
    }


def _reduce_train_epoch_metrics(metrics: dict, device: torch.device, n_embeddings: int) -> dict:
    """All-reduce sample-weighted losses and grouped code-assignment counts."""
    code_counts = metrics["code_counts"]
    scalar = torch.tensor(
        [
            metrics["recon_sum"],
            metrics["total_sum"],
            float(metrics["count"]),
            metrics.get("celltype_loss_sum", 0.0),
            metrics.get("celltype_correct_sum", 0.0),
            metrics.get("contrastive_loss_sum", 0.0),
        ],
        dtype=torch.float64,
        device=device,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(scalar, op=dist.ReduceOp.SUM)
        dist.all_reduce(code_counts, op=dist.ReduceOp.SUM)

    n_total = max(float(scalar[2].item()), 1.0)
    codebook_stats = _codebook_usage_statistics(code_counts, n_embeddings)
    return {
        "train_recon_loss":           float(scalar[0].item() / n_total),
        "train_total_loss":           float(scalar[1].item() / n_total),
        "train_perplexity":           codebook_stats["perplexity_mean"],
        "train_perplexity_min":       codebook_stats["perplexity_min"],
        "train_perplexity_max":       codebook_stats["perplexity_max"],
        "train_perplexity_per_codebook":
            codebook_stats["perplexity_per_codebook"],
        "train_codebook_utilization": codebook_stats["utilization_mean"],
        "train_codebook_utilization_min": codebook_stats["utilization_min"],
        "train_codebook_utilization_max": codebook_stats["utilization_max"],
        "train_utilization_per_codebook":
            codebook_stats["utilization_per_codebook"],
        "train_celltype_loss": float(scalar[3].item() / n_total),
        "train_celltype_accuracy": float(scalar[4].item() / n_total),
        "train_contrastive_loss": float(scalar[5].item() / n_total),
        "train_active_codes":         codebook_stats["active_total"],
        "train_active_codes_min":     codebook_stats["active_min"],
        "train_active_codes_max":     codebook_stats["active_max"],
        "train_active_codes_per_codebook":
            codebook_stats["active_per_codebook"],
    }


def train_vqvae_scrna(cfg: dict) -> None:
    """
    DDP training loop for scRNA VQ-VAE.
    Mirrors train_vqvae except uses dict batch contract and _validate_vqvae_scrna.
    Patches cfg["decoder"]["out_dim"] from dataset.n_predict before building the model.
    """
    local_rank = setup_ddp()
    device     = torch.device(f"cuda:{local_rank}")
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    is_main    = global_rank == 0
    train_cfg = cfg.get("training", {})
    checkpoint_path = train_cfg.get("checkpoint_path", "results/vqvae_scrna_best.pt")
    log_dir = train_cfg.get("log_dir", "logs/vqvae_scrna")
    split_path = train_cfg.get("split_path")

    train_loader, val_loader, test_loader, train_sampler, x_train_var, dataset, pca_state, split_info = \
        load_scrna(cfg, global_rank, world_size, device=device)
    classifier_cfg = cfg.get("celltype_classifier") or {}
    celltype_loss_weight = float(classifier_cfg.get("loss_weight", 0.0))
    celltype_class_weights = None
    if celltype_loss_weight < 0.0:
        raise ValueError("celltype_classifier.loss_weight must be non-negative")
    if celltype_loss_weight > 0.0:
        observed_classes = getattr(dataset, "celltype_classes", None)
        if not observed_classes:
            raise ValueError(
                "celltype_classifier requires non-empty celltype labels"
            )
        expected_classes = int(
            classifier_cfg.get("num_classes") or len(observed_classes)
        )
        if len(observed_classes) != expected_classes:
            raise ValueError(
                "celltype_classifier class count does not match metadata: "
                f"configured {expected_classes}, observed {len(observed_classes)}"
            )
        # Let the first-version metadata define the classifier width when it
        # is omitted from the config.
        classifier_cfg["num_classes"] = expected_classes
        class_counts = _training_celltype_counts(
            dataset, split_info["train_idx"], expected_classes
        )
        class_weighting = str(classifier_cfg.get("class_weighting", "none"))
        max_class_weight = float(classifier_cfg.get("max_class_weight", 5.0))
        celltype_class_weights = compute_celltype_class_weights(
            class_counts,
            strategy=class_weighting,
            max_class_weight=max_class_weight,
        ).to(device)
        if is_main:
            print(
                f"[celltype classifier] classes={expected_classes} "
                f"loss_weight={celltype_loss_weight} "
                f"class_weighting={class_weighting} "
                f"max_class_weight={max_class_weight}: {observed_classes}",
                flush=True,
            )
            for name, count, weight in zip(
                observed_classes, class_counts.tolist(), celltype_class_weights.tolist()
            ):
                print(
                    f"[celltype class weight] {name}: count={count} "
                    f"weight={weight:.6f}",
                    flush=True,
                )
    contrastive_cfg = cfg.get("contrastive") or {}
    contrastive_loss_weight = float(contrastive_cfg.get("loss_weight", 0.0))
    contrastive_temperature = float(contrastive_cfg.get("temperature", 0.1))
    contrastive_gather = bool(contrastive_cfg.get("gather_distributed", True))
    if contrastive_loss_weight < 0.0:
        raise ValueError("contrastive.loss_weight must be non-negative")
    if contrastive_loss_weight > 0.0:
        if getattr(dataset, "celltype_classes", None) is None:
            raise ValueError("contrastive learning requires non-empty celltype labels")
        if is_main:
            print(
                f"[supervised contrastive] weight={contrastive_loss_weight} "
                f"temperature={contrastive_temperature} "
                f"source={contrastive_cfg.get('source', 'z_e')} "
                f"warmup={contrastive_cfg.get('warmup_epochs', 0)} "
                f"ramp={contrastive_cfg.get('ramp_epochs', 0)}",
                flush=True,
            )
    if is_main:
        if split_path is None:
            split_path = os.path.join(
                os.path.dirname(cfg["data"]["embedding_npy_path"]),
                "vqvae_scrna_split.npz",
            )
        os.makedirs(os.path.dirname(split_path), exist_ok=True)
        np.savez(split_path, **split_info)
        print(f"Saved split: {split_path}", flush=True)
    # Decoder output is PCA-coefficient space when recon_target='pca', else gene space.
    if cfg["data"].get("recon_target", "mse") == "pca":
        cfg["decoder"]["out_dim"] = (
            pca_state["n_components"] if pca_state is not None else dataset.target_dim
        )
    elif pca_state is not None:
        cfg["decoder"]["out_dim"] = pca_state["n_components"]
    else:
        cfg["decoder"]["out_dim"] = dataset.n_predict
    if is_main:
        print(
            _reconstruction_task_summary(
                cfg["data"].get("recon_target", "mse"),
                dataset.n_predict,
                cfg["decoder"]["out_dim"],
                pca_state,
            ),
            flush=True,
        )

    model = build_vqvae(cfg).to(device)
    init_info = None
    init_checkpoint = train_cfg.get("init_checkpoint")
    if init_checkpoint:
        init_info = load_vqvae_initial_checkpoint(model, init_checkpoint, device)
        if is_main:
            print(
                f"Initialized Stage-1 from {init_info['checkpoint']} "
                f"({init_info['loaded_keys']} keys)",
                flush=True,
            )
    model = DDP(model, device_ids=[local_rank])

    # Codebook init from downsample embeddings (optional, controlled by config).
    cb_cfg = cfg.get("codebook_init", {})
    if should_run_codebook_init(cfg):
        init_codebook_from_embeddings(
            model         = model.module,
            npy_path      = cb_cfg["npy_path"],
            meta_csv      = cb_cfg["meta_csv"],
            celltype_col  = cb_cfg.get("celltype_col", "celltype"),
            embedding_dim = getattr(
                model.module.quantizer,
                "vocab_width",
                getattr(
                    model.module.quantizer,
                    "embedding_dim",
                    cfg["quantizer"].get("embedding_dim"),
                ),
            ),
            seed          = cfg["data"].get("split_seed", 42),
            noise_scale   = cb_cfg.get("noise_scale", 0.01),
            rank          = global_rank,
        )

    optimizer = build_vqvae_finetune_optimizer(model.module, train_cfg)

    sched_cfg = cfg["training"].get("scheduler", None)
    if sched_cfg and sched_cfg.get("type") == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=sched_cfg.get("factor", 0.5),
            patience=sched_cfg.get("patience", 10),
            min_lr=sched_cfg.get("min_lr", 1e-6),
        )
    else:
        scheduler = None

    writer          = SummaryWriter(log_dir=log_dir) if is_main else None
    global_step     = 0
    best_val_total  = float("inf")
    patience_counter = 0
    patience        = cfg["training"].get("early_stop_patience", None)

    for epoch in range(cfg["training"]["epochs_vqvae"]):
        model.train()
        train_sampler.set_epoch(epoch)
        should_stop = False
        n_e = model.module.quantizer.n_embeddings
        n_codebooks = int(getattr(model.module.quantizer, "num_codebooks", 1))
        epoch_train = {"recon_sum": 0.0, "total_sum": 0.0, "count": 0,
                       "celltype_loss_sum": 0.0, "celltype_correct_sum": 0.0,
                       "contrastive_loss_sum": 0.0,
                       "code_counts": torch.zeros(
                           n_codebooks, n_e, dtype=torch.long, device=device
                       )}

        pbar = tqdm(
            train_loader,
            desc=f"epoch {epoch}",
            disable=not is_main,
            dynamic_ncols=True,
        )
        for batch in pbar:
            x      = batch["x"].to(device)
            target = batch["target"].to(device)
            if pca_state is not None:
                target = project_pca(target, pca_state)
            optimizer.zero_grad()
            x_hat, losses = model(x)
            recon = reconstruction_loss(x_hat, target, x_train_var)
            quant_loss = _quantizer_training_loss(losses, model.module._quantizer_loss_weights)
            total = recon + quant_loss
            if celltype_loss_weight > 0.0:
                labels = batch["celltype_label"].to(device)
                logits = losses["celltype_logits"]
                celltype_loss = celltype_cross_entropy(
                    logits, labels, celltype_class_weights
                )
                total = total + celltype_loss_weight * celltype_loss
            active_contrastive_weight = _scheduled_contrastive_weight(cfg, epoch)
            # Compute the auxiliary loss during warmup as well.  Multiplying
            # it by zero keeps the reconstruction-only warmup objective while
            # ensuring DDP sees the projection-head parameters in the graph.
            if contrastive_loss_weight > 0.0:
                labels = batch["celltype_label"].to(device)
                contrastive_loss = supervised_contrastive_loss(
                    losses["contrastive_embedding"],
                    labels,
                    temperature=contrastive_temperature,
                    gather_distributed=contrastive_gather,
                )
                total = total + active_contrastive_weight * contrastive_loss
            total.backward()
            optimizer.step()
            with torch.no_grad():
                indices = model.module.encode(x)
                batch_n = int(x.shape[0])
                epoch_train["recon_sum"] += recon.item() * batch_n
                epoch_train["total_sum"] += total.item() * batch_n
                epoch_train["count"] += batch_n
                if celltype_loss_weight > 0.0:
                    epoch_train["celltype_loss_sum"] += celltype_loss.item() * batch_n
                    epoch_train["celltype_correct_sum"] += (
                        logits.argmax(dim=-1).eq(labels).sum().item()
                    )
                if contrastive_loss_weight > 0.0:
                    epoch_train["contrastive_loss_sum"] += contrastive_loss.item() * batch_n
                _accumulate_code_counts(
                    epoch_train["code_counts"],
                    indices.detach(),
                )

            if is_main:
                if global_step % 1000 == 0:
                    writer.add_scalar("train/recon_loss",      recon.item(),              global_step)
                    writer.add_scalar("train/embedding_loss",  losses["embedding"].item(), global_step)
                    writer.add_scalar("train/vq_loss",         losses["vq"].item(),        global_step)
                    writer.add_scalar("train/entropy_loss",    losses["entropy"].item(),   global_step)
                    writer.add_scalar("train/commitment_loss", losses["commitment"].item(), global_step)
                    writer.add_scalar("train/codebook_loss",   losses["codebook"].item(),  global_step)
                    if celltype_loss_weight > 0.0:
                        writer.add_scalar("train/celltype_loss", celltype_loss.item(), global_step)
                        writer.add_scalar(
                            "train/celltype_accuracy",
                            logits.argmax(dim=-1).eq(labels).float().mean().item(),
                            global_step,
                        )
                    if contrastive_loss_weight > 0.0:
                        writer.add_scalar("train/contrastive_loss", contrastive_loss.item(), global_step)
                        writer.add_scalar(
                            "train/contrastive_weight",
                            active_contrastive_weight,
                            global_step,
                        )
                    writer.add_scalar("train/total_loss",      total.item(),              global_step)
                pbar.set_postfix(recon=f"{recon.item():.4f}", total=f"{total.item():.4f}")
            global_step += 1

        train_metrics = _reduce_train_epoch_metrics(
            epoch_train, device, model.module.quantizer.n_embeddings
        )

        # Every rank validates its fixed shard.  The helper all-reduces
        # sample-weighted loss sums and global code-assignment counts.
        val_metrics = _validate_vqvae_scrna(
            model.module, val_loader, x_train_var, device, pca_state,
            celltype_loss_weight=celltype_loss_weight,
            celltype_class_weights=celltype_class_weights,
            contrastive_loss_weight=_scheduled_contrastive_weight(cfg, epoch),
            contrastive_temperature=contrastive_temperature,
            contrastive_gather=contrastive_gather,
        )
        if is_main:
            writer.add_scalar("train_epoch/recon_loss",           train_metrics["train_recon_loss"], epoch)
            writer.add_scalar("train_epoch/total_loss",           train_metrics["train_total_loss"], epoch)
            writer.add_scalar("train_epoch/perplexity",           train_metrics["train_perplexity"], epoch)
            writer.add_scalar("train_epoch/codebook_utilization", train_metrics["train_codebook_utilization"], epoch)
            writer.add_scalar("train_epoch/active_codes",         train_metrics["train_active_codes"], epoch)
            writer.add_scalar("val/recon_loss",           val_metrics["val_recon_loss"],          epoch)
            writer.add_scalar("val/total_loss",           val_metrics["val_total_loss"],          epoch)
            writer.add_scalar("val/perplexity",           val_metrics["val_perplexity"],          epoch)
            writer.add_scalar("val/codebook_utilization", val_metrics["val_codebook_utilization"], epoch)
            writer.add_scalar("val/active_codes",         val_metrics["val_active_codes"],        epoch)
            if celltype_loss_weight > 0.0:
                writer.add_scalar("train_epoch/celltype_loss", train_metrics["train_celltype_loss"], epoch)
                writer.add_scalar("train_epoch/celltype_accuracy", train_metrics["train_celltype_accuracy"], epoch)
                writer.add_scalar("val/celltype_loss", val_metrics["val_celltype_loss"], epoch)
                writer.add_scalar("val/celltype_accuracy", val_metrics["val_celltype_accuracy"], epoch)
            if contrastive_loss_weight > 0.0:
                writer.add_scalar("train_epoch/contrastive_loss", train_metrics["train_contrastive_loss"], epoch)
                writer.add_scalar("val/contrastive_loss", val_metrics["val_contrastive_loss"], epoch)
            for group_idx, group_perplexity in enumerate(
                train_metrics["train_perplexity_per_codebook"]
            ):
                writer.add_scalar(
                    f"train_epoch/codebook_{group_idx}/perplexity",
                    group_perplexity,
                    epoch,
                )
                writer.add_scalar(
                    f"train_epoch/codebook_{group_idx}/active_codes",
                    train_metrics["train_active_codes_per_codebook"][group_idx],
                    epoch,
                )
                writer.add_scalar(
                    f"train_epoch/codebook_{group_idx}/utilization",
                    train_metrics["train_utilization_per_codebook"][group_idx],
                    epoch,
                )
            for group_idx, group_perplexity in enumerate(
                val_metrics["val_perplexity_per_codebook"]
            ):
                writer.add_scalar(
                    f"val/codebook_{group_idx}/perplexity",
                    group_perplexity,
                    epoch,
                )
                writer.add_scalar(
                    f"val/codebook_{group_idx}/active_codes",
                    val_metrics["val_active_codes_per_codebook"][group_idx],
                    epoch,
                )
                writer.add_scalar(
                    f"val/codebook_{group_idx}/utilization",
                    val_metrics["val_utilization_per_codebook"][group_idx],
                    epoch,
                )

            if patience is not None:
                if val_metrics["val_total_loss"] < best_val_total:
                    best_val_total   = val_metrics["val_total_loss"]
                    patience_counter = 0
                    save_vqvae_finetune_checkpoint(
                        checkpoint_path,
                        model.module,
                        init_info,
                        optimizer,
                    )
                    improved = True
                else:
                    patience_counter += 1
                    improved = False
                    if patience_counter >= patience:
                        print(
                            f"Early stop at epoch {epoch} "
                            f"(no improvement for {patience} epochs)"
                        )
                        should_stop = True
            else:
                improved = val_metrics["val_total_loss"] < best_val_total
                best_val_total = min(best_val_total, val_metrics["val_total_loss"])

            patience_text = (
                f" patience={patience_counter}/{patience}" if patience is not None else ""
            )
            classifier_text = ""
            if celltype_loss_weight > 0.0:
                classifier_text = (
                    f" | train_celltype_ce={train_metrics['train_celltype_loss']:.4f} "
                    f"train_celltype_acc={train_metrics['train_celltype_accuracy']:.4f} "
                    f"val_celltype_ce={val_metrics['val_celltype_loss']:.4f} "
                    f"val_celltype_acc={val_metrics['val_celltype_accuracy']:.4f}"
                )
            contrastive_text = ""
            if contrastive_loss_weight > 0.0:
                contrastive_text = (
                    f" | train_supcon={train_metrics['train_contrastive_loss']:.4f} "
                    f"val_supcon={val_metrics['val_contrastive_loss']:.4f} "
                    f"supcon_w={_scheduled_contrastive_weight(cfg, epoch):.4f}"
                )
            print(
                f"[scRNA VQVAE epoch {epoch}] "
                f"train_recon={train_metrics['train_recon_loss']:.4f} "
                f"train_total={train_metrics['train_total_loss']:.4f} "
                f"train_ppl={train_metrics['train_perplexity']:.1f} "
                f"train_active={train_metrics['train_active_codes']} "
                f"train_ppl_by_cb="
                f"{[round(x, 1) for x in train_metrics['train_perplexity_per_codebook']]} "
                f"train_active_by_cb="
                f"{train_metrics['train_active_codes_per_codebook']} | "
                f"val_recon={val_metrics['val_recon_loss']:.4f} "
                f"val_total={val_metrics['val_total_loss']:.4f} "
                f"val_ppl={val_metrics['val_perplexity']:.1f} "
                f"val_active={val_metrics['val_active_codes']} "
                f"val_ppl_by_cb="
                f"{[round(x, 1) for x in val_metrics['val_perplexity_per_codebook']]} "
                f"val_active_by_cb="
                f"{val_metrics['val_active_codes_per_codebook']} "
                f"best_val_total={best_val_total:.4f}"
                f"{' *' if improved else ''}"
                f"{classifier_text}"
                f"{contrastive_text}"
                f"{patience_text}",
                flush=True,
            )

        # scheduler step（所有 rank 都执行，保持参数同步）
        if scheduler is not None:
            scheduler.step(val_metrics["val_total_loss"])
        if is_main and writer is not None:
            log_vqvae_optimizer_lrs(writer, optimizer, epoch)

        if sync_should_stop(should_stop, device):
            break

    if is_main:
        writer.close()
        if patience is None:
            save_vqvae_finetune_checkpoint(
                checkpoint_path,
                model.module,
                init_info,
                optimizer,
            )
        print(f"Best val_total_loss: {best_val_total:.4f} -> {checkpoint_path}")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    # Final evaluation on held-out test set (no leakage — split before training).
    best_payload = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    best_state = _state_dict_from_checkpoint(best_payload)
    model.module.load_state_dict(best_state, strict=True)
    test_metrics = _validate_vqvae_scrna(
        model.module, test_loader, x_train_var, device, pca_state,
        celltype_loss_weight=celltype_loss_weight,
        celltype_class_weights=celltype_class_weights,
        contrastive_loss_weight=contrastive_loss_weight,
        contrastive_temperature=contrastive_temperature,
        contrastive_gather=contrastive_gather,
    )
    if is_main:
        writer_test = SummaryWriter(log_dir=log_dir)
        writer_test.add_scalar("test/recon_loss",           test_metrics["val_recon_loss"],           0)
        writer_test.add_scalar("test/total_loss",           test_metrics["val_total_loss"],           0)
        writer_test.add_scalar("test/perplexity",           test_metrics["val_perplexity"],           0)
        writer_test.add_scalar("test/codebook_utilization", test_metrics["val_codebook_utilization"], 0)
        writer_test.add_scalar("test/active_codes",         test_metrics["val_active_codes"],         0)
        if celltype_loss_weight > 0.0:
            writer_test.add_scalar("test/celltype_loss", test_metrics["val_celltype_loss"], 0)
            writer_test.add_scalar("test/celltype_accuracy", test_metrics["val_celltype_accuracy"], 0)
        if contrastive_loss_weight > 0.0:
            writer_test.add_scalar("test/contrastive_loss", test_metrics["val_contrastive_loss"], 0)
        writer_test.close()
        print(
            f"[Test] recon={test_metrics['val_recon_loss']:.4f} "
            f"total={test_metrics['val_total_loss']:.4f} "
            f"perplexity={test_metrics['val_perplexity']:.1f} "
            f"active_codes={test_metrics['val_active_codes']}"
            + (
                f" celltype_ce={test_metrics['val_celltype_loss']:.4f} "
                f"celltype_acc={test_metrics['val_celltype_accuracy']:.4f}"
                if celltype_loss_weight > 0.0 else ""
            )
            + (
                f" contrastive={test_metrics['val_contrastive_loss']:.4f}"
                if contrastive_loss_weight > 0.0 else ""
            )
        )

    teardown_ddp()
