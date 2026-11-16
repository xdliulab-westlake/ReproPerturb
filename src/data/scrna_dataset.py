import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def _load_gene_list(gene_index_path: str) -> list[str]:
    with open(gene_index_path) as f:
        first_line = f.readline()

    if "\t" in first_line:
        gene_df = pd.read_csv(gene_index_path, sep="\t")
        if "gene_name" in gene_df.columns:
            return gene_df["gene_name"].dropna().astype(str).tolist()

    with open(gene_index_path) as f:
        return [line.strip() for line in f if line.strip()]


def _decode_string_array(values):
    arr = np.asarray(values)
    if arr.dtype.kind == "S":
        return np.char.decode(arr, "utf-8").astype(str)
    return arr.astype(str)


def _read_index(group):
    if "_index" in group:
        return _decode_string_array(group["_index"][:])
    if "index" in group:
        return _decode_string_array(group["index"][:])
    raise KeyError(f"No _index/index under {group.name}")


def _read_column(group, key):
    obj = group[key]
    if isinstance(obj, h5py.Dataset):
        return _decode_string_array(obj[:])

    encoding = obj.attrs.get("encoding-type", "")
    if isinstance(encoding, bytes):
        encoding = encoding.decode()
    if encoding == "categorical":
        codes = np.asarray(obj["codes"][:], dtype=np.int64)
        categories = _decode_string_array(obj["categories"][:])
        out = np.empty(len(codes), dtype=object)
        valid = codes >= 0
        out[valid] = categories[codes[valid]]
        out[~valid] = ""
        return out.astype(str)

    raise NotImplementedError(f"Unsupported h5ad column encoding for {group.name}/{key}: {encoding!r}")


class _DenseX:
    def __init__(self, dataset):
        self.dataset = dataset
        self.shape = dataset.shape

    def get_row(self, row: int) -> np.ndarray:
        return np.asarray(self.dataset[row, :], dtype=np.float32)


class _CSRX:
    def __init__(self, group):
        self.data = group["data"]
        self.indices = group["indices"]
        self.indptr = group["indptr"]
        self.shape = tuple(group.attrs["shape"])

    def get_row(self, row: int) -> np.ndarray:
        start = int(self.indptr[row])
        end = int(self.indptr[row + 1])
        mat = csr_matrix(
            (
                np.asarray(self.data[start:end], dtype=np.float32),
                np.asarray(self.indices[start:end], dtype=np.int64),
                np.array([0, end - start], dtype=np.int64),
            ),
            shape=(1, self.shape[1]),
        )
        return mat.toarray()[0].astype(np.float32, copy=False)


class _LazyMatrix:
    def __init__(self, dataset: "ScRNADataset"):
        self.dataset = dataset

    def __getitem__(self, i):
        if isinstance(i, slice):
            return np.stack([self.dataset._raw_expr_by_dataset_row(j) for j in range(*i.indices(len(self)))])
        return self.dataset._raw_expr_by_dataset_row(int(i))

    def __len__(self):
        return len(self.dataset)

    @property
    def shape(self):
        return (len(self.dataset), self.dataset.n_genes)


class ScRNADataset(Dataset):
    def __init__(
        self,
        h5ad_path: str | None,
        gene_index_path: str | None,
        predict_genes: str = "hvg",
        hvg_genes_path: str | None = None,
        tgthighres: str = "a5",
        embedding_npy_path: str | None = None,
        embedding_meta_csv: str | None = None,
        target_npy_path: str | None = None,
        target_npy_expected_dim: int | None = None,
        celltype_key: str | None = "cell_type",
        precomputed_only: bool = False,
    ):
        if tgthighres != "a5":
            raise ValueError(f"Only tgthighres='a5' is supported, got '{tgthighres}'")
        if predict_genes not in ("hvg", "full"):
            raise ValueError(f"predict_genes must be 'hvg' or 'full', got '{predict_genes}'")
        if embedding_npy_path is not None and embedding_meta_csv is None:
            raise ValueError("embedding_meta_csv is required when embedding_npy_path is set")

        self.h5ad_path = h5ad_path
        self.predict_genes = predict_genes
        self.embedding_npy_path = embedding_npy_path
        self.target_npy_path = target_npy_path
        self.target_npy_expected_dim = target_npy_expected_dim
        self.target_dim = None
        self.embeddings = None
        self.targets = None
        self._h5 = None
        self._x_reader = None
        self.precomputed_only = precomputed_only
        self.preloaded_rows = None
        self._preloaded_row_map = None
        self._preloaded_embeddings = None
        self._preloaded_targets = None

        if predict_genes == "hvg" and hvg_genes_path is None:
            raise ValueError("hvg_genes_path is required when predict_genes='hvg'")
        if predict_genes == "hvg":
            with open(hvg_genes_path) as f:
                self.hvg_genes = [line.strip() for line in f if line.strip()]
        else:
            self.hvg_genes = None

        if precomputed_only:
            self._init_precomputed_only(
                embedding_npy_path=embedding_npy_path,
                embedding_meta_csv=embedding_meta_csv,
                target_npy_path=target_npy_path,
                celltype_key=celltype_key,
            )
            return

        if h5ad_path is None:
            raise ValueError("h5ad_path is required unless precomputed_only=True")
        if gene_index_path is None:
            raise ValueError("gene_index_path is required unless precomputed_only=True")

        self.gene_list = _load_gene_list(gene_index_path)
        self.n_genes = len(self.gene_list)

        with h5py.File(h5ad_path, "r") as h5:
            obs = h5["obs"]
            var = h5["var"]
            obs_names = _read_index(obs)
            if "gene_name" in var:
                adata_genes = _read_column(var, "gene_name")
            else:
                adata_genes = _read_index(var)
            celltypes_all = _read_column(obs, celltype_key) if celltype_key and celltype_key in obs else None

            gene_to_col = {}
            for col_idx, gene in enumerate(adata_genes.astype(str)):
                if gene not in gene_to_col:
                    gene_to_col[gene] = col_idx
            pairs = [(i, gene_to_col[gene]) for i, gene in enumerate(self.gene_list) if gene in gene_to_col]
            self.target_idx = np.array([p[0] for p in pairs], dtype=np.int64)
            self.source_idx = np.array([p[1] for p in pairs], dtype=np.int64)

            if predict_genes == "hvg":
                missing = [gene for gene in self.hvg_genes if gene not in gene_to_col]
                if missing:
                    raise ValueError(
                        f"{len(missing)} HVG gene(s) not found in h5ad var. "
                        f"First up to 20: {missing[:20]}"
                    )
                self.hvg_indices = np.array(
                    [gene_to_col[gene] for gene in self.hvg_genes],
                    dtype=np.int64,
                )
                self.n_predict = len(self.hvg_genes)
            else:
                self.hvg_indices = None
                self.n_predict = len(adata_genes)
            self.target_dim = self.n_predict

            keep_rows, kept_ids, kept_celltypes = self._find_nonzero_rows(
                h5["X"], obs_names, celltypes_all
            )

        self.h5_rows = keep_rows
        self.cell_ids = kept_ids
        self.celltypes = kept_celltypes
        self.celltype_classes = (
            sorted(set(self.celltypes)) if self.celltypes is not None else None
        )
        self.celltype_to_index = (
            {name: index for index, name in enumerate(self.celltype_classes)}
            if self.celltype_classes is not None
            else None
        )

        if embedding_npy_path is not None:
            emb_meta = pd.read_csv(embedding_meta_csv)
            emb_cell_ids = emb_meta["cell_id"].astype(str).tolist()
            emb_id_to_row = {cid: i for i, cid in enumerate(emb_cell_ids)}

            valid_positions = [i for i, cid in enumerate(self.cell_ids) if cid in emb_id_to_row]
            n_missing = len(self.cell_ids) - len(valid_positions)
            if n_missing > 0:
                logger.warning(f"{n_missing} cells in h5ad not found in embedding, dropping them.")

            self.h5_rows = self.h5_rows[valid_positions]
            self.cell_ids = [self.cell_ids[i] for i in valid_positions]
            if self.celltypes is not None:
                self.celltypes = [self.celltypes[i] for i in valid_positions]
            self.embedding_rows = np.array([emb_id_to_row[cid] for cid in self.cell_ids], dtype=np.int64)
        else:
            self.embedding_rows = None
        self.target_rows = self.embedding_rows

        self.matrix = _LazyMatrix(self)
        self._target_cache = None  # lazy-loaded normalized+HVG targets

    def _init_precomputed_only(
        self,
        *,
        embedding_npy_path: str | None,
        embedding_meta_csv: str | None,
        target_npy_path: str | None,
        celltype_key: str | None,
    ) -> None:
        if embedding_npy_path is None or embedding_meta_csv is None or target_npy_path is None:
            raise ValueError(
                "precomputed_only=True requires embedding_npy_path, "
                "embedding_meta_csv, and target_npy_path"
            )

        emb_meta = pd.read_csv(embedding_meta_csv)
        if "cell_id" not in emb_meta.columns:
            raise ValueError(f"{embedding_meta_csv} must contain a cell_id column")

        emb = np.load(embedding_npy_path, mmap_mode="r")
        target = np.load(target_npy_path, mmap_mode="r")
        emb_shape = tuple(emb.shape)
        target_shape = tuple(target.shape)
        del emb, target

        if len(emb_shape) != 2:
            raise ValueError(f"embedding_npy_path must be 2D, got shape {emb_shape}")
        if len(target_shape) != 2:
            raise ValueError(f"target_npy_path must be 2D, got shape {target_shape}")
        if emb_shape[0] != len(emb_meta):
            raise ValueError(
                f"embedding rows ({emb_shape[0]}) do not match metadata rows "
                f"({len(emb_meta)})"
            )
        if target_shape[0] != len(emb_meta):
            raise ValueError(
                f"target rows ({target_shape[0]}) do not match metadata rows "
                f"({len(emb_meta)})"
            )

        expected_dim = (
            self.target_npy_expected_dim
            or (len(self.hvg_genes) if self.predict_genes == "hvg" else target_shape[1])
        )
        if target_shape[1] != expected_dim:
            raise ValueError(
                f"target_npy_path second dim {target_shape[1]} != "
                f"expected target dim {expected_dim}"
            )

        self.gene_list = []
        self.n_genes = 0
        self.hvg_indices = None
        self.target_idx = np.array([], dtype=np.int64)
        self.source_idx = np.array([], dtype=np.int64)
        self.n_predict = int(expected_dim)
        self.target_dim = int(target_shape[1])
        self.h5_rows = None
        self.cell_ids = emb_meta["cell_id"].astype(str).tolist()
        self.celltypes = (
            emb_meta[celltype_key].astype(str).tolist()
            if celltype_key and celltype_key in emb_meta.columns
            else None
        )
        self.celltype_classes = (
            sorted(set(self.celltypes)) if self.celltypes is not None else None
        )
        self.celltype_to_index = (
            {name: index for index, name in enumerate(self.celltype_classes)}
            if self.celltype_classes is not None
            else None
        )
        self.embedding_rows = np.arange(len(emb_meta), dtype=np.int64)
        self.target_rows = self.embedding_rows
        self.matrix = None
        self._target_cache = None

    def preload_rows(self, rows) -> None:
        """Copy only selected dataset rows into this rank's RAM.

        DDP workers can call this with their fixed shard.  The source arrays
        remain read-only memmaps, while the selected rows are materialized
        once and reused by every epoch on that rank.
        """
        if self.embedding_npy_path is None or self.target_npy_path is None:
            raise ValueError("preload_rows requires precomputed embedding and target arrays")

        rows = np.asarray(rows, dtype=np.int64)
        if rows.ndim != 1:
            raise ValueError(f"preload rows must be 1-D, got shape {rows.shape}")
        if len(rows) and (rows.min() < 0 or rows.max() >= len(self)):
            raise IndexError("preload row is outside the dataset")
        rows = np.unique(rows)

        self._open_memmaps()
        embedding_rows = self.embedding_rows[rows]
        target_rows = self.target_rows[rows]
        self._preloaded_embeddings = np.asarray(
            self.embeddings[embedding_rows], dtype=np.float32
        )
        self._preloaded_targets = np.asarray(
            self.targets[target_rows], dtype=np.float32
        )
        self._preloaded_row_map = np.full(len(self), -1, dtype=np.int64)
        self._preloaded_row_map[rows] = np.arange(len(rows), dtype=np.int64)
        self.preloaded_rows = rows
        logger.info(
            "Preloaded %d/%d rows into rank-local RAM (embedding %.2f GB, target %.2f GB)",
            len(rows),
            len(self),
            self._preloaded_embeddings.nbytes / 1e9,
            self._preloaded_targets.nbytes / 1e9,
        )

    def _open_memmaps(self):
        if self.embedding_npy_path is not None and self.embeddings is None:
            logger.info(f"Opening embeddings as read-only memmap: {self.embedding_npy_path}")
            self.embeddings = np.load(self.embedding_npy_path, mmap_mode="r")
            logger.info(
                f"Embeddings opened: {self.embeddings.shape}, "
                f"{self.embeddings.nbytes / 1e9:.1f} GB virtual size"
            )
        if self.target_npy_path is not None and self.targets is None:
            logger.info(f"Opening targets as read-only memmap: {self.target_npy_path}")
            self.targets = np.load(self.target_npy_path, mmap_mode="r")
            logger.info(
                f"Targets opened: {self.targets.shape}, "
                f"{self.targets.nbytes / 1e9:.1f} GB virtual size"
            )
            expected_dim = self.target_npy_expected_dim or self.n_predict
            if self.targets.shape[1] != expected_dim:
                raise ValueError(
                    f"target_npy_path second dim {self.targets.shape[1]} != "
                    f"expected target dim {expected_dim}"
                )
            self.target_dim = int(self.targets.shape[1])
            if (
                self.target_rows is not None
                and len(self.target_rows) > 0
                and int(np.max(self.target_rows)) >= self.targets.shape[0]
            ):
                raise ValueError(
                    f"target_npy_path has {self.targets.shape[0]} rows but "
                    f"metadata requires row {int(np.max(self.target_rows))}"
                )

    def _open_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5ad_path, "r", swmr=True)  # Single-writer-multiple-reader 模式，减少锁竞争
            x = self._h5["X"]
            if isinstance(x, h5py.Dataset):
                self._x_reader = _DenseX(x)
            else:
                encoding = x.attrs.get("encoding-type", "")
                if isinstance(encoding, bytes):
                    encoding = encoding.decode()
                if encoding != "csr_matrix":
                    raise NotImplementedError(f"Only dense or CSR h5ad X is supported, got {encoding!r}")
                self._x_reader = _CSRX(x)

    def _find_nonzero_rows(self, x_obj, obs_names, celltypes_all):
        if isinstance(x_obj, h5py.Dataset):
            row_sums = np.asarray(x_obj[:], dtype=np.float32).sum(axis=1)
        else:
            indptr = np.asarray(x_obj["indptr"][:], dtype=np.int64)
            row_sums = np.zeros(len(indptr) - 1, dtype=np.float64)
            data = x_obj["data"]
            for row in range(len(row_sums)):
                row_sums[row] = np.asarray(data[indptr[row]:indptr[row + 1]], dtype=np.float32).sum()

        keep = row_sums > 0
        n_removed = int((~keep).sum())
        if n_removed > 0:
            logger.warning(f"Filtered {n_removed} zero-count cell(s) from {self.h5ad_path}")
        kept_rows = np.where(keep)[0].astype(np.int64)
        kept_ids = obs_names[keep].astype(str).tolist()
        kept_celltypes = celltypes_all[keep].astype(str).tolist() if celltypes_all is not None else None
        return kept_rows, kept_ids, kept_celltypes

    def _raw_expr_by_dataset_row(self, dataset_row: int) -> np.ndarray:
        self._open_h5()
        raw = self._x_reader.get_row(int(self.h5_rows[dataset_row]))
        aligned = np.zeros(self.n_genes, dtype=np.float32)
        aligned[self.target_idx] = raw[self.source_idx]
        return aligned

    def _raw_h5ad_row(self, dataset_row: int) -> np.ndarray:
        self._open_h5()
        return self._x_reader.get_row(int(self.h5_rows[dataset_row]))

    def __len__(self) -> int:
        return len(self.cell_ids)

    def __getitem__(self, i: int) -> dict:
        if self.embedding_rows is not None:
            if self._preloaded_row_map is not None:
                local_row = int(self._preloaded_row_map[i])
                if local_row < 0:
                    raise RuntimeError(
                        f"dataset row {i} was not included in rank-local preload"
                    )
                x = self._preloaded_embeddings[local_row]
            else:
                self._open_memmaps()
                x = np.asarray(self.embeddings[int(self.embedding_rows[i])], dtype=np.float32)
        else:
            expr = self._raw_expr_by_dataset_row(i)
            total_count = float(expr.sum())
            normalized = np.log1p(expr / total_count * 1e4).astype(np.float32)
            S = np.float32(np.log10(total_count))
            T = np.float32(S + 5.0)
            x = np.concatenate([normalized, np.array([T, S], dtype=np.float32)]).astype(np.float32)

        if self.target_npy_path is not None:
            if self._preloaded_row_map is not None:
                local_row = int(self._preloaded_row_map[i])
                target = self._preloaded_targets[local_row]
            else:
                self._open_memmaps()
                row = int(self.target_rows[i]) if self.target_rows is not None else i
                target = np.asarray(self.targets[row], dtype=np.float32)
        else:
            expr = self._raw_h5ad_row(i)
            total_count = float(expr.sum())
            normalized = np.log1p(expr / total_count * 1e4).astype(np.float32)
            if self.predict_genes == "hvg":
                target = normalized[self.hvg_indices]
            else:
                target = normalized

        result = {
            "x": torch.tensor(x, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "cell_id": self.cell_ids[i],
        }
        if self.celltype_to_index is not None:
            result["celltype_label"] = torch.tensor(
                self.celltype_to_index[self.celltypes[i]], dtype=torch.long
            )
        return result
