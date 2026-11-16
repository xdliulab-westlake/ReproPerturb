"""Stage-1 SoftVQ-VAE on HVG2000 with v7 cell-type CE supervision."""


CONFIG = {
    "encoder": {
        "name": "scfoundation",
        "ckpt_path": "/liaozizhuo/iPStem/data/iPStem/models.ckpt",
        "gene_index_path": (
            "/liaozizhuo/iPStem/data/iPStem/OS_scRNA_gene_index.19264.tsv"
        ),
        "mlp_dims": [1024, 256, 128],
        "use_precomputed": True,
        "use_attention": False,
    },
    "pre_quantization": {
        "flat": True,
        "in_features": 128,
        "embedding_dim": 128,
    },
    "decoder": {
        "name": "mlp",
        "embedding_dim": 128,
        "hidden_dims": [256, 1024],
        "out_dim": -1,
    },
    "quantizer": {
        "name": "soft_vq",
        "n_embeddings": 128,
        "embedding_dim": 32,
        "entropy_loss_ratio": 0.01,
        "tau": 0.07,
        "num_codebooks": 4,
        "l2_norm": True,
        "show_usage": False,
        "latent": {"format": "sequence", "projection": "reshape"},
        "loss": {"embedding": 1.0},
    },
    "data": {
        "h5ad_path": "/liaozizhuo/hypoblast/data/hypo/hypo_final_20260313.h5ad",
        "precomputed_only": True,
        "preload_rank_shards": True,
        "predict_genes": "hvg",
        "hvg_genes_path": (
            "/liaozizhuo/hypoblast/data/hypo/hypo_hvg_top2000.txt"
        ),
        "tgthighres": "a5",
        "embedding_npy_path": (
            "/liaozizhuo/hypoblast/output/"
            "scfoundation_hypo_final_20260313_full/cell_embeddings_3072.npy"
        ),
        "embedding_meta_csv": (
            "/liaozizhuo/hypoblast/output/"
            "scfoundation_hypo_final_20260313_full/metadata_updated_v7.csv"
        ),
        "target_npy_path": (
            "/liaozizhuo/hypoblast/output/"
            "scfoundation_hypo_final_20260313_full/targets_hvg2000.npy"
        ),
        "celltype_key": "celltype",
        "recon_target": "mse",
        "pca_n_components": 50,
        "pca_fit_n": 50000,
        "split_seed": 42,
        "test_group_key": "condition",
        "test_group_values": ["H226", "H227", "H240"],
        "test_fraction": 0.1,
        "val_fraction": 0.1,
    },
    "training": {
        "batch_size": 64,
        "lr": 5e-4,
        "epochs_vqvae": 2000,
        "num_workers": 4,
        "early_stop_patience": 30,
        "checkpoint_path": (
            "/liaozizhuo/hypoblast/results/"
            "stage1_softvq_hypo_hvg2000_celltype_updated_v7_best.pt"
        ),
        "log_dir": (
            "/liaozizhuo/hypoblast/logs/"
            "stage1_softvq_hypo_hvg2000_celltype_updated_v7"
        ),
        "split_path": (
            "/liaozizhuo/hypoblast/output/"
            "scfoundation_hypo_final_20260313_full/"
            "vqvae_scrna_split_hypo_hvg2000_metadata_v7.npz"
        ),
    },
    "codebook_init": {},
    "celltype_classifier": {"num_classes": 10, "loss_weight": 0.25},
}
