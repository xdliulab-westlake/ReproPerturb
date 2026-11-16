# Stage-1: cell representation

- `model.py`: VQVAE architecture.
- `components/encoders/scfoundation_encoder.py`: scFoundation feature encoder.
- `components/quantizers/soft_vq_quantizer.py`: adopted SoftVQ quantizer.
- `components/decoders/mlp_decoder.py`: expression reconstruction decoder.
- `train.py`: `train_vqvae_scrna` training implementation.
- `evaluation/`: validation and reconstruction losses.
- `registry.py`: model construction. Other registry components remain because the loader imports them.

Run `torchrun --standalone --nproc_per_node=1 train_stage1.py` from the repository root after adapting `src/configs/stage1/stage1_softvq_hypo_hvg2000_celltype_updated_v7.py` and providing external data.
