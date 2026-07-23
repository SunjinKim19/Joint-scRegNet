# scRegNet: Prediction of Gene Regulatory Connections with Joint Single-Cell Foundation Models and Graph-Based Learning

We provide PyTorch implementation for scRegNet that combines single-cell foundation models and graph-based learning to predict gene regulatory connections.

<p align="center">
  <img src="./figs/Overview.jpg" width="1000" title="scRegNet framework overview" alt="">
</p>

## Installation

For training, a GPU is strongly recommended.

#### PyTorch

The code is based on PyTorch. You can find installation instructions [here](http://pytorch.org/).

#### Dependencies
* Python == 3.10
* PyTorch == 2.4.1
* scikit-learn == 1.5.2
* numpy == 1.20.3
* optuna == 4.0.0

[Optional] We recommend using [wandb](https://wandb.ai/) for logging and visualization.

```bash
pip install wandb
```
**Note: PyTorch 2.4.1 and CUDA 12.4 were used during development.**

## Desription

We use seven publicly available scRNA-seq benchmark datasets by BEELINE (Pratapa et al., 2020) for gene regulatory link prediction from single-cell transcriptomic data. We use the same data split in paper [GENELink](https://github.com/zpliulab/GENELink/tree/main) for a fair comparision. The repository is organised as follows:

* data/: contains the benchmark datasets for running demo experiments
* out/: contains our trained model weights for scRegNet(w/ Geneformer) using GCN as the GNN backbone.
* src/: contains our source code for scRegNet.
  * inference.py: evaluation code for gene regulatory link prediction.
  * models.py: contains our model scRegNet.  
  * utils.py: contains tool functions for preprocessing data, and metrics for evaluation, etc.
  * train.py: code for training a new model.
  * optuna/: sub-directory of codes for hyperparameter tuning using optuna
* scFM/: contains the gene level features extracted from single-cell foundation models. You can download the Geneformer embeddings for demo experiments from [here](https://drive.google.com/drive/folders/1xnh4ixJwx1kzmO98FmGUvy5S7uqLW-yR?usp=sharing)


## Running experiments

### Demo
```bash
$ git clone this-repo-url
$ cd scRegNet
$ python src/inference.py
```

### Train
```bash
$ bash gnn_hp.sh tf_500_mDC GCN mDC 500 Geneformer
```

### Serial cell-guided graph model

`train_cell_guided_graph.py` keeps the legacy joint model intact and trains the
serial path `Cell-M -> GraphConstructor -> Graph-M -> DirectedDecoder`. The
context adjacency remains soft during training and is mixed with a prior built
from positive training-subset edges only; validation and test labels are never
used to construct it. For stronger leakage isolation, an external biological
prior should replace this train-derived prior in future experiments.

One-epoch smoke run:

```bash
python src/train_cell_guided_graph.py \
  --device auto \
  --cell_type hESC \
  --num_TF 500 \
  --llm_type Geneformer \
  --gnn_type GCN \
  --output_dir ./out/cell_guided_graph_hesc_smoke3 \
  --ckpt_dir ./out/cell_guided_graph_hesc_smoke3/ckpt \
  --gnn_epochs 1 \
  --gnn_eval_interval 1 \
  --latent_dim 128 \
  --graph_alpha 0.8 \
  --lambda_sparse 0.0 \
  --graph_constructor_type mlp \
  --early_stop_metric auprc \
  --patience 3
```

`--hard_topk_eval_only K` optionally applies a hard TF-row top-k view only
during validation/test. It is never inserted into the training forward path.

The original fixed-alpha fusion remains the default. A GPU baseline can be run
with:

```bash
python src/train_cell_guided_graph.py \
  --device cuda \
  --cell_type hESC \
  --num_TF 500 \
  --llm_type Geneformer \
  --gnn_type GCN \
  --output_dir ./out/cell_guided_graph_hesc_gpu_fixed_alpha08 \
  --ckpt_dir ./out/cell_guided_graph_hesc_gpu_fixed_alpha08/ckpt \
  --gnn_epochs 5 \
  --gnn_eval_interval 1 \
  --latent_dim 128 \
  --graph_alpha 0.8 \
  --lambda_sparse 0.0 \
  --graph_constructor_type mlp \
  --graph_fusion_type fixed \
  --early_stop_metric auprc \
  --patience 3
```

For adaptive edge-wise prior/context fusion:

```bash
python src/train_cell_guided_graph.py \
  --device cuda \
  --cell_type hESC \
  --num_TF 500 \
  --llm_type Geneformer \
  --gnn_type GCN \
  --output_dir ./out/cell_guided_graph_hesc_gpu_edge_gate_alpha08 \
  --ckpt_dir ./out/cell_guided_graph_hesc_gpu_edge_gate_alpha08/ckpt \
  --gnn_epochs 15 \
  --gnn_eval_interval 1 \
  --latent_dim 128 \
  --graph_alpha 0.8 \
  --lambda_sparse 0.0 \
  --graph_constructor_type mlp \
  --graph_fusion_type edge_gate \
  --gate_hidden_dim 32 \
  --gate_dropout 0.0 \
  --gate_temperature 1.0 \
  --gate_init_from_alpha \
  --early_stop_metric auprc \
  --patience 5
```

### True scFM fine-tuning

The repository's original Geneformer path uses precomputed CSV gene embeddings;
it does not load or fine-tune a Geneformer backbone. The explicit modes are:

- `precomputed`: existing CSV baseline; no scFM backbone is loaded.
- `online_frozen`: run a real HuggingFace-compatible scFM checkpoint online,
  with all backbone parameters frozen.
- `online_lora`: inject PEFT LoRA modules and train them with the link loss.
- `online_topk`: unfreeze only the requested top transformer layers.

Online modes require both a checkpoint (`--scfm_model_path`) and tokenized cell
input (`--scfm_tokenized_path`). The tokenized `.pt`, `.pth`, or `.pkl` file
must be a dictionary containing `input_ids`, optionally `attention_mask` and
`token_type_ids`, and one token-to-gene mapping: `token_gene_indices`,
`gene_indices`, or `gene_ids`. Direct mappings use downstream 0-based gene
indices. Vocabulary-style `gene_ids` additionally require `target_gene_ids` in
the downstream 910-gene order. Training fails rather than inventing a mapping
when these identities are unavailable.

Recommended first online GPU check:

```bash
python src/train_cell_guided_graph.py \
  --device cuda \
  --cell_type hESC \
  --num_TF 500 \
  --llm_type Geneformer \
  --gnn_type GCN \
  --scfm_mode online_frozen \
  --scfm_model_path <PATH_OR_MODEL_NAME> \
  --scfm_tokenized_path <TOKENIZED_INPUT_PATH> \
  --output_dir ./out/ft_smoke_online_frozen \
  --ckpt_dir ./out/ft_smoke_online_frozen/ckpt \
  --gnn_epochs 1 \
  --gnn_eval_interval 1 \
  --latent_dim 128 \
  --graph_alpha 0.8 \
  --graph_fusion_type fixed \
  --early_stop_metric auprc \
  --patience 1
```

Then verify true fine-tuning with LoRA:

```bash
python src/train_cell_guided_graph.py \
  --device cuda \
  --cell_type hESC \
  --num_TF 500 \
  --llm_type Geneformer \
  --gnn_type GCN \
  --scfm_mode online_lora \
  --scfm_model_path <PATH_OR_MODEL_NAME> \
  --scfm_tokenized_path <TOKENIZED_INPUT_PATH> \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --scfm_lr 1e-5 \
  --downstream_lr 1e-3 \
  --output_dir ./out/ft_smoke_lora_r8 \
  --ckpt_dir ./out/ft_smoke_lora_r8/ckpt \
  --gnn_epochs 1 \
  --gnn_eval_interval 1 \
  --latent_dim 128 \
  --graph_alpha 0.8 \
  --graph_fusion_type fixed \
  --early_stop_metric auprc \
  --patience 1
```

Top-layer fine-tuning uses `--scfm_mode online_topk` with
`--train_scfm_top_layers 1`. For memory control use `--max_scfm_cells`,
`--amp`, and `--grad_clip_norm`. Online LoRA/top-k outputs are never cached or
detached; the first training batch checks for a non-zero scFM gradient after
the normal backward pass.

## Acknowledgements

We sincerely thank the authors of following open-source projects:

- [Geneformer](https://huggingface.co/ctheodoris/Geneformer)
- [scFoundation](https://github.com/biomap-research/scFoundation)
- [scBERT](https://github.com/TencentAILabHealthcare/scBERT)
- [Optuna](https://github.com/optuna/optuna)
- [BEELINE](https://github.com/Murali-group/Beeline)

## Citation
If you find this repository useful, please cite the following paper:
```
@article {Kommu2024.12.16.628715,
	author = {Kommu, Sindhura and Wang, Yizhi and Wang, Yue and Wang, Xuan},
	title = {Prediction of Gene Regulatory Connections with Joint Single-Cell Foundation Models and Graph-Based Learning},
	elocation-id = {2024.12.16.628715},
	year = {2025},
	doi = {10.1101/2024.12.16.628715},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/01/29/2024.12.16.628715},
	eprint = {https://www.biorxiv.org/content/early/2025/01/29/2024.12.16.628715.full.pdf},
	journal = {bioRxiv}
}
```
