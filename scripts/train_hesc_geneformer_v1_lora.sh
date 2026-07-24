#!/usr/bin/env bash
set -euo pipefail

ARTIFACT="${ARTIFACT:-./artifacts/hESC_500_geneformer_v1_tokens.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-./out/hesc_geneformer_v1_lora_r8}"
SCFM_GENE_POOLING="${SCFM_GENE_POOLING:-mean}"

if [[ ! -f "$ARTIFACT" ]]; then
  echo "Missing token artifact: $ARTIFACT" >&2
  echo "Run src/prepare_geneformer_tokens.py first." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR/ckpt"

python src/train_cell_guided_graph.py \
  --device cuda \
  --cell_type hESC \
  --num_TF 500 \
  --llm_type Geneformer \
  --gnn_type GCN \
  --scfm_mode online_lora \
  --scfm_model_repo ctheodoris/Geneformer \
  --scfm_model_subfolder Geneformer-V1-10M \
  --scfm_model_version V1 \
  --scfm_tokenized_path "$ARTIFACT" \
  --scfm_gene_pooling "${SCFM_GENE_POOLING}" \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_target_modules query,value \
  --scfm_lr 1e-4 \
  --scfm_weight_decay 0.0 \
  --downstream_lr 1e-3 \
  --downstream_weight_decay 1e-4 \
  --max_scfm_cells 64 \
  --scfm_cell_batch_size 4 \
  --scfm_cell_sampling fixed_random \
  --output_dir "$OUTPUT_DIR" \
  --ckpt_dir "$OUTPUT_DIR/ckpt" \
  --gnn_epochs 15 \
  --gnn_eval_interval 1 \
  --latent_dim 128 \
  --graph_alpha 0.8 \
  --lambda_sparse 0.0 \
  --graph_constructor_type mlp \
  --graph_fusion_type fixed \
  --amp \
  --grad_clip_norm 1.0 \
  --early_stop_metric auprc \
  --patience 5
