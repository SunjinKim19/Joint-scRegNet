#!/usr/bin/env bash
set -euo pipefail

ARTIFACT="${ARTIFACT:-./artifacts/hESC_500_geneformer_v1_tokens.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-./out/ft_smoke_geneformer_v1_lora}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "$(dirname "$ARTIFACT")" "$OUTPUT_DIR/ckpt"

if [[ ! -f "$ARTIFACT" ]]; then
  python src/prepare_geneformer_tokens.py \
    --expression_path ./data/hESC/TFs+500/BL--ExpressionData.csv \
    --mapping_path ./scFM/Geneformer/hESC_500.csv \
    --output_path "$ARTIFACT" \
    --scfm_model_version V1
fi

python src/train_cell_guided_graph.py \
  --device "$DEVICE" \
  --cell_type hESC \
  --num_TF 500 \
  --llm_type Geneformer \
  --gnn_type GCN \
  --scfm_mode online_lora \
  --scfm_model_repo ctheodoris/Geneformer \
  --scfm_model_subfolder Geneformer-V1-10M \
  --scfm_model_version V1 \
  --scfm_tokenized_path "$ARTIFACT" \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_target_modules query,value \
  --scfm_lr 1e-4 \
  --downstream_lr 1e-3 \
  --max_scfm_cells 8 \
  --scfm_cell_batch_size 2 \
  --scfm_cell_sampling fixed_random \
  --limit_train_edges 128 \
  --limit_valid_edges 128 \
  --limit_test_edges 128 \
  --output_dir "$OUTPUT_DIR" \
  --ckpt_dir "$OUTPUT_DIR/ckpt" \
  --gnn_epochs 1 \
  --gnn_eval_interval 1 \
  --latent_dim 128 \
  --graph_alpha 0.8 \
  --graph_fusion_type fixed \
  --amp \
  --grad_clip_norm 1.0 \
  --early_stop_metric auprc \
  --patience 1

test -f "$OUTPUT_DIR/ckpt/cell_guided_graph_seed0.pt"
