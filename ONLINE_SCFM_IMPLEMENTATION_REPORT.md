# Online scFM Implementation Report

## 1. Final model data flow

`BL--ExpressionData.csv` → custom rank-based 910-gene token artifact → actual
Geneformer token forward → `gene_index_map` scatter pooling → 910 gene
representations (trainable fallback for unobserved genes) → Cell-M →
GraphConstructor/Graph-M → DirectedDecoder → TF-target BCE.

`online_lora` and `online_topk` keep this graph connected to trainable backbone
parameters. `precomputed` remains the CSV baseline.

## 2. Modified and added files

- `src/geneformer_assets.py`
- `src/prepare_geneformer_tokens.py`
- `src/scfm_encoder.py`
- `src/args.py`
- `src/models_cell_guided_graph.py`
- `src/train_cell_guided_graph.py`
- `src/train_condition_joint.py`
- `requirements-online-scfm.txt`
- `scripts/smoke_hesc_geneformer_lora.sh`
- `scripts/train_hesc_geneformer_v1_lora.sh`
- online asset/tokenization/pooling tests under `tests/`
- `README.md`

## 3. Token artifact format

The version-1 `torch.save` dictionary contains `input_ids`, `attention_mask`,
`gene_index_map`, original cell/gene identities, `gene_token_ids`,
`tokenizable_gene_mask`, and preprocessing metadata/fingerprints. Padding
positions use `gene_index_map=-1`.

## 4. Invalid and non-tokenizable genes

Only IDs matching `^ENSG[0-9]{11}$` and present in the Geneformer token
dictionary are ranked. ERCC/`ENSG` placeholders and absent dictionary entries
are excluded before sorting and recorded in metadata. They remain in the
downstream 910-gene space through trainable fallback embeddings.

## 5. 910-gene index preservation

Expression and mapping row order must match exactly. Each token stores its
original GRN row index. Pooling scatters hidden states by this index and always
returns `[original_gene_count, hidden_size]`. TF/train/valid/test indices are
range-checked without remapping.

## 6. Geneformer identity

Default: `ctheodoris/Geneformer`, subfolder `Geneformer-V1-10M`, V1 maximum
length 2048, no manually inserted special tokens. Local checkpoints take
precedence. Hidden size is read from model config.

## 7. LoRA configuration

Default PEFT configuration: rank 8, alpha 16, dropout 0.05, targets
`query,value`, bias none, task type `FEATURE_EXTRACTION`. Zero matched/trainable
LoRA parameters are fatal.

## 8. Optimizer groups

Downstream/Cell-M/fallback parameters use downstream LR and weight decay.
LoRA or unfrozen top-layer backbone parameters use scFM LR and weight decay.
Duplicate IDs and missing trainable parameters are fatal.

## 9. Gradient checks

After the normal first backward and AMP unscale, training requires finite,
non-zero downstream gradients. LoRA/top-k additionally require finite,
non-zero trainable backbone gradients. No helper invokes backward.

## 10. Unit and integration tests

`python -m unittest discover -s tests -v`: **PASS (21 tests at the time of this
report)**. Tests cover assets, deterministic rank tokenization, scatter pooling,
fallback gradients, fixed/edge graph paths, and a synthetic online top-layer
backbone-to-link-loss gradient.

`pytest` is not installed in the current environment, so the equivalent
standard-library unittest suite was used.

## 11. Actual V1 smoke result

**SKIPPED/BLOCKED.** The current environment has no `transformers`, `peft`,
`huggingface_hub`, local Geneformer checkpoint, or local token dictionary.
Running the smoke script correctly stops with the actionable missing
`huggingface_hub`/token-dictionary error. Actual V1 LoRA success is not claimed.

## 12. Precomputed regression

**PASS.** A real hESC one-epoch CPU run completed with 910×896 precomputed
input, downstream backward, checkpoint save, evaluation, and checkpoint resume.

## 13. Commands

```bash
pip install -r requirements-online-scfm.txt
python src/prepare_geneformer_tokens.py \
  --expression_path ./data/hESC/TFs+500/BL--ExpressionData.csv \
  --mapping_path ./scFM/Geneformer/hESC_500.csv \
  --output_path ./artifacts/hESC_500_geneformer_v1_tokens.pt \
  --scfm_model_version V1
bash scripts/smoke_hesc_geneformer_lora.sh
bash scripts/train_hesc_geneformer_v1_lora.sh
```

## 14. Known limitation

The benchmark is a normalized, feature-selected 910-gene matrix, not raw
full-transcriptome counts. Tokenization is therefore a custom deterministic
rank-based procedure over available genes, not the official Geneformer
TranscriptomeTokenizer pipeline.

## 15. Remaining blockers

- Install optional online dependencies.
- Download/provide the V1-10M checkpoint and 30M token dictionary.
- Generate the real hESC token artifact.
- Run and verify actual V1 LoRA gradients and online checkpoint round-trip on
  the target GPU server.
- Online checkpoints currently prioritize straightforward state-dict
  reproducibility and may include frozen base-model tensors; adapter-only PEFT
  checkpoint compaction remains a follow-up after the real V1 round-trip test.
