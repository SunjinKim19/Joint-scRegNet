"""Explicit precomputed and online scFM encoding paths.

Online modes require a real HuggingFace-compatible checkpoint plus tokenized
cell sequences carrying a token-to-downstream-gene mapping. This module never
silently substitutes precomputed CSV embeddings for requested fine-tuning.
"""

import logging
import os
import pickle
import hashlib

import torch
import torch.nn as nn

from src.geneformer_assets import resolve_geneformer_assets

logger = logging.getLogger(__name__)


class GeneRepresentationPooler(nn.Module):
    """Scatter token hidden states back to the immutable GRN gene index."""

    def __init__(self, num_genes, hidden_dim):
        super().__init__()
        self.num_genes = num_genes
        self.hidden_dim = hidden_dim
        self.fallback_gene_embeddings = nn.Embedding(num_genes, hidden_dim)
        nn.init.normal_(self.fallback_gene_embeddings.weight, std=0.02)

    def accumulate(self, hidden, attention_mask, gene_index_map):
        if hidden.dim() != 3:
            raise ValueError(f"hidden must be [B, L, H], got {tuple(hidden.shape)}")
        if attention_mask.shape != hidden.shape[:2]:
            raise ValueError("attention_mask shape must match hidden [B, L]")
        if gene_index_map.shape != hidden.shape[:2]:
            raise ValueError("gene_index_map shape must match hidden [B, L]")
        valid = attention_mask.bool() & gene_index_map.ge(0)
        valid = valid & gene_index_map.lt(self.num_genes)
        flat_indices = gene_index_map[valid].long()
        flat_hidden = hidden[valid]
        gene_sum = hidden.new_zeros((self.num_genes, self.hidden_dim))
        gene_count = hidden.new_zeros((self.num_genes, 1))
        gene_sum.index_add_(0, flat_indices, flat_hidden)
        gene_count.index_add_(
            0,
            flat_indices,
            torch.ones(
                (flat_indices.numel(), 1),
                dtype=hidden.dtype,
                device=hidden.device,
            ),
        )
        return gene_sum, gene_count

    def finalize(self, gene_sum, gene_count):
        pooled = gene_sum / gene_count.clamp_min(1)
        observed = gene_count.squeeze(-1).gt(0)
        return torch.where(
            observed.unsqueeze(-1), pooled, self.fallback_gene_embeddings.weight
        )

    def forward(self, hidden, attention_mask, gene_index_map):
        return self.finalize(
            *self.accumulate(hidden, attention_mask, gene_index_map)
        )


class ScFMEncoder(nn.Module):
    ONLINE_MODES = {"online_frozen", "online_lora", "online_topk"}
    MODEL_INPUT_KEYS = ("input_ids", "attention_mask", "token_type_ids")
    GENE_MAPPING_KEYS = (
        "gene_index_map",
        "token_gene_indices",
        "gene_indices",
        "gene_ids",
    )

    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.mode = args.scfm_mode
        self.device = device
        self.model = None
        self.tokenized_data = None
        self.output_dim = None
        self.num_original_genes = None
        self.gene_pooler = None
        self._cached_gene_embeddings = None
        self.last_diagnostics = {}
        self.assets = None
        self.artifact_fingerprint = None
        self.artifact_metadata = {}

        if self.mode == "precomputed":
            logger.info(
                "scFM mode=precomputed: using frozen gene embeddings; no scFM "
                "backbone is loaded."
            )
            return
        if self.mode not in self.ONLINE_MODES:
            raise ValueError(f"Unsupported scfm_mode: {self.mode}")
        self._validate_online_paths()
        self.assets = resolve_geneformer_assets(
            args, require_token_dictionary=False
        )
        self.tokenized_data = self._load_tokenized_data(args.scfm_tokenized_path)
        self.artifact_fingerprint = self._sha256(args.scfm_tokenized_path)
        self.artifact_metadata = dict(self.tokenized_data.get("metadata", {}))
        self.model = self._load_huggingface_model(self.assets, args.hf_cache_dir)
        self.model.to(device)
        self.output_dim = self._infer_hidden_size()
        self.num_original_genes = self._infer_original_gene_count()
        self.gene_pooler = GeneRepresentationPooler(
            self.num_original_genes, self.output_dim
        ).to(device)
        self._configure_trainable_parameters()
        self._configure_gradient_checkpointing()
        self._validate_cache_policy()
        self.selected_cell_indices = self._select_cells()
        self._log_parameter_summary()

    @property
    def backbone_loaded(self):
        return self.model is not None

    def _validate_online_paths(self):
        missing = []
        if not self.args.scfm_model_path and not self.args.scfm_model_repo:
            missing.append("--scfm_model_path or --scfm_model_repo")
        if not self.args.scfm_tokenized_path:
            missing.append("--scfm_tokenized_path (tokenized cell input)")
        if missing:
            raise ValueError(
                f"scfm_mode={self.mode} requires true online scFM inputs. Missing: "
                + ", ".join(missing)
                + ". Precomputed CSV embeddings cannot be used as a substitute."
            )
        if not os.path.exists(self.args.scfm_tokenized_path):
            raise ValueError(
                "Tokenized scFM input was not found at "
                f"'{self.args.scfm_tokenized_path}'. Provide a .pt/.pth, .pkl, "
                "or HuggingFace Dataset directory containing input_ids and a "
                "token-to-gene mapping."
            )

    @staticmethod
    def _load_huggingface_model(assets, cache_dir):
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError(
                "Online scFM modes require HuggingFace transformers. Install it "
                "with `pip install transformers`, then provide a compatible "
                "--scfm_model_path."
            ) from exc
        try:
            if assets.model_path:
                return AutoModel.from_pretrained(
                    assets.model_path, cache_dir=cache_dir
                )
            return AutoModel.from_pretrained(
                assets.model_repo,
                subfolder=assets.model_subfolder,
                cache_dir=cache_dir,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load the scFM model from '{assets.model_identity}' with "
                "transformers.AutoModel. Verify that it is a local checkpoint or "
                "a reachable HuggingFace model and that its custom dependencies "
                f"are installed. Original error: {exc}"
            ) from exc

    @staticmethod
    def _load_tokenized_data(path):
        suffix = os.path.splitext(path)[1].lower()
        try:
            if suffix in (".pt", ".pth"):
                try:
                    data = torch.load(path, map_location="cpu", weights_only=False)
                except TypeError:
                    data = torch.load(path, map_location="cpu")
            elif suffix in (".pkl", ".pickle"):
                with open(path, "rb") as handle:
                    data = pickle.load(handle)
            elif os.path.isdir(path):
                try:
                    from datasets import load_from_disk
                except ImportError as exc:
                    raise ImportError(
                        "Loading a HuggingFace Dataset directory requires "
                        "`pip install datasets`. Alternatively provide a .pt file."
                    ) from exc
                dataset = load_from_disk(path)
                data = {column: dataset[column] for column in dataset.column_names}
            else:
                raise ValueError(
                    "Unsupported tokenized input format. Expected .pt/.pth, "
                    ".pkl/.pickle, or a HuggingFace Dataset directory."
                )
        except (ImportError, ValueError):
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load tokenized scFM input from '{path}': {exc}"
            ) from exc

        if isinstance(data, dict) and isinstance(data.get("batch"), dict):
            data = data["batch"]
        if not isinstance(data, dict):
            raise ValueError(
                "Tokenized scFM input must load as a dictionary. "
                f"Received {type(data).__name__}."
            )
        available = sorted(data.keys())
        if "input_ids" not in data:
            raise ValueError(
                "Tokenized input is missing required key 'input_ids'. "
                f"Available keys: {available}"
            )
        if not any(key in data for key in ScFMEncoder.GENE_MAPPING_KEYS):
            raise ValueError(
                "Cannot construct gene-level embeddings because token-to-gene "
                "mapping is missing. Provide one of "
                f"{ScFMEncoder.GENE_MAPPING_KEYS}. Available keys: {available}"
            )
        normalized = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                normalized[key] = value
            elif key in (
                *ScFMEncoder.MODEL_INPUT_KEYS,
                *ScFMEncoder.GENE_MAPPING_KEYS,
                "target_gene_ids",
            ):
                try:
                    normalized[key] = ScFMEncoder._to_tensor_with_padding(
                        value,
                        padding_value=-1
                        if key in ScFMEncoder.GENE_MAPPING_KEYS
                        else 0,
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Tokenized key '{key}' could not be converted to a tensor."
                    ) from exc
            else:
                normalized[key] = value
        return normalized

    @staticmethod
    def _sha256(path):
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _to_tensor_with_padding(value, padding_value=0):
        try:
            return torch.as_tensor(value)
        except (TypeError, ValueError):
            if not isinstance(value, (list, tuple)) or not value:
                raise
            rows = [torch.as_tensor(row) for row in value]
            if any(row.dim() != 1 for row in rows):
                raise ValueError("Only one-dimensional variable-length rows are supported")
            return nn.utils.rnn.pad_sequence(
                rows, batch_first=True, padding_value=padding_value
            )

    def _infer_hidden_size(self):
        config = getattr(self.model, "config", None)
        for name in ("hidden_size", "d_model", "n_embd"):
            value = getattr(config, name, None)
            if isinstance(value, int) and value > 0:
                return value
        raise ValueError(
            "Could not infer the scFM output dimension from model.config. "
            "Expected hidden_size, d_model, or n_embd."
        )

    def _infer_original_gene_count(self):
        metadata_count = self.tokenized_data.get("metadata", {}).get(
            "original_gene_count"
        )
        candidates = [
            metadata_count,
            len(self.tokenized_data.get("gene_symbols", [])) or None,
            (
                int(self.tokenized_data["gene_token_ids"].numel())
                if isinstance(self.tokenized_data.get("gene_token_ids"), torch.Tensor)
                else None
            ),
        ]
        mapping_key = next(
            key for key in self.GENE_MAPPING_KEYS if key in self.tokenized_data
        )
        mapping = self.tokenized_data[mapping_key]
        valid = mapping[mapping >= 0]
        candidates.append(int(valid.max()) + 1 if valid.numel() else None)
        count = next(
            (int(value) for value in candidates if value is not None and int(value) > 0),
            None,
        )
        if count is None:
            raise ValueError("Could not infer original gene count from token artifact")
        return count

    def _configure_gradient_checkpointing(self):
        if not self.args.gradient_checkpointing:
            return
        enable = getattr(self.model, "gradient_checkpointing_enable", None)
        if not callable(enable):
            raise ValueError(
                "--gradient_checkpointing was requested but this scFM model "
                "does not expose gradient_checkpointing_enable()."
            )
        enable()
        logger.info("Enabled scFM gradient checkpointing")

    def _select_cells(self):
        cell_count = self.tokenized_data["input_ids"].size(0)
        maximum = self.args.max_scfm_cells
        selected_count = cell_count if maximum <= 0 else min(maximum, cell_count)
        if self.args.scfm_cell_sampling == "all":
            indices = torch.arange(selected_count)
        else:
            generator = torch.Generator().manual_seed(self.args.scfm_seed)
            indices = torch.randperm(cell_count, generator=generator)[:selected_count]
            indices = indices.sort().values
        logger.info(
            "Selected %d/%d scFM cells using %s (seed=%d)",
            selected_count,
            cell_count,
            self.args.scfm_cell_sampling,
            self.args.scfm_seed,
        )
        return indices

    def _configure_trainable_parameters(self):
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        if self.mode == "online_frozen":
            self.model.eval()
            logger.info("scFM mode=online_frozen; backbone parameters are frozen.")
            return
        if self.mode == "online_lora":
            self._enable_lora()
            return
        self._unfreeze_top_layers()

    def _enable_lora(self):
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise ImportError(
                "online_lora requires peft. Install with `pip install peft` or "
                "use --scfm_mode online_topk."
            ) from exc
        targets = [
            item.strip()
            for item in self.args.lora_target_modules.split(",")
            if item.strip()
        ]
        if not targets:
            raise ValueError("--lora_target_modules must contain at least one name")
        module_names = [name for name, _ in self.model.named_modules()]
        matched = [
            name
            for name in module_names
            if any(name.endswith(target) or target in name for target in targets)
        ]
        if not matched:
            candidates = sorted(
                name
                for name in module_names
                if any(
                    token in name.lower()
                    for token in (
                        "query",
                        "key",
                        "value",
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "attention",
                    )
                )
            )
            raise ValueError(
                "No modules matched --lora_target_modules="
                f"'{self.args.lora_target_modules}'. Candidate attention modules: "
                f"{candidates[:80]}"
            )
        config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=self.args.lora_r
            if self.args.lora_r is not None
            else self.args.lora_rank,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            target_modules=targets,
            bias="none",
        )
        self.model = get_peft_model(self.model, config)
        if not any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("PEFT created zero trainable LoRA parameters")
        logger.info(
            "LoRA enabled: rank=%d alpha=%d dropout=%.4f targets=%s",
            self.args.lora_r
            if self.args.lora_r is not None
            else self.args.lora_rank,
            self.args.lora_alpha,
            self.args.lora_dropout,
            targets,
        )

    def _unfreeze_top_layers(self):
        k = self.args.train_scfm_top_layers
        if k <= 0:
            raise ValueError(
                "scfm_mode=online_topk requires --train_scfm_top_layers > 0"
            )
        candidates = (
            ("encoder.layer", lambda model: getattr(model, "encoder", None)),
            ("bert.encoder.layer", lambda model: getattr(model, "bert", None)),
            ("transformer.h", lambda model: getattr(model, "transformer", None)),
            ("layers", lambda model: model),
        )
        layers = None
        selected_path = None
        for path, root_getter in candidates:
            root = root_getter(self.model)
            try:
                if path == "encoder.layer":
                    value = root.layer
                elif path == "bert.encoder.layer":
                    value = root.encoder.layer
                elif path == "transformer.h":
                    value = root.h
                else:
                    value = root.layers
            except AttributeError:
                continue
            if isinstance(value, (nn.ModuleList, list, tuple)):
                layers = value
                selected_path = path
                break
        if layers is None:
            top_level = [name for name, _ in self.model.named_children()]
            raise ValueError(
                "Could not locate transformer layers for online_topk. Tried "
                "model.encoder.layer, model.bert.encoder.layer, "
                "model.transformer.h, and model.layers. Top-level modules: "
                f"{top_level}"
            )
        if k > len(layers):
            raise ValueError(
                f"Requested top {k} layers but {selected_path} contains only "
                f"{len(layers)} layers."
            )
        for layer in layers[-k:]:
            for parameter in layer.parameters():
                parameter.requires_grad_(True)
        logger.info(
            "scFM mode=online_topk; unfreezing top %d transformer layer(s) at %s",
            k,
            selected_path,
        )

    def _validate_cache_policy(self):
        if (
            self.args.cache_online_scfm_outputs
            and self.mode in ("online_lora", "online_topk")
        ):
            raise ValueError(
                "--cache_online_scfm_outputs cannot be used with trainable scFM "
                f"mode {self.mode}: caching would reuse or detach the autograd graph."
            )

    def parameter_counts(self):
        if self.model is None:
            return 0, 0
        total = sum(parameter.numel() for parameter in self.model.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        return total, trainable

    def backbone_trainable_parameters(self):
        if self.model is None:
            return []
        return [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]

    def fallback_parameters(self):
        if self.gene_pooler is None:
            return []
        return list(self.gene_pooler.fallback_gene_embeddings.parameters())

    def _log_parameter_summary(self):
        total, trainable = self.parameter_counts()
        percentage = 100.0 * trainable / max(1, total)
        logger.info("scFM mode=%s", self.mode)
        logger.info("Loaded scFM model from %s", self.assets.model_identity)
        logger.info(
            "Actual scFM model class=%s hidden_size=%d token_artifact=%s "
            "original_gene_count=%d",
            self.model.__class__.__name__,
            self.output_dim,
            self.args.scfm_tokenized_path,
            self.num_original_genes,
        )
        logger.info(
            "Trainable scFM parameters: %d / %d (%.4f%%)",
            trainable,
            total,
            percentage,
        )

    def train(self, mode=True):
        super().train(mode)
        if self.mode == "online_frozen" and self.model is not None:
            self.model.eval()
        return self

    def _extract_hidden_state(self, outputs):
        output_layer = self.args.scfm_output_layer
        if output_layer == "last_hidden":
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None and isinstance(outputs, dict):
                hidden = outputs.get("last_hidden_state")
        elif output_layer.startswith("hidden_states:"):
            try:
                index = int(output_layer.split(":", 1)[1])
            except ValueError as exc:
                raise ValueError(
                    "--scfm_output_layer hidden-state syntax is "
                    "'hidden_states:<integer>'"
                ) from exc
            hidden_states = getattr(outputs, "hidden_states", None)
            if hidden_states is None and isinstance(outputs, dict):
                hidden_states = outputs.get("hidden_states")
            hidden = None if hidden_states is None else hidden_states[index]
        else:
            raise ValueError(
                "Unsupported --scfm_output_layer. Use 'last_hidden' or "
                "'hidden_states:<integer>'."
            )
        if hidden is None:
            if isinstance(outputs, dict):
                available = sorted(outputs.keys())
            else:
                available = [
                    name
                    for name in ("last_hidden_state", "hidden_states", "logits")
                    if hasattr(outputs, name)
                ]
            raise ValueError(
                f"scFM output does not provide '{output_layer}'. Available "
                f"outputs/attributes: {available}"
            )
        if hidden.dim() != 3:
            raise ValueError(
                "Expected token representations [cells, tokens, hidden], got "
                f"{tuple(hidden.shape)}"
            )
        return hidden

    def _direct_gene_indices(self, num_genes, cell_indices, token_count):
        mapping_key = next(
            key for key in self.GENE_MAPPING_KEYS if key in self.tokenized_data
        )
        mapping = self.tokenized_data[mapping_key].index_select(0, cell_indices)
        if mapping.shape != (cell_indices.numel(), token_count):
            raise ValueError(
                f"Token-to-gene mapping '{mapping_key}' must have shape "
                f"[{cell_indices.numel()}, {token_count}], got {tuple(mapping.shape)}"
            )
        mapping = mapping.long()
        if mapping_key == "gene_ids" and "target_gene_ids" in self.tokenized_data:
            targets = self.tokenized_data["target_gene_ids"].view(-1).tolist()
            if len(targets) != num_genes:
                raise ValueError(
                    "target_gene_ids length must equal downstream num_genes "
                    f"({num_genes}), got {len(targets)}"
                )
            lookup = {int(gene_id): index for index, gene_id in enumerate(targets)}
            flattened = mapping.view(-1).tolist()
            mapping = torch.tensor(
                [lookup.get(int(gene_id), -1) for gene_id in flattened],
                dtype=torch.long,
            ).view_as(mapping)
        valid_values = mapping[mapping >= 0]
        if valid_values.numel() and valid_values.max() >= num_genes:
            raise ValueError(
                f"Mapping '{mapping_key}' contains value "
                f"{int(valid_values.max())}, outside downstream gene range "
                f"[0, {num_genes - 1}]. If these are vocabulary IDs, also "
                "provide target_gene_ids for an explicit mapping."
            )
        return mapping

    def _pool_batch(self, hidden, num_genes, cell_indices):
        if self.args.scfm_pooling != "gene":
            raise ValueError(
                "CellGuidedGraphScRegNet requires gene-level [num_genes, dim] "
                f"embeddings, but --scfm_pooling={self.args.scfm_pooling} was "
                "requested. Use --scfm_pooling gene with a token-to-gene mapping."
            )
        _, tokens, _ = hidden.shape
        mapping = self._direct_gene_indices(
            num_genes, cell_indices, tokens
        ).to(hidden.device)
        attention_mask = self.tokenized_data.get("attention_mask")
        if attention_mask is None:
            attention_mask = mapping.ge(0)
        else:
            attention_mask = attention_mask.index_select(0, cell_indices)
        attention_mask = attention_mask.to(hidden.device)
        return self.gene_pooler.accumulate(hidden, attention_mask, mapping)

    def _model_forward_for_cells(self, cell_indices):
        model_inputs = {}
        for key in self.MODEL_INPUT_KEYS:
            value = self.tokenized_data.get(key)
            if value is not None:
                model_inputs[key] = value.index_select(0, cell_indices).to(self.device)
        try:
            outputs = self.model(
                **model_inputs, output_hidden_states=True, return_dict=True
            )
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError(
                "scFM forward ran out of GPU memory. Reduce --max_scfm_cells or "
                "--scfm_cell_batch_size, and prefer online_lora over online_topk."
            ) from exc
        return self._extract_hidden_state(outputs)

    def _pool_selected_cells(self, num_genes):
        if num_genes != self.num_original_genes:
            raise ValueError(
                "Token artifact original gene count does not match downstream "
                f"graph: {self.num_original_genes} vs {num_genes}. The 910-gene "
                "GRN index space must not be reordered or reduced."
            )
        batch_size = self.args.scfm_cell_batch_size
        if batch_size <= 0:
            raise ValueError("--scfm_cell_batch_size must be positive")
        total_sum = None
        total_count = None
        for cell_indices in self.selected_cell_indices.split(batch_size):
            hidden = self._model_forward_for_cells(cell_indices)
            batch_sum, batch_count = self._pool_batch(
                hidden, num_genes, cell_indices
            )
            total_sum = batch_sum if total_sum is None else total_sum + batch_sum
            total_count = (
                batch_count if total_count is None else total_count + batch_count
            )
        output = self.gene_pooler.finalize(total_sum, total_count)
        observed = total_count.squeeze(-1).gt(0)
        tokenizable = self.tokenized_data.get("tokenizable_gene_mask")
        tokenizable_count = (
            int(tokenizable.sum())
            if isinstance(tokenizable, torch.Tensor)
            else int(observed.sum())
        )
        self.last_diagnostics = {
            "pooled_gene_count": int(observed.sum()),
            "fallback_gene_count": int((~observed).sum()),
            "tokenizable_gene_count": tokenizable_count,
            "selected_cell_count": int(self.selected_cell_indices.numel()),
            "minimum_observation_count": float(total_count.min()),
            "maximum_observation_count": float(total_count.max()),
            "output_shape": tuple(output.shape),
        }
        logger.info("Gene pooling diagnostics: %s", self.last_diagnostics)
        return output

    def forward(self, context):
        if self.mode == "precomputed":
            embeddings = (
                context.get("precomputed_embeddings")
                if isinstance(context, dict)
                else context
            )
            if not isinstance(embeddings, torch.Tensor):
                raise ValueError(
                    "precomputed mode requires a precomputed_embeddings tensor"
                )
            return embeddings

        if self._cached_gene_embeddings is not None:
            return self._cached_gene_embeddings
        num_genes = context.get("num_genes") if isinstance(context, dict) else None
        if not isinstance(num_genes, int) or num_genes <= 0:
            raise ValueError("Online scFM forward requires integer context['num_genes']")
        gene_embeddings = self._pool_selected_cells(num_genes)
        if self.args.cache_online_scfm_outputs:
            # Only online_frozen reaches this branch; no trainable graph is detached.
            self._cached_gene_embeddings = gene_embeddings.detach()
        return gene_embeddings
