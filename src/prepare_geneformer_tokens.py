"""Custom rank-based tokenization over scRegNet's available 910 genes.

This intentionally does not claim to reproduce Geneformer's official raw-count,
full-transcriptome TranscriptomeTokenizer pipeline.
"""

import argparse
import hashlib
import json
import os
import pickle
import re
import sys
from types import SimpleNamespace

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

from src.geneformer_assets import get_model_spec, resolve_token_dictionary

ENSEMBL_PATTERN = re.compile(r"^ENSG[0-9]{11}$")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_token_dictionary(path):
    with open(path, "rb") as handle:
        dictionary = pickle.load(handle)
    if not isinstance(dictionary, dict):
        raise ValueError("Geneformer token dictionary must be a dictionary")
    return dictionary


def validate_token_artifact(artifact):
    required = {
        "input_ids",
        "attention_mask",
        "gene_index_map",
        "gene_symbols",
        "gene_token_ids",
        "tokenizable_gene_mask",
        "metadata",
    }
    missing = required - set(artifact)
    if missing:
        raise ValueError(f"Token artifact is missing keys: {sorted(missing)}")
    input_ids = artifact["input_ids"]
    attention = artifact["attention_mask"]
    mapping = artifact["gene_index_map"]
    if input_ids.dtype != torch.long or attention.dtype != torch.long:
        raise ValueError("input_ids and attention_mask must be LongTensor")
    if mapping.dtype != torch.long or input_ids.shape != attention.shape:
        raise ValueError("gene_index_map must be long and match input_ids shape")
    if mapping.shape != input_ids.shape:
        raise ValueError("gene_index_map shape must match input_ids")
    if not torch.equal(mapping[attention == 0], torch.full_like(mapping[attention == 0], -1)):
        raise ValueError("Padding positions in gene_index_map must be -1")
    num_genes = len(artifact["gene_symbols"])
    valid = mapping[attention.bool()]
    if valid.numel() and (valid.min() < 0 or valid.max() >= num_genes):
        raise ValueError("Real token gene indices are outside the original gene range")
    return {
        "input_ids_shape": tuple(input_ids.shape),
        "attention_mask_shape": tuple(attention.shape),
        "gene_index_map_shape": tuple(mapping.shape),
        "original_gene_count": num_genes,
        "cell_count": input_ids.size(0),
        "padding_consistent": True,
        "token_id_min": int(input_ids.min()) if input_ids.numel() else None,
        "token_id_max": int(input_ids.max()) if input_ids.numel() else None,
    }


def prepare_geneformer_tokens(
    expression_path,
    mapping_path,
    output_path,
    token_dictionary_path,
    model_version="V1",
    max_length=None,
    pad_token_id=None,
    overwrite=False,
):
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Pass --overwrite to replace it."
        )
    spec = get_model_spec(model_version)
    if model_version == "V1" and spec.special_tokens:
        raise RuntimeError("Geneformer V1 must not add special tokens")
    maximum = spec.maximum_sequence_length if max_length is None else max_length
    if maximum <= 0 or maximum > spec.maximum_sequence_length:
        raise ValueError(
            f"max_length must be in [1, {spec.maximum_sequence_length}] for "
            f"Geneformer {model_version}"
        )
    padding_id = spec.default_pad_token_id if pad_token_id is None else pad_token_id

    expression_df = pd.read_csv(expression_path)
    mapping_df = pd.read_csv(mapping_path)
    if expression_df.shape[1] < 2:
        raise ValueError("Expression CSV must contain a gene column and cell columns")
    gene_symbols = expression_df.iloc[:, 0].astype(str).tolist()
    if len(gene_symbols) != len(set(gene_symbols)):
        raise ValueError("Expression gene symbols contain duplicates")
    if "gene" not in mapping_df or "ensembl_id" not in mapping_df:
        raise ValueError("Mapping CSV must contain columns 'gene' and 'ensembl_id'")
    mapping_genes = mapping_df["gene"].astype(str).tolist()
    if len(mapping_genes) != len(gene_symbols):
        raise ValueError(
            "Mapping row count does not equal expression gene count: "
            f"{len(mapping_genes)} vs {len(gene_symbols)}"
        )
    if mapping_genes != gene_symbols:
        mismatch = next(
            index
            for index, (left, right) in enumerate(zip(gene_symbols, mapping_genes))
            if left != right
        )
        raise ValueError(
            "Expression and mapping gene order differ at index "
            f"{mismatch}: {gene_symbols[mismatch]} != {mapping_genes[mismatch]}"
        )
    ensembl_ids = mapping_df["ensembl_id"].fillna("").astype(str).tolist()
    valid_ensembl = np.array(
        [bool(ENSEMBL_PATTERN.fullmatch(value)) for value in ensembl_ids],
        dtype=bool,
    )
    valid_values = [
        value for value, valid in zip(ensembl_ids, valid_ensembl) if valid
    ]
    if len(valid_values) != len(set(valid_values)):
        raise ValueError("Valid Ensembl IDs contain duplicates")

    expression = expression_df.iloc[:, 1:].to_numpy(dtype=np.float64)
    if not np.isfinite(expression).all():
        raise ValueError("Expression matrix contains NaN or Inf")
    if (expression < 0).any():
        raise ValueError("Expression matrix contains negative values")
    token_dictionary = load_token_dictionary(token_dictionary_path)
    gene_token_ids = torch.full((len(gene_symbols),), -1, dtype=torch.long)
    for index, ensembl_id in enumerate(ensembl_ids):
        if valid_ensembl[index] and ensembl_id in token_dictionary:
            gene_token_ids[index] = int(token_dictionary[ensembl_id])
    tokenizable = gene_token_ids.ge(0)
    tokenizable_indices = np.flatnonzero(tokenizable.numpy())

    ranked_token_rows = []
    ranked_gene_rows = []
    detected_lengths = []
    for cell_index in range(expression.shape[1]):
        values = expression[tokenizable_indices, cell_index]
        positive = values > 0
        candidate_indices = tokenizable_indices[positive]
        candidate_values = values[positive]
        order = np.argsort(-candidate_values, kind="stable")
        ranked_indices = candidate_indices[order][:maximum]
        if ranked_indices.size == 0:
            raise ValueError(
                f"Cell '{expression_df.columns[cell_index + 1]}' is empty after "
                "valid/tokenizable/positive-expression filtering"
            )
        ranked_gene_rows.append(torch.as_tensor(ranked_indices, dtype=torch.long))
        ranked_token_rows.append(gene_token_ids[ranked_indices])
        detected_lengths.append(len(ranked_indices))

    saved_length = min(max(detected_lengths), maximum)
    cell_count = len(ranked_token_rows)
    input_ids = torch.full(
        (cell_count, saved_length), int(padding_id), dtype=torch.long
    )
    attention_mask = torch.zeros((cell_count, saved_length), dtype=torch.long)
    gene_index_map = torch.full((cell_count, saved_length), -1, dtype=torch.long)
    for cell_index, (tokens, indices) in enumerate(
        zip(ranked_token_rows, ranked_gene_rows)
    ):
        length = min(tokens.numel(), saved_length)
        input_ids[cell_index, :length] = tokens[:length]
        attention_mask[cell_index, :length] = 1
        gene_index_map[cell_index, :length] = indices[:length]

    removed = [
        {
            "index": index,
            "gene": gene_symbols[index],
            "ensembl_id": ensembl_ids[index],
            "reason": (
                "invalid_ensembl"
                if not valid_ensembl[index]
                else "not_in_token_dictionary"
            ),
        }
        for index in range(len(gene_symbols))
        if not tokenizable[index]
    ]
    metadata = {
        "expression_path": os.path.abspath(expression_path),
        "mapping_path": os.path.abspath(mapping_path),
        "token_dictionary_source": os.path.abspath(token_dictionary_path),
        "model_version": model_version,
        "original_gene_count": len(gene_symbols),
        "cell_count": cell_count,
        "valid_ensembl_count": int(valid_ensembl.sum()),
        "invalid_ensembl_count": int((~valid_ensembl).sum()),
        "tokenizable_gene_count": int(tokenizable.sum()),
        "non_tokenizable_valid_ensembl_count": int(
            (valid_ensembl & ~tokenizable.numpy()).sum()
        ),
        "maximum_detected_sequence_length": max(detected_lengths),
        "saved_sequence_length": saved_length,
        "pad_token_id": int(padding_id),
        "special_tokens_added": False,
        "removed_genes": removed,
        "expression_sha256": sha256_file(expression_path),
        "mapping_sha256": sha256_file(mapping_path),
        "token_dictionary_sha256": sha256_file(token_dictionary_path),
        "preprocessing_limitation": (
            "Custom rank-based tokenization over the 910 genes available in "
            "the normalized, feature-selected scRegNet dataset; not the official "
            "raw-count full-transcriptome Geneformer tokenization pipeline."
        ),
    }
    artifact = {
        "format_version": 1,
        "model_version": model_version,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "gene_index_map": gene_index_map,
        "cell_ids": [str(value) for value in expression_df.columns[1:]],
        "gene_symbols": gene_symbols,
        "ensembl_ids": ensembl_ids,
        "gene_token_ids": gene_token_ids,
        "tokenizable_gene_mask": tokenizable,
        "metadata": metadata,
    }
    diagnostics = validate_token_artifact(artifact)
    output_directory = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_directory, exist_ok=True)
    torch.save(artifact, output_path)
    summary_path = os.path.splitext(output_path)[0] + ".summary.json"
    with open(summary_path, "w") as handle:
        json.dump({**metadata, **diagnostics}, handle, indent=2)
    return artifact, summary_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expression_path", required=True)
    parser.add_argument("--mapping_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--scfm_model_version", choices=["V1", "V2"], default="V1")
    parser.add_argument("--scfm_token_dictionary_path", default=None)
    parser.add_argument(
        "--scfm_model_repo", default="ctheodoris/Geneformer"
    )
    parser.add_argument(
        "--scfm_token_dictionary_file",
        default="geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl",
    )
    parser.add_argument("--hf_cache_dir", default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--pad_token_id", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    resolver_args = SimpleNamespace(
        scfm_token_dictionary_path=args.scfm_token_dictionary_path,
        scfm_model_repo=args.scfm_model_repo,
        scfm_token_dictionary_file=args.scfm_token_dictionary_file,
        hf_cache_dir=args.hf_cache_dir,
    )
    dictionary_path = resolve_token_dictionary(resolver_args, required=True)
    artifact, summary_path = prepare_geneformer_tokens(
        expression_path=args.expression_path,
        mapping_path=args.mapping_path,
        output_path=args.output_path,
        token_dictionary_path=dictionary_path,
        model_version=args.scfm_model_version,
        max_length=args.max_length,
        pad_token_id=args.pad_token_id,
        overwrite=args.overwrite,
    )
    print(validate_token_artifact(artifact))
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
