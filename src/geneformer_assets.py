"""Resolve Geneformer checkpoint and token-dictionary assets consistently."""

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeneformerModelSpec:
    version: str
    maximum_sequence_length: int
    special_tokens: bool
    default_pad_token_id: int


MODEL_SPECS = {
    "V1": GeneformerModelSpec("V1", 2048, False, 0),
    # V2 checkpoints use explicit special-token-aware model configurations.
    "V2": GeneformerModelSpec("V2", 4096, True, 0),
}


@dataclass(frozen=True)
class ResolvedGeneformerAssets:
    model_path: str | None
    model_repo: str | None
    model_subfolder: str | None
    token_dictionary_path: str | None
    model_spec: GeneformerModelSpec

    @property
    def model_identity(self):
        if self.model_path:
            return self.model_path
        return f"{self.model_repo}/{self.model_subfolder}"


def get_model_spec(version):
    try:
        return MODEL_SPECS[version]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported Geneformer model version '{version}'. "
            f"Expected one of {sorted(MODEL_SPECS)}."
        ) from exc


def resolve_model_asset(args, required=True):
    local_path = getattr(args, "scfm_model_path", None)
    if local_path:
        if not os.path.isdir(local_path):
            raise ValueError(
                f"Local Geneformer checkpoint directory does not exist: {local_path}"
            )
        logger.info("Geneformer checkpoint source: local path %s", local_path)
        return local_path, None, None
    repo = getattr(args, "scfm_model_repo", None)
    subfolder = getattr(args, "scfm_model_subfolder", None)
    if required and not repo:
        raise ValueError(
            "Online scFM mode requires --scfm_model_path or --scfm_model_repo."
        )
    if repo:
        logger.info(
            "Geneformer checkpoint source: HuggingFace repo=%s subfolder=%s",
            repo,
            subfolder,
        )
    return None, repo, subfolder


def resolve_token_dictionary(args, required=True, downloader=None):
    local_path = getattr(args, "scfm_token_dictionary_path", None)
    if local_path:
        if not os.path.isfile(local_path):
            raise ValueError(
                f"Local Geneformer token dictionary does not exist: {local_path}"
            )
        logger.info("Geneformer token dictionary source: local path %s", local_path)
        return local_path
    if not required:
        return None
    repo = getattr(args, "scfm_model_repo", None)
    filename = getattr(args, "scfm_token_dictionary_file", None)
    if not repo or not filename:
        raise ValueError(
            "Tokenization requires --scfm_token_dictionary_path or both "
            "--scfm_model_repo and --scfm_token_dictionary_file."
        )
    if downloader is None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "Downloading the Geneformer token dictionary requires "
                "`pip install huggingface_hub`, or provide "
                "--scfm_token_dictionary_path."
            ) from exc
        downloader = hf_hub_download
    try:
        resolved = downloader(
            repo_id=repo,
            filename=filename,
            cache_dir=getattr(args, "hf_cache_dir", None),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download token dictionary '{filename}' from '{repo}'. "
            "Provide a valid local --scfm_token_dictionary_path or verify Hub "
            f"access. Original error: {exc}"
        ) from exc
    logger.info(
        "Geneformer token dictionary source: HuggingFace %s/%s -> %s",
        repo,
        filename,
        resolved,
    )
    return resolved


def resolve_geneformer_assets(args, require_token_dictionary=False):
    mode = getattr(args, "scfm_mode", "precomputed")
    spec = get_model_spec(getattr(args, "scfm_model_version", "V1"))
    if mode == "precomputed":
        return ResolvedGeneformerAssets(None, None, None, None, spec)
    model_path, repo, subfolder = resolve_model_asset(args, required=True)
    token_path = resolve_token_dictionary(
        args, required=require_token_dictionary
    )
    return ResolvedGeneformerAssets(
        model_path=model_path,
        model_repo=repo,
        model_subfolder=subfolder,
        token_dictionary_path=token_path,
        model_spec=spec,
    )
