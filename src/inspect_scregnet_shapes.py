import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.args import parse_args
from src.device_utils import get_device
from src.train_condition_joint import (
    adj_edge_count,
    available_cell_types,
    load_condition_bundle,
    to_binary_label,
)


def type_description(value):
    parts = [type(value).__name__]
    if isinstance(value, torch.Tensor):
        parts.append("torch.Tensor")
        if value.is_sparse:
            parts.append("sparse tensor")
        else:
            parts.append("dense tensor")
    elif isinstance(value, np.ndarray):
        parts.append("numpy array")
    elif isinstance(value, DataLoader):
        parts.append("DataLoader")
    return ", ".join(parts)


def shape_description(value):
    if isinstance(value, torch.Tensor):
        return tuple(value.shape)
    if isinstance(value, np.ndarray):
        return value.shape
    if hasattr(value, "train_set"):
        return value.train_set.shape
    if hasattr(value, "__len__"):
        return f"len={len(value)}"
    return "n/a"


def label_format(y):
    if not isinstance(y, torch.Tensor):
        y = torch.as_tensor(y)
    if y.dim() == 1:
        return "[B]"
    if y.dim() == 2 and y.size(1) == 1:
        return "[B, 1]"
    if y.dim() == 2 and y.size(1) == 2:
        row_sums = y.sum(dim=1)
        one_hot = torch.all((y == 0) | (y == 1)) and torch.allclose(
            row_sums.float(), torch.ones_like(row_sums).float()
        )
        return "[B, 2] one-hot" if one_hot else "[B, 2]"
    return str(list(y.shape))


def edge_pair_format(x):
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    if x.dim() == 2:
        return f"[B, {x.size(1)}]"
    return str(list(x.shape))


def describe(lines, name, value):
    lines.append(f"{name}:")
    lines.append(f"  type: {type_description(value)}")
    lines.append(f"  shape: {shape_description(value)}")
    if isinstance(value, torch.Tensor):
        lines.append(f"  dtype: {value.dtype}")
        lines.append(f"  device: {value.device}")
        lines.append(f"  is_sparse: {value.is_sparse}")


def main():
    args = parse_args()
    device = get_device(args.device)
    bundle = load_condition_bundle(args, args.cell_type, condition_id=0, device=device)
    train_loader = DataLoader(
        bundle.train_dataset, batch_size=args.batch_size, shuffle=False
    )
    train_x, train_y = next(iter(train_loader))
    train_x = train_x.to(device)
    train_y = train_y.to(device)

    lines = []
    lines.append("scRegNet shape inspection")
    lines.append(
        "Data preparation mirrors src/train.py file conventions and reuses the "
        "ConditionJoint loader to avoid importing torch_geometric during inspection."
    )
    lines.append(f"cell_type: {args.cell_type}")
    lines.append(f"num_TF: {args.num_TF}")
    lines.append(f"llm_type: {args.llm_type}")
    lines.append(f"gnn_type: {args.gnn_type}")
    lines.append(f"device: {device}")
    lines.append(f"available cell types for TFs+{args.num_TF}: {available_cell_types(args)}")
    lines.append("")

    describe(lines, "data_feature1 / scFM gene embeddings", bundle.data_feature1)
    describe(lines, "data_feature2 / graph node expression features", bundle.data_feature2)
    describe(lines, "adj", bundle.adj)
    lines.append(f"  edge_count: {adj_edge_count(bundle.adj)}")
    describe(lines, "train dataset", bundle.train_dataset)
    describe(lines, "train DataLoader", train_loader)
    describe(lines, "train_x", train_x)
    describe(lines, "train_y", train_y)
    lines.append(f"  label format: {label_format(train_y)}")
    lines.append(f"  binary positive ratio: {float(to_binary_label(train_y).mean()):.4f}")
    lines.append(f"  edge pair format: {edge_pair_format(train_x)}")
    describe(lines, "validation data", bundle.valid_data)
    lines.append(f"  label format: {label_format(bundle.valid_data[:, -1])}")
    lines.append(f"  edge pair format: {edge_pair_format(bundle.valid_data[:, :2])}")
    lines.append(
        f"  binary positive ratio: {float(to_binary_label(bundle.valid_data[:, -1]).mean()):.4f}"
    )
    describe(lines, "test_data", bundle.test_data)
    lines.append(f"  label format: {label_format(bundle.test_data[:, -1])}")
    lines.append(f"  edge pair format: {edge_pair_format(bundle.test_data[:, :2])}")
    lines.append(
        f"  binary positive ratio: {float(to_binary_label(bundle.test_data[:, -1]).mean()):.4f}"
    )
    lines.append("")

    if bundle.raw_expr is None:
        lines.append("Raw cell × gene matrix not found in current pipeline.")
    else:
        lines.append("Raw cell × gene matrix found in current pipeline.")
        describe(lines, "raw_expr / cell x gene matrix", bundle.raw_expr)
    lines.append(f"gene names: count={len(bundle.gene_names)} sample={bundle.gene_names[:5]}")
    lines.append(f"TF names: count={len(bundle.tf_names)} sample={bundle.tf_names[:5]}")
    lines.append(
        f"target names: count={len(bundle.target_names)} sample={bundle.target_names[:5]}"
    )
    lines.append("")
    lines.append(f"full train set before validation split: {bundle.train_full_shape}")
    lines.append(f"train set after validation split: {bundle.train_dataset.train_set.shape}")
    lines.append(f"validation split: {tuple(bundle.valid_data.shape)}")
    lines.append(f"test set: {tuple(bundle.test_data.shape)}")

    output = "\n".join(lines)
    print(output)
    out_dir = os.path.join(".", "out", "shape_inspection")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir, f"{args.cell_type}_{args.num_TF}_{args.llm_type}_shapes.txt"
    )
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(output + "\n")
    print(f"\nShape inspection written to {out_path}")


if __name__ == "__main__":
    main()
