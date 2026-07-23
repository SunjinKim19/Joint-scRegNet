import argparse
import json
import logging
import os

logger = logging.getLogger(__name__)

GNN_LIST = ["GraphSAGE", "GCN", "GAT"]


def str2bool(value):
    """Parse explicit true/false CLI values without argparse's bool pitfall."""
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("expected one of: true, false, 1, 0, yes, no")

def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument("--single_gpu", type=int, default=0)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--start_seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--suffix", type=str, default="main")
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument(
        "--device", choices=["auto", "cuda", "cpu"], default="auto"
    )

    # parameters for data and model storage
    parser.add_argument("--data_folder", type=str, default="./data")
    parser.add_argument("--scFM_folder", type=str, default="./scFM")
    parser.add_argument("--task_type", type=str, default="link_pred")
    parser.add_argument("--output_dir", type=str)  # output dir
    parser.add_argument("--ckpt_dir", type=str)  # ckpt path to save
    parser.add_argument(
        "--ckpt_name", type=str, default="model.pt"
    )  # ckpt name to be loaded
   

    # training hyperparameters
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--accum_interval", type=int, default=1)
    parser.add_argument("--attention_dropout_prob", type=float, default=0.1)
    parser.add_argument("--label_smoothing", type=float, default=0.3)
    parser.add_argument("--warmup_ratio", type=float, default=0.6)
    parser.add_argument("--num_iterations", type=int, default=4)
    parser.add_argument("--optimizer_name", type=str, default="Adam")
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        choices=["linear", "constant"],
    )

    # module hyperparameters
    # gnn parameters
    parser.add_argument("--gnn_epochs", type=int, default=300)
    parser.add_argument("--gnn_eval_interval", type=int, default=5)
    parser.add_argument("--gnn_label_smoothing", type=float, default=0.1)
    parser.add_argument("--gnn_warmup_ratio", type=float, default=0.25)
    parser.add_argument("--gnn_num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--gnn_dim_hidden", type=int, default=256)
    parser.add_argument("--gnn_lr", type=float, default=5e-4)
    parser.add_argument("--gnn_weight_decay", type=float, default=1e-5)
    parser.add_argument(
        "--gnn_lr_scheduler_type",
        type=str,
        default="constant",
        choices=["constant", "linear"],
    )

    # optuna hyperparameters
    parser.add_argument("--expected_valid_acc", type=float, default=0.6)
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--load_study", action="store_true", default=False)

    # grn parameters
    parser.add_argument("--dataset", type=str, default="tf_500")
    parser.add_argument("--gnn_type", type=str, default="GCN")
    parser.add_argument("--llm_type", type=str, default="Geneformer")
    parser.add_argument("--cell_type", type=str, default="hESC")
    parser.add_argument("--num_TF", type=str, default="500")
    parser.add_argument("--flag", type=bool, default=True) 
    parser.add_argument("--loop", type=bool, default=False)
    parser.add_argument("--type", type=str, default="MLP") # score metric
    parser.add_argument("--mlp_dim_hidden", type=int, default=64) 
    parser.add_argument("--mlp_num_layers", type=int, default=2)

    #params for GAT
    parser.add_argument("--reduction", type=str, default="concate")
    parser.add_argument("--num_heads", type=int, default=3)
    parser.add_argument("--alpha", type=int, default=0.2)

    #params for joint
    parser.add_argument("--joint_latent_dim", type=int, default=128)
    parser.add_argument("--cell_hidden_dim", type=int, default=256)
    parser.add_argument("--link_hidden_dim", type=int, default=128)

    parser.add_argument("--lambda_recon", type=float, default=0.005)
    parser.add_argument("--lambda_align", type=float, default=0.01)
    parser.add_argument("--max_recon_cells", type=int, default=256)

    # cross-model attention fusion
    parser.add_argument(
        "--fusion_mode",
        type=str,
        default="gnn_to_scfm",
        choices=["gated", "gnn_to_scfm", "bidirectional"],
    )
    parser.add_argument("--fusion_layers", type=int, default=1)
    parser.add_argument("--fusion_heads", type=int, default=4)
    parser.add_argument("--fusion_dropout", type=float, default=0.2)
    parser.add_argument("--cross_latent_dim", type=int, default=128)
    parser.add_argument("--directed_link_predictor", type=int, default=1)
    parser.add_argument(
        "--loss_type",
        type=str,
        default="bce",
        choices=["bce", "pos_weight_bce", "focal"],
    )
    parser.add_argument("--focal_alpha", type=float, default=0.75)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument(
        "--early_stop_metric",
        type=str,
        default="auprc",
        choices=["auprc", "auroc"],
    )
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min_delta", type=float, default=0.0)

    # GeSubNet-inspired Cell/Condition-M + Graph-M + Infer-M path
    parser.add_argument(
        "--cell_types",
        type=str,
        default="",
        help="Comma-separated cell types for multi-condition training.",
    )
    parser.add_argument("--condition_joint_latent_dim", type=int, default=128)
    parser.add_argument("--condition_hidden_dim", type=int, default=256)
    parser.add_argument(
        "--context_recon_target",
        type=str,
        default="projected_scfm",
        choices=["none", "scfm", "projected_scfm"],
    )
    parser.add_argument("--lambda_context", type=float, default=0.001)
    parser.add_argument("--lambda_condition", type=float, default=0.0)
    parser.add_argument("--lambda_graph_pre", type=float, default=0.0)
    parser.add_argument(
        "--infer_fusion_mode",
        type=str,
        default="bidirectional",
        choices=["gated", "bidirectional"],
    )
    parser.add_argument("--infer_layers", type=int, default=1)
    parser.add_argument("--infer_heads", type=int, default=4)
    parser.add_argument("--infer_dropout", type=float, default=0.2)
    parser.add_argument(
        "--scfm_tune_mode",
        type=str,
        default="adapter",
        choices=["frozen_embedding", "adapter", "top", "full"],
    )

    # Serial Cell-M -> graph construction -> Graph-M path.
    parser.add_argument("--use_cell_guided_graph", type=str2bool, default=True)
    parser.add_argument("--graph_alpha", type=float, default=0.8)
    parser.add_argument(
        "--graph_fusion_type",
        type=str,
        default="fixed",
        choices=["fixed", "edge_gate"],
    )
    parser.add_argument("--gate_hidden_dim", type=int, default=32)
    parser.add_argument("--gate_dropout", type=float, default=0.0)
    parser.add_argument("--gate_temperature", type=float, default=1.0)
    parser.add_argument(
        "--gate_init_from_alpha",
        dest="gate_init_from_alpha",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no_gate_init_from_alpha",
        dest="gate_init_from_alpha",
        action="store_false",
        help="Use the gate module's default random initialization.",
    )
    parser.add_argument("--lambda_sparse", type=float, default=0.0)
    parser.add_argument(
        "--graph_constructor_type",
        type=str,
        default="mlp",
        choices=["mlp", "bilinear"],
    )
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument(
        "--hard_topk_eval_only",
        type=int,
        default=0,
        help="Optional outgoing edges per TF for evaluation/visualization only; 0 disables it.",
    )
    parser.add_argument("--freeze_scfm", type=str2bool, default=True)
    parser.add_argument("--train_scfm_adapter", type=str2bool, default=True)
    parser.add_argument("--train_scfm_top_layers", type=int, default=0)
    parser.add_argument(
        "--scfm_mode",
        type=str,
        default="precomputed",
        choices=["precomputed", "online_frozen", "online_lora", "online_topk"],
    )
    parser.add_argument("--scfm_model_path", type=str, default=None)
    parser.add_argument("--scfm_tokenized_path", type=str, default=None)
    parser.add_argument("--scfm_output_layer", type=str, default="last_hidden")
    parser.add_argument(
        "--scfm_pooling",
        type=str,
        default="gene",
        choices=["gene", "cell", "mean"],
    )
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules", type=str, default="query,value"
    )
    parser.add_argument("--scfm_lr", type=float, default=1e-5)
    parser.add_argument("--downstream_lr", type=float, default=1e-3)
    parser.add_argument("--scfm_weight_decay", type=float, default=0.01)
    parser.add_argument("--downstream_weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_scfm_cells", type=int, default=0)
    parser.add_argument(
        "--cache_online_scfm_outputs", action="store_true", default=False
    )
    parser.add_argument(
        "--no_cache_online_scfm_outputs",
        dest="cache_online_scfm_outputs",
        action="store_false",
    )
    parser.add_argument("--amp", dest="amp", action="store_true", default=False)
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)

    args = parser.parse_args()
    return args


def save_args(args, dir):
    # if int(os.getenv("RANK", -1)) <= 0:
    FILE_NAME = "args.json"
    with open(os.path.join(dir, FILE_NAME), "w") as f:
        json.dump(args.__dict__, f, indent=2)
    logger.info("args saved to {}".format(os.path.join(dir, FILE_NAME)))


def load_args(dir):
    with open(os.path.join(dir, "args.json"), "r") as f:
        args = argparse.Namespace(**json.load(f))
    return args


if __name__ == "__main__":
    args = parse_args()
