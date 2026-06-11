# common/config.py
import argparse
import os
from pathlib import Path


MODEL_NAMES = [
    "casrel",#1
    "tplinker",#2
    "spn",
    "care",
    "pfn",
    "biaffine",
    "gplinker",#3
    "spn",
    "bicon",
    "spert",
    "rsan",
    "model1",
    "model2",
    "model3",
]


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1", "y"):
        return True
    if v.lower() in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument("--model", type=str, required=True, choices=MODEL_NAMES)

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--dev_file", type=str, default=None)
    parser.add_argument("--test_file", type=str, default=None)
    parser.add_argument("--rel2id_file", type=str, default=None)
    parser.add_argument("--id2rel_file", type=str, default=None)

    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)

    parser.add_argument("--pretrained_model", type=str, default="bert-base-chinese")
    parser.add_argument("--max_len", type=int, default=512)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--early_stop", type=int, default=8)

    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--fp16", action="store_true")

    parser.add_argument("--log_steps", type=int, default=20)
    parser.add_argument("--eval_source_level", type=str2bool, default=True)

    return parser


def post_process_args(args):
    args.model = args.model.lower()

    data_dir = Path(args.data_dir)

    if args.train_file is None:
        args.train_file = str(data_dir / "train_chunks.json")
    if args.dev_file is None:
        args.dev_file = str(data_dir / "dev_chunks.json")
    if args.test_file is None:
        args.test_file = str(data_dir / "test_chunks.json")
    if args.rel2id_file is None:
        args.rel2id_file = str(data_dir / "rel2id.json")
    if args.id2rel_file is None:
        args.id2rel_file = str(data_dir / "id2rel.json")

    if args.save_dir is None:
        args.save_dir = os.path.join(args.output_dir, args.model)

    if args.checkpoint is None:
        args.checkpoint = os.path.join(args.save_dir, "best_model.pt")

    return args


def get_train_args():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()
    return post_process_args(args)


def get_test_args():
    parser = argparse.ArgumentParser()
    add_common_args(parser)

    parser.add_argument("--prediction_dir", type=str, default=None)
    parser.add_argument("--save_predictions", type=str2bool, default=True)

    args = parser.parse_args()
    args = post_process_args(args)

    if args.prediction_dir is None:
        args.prediction_dir = os.path.join(args.save_dir, "predictions")

    return args