# run_experiments.py
import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_MODELS = [
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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="all 或逗号分隔模型名，例如 casrel,tplinker,gplinker"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["train", "test", "both"]
    )

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--pretrained_model", type=str, default="bert-base-chinese")
    parser.add_argument("--max_len", type=int, default=512)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--early_stop", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--fp16", action="store_true")

    return parser.parse_args()


def get_model_list(model_arg):
    if model_arg == "all":
        return DEFAULT_MODELS

    return [x.strip().lower() for x in model_arg.split(",") if x.strip()]


def run_cmd(cmd):
    print("=" * 100)
    print("Running command:")
    print(" ".join(cmd))
    print("=" * 100)

    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}: {' '.join(cmd)}")


def build_common_args(args, model):
    common_args = [
        "--model", model,
        "--data_dir", args.data_dir,
        "--output_dir", args.output_dir,
        "--pretrained_model", args.pretrained_model,
        "--max_len", str(args.max_len),
        "--batch_size", str(args.batch_size),
        "--eval_batch_size", str(args.eval_batch_size),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--early_stop", str(args.early_stop),
        "--seed", str(args.seed),
        "--device", args.device,
    ]

    if args.fp16:
        common_args.append("--fp16")

    return common_args


def main():
    args = parse_args()
    models = get_model_list(args.models)

    print(f"Models: {models}")
    print(f"Mode: {args.mode}")

    for model in models:
        if args.mode in ["train", "both"]:
            train_cmd = [
                sys.executable,
                "train.py",
            ] + build_common_args(args, model)

            run_cmd(train_cmd)

        if args.mode in ["test", "both"]:
            test_cmd = [
                sys.executable,
                "test.py",
            ] + build_common_args(args, model)

            run_cmd(test_cmd)


if __name__ == "__main__":
    main()