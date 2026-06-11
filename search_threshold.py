# search_threshold.py
import argparse
import importlib
import os

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from common.config import add_common_args, post_process_args
from common.data_reader import (
    load_rel_maps,
    gold_by_source_from_metas,
    standardize_decode_output,
)
from common.metrics import (
    aggregate_chunk_records,
    compute_prf,
    compute_relation_prf,
    format_metric_line,
)
from common.utils import (
    set_seed,
    get_device,
    get_logger,
    ensure_dir,
    move_to_device,
    get_model_inputs,
    load_checkpoint,
    save_json,
)


def import_model_components(model_name: str):
    dataset_module = importlib.import_module(f"models.{model_name}.dataset")
    model_module = importlib.import_module(f"models.{model_name}.model")
    decode_module = importlib.import_module(f"models.{model_name}.decode")

    dataset_cls = getattr(dataset_module, "REDataset")
    model_cls = getattr(model_module, "REModel")
    decode_batch = getattr(decode_module, "decode_batch")

    return dataset_cls, model_cls, decode_batch


def parse_args():
    parser = argparse.ArgumentParser()
    add_common_args(parser)

    parser.add_argument(
        "--eval_file",
        type=str,
        default=None,
        help="用于搜索阈值的数据集，默认使用 dev_file"
    )

    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70",
        help="逗号分隔的阈值列表"
    )

    parser.add_argument(
        "--search_output",
        type=str,
        default=None,
        help="阈值搜索结果保存路径"
    )

    args = parser.parse_args()
    args = post_process_args(args)

    if args.eval_file is None:
        args.eval_file = args.dev_file

    if args.search_output is None:
        args.search_output = os.path.join(args.save_dir, "threshold_search_dev.json")

    return args


@torch.no_grad()
def evaluate_with_threshold(model, data_loader, decode_batch, args, id2rel, device):
    model.eval()

    all_pred_records = []
    all_metas = []

    for batch in data_loader:
        batch = move_to_device(batch, device)
        metas = batch.get("meta", [])
        model_inputs = get_model_inputs(batch)

        outputs = model(**model_inputs)

        decoded = decode_batch(
            outputs=outputs,
            batch=batch,
            args=args,
            id2rel=id2rel,
        )

        pred_records = standardize_decode_output(decoded, metas)

        all_pred_records.extend(pred_records)
        all_metas.extend(metas)

    gold_by_source = gold_by_source_from_metas(all_metas)
    pred_by_source = aggregate_chunk_records(all_pred_records)

    overall = compute_prf(gold_by_source, pred_by_source)
    by_relation = compute_relation_prf(gold_by_source, pred_by_source)

    return overall, by_relation


def main():
    args = parse_args()

    ensure_dir(args.save_dir)

    logger = get_logger(
        name=f"THRESHOLD-{args.model}",
        log_file=os.path.join(args.save_dir, "threshold_search.log")
    )

    set_seed(args.seed)
    device = get_device(args.device)

    logger.info(f"Model: {args.model}")
    logger.info(f"Eval file: {args.eval_file}")
    logger.info(f"Checkpoint: {args.checkpoint}")

    rel2id, id2rel = load_rel_maps(args.rel2id_file, args.id2rel_file)

    dataset_cls, model_cls, decode_batch = import_model_components(args.model)

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model,
        local_files_only=True
    )

    eval_dataset = dataset_cls(
        data_path=args.eval_file,
        tokenizer=tokenizer,
        rel2id=rel2id,
        args=args,
        is_train=False,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=eval_dataset.collate_fn,
    )

    model = model_cls(
        args=args,
        rel2id=rel2id,
        id2rel=id2rel,
    )

    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)

    threshold_list = [
        float(x.strip())
        for x in args.thresholds.split(",")
        if x.strip()
    ]

    all_results = []
    best_result = None

    logger.info("=" * 80)
    logger.info("Start threshold search")
    logger.info("=" * 80)

    for th in threshold_list:
        args.threshold = th

        overall, by_relation = evaluate_with_threshold(
            model=model,
            data_loader=eval_loader,
            decode_batch=decode_batch,
            args=args,
            id2rel=id2rel,
            device=device,
        )

        record = {
            "threshold": th,
            "overall": overall,
            "by_relation": by_relation,
        }

        all_results.append(record)

        logger.info(f"Threshold={th:.2f} | {format_metric_line(overall)}")

        if best_result is None or overall["f1"] > best_result["overall"]["f1"]:
            best_result = record

    logger.info("=" * 80)
    logger.info("Best threshold")
    logger.info("=" * 80)
    logger.info(
        f"Best threshold={best_result['threshold']:.2f} | "
        f"{format_metric_line(best_result['overall'])}"
    )

    save_json(
        {
            "best": best_result,
            "all_results": all_results,
        },
        args.search_output
    )

    logger.info(f"Threshold search result saved to: {args.search_output}")


if __name__ == "__main__":
    main()