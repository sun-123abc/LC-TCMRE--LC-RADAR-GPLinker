# test.py
import importlib
import os

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from common.config import get_test_args
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
    build_source_prediction_records,
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


@torch.no_grad()
def run_test(model, data_loader, decode_batch, args, id2rel, device):
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

    return overall, by_relation, all_pred_records, pred_by_source, gold_by_source


def main():
    args = get_test_args()

    ensure_dir(args.prediction_dir)

    logger = get_logger(
        name=f"TEST-{args.model}",
        log_file=os.path.join(args.prediction_dir, "test.log")
    )

    set_seed(args.seed)
    device = get_device(args.device)

    logger.info(f"Model: {args.model}")
    logger.info(f"Device: {device}")
    logger.info(f"Checkpoint: {args.checkpoint}")

    rel2id, id2rel = load_rel_maps(args.rel2id_file, args.id2rel_file)

    dataset_cls, model_cls, decode_batch = import_model_components(args.model)

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model,
        local_files_only=True
    )

    test_dataset = dataset_cls(
        data_path=args.test_file,
        tokenizer=tokenizer,
        rel2id=rel2id,
        args=args,
        is_train=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=test_dataset.collate_fn,
    )

    model = model_cls(
        args=args,
        rel2id=rel2id,
        id2rel=id2rel,
    )

    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)

    overall, by_relation, chunk_pred_records, pred_by_source, gold_by_source = run_test(
        model=model,
        data_loader=test_loader,
        decode_batch=decode_batch,
        args=args,
        id2rel=id2rel,
        device=device,
    )

    logger.info(f"Test overall: {format_metric_line(overall)}")

    for rel, m in by_relation.items():
        logger.info(f"Test {rel}: {format_metric_line(m)}")

    result = {
        "overall": overall,
        "by_relation": by_relation,
    }

    save_json(result, os.path.join(args.prediction_dir, "test_metrics.json"))

    if args.save_predictions:
        source_pred_records = build_source_prediction_records(pred_by_source)
        source_gold_records = build_source_prediction_records(gold_by_source)

        save_json(
            chunk_pred_records,
            os.path.join(args.prediction_dir, "chunk_predictions.json")
        )

        save_json(
            source_pred_records,
            os.path.join(args.prediction_dir, "source_predictions.json")
        )

        save_json(
            source_gold_records,
            os.path.join(args.prediction_dir, "source_gold.json")
        )

    logger.info(f"Test finished. Results saved to: {args.prediction_dir}")


if __name__ == "__main__":
    main()