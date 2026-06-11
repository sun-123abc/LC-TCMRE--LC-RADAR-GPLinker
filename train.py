# -*- coding: utf-8 -*-
"""
train.py with FGM adversarial training for model2.

用法：
    直接用本文件替换项目根目录下的 train.py。

默认行为：
    - 当 args.model == "model2" 时，自动启用 FGM；
    - 其他模型默认不启用 FGM，避免影响 baseline/model1 对比。

可选环境变量：
    PowerShell 开启 FGM：$env:USE_FGM="1"
    PowerShell 关闭 FGM：$env:USE_FGM="0"
    PowerShell 设置 epsilon：$env:FGM_EPSILON="1.0"
"""

import importlib
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from common.config import get_train_args
from common.data_reader import (
    load_rel_maps,
    load_chunks,
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
    save_checkpoint,
    count_parameters,
    AverageMeter,
    save_json,
)


# ============================================================
# FGM：Fast Gradient Method 对抗训练
# ============================================================

class FGM:
    """
    对 embedding 参数施加一次梯度方向扰动。

    典型流程：
        1. 正常 forward/backward；
        2. attack：在 embedding 上加扰动；
        3. 再 forward/backward；
        4. restore：恢复原参数；
        5. optimizer.step()。
    """

    def __init__(self, model, emb_name="word_embeddings", epsilon=1.0):
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.backup = {}

    def attack(self):
        self.backup = {}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if self.emb_name not in name:
                continue
            if param.grad is None:
                continue

            self.backup[name] = param.data.clone()
            norm = torch.norm(param.grad)

            if norm is not None and norm != 0 and not torch.isnan(norm):
                r_at = self.epsilon * param.grad / norm
                param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


def should_use_fgm(args):
    """
    默认只对 model2 启用 FGM。
    也可以通过环境变量 USE_FGM 强制开关：
        USE_FGM=1 开启
        USE_FGM=0 关闭
    """
    env_flag = os.environ.get("USE_FGM")

    if env_flag is not None:
        return env_flag.strip() in {"1", "true", "True", "yes", "YES"}

    return getattr(args, "model", "") in {"model2", "model3"}


def get_fgm_epsilon():
    value = os.environ.get("FGM_EPSILON", "1.0")
    try:
        return float(value)
    except Exception:
        return 1.0


def import_model_components(model_name: str):
    """
    每个模型目录需要有：
    models/xxx/dataset.py  -> REDataset
    models/xxx/model.py    -> REModel
    models/xxx/decode.py   -> decode_batch
    """
    dataset_module = importlib.import_module(f"models.{model_name}.dataset")
    model_module = importlib.import_module(f"models.{model_name}.model")
    decode_module = importlib.import_module(f"models.{model_name}.decode")

    dataset_cls = getattr(dataset_module, "REDataset")
    model_cls = getattr(model_module, "REModel")
    decode_batch = getattr(decode_module, "decode_batch")

    return dataset_cls, model_cls, decode_batch


@torch.no_grad()
def evaluate(model, data_loader, decode_batch, args, id2rel, device):
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

    return overall, by_relation, all_pred_records


def train_one_epoch(model, data_loader, optimizer, scheduler, scaler, args, device, logger, epoch, fgm=None):
    model.train()

    loss_meter = AverageMeter()
    use_fgm = fgm is not None

    for step, batch in enumerate(data_loader, start=1):
        batch = move_to_device(batch, device)
        model_inputs = get_model_inputs(batch)

        optimizer.zero_grad()

        if args.fp16:
            # 正常 forward/backward
            with torch.cuda.amp.autocast():
                outputs = model(**model_inputs)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

            scaler.scale(loss).backward()

            # FGM 对抗 forward/backward
            if use_fgm:
                fgm.attack()
                with torch.cuda.amp.autocast():
                    outputs_adv = model(**model_inputs)
                    loss_adv = outputs_adv["loss"] if isinstance(outputs_adv, dict) else outputs_adv[0]
                scaler.scale(loss_adv).backward()
                fgm.restore()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # 正常 forward/backward
            outputs = model(**model_inputs)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            loss.backward()

            # FGM 对抗 forward/backward
            if use_fgm:
                fgm.attack()
                outputs_adv = model(**model_inputs)
                loss_adv = outputs_adv["loss"] if isinstance(outputs_adv, dict) else outputs_adv[0]
                loss_adv.backward()
                fgm.restore()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

        scheduler.step()

        loss_meter.update(loss.item(), n=1)

        if step % args.log_steps == 0:
            if use_fgm:
                logger.info(
                    f"Epoch {epoch} | Step {step}/{len(data_loader)} | "
                    f"Loss {loss_meter.avg:.6f} | FGM on"
                )
            else:
                logger.info(
                    f"Epoch {epoch} | Step {step}/{len(data_loader)} | "
                    f"Loss {loss_meter.avg:.6f}"
                )

    return loss_meter.avg


def main():
    args = get_train_args()

    ensure_dir(args.save_dir)
    logger = get_logger(
        name=f"TRAIN-{args.model}",
        log_file=os.path.join(args.save_dir, "train.log")
    )

    set_seed(args.seed)
    device = get_device(args.device)

    logger.info(f"Model: {args.model}")
    logger.info(f"Device: {device}")
    logger.info(f"Save dir: {args.save_dir}")

    rel2id, id2rel = load_rel_maps(args.rel2id_file, args.id2rel_file)
    logger.info(f"Relation num: {len(rel2id)}")
    logger.info(f"Relations: {list(rel2id.keys())}")

    dataset_cls, model_cls, decode_batch = import_model_components(args.model)

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model,
        local_files_only=True
    )

    train_dataset = dataset_cls(
        data_path=args.train_file,
        tokenizer=tokenizer,
        rel2id=rel2id,
        args=args,
        is_train=True,
    )

    dev_dataset = dataset_cls(
        data_path=args.dev_file,
        tokenizer=tokenizer,
        rel2id=rel2id,
        args=args,
        is_train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=train_dataset.collate_fn,
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dev_dataset.collate_fn,
    )

    model = model_cls(
        args=args,
        rel2id=rel2id,
        id2rel=id2rel,
    )
    model.to(device)

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Dev samples: {len(dev_dataset)}")
    logger.info(f"Trainable parameters: {count_parameters(model):,}")

    use_fgm = should_use_fgm(args)
    fgm = None

    if use_fgm:
        epsilon = get_fgm_epsilon()
        fgm = FGM(model=model, emb_name="word_embeddings", epsilon=epsilon)
        logger.info(f"FGM adversarial training: ENABLED | epsilon={epsilon}")
    else:
        logger.info("FGM adversarial training: DISABLED")

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    best_f1 = -1.0
    bad_epochs = 0

    train_config_path = os.path.join(args.save_dir, "train_config.json")
    save_json(vars(args), train_config_path)

    # 额外记录 FGM 配置，方便论文和复现实验。
    save_json(
        {
            "use_fgm": use_fgm,
            "fgm_epsilon": get_fgm_epsilon() if use_fgm else None,
            "fgm_emb_name": "word_embeddings" if use_fgm else None,
        },
        os.path.join(args.save_dir, "fgm_config.json")
    )

    for epoch in range(1, args.epochs + 1):
        logger.info("=" * 80)
        logger.info(f"Epoch {epoch}/{args.epochs}")

        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            args=args,
            device=device,
            logger=logger,
            epoch=epoch,
            fgm=fgm,
        )

        logger.info(f"Epoch {epoch} train loss: {train_loss:.6f}")

        overall, by_relation, _ = evaluate(
            model=model,
            data_loader=dev_loader,
            decode_batch=decode_batch,
            args=args,
            id2rel=id2rel,
            device=device,
        )

        logger.info(f"Dev overall: {format_metric_line(overall)}")

        for rel, m in by_relation.items():
            logger.info(f"Dev {rel}: {format_metric_line(m)}")

        current_f1 = overall["f1"]

        if current_f1 > best_f1:
            best_f1 = current_f1
            bad_epochs = 0

            best_path = os.path.join(args.save_dir, "best_model.pt")

            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_name": args.model,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_f1": best_f1,
                    "args": vars(args),
                    "rel2id": rel2id,
                    "id2rel": id2rel,
                    "dev_metrics": overall,
                    "fgm": {
                        "enabled": use_fgm,
                        "epsilon": get_fgm_epsilon() if use_fgm else None,
                        "emb_name": "word_embeddings" if use_fgm else None,
                    },
                },
                best_path,
            )

            save_json(
                {
                    "epoch": epoch,
                    "overall": overall,
                    "by_relation": by_relation,
                },
                os.path.join(args.save_dir, "best_dev_metrics.json")
            )

            logger.info(f"New best model saved: {best_path}")
        else:
            bad_epochs += 1
            logger.info(f"No improvement. bad_epochs={bad_epochs}/{args.early_stop}")

        if bad_epochs >= args.early_stop:
            logger.info("Early stopping triggered.")
            break

    logger.info(f"Training finished. Best dev F1 = {best_f1 * 100:.2f}")


if __name__ == "__main__":
    main()
