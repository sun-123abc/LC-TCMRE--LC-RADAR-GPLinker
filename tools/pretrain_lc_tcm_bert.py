# -*- coding: utf-8 -*-
"""
领域自适应继续预训练脚本：LC-TCM-DAPT

输入：
    1. 一个本地中文预训练模型路径，如 bert-base-chinese 本地文件夹
    2. 一个 txt 语料文件，每行一条医案文本或 chunk 文本

输出：
    一个可以直接传给 --pretrained_model 的本地模型文件夹

示例：
python tools/pretrain_lc_tcm_bert.py ^
  --base_model "F:\pretrained_models\bert-base-chinese" ^
  --train_file data/lc_tcm_mlm_train.txt ^
  --output_dir pretrained_models/LC-TCM-BERT ^
  --epochs 8 ^
  --batch_size 8 ^
  --grad_accum 4
"""

import argparse
from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model", required=True, help="本地预训练模型路径")
    parser.add_argument("--train_file", required=True, help="MLM训练文本，每行一条")
    parser.add_argument("--output_dir", required=True, help="输出模型目录")

    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--fp16", action="store_true")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        local_files_only=True,
    )

    model = AutoModelForMaskedLM.from_pretrained(
        args.base_model,
        local_files_only=True,
    )

    raw_dataset = load_dataset(
        "text",
        data_files={"train": args.train_file},
    )

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
            return_special_tokens_mask=True,
        )

    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )

    # mlm_probability=0.15 是常用的遮蔽比例；DataCollatorForLanguageModeling 会动态构造 MLM 样本。
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        fp16=args.fp16,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        prediction_loss_only=True,
        report_to="none",

        # 关键：避免 safetensors 因 non-contiguous tensor 保存失败
        save_safetensors=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"LC-TCM-DAPT model saved to: {output_dir}")


if __name__ == "__main__":
    main()