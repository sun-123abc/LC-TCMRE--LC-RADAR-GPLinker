# -*- coding: utf-8 -*-
"""
构建 LC-TCM-BERT 领域继续预训练语料

默认合并三类语料：
1. 当前关系抽取训练集：data/train_chunks.json
2. NER 训练集：E:/PycharmProjects/医案命名实体识别/dataset_final/train_final.txt
3. 原始 753 篇医案 txt：E:/PycharmProjects/医案命名实体化模型构建/dataset/train

输出：
data/lc_tcm_mlm_train_all.txt

说明：
- 只使用训练集相关文本，不加入 dev/test；
- 自动去重；
- 支持普通 txt；
- 支持 BIO 格式 NER txt，如：字 标签，空行分句；
- 支持文件夹递归读取所有 .txt 文件；
- 长文本会按标点和长度切分，避免一整篇医案被 tokenizer 截断。
"""

import json
import argparse
import re
from pathlib import Path


# =========================
# 默认路径：按你的本地目录写死
# =========================

DEFAULT_RE_TRAIN_JSON = r"E:/PycharmProjects/关系抽取/data/train_chunks.json"

DEFAULT_NER_TRAIN_TXT = r"E:/PycharmProjects/医案命名实体识别/dataset_final/train_final.txt"

DEFAULT_RAW_CASE_DIR = r"E:/PycharmProjects/医案命名实体化模型构建/dataset/train"

DEFAULT_OUTPUT = r"E:/PycharmProjects/关系抽取/data/lc_tcm_mlm_train_all.txt"


def read_text_file(path: Path):
    """兼容 utf-8 / gbk / utf-8-sig。"""
    for enc in ["utf-8", "utf-8-sig", "gbk", "gb18030"]:
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def clean_text(text: str):
    text = text.replace("\ufeff", "")
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_long_text(text: str, max_chars=450, min_chars=20):
    """
    将长文本切成适合 MLM 的片段。
    max_chars 不需要等于 512 token，因为中文 tokenizer 后大致接近字符长度。
    """
    text = clean_text(text)
    if not text:
        return []

    # 先按中文标点切分
    pieces = re.split(r"([。！？；;])", text)

    sentences = []
    buf = ""

    for i in range(0, len(pieces), 2):
        sent = pieces[i]
        punct = pieces[i + 1] if i + 1 < len(pieces) else ""
        sent = (sent + punct).strip()
        if sent:
            sentences.append(sent)

    if not sentences:
        sentences = [text]

    chunks = []
    cur = ""

    for sent in sentences:
        if len(cur) + len(sent) <= max_chars:
            cur += sent
        else:
            if len(cur) >= min_chars:
                chunks.append(cur)
            cur = sent

            # 如果单句本身过长，则硬切
            while len(cur) > max_chars:
                part = cur[:max_chars]
                if len(part) >= min_chars:
                    chunks.append(part)
                cur = cur[max_chars:]

    if len(cur) >= min_chars:
        chunks.append(cur)

    return chunks


def load_re_train_json(path: Path):
    """读取关系抽取 train_chunks.json。"""
    if not path.exists():
        print(f"[WARN] RE train json not found: {path}")
        return []

    data = json.loads(read_text_file(path))
    texts = []

    for item in data:
        text = item.get("text", "")
        text = clean_text(text)
        if text:
            texts.extend(split_long_text(text))

    print(f"[OK] RE train json: {path}, raw_items={len(data)}, mlm_chunks={len(texts)}")
    return texts


def looks_like_bio_line(line: str):
    """
    判断是否像 NER BIO 行：
    常见格式：
    肺 B-疾病
    癌 I-疾病
    ， O
    """
    parts = line.strip().split()
    if len(parts) < 2:
        return False

    label = parts[-1]
    if label == "O":
        return True
    if label.startswith("B-") or label.startswith("I-") or label.startswith("S-") or label.startswith("E-"):
        return True
    if label.startswith("B_") or label.startswith("I_") or label.startswith("S_") or label.startswith("E_"):
        return True

    return False


def load_ner_train_txt(path: Path):
    """
    读取 NER train_final.txt。
    如果是 BIO 格式，则按空行还原句子；
    如果是普通文本，则按普通 txt 处理。
    """
    if not path.exists():
        print(f"[WARN] NER train txt not found: {path}")
        return []

    content = read_text_file(path)
    lines = content.splitlines()

    non_empty = [x for x in lines if x.strip()]
    if not non_empty:
        return []

    bio_like_count = sum(1 for x in non_empty[:1000] if looks_like_bio_line(x))
    is_bio = bio_like_count / max(1, min(len(non_empty), 1000)) > 0.5

    texts = []

    if is_bio:
        sent_chars = []

        for line in lines:
            line = line.strip()

            if not line:
                if sent_chars:
                    sent = "".join(sent_chars)
                    texts.extend(split_long_text(sent))
                    sent_chars = []
                continue

            parts = line.split()
            if len(parts) >= 2:
                char = parts[0]
                sent_chars.append(char)

        if sent_chars:
            sent = "".join(sent_chars)
            texts.extend(split_long_text(sent))

        print(f"[OK] NER train txt(BIO): {path}, mlm_chunks={len(texts)}")

    else:
        # 普通文本：按行读，也可处理长行
        for line in lines:
            line = clean_text(line)
            if line:
                texts.extend(split_long_text(line))

        print(f"[OK] NER train txt(plain): {path}, mlm_chunks={len(texts)}")

    return texts


def load_raw_case_txt_dir(folder: Path):
    """读取原始医案 train 文件夹中的所有 txt。"""
    if not folder.exists():
        print(f"[WARN] raw case dir not found: {folder}")
        return []

    txt_files = sorted(folder.rglob("*.txt"))
    texts = []

    for file in txt_files:
        content = read_text_file(file)
        content = clean_text(content)
        if content:
            texts.extend(split_long_text(content))

    print(f"[OK] raw case dir: {folder}, txt_files={len(txt_files)}, mlm_chunks={len(texts)}")
    return texts


def deduplicate_texts(texts):
    """去重，保留原始顺序。"""
    seen = set()
    output = []

    for text in texts:
        text = clean_text(text)
        if not text:
            continue

        # 太短的文本对 MLM 帮助不大
        if len(text) < 10:
            continue

        if text in seen:
            continue

        seen.add(text)
        output.append(text)

    return output


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--re_train_json", default=DEFAULT_RE_TRAIN_JSON)
    parser.add_argument("--ner_train_txt", default=DEFAULT_NER_TRAIN_TXT)
    parser.add_argument("--raw_case_dir", default=DEFAULT_RAW_CASE_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)

    args = parser.parse_args()

    all_texts = []

    all_texts.extend(load_re_train_json(Path(args.re_train_json)))
    all_texts.extend(load_ner_train_txt(Path(args.ner_train_txt)))
    all_texts.extend(load_raw_case_txt_dir(Path(args.raw_case_dir)))

    before = len(all_texts)
    all_texts = deduplicate_texts(all_texts)
    after = len(all_texts)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for text in all_texts:
            f.write(text + "\n")

    print("=" * 80)
    print("MLM corpus build finished")
    print("=" * 80)
    print(f"Before dedup: {before}")
    print(f"After dedup : {after}")
    print(f"Output      : {output_path}")


if __name__ == "__main__":
    main()