# common/data_reader.py
from collections import defaultdict
from typing import Dict, List, Tuple, Any

from common.utils import load_json


Triple = Tuple[str, str, str]


def load_rel_maps(rel2id_file: str, id2rel_file: str = None):
    rel2id = load_json(rel2id_file)

    if id2rel_file is not None:
        id2rel = load_json(id2rel_file)
    else:
        id2rel = {str(v): k for k, v in rel2id.items()}

    # 统一 key 为 str，避免 json 读入后 key 类型混乱
    id2rel = {str(k): v for k, v in id2rel.items()}

    return rel2id, id2rel


def load_chunks(path: str):
    return load_json(path)


def normalize_text(s):
    if s is None:
        return ""
    return str(s).strip()


def normalize_triple(triple) -> Triple:
    """
    支持：
    ["subject", "predicate", "object"]
    {"subject": ..., "predicate": ..., "object": ...}
    """
    if isinstance(triple, dict):
        s = triple.get("subject", "")
        p = triple.get("predicate", "")
        o = triple.get("object", "")
    else:
        s, p, o = triple[0], triple[1], triple[2]

    return normalize_text(s), normalize_text(p), normalize_text(o)


def get_gold_triples_from_sample(sample: Dict[str, Any]) -> List[Triple]:
    if "triple_list" in sample and sample["triple_list"] is not None:
        triples = sample["triple_list"]
        return [normalize_triple(t) for t in triples]

    spo_list = sample.get("spo_list", [])
    return [normalize_triple(spo) for spo in spo_list]


def get_meta_from_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    每个 Dataset 最好把这个 meta 放进 batch["meta"]。
    """
    return {
        "source_id": sample.get("source_id", sample.get("id")),
        "chunk_id": sample.get("chunk_id", 0),
        "offset_start": sample.get("offset_start", 0),
        "offset_end": sample.get("offset_end", len(sample.get("text", ""))),
        "text": sample.get("text", ""),
        "triple_list": get_gold_triples_from_sample(sample),
        "spo_list": sample.get("spo_list", []),
    }


def gold_by_source_from_samples(samples: List[Dict[str, Any]]):
    """
    从 chunk 样本聚合回原始 source_id。
    由于滑动窗口可能重复包含同一三元组，所以用 set 去重。
    """
    gold_by_source = defaultdict(set)

    for sample in samples:
        source_id = str(sample.get("source_id", sample.get("id")))
        triples = get_gold_triples_from_sample(sample)

        for triple in triples:
            gold_by_source[source_id].add(normalize_triple(triple))

    return dict(gold_by_source)


def gold_by_source_from_metas(metas: List[Dict[str, Any]]):
    gold_by_source = defaultdict(set)

    for meta in metas:
        source_id = str(meta.get("source_id"))
        triples = meta.get("triple_list", [])

        for triple in triples:
            gold_by_source[source_id].add(normalize_triple(triple))

    return dict(gold_by_source)


def standardize_decode_output(decoded, metas: List[Dict[str, Any]]):
    """
    将各模型 decode_batch 的输出统一成：
    [
      {
        "source_id": ...,
        "chunk_id": ...,
        "triple_list": [...]
      }
    ]

    decode_batch 可以返回两种形式：
    1. [{"triple_list": [...], ...}, ...]
    2. [[triple1, triple2], [triple1, triple2], ...]
    """
    records = []

    if decoded is None:
        decoded = [[] for _ in metas]

    if len(decoded) != len(metas):
        raise ValueError(
            f"decode 输出数量与 batch 样本数不一致：decoded={len(decoded)}, metas={len(metas)}"
        )

    for item, meta in zip(decoded, metas):
        source_id = meta.get("source_id")
        chunk_id = meta.get("chunk_id")

        if isinstance(item, dict):
            triples = item.get("triple_list", item.get("pred_triples", []))
            source_id = item.get("source_id", source_id)
            chunk_id = item.get("chunk_id", chunk_id)
        else:
            triples = item

        records.append({
            "source_id": source_id,
            "chunk_id": chunk_id,
            "triple_list": [normalize_triple(t) for t in triples],
        })

    return records