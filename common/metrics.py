# common/metrics.py
from collections import defaultdict
from typing import Dict, Set, Tuple, List, Any

from common.data_reader import normalize_triple


Triple = Tuple[str, str, str]


def aggregate_chunk_records(records: List[Dict[str, Any]]) -> Dict[str, Set[Triple]]:
    """
    chunk 级预测结果聚合回 source_id。
    滑动窗口产生的重复预测在这里去重。
    """
    pred_by_source = defaultdict(set)

    for record in records:
        source_id = str(record.get("source_id"))
        triples = record.get("triple_list", [])

        for triple in triples:
            pred_by_source[source_id].add(normalize_triple(triple))

    return dict(pred_by_source)


def compute_prf(gold_by_source: Dict[str, Set[Triple]],
                pred_by_source: Dict[str, Set[Triple]]):
    all_source_ids = set(gold_by_source.keys()) | set(pred_by_source.keys())

    gold_total = 0
    pred_total = 0
    correct_total = 0

    for source_id in all_source_ids:
        gold_set = set(gold_by_source.get(source_id, set()))
        pred_set = set(pred_by_source.get(source_id, set()))

        gold_total += len(gold_set)
        pred_total += len(pred_set)
        correct_total += len(gold_set & pred_set)

    precision = correct_total / pred_total if pred_total > 0 else 0.0
    recall = correct_total / gold_total if gold_total > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "gold": gold_total,
        "pred": pred_total,
        "correct": correct_total,
    }


def compute_relation_prf(gold_by_source: Dict[str, Set[Triple]],
                         pred_by_source: Dict[str, Set[Triple]]):
    relations = set()

    for triples in gold_by_source.values():
        for _, p, _ in triples:
            relations.add(p)

    for triples in pred_by_source.values():
        for _, p, _ in triples:
            relations.add(p)

    result = {}

    for rel in sorted(relations):
        rel_gold = {}
        rel_pred = {}

        for source_id, triples in gold_by_source.items():
            rel_gold[source_id] = {t for t in triples if t[1] == rel}

        for source_id, triples in pred_by_source.items():
            rel_pred[source_id] = {t for t in triples if t[1] == rel}

        result[rel] = compute_prf(rel_gold, rel_pred)

    return result


def format_metric_line(metrics: Dict[str, float]) -> str:
    return (
        f"P={metrics['precision'] * 100:.2f}, "
        f"R={metrics['recall'] * 100:.2f}, "
        f"F1={metrics['f1'] * 100:.2f}, "
        f"Gold={metrics['gold']}, "
        f"Pred={metrics['pred']}, "
        f"Correct={metrics['correct']}"
    )


def build_source_prediction_records(pred_by_source: Dict[str, Set[Triple]]):
    records = []

    for source_id, triples in pred_by_source.items():
        records.append({
            "source_id": source_id,
            "triple_list": [list(t) for t in sorted(triples)]
        })

    return records