# models/model1/decode.py
# -*- coding: utf-8 -*-
"""
model1: RSCD-v3 选择性关系约束解码模块
Relation-selective Constraint Decoding

目标：保留 v1 中对「方剂-处方用药、治法-方剂、证型-方剂、证型-治法」的有效提升，
同时放开 v1 中被误伤的「检查-检查结果、症状-证型」。

说明：
- dataset.py/model.py 不改，直接复用 GPLinker baseline checkpoint；
- args.threshold 仍然有效，可以继续用 search_threshold.py 搜阈值；
- 这版不是全面收紧，而是只对已经验证有效的关系收紧。
"""

import torch


# =========================
# 1. 关系特异性阈值偏移
# =========================
# 实际关系阈值 = args.threshold + delta
# v1 中「检查-检查结果」「症状-证型」被误伤，所以这两类不再提高阈值。
REL_THRESHOLD_DELTA = {
    "中医病名-证型": 0.00,
    "症状-证型": 0.00,       # 放开：该类样本最多，不能随便降召回
    "检查-检查结果": 0.00,   # 放开：v1 中该类 F1 明显下降
    "证型-治法": 0.00,       # v1 有小幅提升，保留轻约束但不提阈值
    "方剂-处方用药": 0.02,   # v1 明显提升，保留
    "治法-方剂": 0.03,       # v1 明显提升，保留
    "证型-方剂": 0.03,       # v1 明显提升，保留
}


# =========================
# 2. 关系特异性最大距离
# =========================
# 对被误伤的关系设置为极大值，相当于不做距离过滤。
REL_MAX_DISTANCE = {
    "中医病名-证型": 999999,
    "症状-证型": 999999,
    "检查-检查结果": 999999,
    "证型-治法": 160,
    "方剂-处方用药": 140,
    "治法-方剂": 100,
    "证型-方剂": 120,
}


# =========================
# 3. subject + predicate 的 object top-k
# =========================
# 对被误伤的关系设置为大值，相当于不限制。
SUBJECT_OBJECT_TOPK = {
    "中医病名-证型": 999,
    "症状-证型": 999,
    "检查-检查结果": 999,
    "证型-治法": 4,
    "方剂-处方用药": 40,
    "治法-方剂": 3,
    "证型-方剂": 3,
}


# =========================
# 4. 距离惩罚权重
# =========================
# 对被误伤的关系不做距离惩罚，只保留 v1 中有帮助的关系。
REL_DISTANCE_PENALTY = {
    "中医病名-证型": 0.0,
    "症状-证型": 0.0,
    "检查-检查结果": 0.0,
    "证型-治法": 0.0010,
    "方剂-处方用药": 0.0010,
    "治法-方剂": 0.0015,
    "证型-方剂": 0.0015,
}


def _get_rel_name(id2rel, rel_id):
    if rel_id in id2rel:
        return id2rel[rel_id]
    if str(rel_id) in id2rel:
        return id2rel[str(rel_id)]
    return None


def _get_relation_threshold(predicate, base_threshold):
    delta = REL_THRESHOLD_DELTA.get(predicate, 0.0)
    threshold = float(base_threshold) + float(delta)
    threshold = max(0.05, min(0.995, threshold))
    return threshold


def _token_span_to_text(text, offset_mapping, token_start, token_end):
    if token_start >= len(offset_mapping) or token_end >= len(offset_mapping):
        return ""

    char_start = offset_mapping[token_start][0]
    char_end = offset_mapping[token_end][1]

    if char_start == char_end:
        return ""

    if char_start < 0 or char_end > len(text) or char_start >= char_end:
        return ""

    return text[char_start:char_end].strip()


def _span_token_distance(span1, span2):
    """
    计算两个 token span 的最小间隔距离。
    span1/span2: (start, end)
    重叠或相邻时距离接近 0。
    """
    s1, e1 = span1
    s2, e2 = span2

    if e1 < s2:
        return s2 - e1
    if e2 < s1:
        return s1 - e2
    return 0


def _extract_spans(
    role_probs,
    valid_mask,
    threshold,
    max_span_len=80,
    max_spans=300,
    max_spans_per_head=3,
):
    """
    role_probs: [L, L]
    return:
        [(start, end, score), ...]
    """
    mask = role_probs > threshold
    pos = torch.where(mask)

    if pos[0].numel() == 0:
        return []

    candidates_by_head = {}

    for s, e in zip(pos[0].tolist(), pos[1].tolist()):
        if valid_mask[s].item() <= 0 or valid_mask[e].item() <= 0:
            continue

        if e < s:
            continue

        if e - s + 1 > max_span_len:
            continue

        score = role_probs[s, e].item()
        candidates_by_head.setdefault(s, []).append((s, e, score))

    spans = []

    for _, items in candidates_by_head.items():
        items = sorted(items, key=lambda x: x[2], reverse=True)
        items = items[:max_spans_per_head]
        spans.extend(items)

    spans = sorted(spans, key=lambda x: x[2], reverse=True)
    spans = spans[:max_spans]

    return spans


def _extract_links_relation_specific(
    rel_probs,
    valid_mask,
    base_threshold,
    id2rel,
    max_links=800,
):
    """
    rel_probs: [R, L, L]
    return:
        [(rel_id, i, j, score), ...]

    与原始 GPLinker 不同：这里每个 relation 使用自己的 threshold。
    """
    if not REL_THRESHOLD_DELTA:
        min_threshold = base_threshold
    else:
        min_delta = min(REL_THRESHOLD_DELTA.values())
        min_threshold = max(0.05, float(base_threshold) + float(min_delta))

    mask = rel_probs > min_threshold
    pos = torch.where(mask)

    if pos[0].numel() == 0:
        return []

    rel_ids = pos[0]
    rows = pos[1]
    cols = pos[2]
    scores = rel_probs[rel_ids, rows, cols]

    valid_items = []

    for rel_id, i, j, score in zip(
        rel_ids.tolist(),
        rows.tolist(),
        cols.tolist(),
        scores.tolist(),
    ):
        if valid_mask[i].item() <= 0 or valid_mask[j].item() <= 0:
            continue

        predicate = _get_rel_name(id2rel, rel_id)
        if predicate is None:
            continue

        rel_threshold = _get_relation_threshold(predicate, base_threshold)
        if score < rel_threshold:
            continue

        valid_items.append((rel_id, i, j, score))

    valid_items = sorted(valid_items, key=lambda x: x[3], reverse=True)
    valid_items = valid_items[:max_links]

    return valid_items


def _apply_subject_object_topk(triple_candidates):
    """
    triple_candidates:
        [(score, subject, predicate, object, distance), ...]

    按 subject + predicate 分组，只保留 top-k object。
    """
    grouped = {}

    for item in triple_candidates:
        score, subject, predicate, object_, distance = item
        key = (subject, predicate)
        grouped.setdefault(key, []).append(item)

    kept = []

    for (subject, predicate), items in grouped.items():
        topk = SUBJECT_OBJECT_TOPK.get(predicate, 10)
        items = sorted(items, key=lambda x: x[0], reverse=True)
        kept.extend(items[:topk])

    return kept


def decode_batch(outputs, batch, args, id2rel):
    base_threshold = getattr(args, "threshold", 0.5)

    # entity span 仍然使用统一阈值，避免因为关系阈值校准导致实体召回过低。
    entity_threshold = base_threshold
    rel_base_threshold = base_threshold

    entity_probs = outputs["entity_probs"]
    head_probs = outputs["head_probs"]
    tail_probs = outputs["tail_probs"]

    metas = batch["meta"]
    offset_mappings = batch["offset_mapping"]
    loss_mask = batch["loss_mask"]

    batch_size = entity_probs.size(0)

    max_span_len = 80
    max_entity_spans = 300
    max_rel_links = 1000
    max_triples_per_sample = 500

    decoded_records = []

    for b in range(batch_size):
        text = metas[b]["text"]
        offsets = offset_mappings[b]
        valid_mask = loss_mask[b]

        # 1. 解码 subject / object spans
        subject_spans = _extract_spans(
            entity_probs[b, 0],
            valid_mask,
            entity_threshold,
            max_span_len=max_span_len,
            max_spans=max_entity_spans,
            max_spans_per_head=3,
        )

        object_spans = _extract_spans(
            entity_probs[b, 1],
            valid_mask,
            entity_threshold,
            max_span_len=max_span_len,
            max_spans=max_entity_spans,
            max_spans_per_head=3,
        )

        if not subject_spans or not object_spans:
            decoded_records.append({
                "source_id": metas[b].get("source_id"),
                "chunk_id": metas[b].get("chunk_id"),
                "triple_list": [],
            })
            continue

        subjects_by_head = {}
        for s, e, score in subject_spans:
            subjects_by_head.setdefault(s, []).append((s, e, score))

        objects_by_head = {}
        for s, e, score in object_spans:
            objects_by_head.setdefault(s, []).append((s, e, score))

        # 2. 解码 head links 和 tail links：使用分关系阈值
        head_links = _extract_links_relation_specific(
            head_probs[b],
            valid_mask,
            rel_base_threshold,
            id2rel,
            max_links=max_rel_links,
        )

        tail_links = _extract_links_relation_specific(
            tail_probs[b],
            valid_mask,
            rel_base_threshold,
            id2rel,
            max_links=max_rel_links,
        )

        tail_score = {}
        for rel_id, st, ot, score in tail_links:
            key = (rel_id, st, ot)
            old = tail_score.get(key, 0.0)
            if score > old:
                tail_score[key] = score

        triple_candidates = []

        # 3. 组合三元组 + 关系距离约束
        for rel_id, sh, oh, h_score in head_links:
            predicate = _get_rel_name(id2rel, rel_id)
            if predicate is None:
                continue

            sub_candidates = subjects_by_head.get(sh, [])
            obj_candidates = objects_by_head.get(oh, [])

            if not sub_candidates or not obj_candidates:
                continue

            max_distance = REL_MAX_DISTANCE.get(predicate, 999999)
            penalty_weight = REL_DISTANCE_PENALTY.get(predicate, 0.0)

            for sub_s, sub_e, sub_score in sub_candidates:
                for obj_s, obj_e, obj_score in obj_candidates:
                    if sub_s == obj_s and sub_e == obj_e:
                        continue

                    key = (rel_id, sub_e, obj_e)
                    if key not in tail_score:
                        continue

                    distance = _span_token_distance(
                        (sub_s, sub_e),
                        (obj_s, obj_e),
                    )

                    if distance > max_distance:
                        continue

                    subject = _token_span_to_text(text, offsets, sub_s, sub_e)
                    object_ = _token_span_to_text(text, offsets, obj_s, obj_e)

                    if not subject or not object_:
                        continue

                    if len(subject) > 80 or len(object_) > 120:
                        continue

                    final_score = (
                        h_score
                        + tail_score[key]
                        + sub_score
                        + obj_score
                        - penalty_weight * distance
                    )

                    triple_candidates.append(
                        (final_score, subject, predicate, object_, distance)
                    )

        # 4. 先排序，再做 subject+predicate top-k
        triple_candidates = sorted(
            triple_candidates,
            key=lambda x: x[0],
            reverse=True,
        )

        triple_candidates = _apply_subject_object_topk(triple_candidates)

        triple_candidates = sorted(
            triple_candidates,
            key=lambda x: x[0],
            reverse=True,
        )

        triple_candidates = triple_candidates[:max_triples_per_sample]

        triple_set = set()
        for _, subject, predicate, object_, _ in triple_candidates:
            triple_set.add((subject, predicate, object_))

        decoded_records.append({
            "source_id": metas[b].get("source_id"),
            "chunk_id": metas[b].get("chunk_id"),
            "triple_list": [list(x) for x in sorted(triple_set)],
        })

    return decoded_records
