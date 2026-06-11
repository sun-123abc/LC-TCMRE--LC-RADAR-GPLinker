# models/gplinker/decode.py
import torch


def _get_rel_name(id2rel, rel_id):
    if rel_id in id2rel:
        return id2rel[rel_id]
    if str(rel_id) in id2rel:
        return id2rel[str(rel_id)]
    return None


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

    for head, items in candidates_by_head.items():
        items = sorted(items, key=lambda x: x[2], reverse=True)
        items = items[:max_spans_per_head]
        spans.extend(items)

    spans = sorted(spans, key=lambda x: x[2], reverse=True)
    spans = spans[:max_spans]

    return spans


def _extract_links(
    rel_probs,
    valid_mask,
    threshold,
    max_links=800,
):
    """
    rel_probs: [R, L, L]
    return:
        [(rel_id, i, j, score), ...]
    """
    mask = rel_probs > threshold
    pos = torch.where(mask)

    if pos[0].numel() == 0:
        return []

    rel_ids = pos[0]
    rows = pos[1]
    cols = pos[2]
    scores = rel_probs[rel_ids, rows, cols]

    valid_items = []

    for rel_id, i, j, score in zip(rel_ids.tolist(), rows.tolist(), cols.tolist(), scores.tolist()):
        if valid_mask[i].item() <= 0 or valid_mask[j].item() <= 0:
            continue
        valid_items.append((rel_id, i, j, score))

    valid_items = sorted(valid_items, key=lambda x: x[3], reverse=True)
    valid_items = valid_items[:max_links]

    return valid_items


def decode_batch(outputs, batch, args, id2rel):
    threshold = getattr(args, "threshold", 0.5)

    # GPLinker 一般阈值后面通过 search_threshold 搜。
    # 训练中先用 0.5 保持输出稳定。
    entity_threshold = threshold
    rel_threshold = threshold

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

        # 2. 解码 head links 和 tail links
        head_links = _extract_links(
            head_probs[b],
            valid_mask,
            rel_threshold,
            max_links=max_rel_links,
        )

        tail_links = _extract_links(
            tail_probs[b],
            valid_mask,
            rel_threshold,
            max_links=max_rel_links,
        )

        tail_score = {}
        for rel_id, st, ot, score in tail_links:
            key = (rel_id, st, ot)
            old = tail_score.get(key, 0.0)
            if score > old:
                tail_score[key] = score

        triple_candidates = []

        # 3. 组合三元组
        for rel_id, sh, oh, h_score in head_links:
            predicate = _get_rel_name(id2rel, rel_id)
            if predicate is None:
                continue

            sub_candidates = subjects_by_head.get(sh, [])
            obj_candidates = objects_by_head.get(oh, [])

            if not sub_candidates or not obj_candidates:
                continue

            for sub_s, sub_e, sub_score in sub_candidates:
                for obj_s, obj_e, obj_score in obj_candidates:
                    if sub_s == obj_s and sub_e == obj_e:
                        continue

                    key = (rel_id, sub_e, obj_e)
                    if key not in tail_score:
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
                    )

                    triple_candidates.append(
                        (final_score, subject, predicate, object_)
                    )

        triple_candidates = sorted(
            triple_candidates,
            key=lambda x: x[0],
            reverse=True,
        )
        triple_candidates = triple_candidates[:max_triples_per_sample]

        triple_set = set()
        for _, subject, predicate, object_ in triple_candidates:
            triple_set.add((subject, predicate, object_))

        decoded_records.append({
            "source_id": metas[b].get("source_id"),
            "chunk_id": metas[b].get("chunk_id"),
            "triple_list": [list(x) for x in sorted(triple_set)],
        })

    return decoded_records