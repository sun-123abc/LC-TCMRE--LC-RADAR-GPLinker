# models/gplinker/dataset.py
import torch
from torch.utils.data import Dataset

from common.data_reader import load_chunks, get_meta_from_sample


class REDataset(Dataset):
    """
    GPLinker 数据集动态转换器。

    统一输入格式：
    {
        "text": "...",
        "spo_list": [
            {
                "subject": "...",
                "predicate": "...",
                "object": "...",
                "subject_start": ...,
                "subject_end": ...,
                "object_start": ...,
                "object_end": ...
            }
        ]
    }

    构造标签：
    1. entity_labels: [2, L, L]
       0 = subject span
       1 = object span

    2. head_labels: [R, L, L]
       relation 下 subject_head -> object_head

    3. tail_labels: [R, L, L]
       relation 下 subject_tail -> object_tail
    """

    def __init__(self, data_path, tokenizer, rel2id, args, is_train=True):
        self.data = load_chunks(data_path)
        self.tokenizer = tokenizer
        self.rel2id = rel2id
        self.args = args
        self.is_train = is_train
        self.max_len = args.max_len

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _char_span_to_token_span(offset_mapping, char_start, char_end):
        token_indices = []

        for idx, (start, end) in enumerate(offset_mapping):
            if start == end:
                continue

            if not (end <= char_start or start >= char_end):
                token_indices.append(idx)

        if not token_indices:
            return None

        return token_indices[0], token_indices[-1]

    def _encode_text(self, text):
        return self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding=False,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            add_special_tokens=True,
        )

    def _build_token_triples(self, sample, offset_mapping):
        """
        转换为 token 级三元组：
        (sub_head, sub_tail, rel_id, obj_head, obj_tail)
        """
        token_triples = []

        for spo in sample.get("spo_list", []):
            predicate = spo.get("predicate")
            if predicate not in self.rel2id:
                continue

            ss = spo.get("subject_start")
            se = spo.get("subject_end")
            os = spo.get("object_start")
            oe = spo.get("object_end")

            if not all(isinstance(x, int) for x in [ss, se, os, oe]):
                continue

            sub_span = self._char_span_to_token_span(offset_mapping, ss, se)
            obj_span = self._char_span_to_token_span(offset_mapping, os, oe)

            if sub_span is None or obj_span is None:
                continue

            sh, st = sub_span
            oh, ot = obj_span
            rel_id = self.rel2id[predicate]

            token_triples.append((sh, st, rel_id, oh, ot))

        return token_triples

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = sample.get("text", "")
        meta = get_meta_from_sample(sample)

        encoded = self._encode_text(text)

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        token_type_ids = encoded.get("token_type_ids", [0] * len(input_ids))
        offset_mapping = encoded["offset_mapping"]
        special_tokens_mask = encoded["special_tokens_mask"]

        seq_len = len(input_ids)

        loss_mask = [
            1 if attention_mask[i] == 1 and special_tokens_mask[i] == 0 else 0
            for i in range(seq_len)
        ]

        feature = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "special_tokens_mask": special_tokens_mask,
            "loss_mask": loss_mask,
            "offset_mapping": offset_mapping,
            "meta": meta,
        }

        if not self.is_train:
            return feature

        rel_num = len(self.rel2id)
        token_triples = self._build_token_triples(sample, offset_mapping)

        # [2, L, L]
        entity_labels = [
            [
                [0.0 for _ in range(seq_len)]
                for _ in range(seq_len)
            ]
            for _ in range(2)
        ]

        # [R, L, L]
        head_labels = [
            [
                [0.0 for _ in range(seq_len)]
                for _ in range(seq_len)
            ]
            for _ in range(rel_num)
        ]

        tail_labels = [
            [
                [0.0 for _ in range(seq_len)]
                for _ in range(seq_len)
            ]
            for _ in range(rel_num)
        ]

        for sh, st, rel_id, oh, ot in token_triples:
            if max(sh, st, oh, ot) >= seq_len:
                continue

            # entity role
            entity_labels[0][sh][st] = 1.0
            entity_labels[1][oh][ot] = 1.0

            # relation links
            head_labels[rel_id][sh][oh] = 1.0
            tail_labels[rel_id][st][ot] = 1.0

        feature.update({
            "entity_labels": entity_labels,
            "head_labels": head_labels,
            "tail_labels": tail_labels,
        })

        return feature

    @staticmethod
    def _pad_1d(seq, max_len, pad_value=0):
        return seq + [pad_value] * (max_len - len(seq))

    @staticmethod
    def _pad_offsets(offsets, max_len):
        return offsets + [(0, 0)] * (max_len - len(offsets))

    @staticmethod
    def _pad_matrix_labels(labels, head_num, max_len):
        """
        labels: [H, L, L]
        pad to [H, max_len, max_len]
        """
        padded_all = []

        for h in range(head_num):
            mat = labels[h]
            cur_len = len(mat)

            padded = []
            for row in mat:
                padded.append(row + [0.0] * (max_len - len(row)))

            if cur_len < max_len:
                padded += [
                    [0.0 for _ in range(max_len)]
                    for _ in range(max_len - cur_len)
                ]

            padded_all.append(padded)

        return padded_all

    def collate_fn(self, batch):
        batch_max_len = max(len(x["input_ids"]) for x in batch)
        rel_num = len(self.rel2id)

        input_ids = []
        attention_mask = []
        token_type_ids = []
        special_tokens_mask = []
        loss_mask = []
        offset_mapping = []
        metas = []

        has_labels = "entity_labels" in batch[0]

        entity_labels = []
        head_labels = []
        tail_labels = []

        for item in batch:
            input_ids.append(self._pad_1d(item["input_ids"], batch_max_len, 0))
            attention_mask.append(self._pad_1d(item["attention_mask"], batch_max_len, 0))
            token_type_ids.append(self._pad_1d(item["token_type_ids"], batch_max_len, 0))
            special_tokens_mask.append(self._pad_1d(item["special_tokens_mask"], batch_max_len, 1))
            loss_mask.append(self._pad_1d(item["loss_mask"], batch_max_len, 0))
            offset_mapping.append(self._pad_offsets(item["offset_mapping"], batch_max_len))
            metas.append(item["meta"])

            if has_labels:
                entity_labels.append(
                    self._pad_matrix_labels(item["entity_labels"], 2, batch_max_len)
                )
                head_labels.append(
                    self._pad_matrix_labels(item["head_labels"], rel_num, batch_max_len)
                )
                tail_labels.append(
                    self._pad_matrix_labels(item["tail_labels"], rel_num, batch_max_len)
                )

        output = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "special_tokens_mask": torch.tensor(special_tokens_mask, dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.float),
            "offset_mapping": offset_mapping,
            "meta": metas,
        }

        if has_labels:
            output.update({
                "entity_labels": torch.tensor(entity_labels, dtype=torch.float),
                "head_labels": torch.tensor(head_labels, dtype=torch.float),
                "tail_labels": torch.tensor(tail_labels, dtype=torch.float),
            })

        return output