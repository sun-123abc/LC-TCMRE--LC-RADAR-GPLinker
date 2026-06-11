# models/gplinker/model.py
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class GlobalPointer(nn.Module):
    """
    GlobalPointer 层。

    输入：
        hidden_states: [B, L, H]
    输出：
        logits: [B, heads, L, L]
    """

    def __init__(self, hidden_size, heads, head_size=64, use_rope=True, tril_mask=False):
        super().__init__()

        self.heads = heads
        self.head_size = head_size
        self.use_rope = use_rope
        self.tril_mask = tril_mask

        self.dense = nn.Linear(hidden_size, heads * head_size * 2)

    def _sinusoidal_position_embedding(self, batch_size, seq_len, output_dim, device):
        position_ids = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(-1)
        indices = torch.arange(output_dim // 2, dtype=torch.float, device=device)
        indices = torch.pow(10000.0, -2 * indices / output_dim)

        embeddings = position_ids * indices
        embeddings = torch.stack(
            [torch.sin(embeddings), torch.cos(embeddings)],
            dim=-1,
        )
        embeddings = embeddings.reshape(seq_len, output_dim)
        embeddings = embeddings.unsqueeze(0).repeat(batch_size, 1, 1)

        return embeddings

    def _add_rope(self, qw, kw):
        """
        qw, kw: [B, L, heads, head_size]
        """
        batch_size, seq_len = qw.size(0), qw.size(1)
        device = qw.device

        pos_emb = self._sinusoidal_position_embedding(
            batch_size=batch_size,
            seq_len=seq_len,
            output_dim=self.head_size,
            device=device,
        )

        cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)
        sin_pos = pos_emb[..., 0::2].repeat_interleave(2, dim=-1)

        cos_pos = cos_pos.unsqueeze(2)
        sin_pos = sin_pos.unsqueeze(2)

        qw2 = torch.stack(
            [-qw[..., 1::2], qw[..., 0::2]],
            dim=-1,
        )
        qw2 = qw2.reshape_as(qw)

        kw2 = torch.stack(
            [-kw[..., 1::2], kw[..., 0::2]],
            dim=-1,
        )
        kw2 = kw2.reshape_as(kw)

        qw = qw * cos_pos + qw2 * sin_pos
        kw = kw * cos_pos + kw2 * sin_pos

        return qw, kw

    def forward(self, hidden_states, mask=None):
        """
        hidden_states: [B, L, H]
        mask: [B, L]
        """
        batch_size, seq_len, _ = hidden_states.size()

        outputs = self.dense(hidden_states)
        outputs = outputs.view(
            batch_size,
            seq_len,
            self.heads,
            self.head_size * 2,
        )

        qw = outputs[..., : self.head_size]
        kw = outputs[..., self.head_size :]

        if self.use_rope:
            qw, kw = self._add_rope(qw, kw)

        logits = torch.einsum("bmhd,bnhd->bhmn", qw, kw)
        logits = logits / math.sqrt(self.head_size)

        if mask is not None:
            # mask: [B, L]
            mask1 = mask[:, None, :, None]
            mask2 = mask[:, None, None, :]
            pair_mask = mask1 * mask2
            logits = logits * pair_mask - (1.0 - pair_mask) * 1e12

        if self.tril_mask:
            tril_mask = torch.tril(
                torch.ones(seq_len, seq_len, device=hidden_states.device),
                diagonal=-1,
            )
            logits = logits - tril_mask[None, None, :, :] * 1e12

        return logits


class REModel(nn.Module):
    """
    GPLinker 模型。

    统一框架要求模型类名必须是 REModel。

    1. entity_pointer:
       heads = 2
       0 = subject span
       1 = object span

    2. head_pointer:
       heads = rel_num
       subject_head -> object_head

    3. tail_pointer:
       heads = rel_num
       subject_tail -> object_tail
    """

    def __init__(self, args, rel2id, id2rel):
        super().__init__()

        self.args = args
        self.rel2id = rel2id
        self.id2rel = id2rel
        self.rel_num = len(rel2id)

        self.bert = AutoModel.from_pretrained(
            args.pretrained_model,
            local_files_only=True,
        )

        hidden_size = self.bert.config.hidden_size
        dropout_prob = getattr(self.bert.config, "hidden_dropout_prob", 0.1)

        self.dropout = nn.Dropout(dropout_prob)

        pointer_head_size = getattr(args, "pointer_head_size", 64)

        # subject / object span
        self.entity_pointer = GlobalPointer(
            hidden_size=hidden_size,
            heads=2,
            head_size=pointer_head_size,
            use_rope=True,
            tril_mask=True,
        )

        # subject head - object head
        self.head_pointer = GlobalPointer(
            hidden_size=hidden_size,
            heads=self.rel_num,
            head_size=pointer_head_size,
            use_rope=True,
            tril_mask=False,
        )

        # subject tail - object tail
        self.tail_pointer = GlobalPointer(
            hidden_size=hidden_size,
            heads=self.rel_num,
            head_size=pointer_head_size,
            use_rope=True,
            tril_mask=False,
        )

    def _get_bert_outputs(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)

        return sequence_output

    @staticmethod
    def _build_pair_mask(loss_mask):
        """
        loss_mask: [B, L]
        return: [B, 1, L, L]
        """
        pair_mask = (
            loss_mask.unsqueeze(1).unsqueeze(3)
            * loss_mask.unsqueeze(1).unsqueeze(2)
        )
        return pair_mask

    @staticmethod
    def _build_upper_mask(loss_mask):
        """
        entity span 只允许 start <= end。
        loss_mask: [B, L]
        return: [B, 1, L, L]
        """
        batch_size, seq_len = loss_mask.size()

        pair_mask = (
            loss_mask.unsqueeze(1).unsqueeze(3)
            * loss_mask.unsqueeze(1).unsqueeze(2)
        )

        upper_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=loss_mask.device),
            diagonal=0,
        )

        pair_mask = pair_mask * upper_mask.unsqueeze(0).unsqueeze(0)

        return pair_mask

    def _masked_bce_logits_loss(self, logits, labels, mask, pos_weight=30.0):
        """
        logits / labels: [B, H, L, L]
        mask: [B, 1, L, L]
        """
        loss = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            reduction="none",
        )

        weight = torch.ones_like(loss)
        weight = weight + labels * (pos_weight - 1.0)
        loss = loss * weight

        if mask is not None:
            mask = mask.expand_as(loss)
            loss = loss * mask
            denom = mask.sum()
        else:
            denom = torch.numel(loss)

        denom = torch.clamp(
            torch.as_tensor(denom, dtype=torch.float, device=logits.device),
            min=1.0,
        )

        return loss.sum() / denom

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        special_tokens_mask=None,
        loss_mask=None,
        offset_mapping=None,
        entity_labels=None,
        head_labels=None,
        tail_labels=None,
    ):
        if loss_mask is None:
            loss_mask = attention_mask.float()

        sequence_output = self._get_bert_outputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        entity_logits = self.entity_pointer(
            sequence_output,
            mask=loss_mask,
        )

        head_logits = self.head_pointer(
            sequence_output,
            mask=loss_mask,
        )

        tail_logits = self.tail_pointer(
            sequence_output,
            mask=loss_mask,
        )

        output = {
            "entity_logits": entity_logits,
            "head_logits": head_logits,
            "tail_logits": tail_logits,
            "entity_probs": torch.sigmoid(entity_logits),
            "head_probs": torch.sigmoid(head_logits),
            "tail_probs": torch.sigmoid(tail_logits),
        }

        if entity_labels is not None:
            entity_mask = self._build_upper_mask(loss_mask)
            pair_mask = self._build_pair_mask(loss_mask)

            entity_loss = self._masked_bce_logits_loss(
                entity_logits,
                entity_labels,
                entity_mask,
                pos_weight=30.0,
            )

            head_loss = self._masked_bce_logits_loss(
                head_logits,
                head_labels,
                pair_mask,
                pos_weight=80.0,
            )

            tail_loss = self._masked_bce_logits_loss(
                tail_logits,
                tail_labels,
                pair_mask,
                pos_weight=80.0,
            )

            loss = (entity_loss + head_loss + tail_loss) / 3.0

            output.update(
                {
                    "loss": loss,
                    "entity_loss": entity_loss,
                    "head_loss": head_loss,
                    "tail_loss": tail_loss,
                }
            )

        return output