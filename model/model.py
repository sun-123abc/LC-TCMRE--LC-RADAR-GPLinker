# models/model2/model.py
# -*- coding: utf-8 -*-
"""
model2 = model1(RSCD-v3) + HNFS loss

HNFS: Hard Negative Focal Suppression
中文名：难负样本焦点抑制模块

设计目的：
1. 保留 GPLinker 的主体结构，不再加入 relation_classifier，避免 RGF 造成训练偏移；
2. 保留 model1 的 RSCD-v3 解码模块，由 decode.py 实现；
3. 在训练阶段只改 head/tail link 的损失函数，重点压制“合法关系类型内部的错配”；
4. entity span 分支仍使用稳定的加权 BCE，避免实体召回被破坏。

注意：
- 这个 model.py 新增不了额外参数，因此结构和 GPLinker baseline 一致；
- 理论上可以加载 gplinker checkpoint，但正式实验建议重新训练，保证训练损失一致；
- dataset.py 不需要改，decode.py 使用 model1 的 RSCD-v3。
"""

import torch
import torch.nn.functional as F

from models.gplinker.model import REModel as GPLinkerBase


class REModel(GPLinkerBase):
    """
    GPLinker + 难负样本焦点抑制损失。

    原始 GPLinker 的 head_labels/tail_labels 极度稀疏，大量负样本中只有少量是
    真正容易被误判的 hard negatives。普通 BCE 会让大量 easy negatives 主导训练，
    而本模块通过 negative focal factor 让模型更关注高置信错误负例，从而减少 FP。
    """

    def __init__(self, args, rel2id, id2rel):
        super().__init__(args, rel2id, id2rel)

        # 保持实体识别分支稳定。
        self.entity_pos_weight = 30.0

        # head/tail link 使用更强的正样本权重，恢复到你前面更稳的配置。
        self.link_pos_weight = 90.0

        # 负样本 focal 系数：越大越聚焦高置信负例。
        # 建议先用 2.0；如果 recall 掉太多，可改成 1.5。
        self.neg_gamma = 2.0

        # 正样本不做 focal 衰减，避免损伤召回。
        self.pos_gamma = 0.0

        # 对 hard negative loss 轻微放大，目标是压 FP。
        # 如果模型变得太保守，可改成 1.0。
        self.neg_loss_weight = 1.2

    @staticmethod
    def _safe_denom(mask_or_num, device):
        denom = torch.as_tensor(mask_or_num, dtype=torch.float, device=device)
        return torch.clamp(denom, min=1.0)

    def _weighted_bce_logits_loss(self, logits, labels, mask=None, pos_weight=30.0):
        """
        稳定的加权 BCE，用于 entity span 分支。
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

        denom = self._safe_denom(denom, logits.device)
        return loss.sum() / denom

    def _asymmetric_focal_logits_loss(
        self,
        logits,
        labels,
        mask=None,
        pos_weight=90.0,
        neg_gamma=2.0,
        pos_gamma=0.0,
        neg_loss_weight=1.2,
    ):
        """
        用于 head/tail link 的难负样本焦点抑制损失。

        对正样本：
            保持较高 pos_weight，且 pos_gamma=0，尽量不伤 recall。

        对负样本：
            使用 p^gamma 作为权重。
            p 越大，说明模型越容易把负样本误判为正样本，损失越大；
            p 越小，说明是 easy negative，损失被降低。
        """
        probs = torch.sigmoid(logits)

        bce = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            reduction="none",
        )

        positive_mask = labels
        negative_mask = 1.0 - labels

        # 正样本：不做或少做 focal 衰减，保护召回。
        if pos_gamma > 0:
            pos_focal = torch.pow(torch.clamp(1.0 - probs, min=1e-6), pos_gamma)
        else:
            pos_focal = torch.ones_like(probs)

        # 负样本：p 越高越危险，权重越大。
        neg_focal = torch.pow(torch.clamp(probs, min=1e-6), neg_gamma)

        pos_loss = bce * positive_mask * pos_focal * pos_weight
        neg_loss = bce * negative_mask * neg_focal * neg_loss_weight

        loss = pos_loss + neg_loss

        if mask is not None:
            mask = mask.expand_as(loss)
            loss = loss * mask
            denom = mask.sum()
        else:
            denom = torch.numel(loss)

        denom = self._safe_denom(denom, logits.device)
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
        **kwargs,
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

            entity_loss = self._weighted_bce_logits_loss(
                entity_logits,
                entity_labels,
                entity_mask,
                pos_weight=self.entity_pos_weight,
            )

            head_loss = self._asymmetric_focal_logits_loss(
                head_logits,
                head_labels,
                pair_mask,
                pos_weight=self.link_pos_weight,
                neg_gamma=self.neg_gamma,
                pos_gamma=self.pos_gamma,
                neg_loss_weight=self.neg_loss_weight,
            )

            tail_loss = self._asymmetric_focal_logits_loss(
                tail_logits,
                tail_labels,
                pair_mask,
                pos_weight=self.link_pos_weight,
                neg_gamma=self.neg_gamma,
                pos_gamma=self.pos_gamma,
                neg_loss_weight=self.neg_loss_weight,
            )

            loss = (entity_loss + head_loss + tail_loss) / 3.0

            output.update({
                "loss": loss,
                "entity_loss": entity_loss,
                "head_loss": head_loss,
                "tail_loss": tail_loss,
            })

        return output
