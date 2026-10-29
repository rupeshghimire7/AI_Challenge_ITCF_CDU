"""Three-head ABSA model.

Heads:
  category_logits: [batch, num_categories]
  attribute_logits: [batch, num_attributes]
  polarity_logits: [batch, num_categories, num_attributes, num_polarities]

Polarity is predicted per category/attribute pair, rather than once per sentence. This is
important when a sentence contains multiple opinions with different polarities.

Usage: nothing to run on its own. custom_train.py trains it and custom_inference.py loads it; in
Python, from the models/ folder:
    from custom_model import CustomABSAModel
    model = CustomABSAModel("BAAI/bge-small-en-v1.5", num_categories=22, num_attributes=9)
    out = model(input_ids, attention_mask)          # the three logits above
    loss, parts = model.compute_loss(out, batch)    # batch: category_targets, attribute_targets,
                                                    # polarity_targets (-100 where a pair has no label)
"""
from __future__ import annotations

from typing import Optional
import torch
from torch import nn
from transformers import AutoModel


class CustomABSAModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_categories: int,
        num_attributes: int,
        num_polarities: int = 3,
        dropout: float = 0.1,
        pooling: str = "mean",
        trust_remote_code: bool = True,
        encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        if pooling not in {"mean", "cls"}:
            raise ValueError("pooling must be 'mean' or 'cls'")
        self.model_name = model_name
        self.num_categories = num_categories
        self.num_attributes = num_attributes
        self.num_polarities = num_polarities
        self.pooling = pooling
        self.encoder = encoder or AutoModel.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        hidden_size = getattr(self.encoder.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self.encoder.config, "d_model", None)
        if hidden_size is None:
            raise ValueError("Cannot infer encoder hidden size from config.")
        self.dropout = nn.Dropout(dropout)
        self.category_head = nn.Linear(hidden_size, num_categories)
        self.attribute_head = nn.Linear(hidden_size, num_attributes)
        self.polarity_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_categories * num_attributes * num_polarities),
        )

    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        if self.pooling == "cls":
            pooled = hidden[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        return self.dropout(pooled)

    def forward(self, input_ids, attention_mask):
        pooled = self.encode(input_ids, attention_mask)
        b = pooled.shape[0]
        category_logits = self.category_head(pooled)
        attribute_logits = self.attribute_head(pooled)
        polarity_logits = self.polarity_head(pooled).view(
            b, self.num_categories, self.num_attributes, self.num_polarities
        )
        return {
            "category_logits": category_logits,
            "attribute_logits": attribute_logits,
            "polarity_logits": polarity_logits,
        }

    def compute_loss(self, outputs, batch, polarity_class_weights=None,
                     category_weight=1.0, attribute_weight=1.0, polarity_weight=1.0):
        """Targets: category_targets [B,C], attribute_targets [B,A],
        polarity_targets [B,C,A] with -100 for pairs without a polarity label.
        """
        cat_loss = nn.functional.binary_cross_entropy_with_logits(
            outputs["category_logits"], batch["category_targets"].float()
        )
        attr_loss = nn.functional.binary_cross_entropy_with_logits(
            outputs["attribute_logits"], batch["attribute_targets"].float()
        )
        pol_logits = outputs["polarity_logits"].reshape(-1, self.num_polarities)
        pol_targets = batch["polarity_targets"].reshape(-1)
        valid = pol_targets.ne(-100)
        if valid.any():
            pol_loss = nn.functional.cross_entropy(
                pol_logits[valid], pol_targets[valid],
                weight=polarity_class_weights,
            )
        else:
            pol_loss = pol_logits.sum() * 0.0
        total = category_weight * cat_loss + attribute_weight * attr_loss + polarity_weight * pol_loss
        return total, {
            "total": total.detach(),
            "category": cat_loss.detach(),
            "attribute": attr_loss.detach(),
            "polarity": pol_loss.detach(),
        }
