"""Model B: bge-small-en-v1.5 (BAAI, 2023), a BERT encoder loaded with transformers' BertModel.

Architecture (config.json): 12 transformer layers of hidden size 384, 12 attention heads, feed-forward 1536, learned
absolute positions (512 max), 2 token types, 30,522-token uncased WordPiece vocabulary. Post-norm: each layer is
h = LayerNorm(h + attention(h)); h = LayerNorm(h + FFN(h)). The pretrained pooler layer is not loaded: BGE embeddings
use the raw [CLS] state.

Input framing (the tokenizer's template):
  sentence   [CLS] text [SEP]
  context    [CLS] prev [SEP] target [SEP] next [SEP]   context tokens get BERT's token type 1, the target type 0
  padding    [PAD] (id 0) on the right, masked out of attention
  pooling    the [CLS] state (BGE documentation, plan M2.2)
The tokenizer lowercases and has no emoji: "AMAZING" and "amazing" are the same tokens, and an emoji becomes [UNK].
512 positions is a hard limit; Inputs never builds a longer window.

Tweaks (TweakedBGE): changes to the backbone itself. The decision heads are in heads.py and set_detector.py;
layer-wise lr decay is in train.py. Every tweak is off by default and leaves the pretrained model unchanged at step 0.
  context      review context gets BERT's own token type 1 (pretrained for "second segment"); the target keeps type 0
               (plan L4)
  neg_scope    a new zero-initialised embedding added to target tokens inside a negation scope (plan L10, D6.3); BERT's
               type table only has rows 0 and 1, both already in use
  drop_blocks  remove transformer layers by index 0-11 (plan C2 pruning)
  layer_mix    learned softmax weights over every layer's output instead of the last layer only
  grad_ckpt    recompute layer activations during backward (less memory, slower)
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from models.data import Inputs, TokenSpec, hub_file

REPO = "BAAI/bge-small-en-v1.5"
SPEC = TokenSpec(repo=REPO, prefix=("[CLS]",), suffix=("[SEP]",), sep=("[SEP]",), pad="[PAD]", max_positions=512,
                 pooling="cls")


class BGEEncoder(nn.Module):
    """input_ids, attention_mask [W, T] -> token states [W, T, 384]."""

    spec = SPEC

    def __init__(self, bert):
        super().__init__()
        self.bert, self.hidden_size = bert, bert.config.hidden_size

    def forward(self, input_ids, attention_mask, type_ids=None):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    def layer_groups(self) -> list[list[nn.Parameter]]:
        """Parameters from the bottom up (embeddings, each layer): for layer-wise lr decay and freezing."""
        return [list(self.bert.embeddings.parameters()), *[list(l.parameters()) for l in self.bert.encoder.layer]]


def load() -> BGEEncoder:
    from transformers import BertModel
    from transformers.utils import logging

    logging.set_verbosity_error()  # the unused pooler weights would otherwise be listed
    logging.disable_progress_bar()
    hub_file(REPO, "config.json")
    folder = hub_file(REPO, "model.safetensors").rsplit("/", 1)[0]
    bert = BertModel.from_pretrained(folder, add_pooling_layer=False, attn_implementation="sdpa")
    return BGEEncoder(bert).eval()


def save_config(model: BGEEncoder, folder: Path) -> None:
    model.bert.config.to_json_file(str(folder / "encoder_config.json"))


def from_config(folder: Path) -> BGEEncoder:
    """The architecture from a checkpoint's encoder_config.json, without pretrained weights (the checkpoint has them)."""
    from transformers import BertConfig, BertModel

    cfg = BertConfig.from_json_file(str(folder / "encoder_config.json"))
    cfg._attn_implementation = "sdpa"
    return BGEEncoder(BertModel(cfg, add_pooling_layer=False)).eval()


def describe(model: BGEEncoder) -> list[str]:
    n = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    cfg, bert = model.bert.config, model.bert
    out = [f"{REPO}: hidden {cfg.hidden_size}, vocab {cfg.vocab_size}, {len(bert.encoder.layer)} layers, "
           f"{cfg.num_attention_heads} heads, feed-forward {cfg.intermediate_size}, {cfg.max_position_embeddings} positions, "
           f"dropout {cfg.hidden_dropout_prob}",
           f"  embeddings (word {n(bert.embeddings.word_embeddings):,}, position {n(bert.embeddings.position_embeddings):,}, "
           f"type {n(bert.embeddings.token_type_embeddings):,}, LayerNorm)  {n(bert.embeddings):>11,}"]
    for i, layer in enumerate(bert.encoder.layer):
        out.append(f"  layer {i:>2}  attention {n(layer.attention):,} + feed-forward "
                   f"{n(layer.intermediate) + n(layer.output):,}  {n(layer):>12,}")
    out.append(f"  total  {n(model):>12,}")
    return out


@torch.no_grad()
def checks(model: BGEEncoder, inputs: Inputs) -> list[str]:
    """Plan M3 model checks for B: CLS pooling equals the sentence-transformers embedding; padding invariance."""
    from models.data import Example

    model = model.eval()
    texts = ["The So called laptop Runs to Slow and I hate it!", "Do not buy it!", "It is the worst laptop ever."]
    b = inputs.batch([Example(str(i), "check", t) for i, t in enumerate(texts)])
    ours = nn.functional.normalize(model(b.input_ids, b.attention_mask)[:, 0], dim=-1)
    try:
        from sentence_transformers import SentenceTransformer

        st = SentenceTransformer(REPO, device="cpu", local_files_only=True)
        ref = torch.tensor(st.encode(texts, normalize_embeddings=True))
        cos = (ours * ref).sum(-1).min().item()
        out = [f"CLS parity with sentence-transformers  min cosine {cos:.5f}  {'ok' if cos > 0.999 else 'FAILED'}"]
    except Exception as e:  # sentence-transformers files not cached: skip, the rest still runs
        out = [f"CLS parity skipped ({type(e).__name__})"]
    x, m = b.input_ids[1:2], b.attention_mask[1:2]
    n = int(m.sum())
    pad = torch.cat([x[:, :n], torch.full((1, 64 - n), inputs.pad)], 1)
    diff = (model(x[:, :n], m[:, :n])[0] - model(pad, (pad != inputs.pad).long())[0, :n]).abs().max().item()
    out.append(f"padding invariance  same sentence padded to 64: max state difference {diff:.2e}  "
               f"{'ok' if diff < 1e-3 else 'FAILED'}")
    return out


class TweakedBGE(nn.Module):
    def __init__(self, base: BGEEncoder, context: bool = False, neg_scope: bool = False, drop_blocks=(),
                 layer_mix: bool = False, grad_ckpt: bool = False):
        super().__init__()
        self.base, self.spec, self.hidden_size = base, base.spec, base.hidden_size
        bert = base.bert
        self.kept = [i for i in range(len(bert.encoder.layer)) if i not in set(drop_blocks)]
        bert.encoder.layer = nn.ModuleList(bert.encoder.layer[i] for i in self.kept)
        bert.config.num_hidden_layers = len(self.kept)
        self.context = context
        self.scope = nn.Embedding(1, self.hidden_size) if neg_scope else None
        if self.scope is not None:
            nn.init.zeros_(self.scope.weight)
        self.mix = nn.Parameter(torch.zeros(len(self.kept))) if layer_mix else None
        if grad_ckpt:
            bert.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def forward(self, input_ids, attention_mask, type_ids=None):
        bert = self.base.bert
        segment = (type_ids == 1).long() if self.context and type_ids is not None else None
        embeds = bert.embeddings.word_embeddings(input_ids)
        if self.scope is not None and type_ids is not None:
            embeds = embeds + (type_ids == 2).unsqueeze(-1).to(embeds.dtype) * self.scope.weight[0]
        out = bert(inputs_embeds=embeds, attention_mask=attention_mask, token_type_ids=segment,
                   output_hidden_states=self.mix is not None)
        if self.mix is None:
            return out.last_hidden_state
        return (torch.softmax(self.mix, 0)[:, None, None, None] * torch.stack(out.hidden_states[1:])).sum(0)

    def layer_groups(self) -> list[list[nn.Parameter]]:
        return self.base.layer_groups()

    def describe(self) -> list[str]:
        return [f"tweaks: layers kept {len(self.kept)}/12 {self.kept}, context type ids {'on' if self.context else 'off'}, "
                f"negation-scope embedding {'on' if self.scope is not None else 'off'}, "
                f"layer mix {'on' if self.mix is not None else 'off'}"]
