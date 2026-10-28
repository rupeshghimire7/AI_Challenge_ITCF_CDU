"""Model A: LFM2.5-Encoder-230M (LiquidAI, 2026) in our own PyTorch code: no trust_remote_code, no transformers model.

Architecture (config.json): 14 blocks of hidden size 1024 on a 65,536-token vocabulary.
  8 conv blocks       (0 1 3 5 7 9 11 13)  gated short convolution: (B, C, x) = in_proj(h); out_proj(C * conv3(B * x))
  6 attention blocks  (2 4 6 8 10 12)      16 query heads, 8 key/value heads (grouped-query), RoPE, RMSNorm on q and k
  every block         pre-norm residual: h += op(RMSNorm(h)); h += SwiGLU(RMSNorm(h)), feed-forward 1024 -> 2560 -> 1024
  output              final RMSNorm ("embedding_norm"); the MLM head is the embedding matrix itself (tied)
LiquidAI made the causal LFM2 decoder an encoder with two changes, re-implemented here exactly: attention sees every
token (only padding is masked), and the convolution is centred (previous, current and next token) instead of causal.

Input framing (configured from the tokenizer, not guessed):
  sentence   <|startoftext|> text            the tokenizer's own template: BOS only. config.json names <|im_end|> as
                                             eos_token, but that is a chat token the encoder never saw closing its
                                             inputs, so no EOS is added
  context    <|startoftext|> prev <|startoftext|> target <|startoftext|> next   (the tokenizer's pair template separates
                                             with BOS)
  padding    <|pad|> (id 0) on the right: masked out of attention and zeroed before every convolution
  pooling    mean over the target sentence's tokens (the model has no CLS objective, plan M2.2)

Tweaks (TweakedLFM): changes to the backbone itself. The decision heads are in heads.py and set_detector.py;
layer-wise lr decay is in train.py. Every tweak is off by default, and each one is exactly the pretrained model at
step 0 (new weights start at zero, or added kernel taps are zero), so a ladder step measures what training does with
it, not a broken start.
  context      token-role embedding added to the word embeddings: 1 = review context (plan L4). LFM2 has no segment
               embeddings, so this table is new and zero-initialised
  neg_scope    the same table, role 2 = target token inside a negation scope (plan L10, D6.3)
  drop_blocks  remove blocks by index (plan C2 pruning). conv blocks: 0 1 3 5 7 9 11 13; attention: 2 4 6 8 10 12
  conv_kernel  widen the short convolution from 3 tokens to 5, 7, ... by zero-padding the pretrained kernel: same
               outputs at step 0, then a wider local window can be learned
  layer_mix    learned softmax weights over every block's output instead of the last block only (lower blocks keep
               more word-level detail)
  grad_ckpt    recompute block activations during backward: about 30% slower, much less activation memory
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn
from torch.utils.checkpoint import checkpoint

from models.data import Inputs, TokenSpec, hub_file

REPO = "LiquidAI/LFM2.5-Encoder-230M"
SPEC = TokenSpec(repo=REPO, prefix=("<|startoftext|>",), suffix=(), sep=("<|startoftext|>",), pad="<|pad|>",
                 max_positions=8192, pooling="mean")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight, self.eps = nn.Parameter(torch.ones(dim)), eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).to(dtype)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w1, self.w3 = nn.Linear(dim, hidden, bias=False), nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ShortConv(nn.Module):
    """Gated depthwise convolution over the sequence, centred (non-causal) as in the published encoder."""

    def __init__(self, dim: int, kernel: int):
        super().__init__()
        self.in_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.conv = nn.Conv1d(dim, dim, kernel, groups=dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, h, mask, cos=None, sin=None):
        h = h * mask[..., None].to(h.dtype)  # padding contributes nothing to its neighbours
        b, c, x = self.in_proj(h).transpose(1, 2).chunk(3, dim=1)
        k = self.conv.weight.shape[-1]
        y = F.conv1d(b * x, self.conv.weight, padding=k // 2, groups=x.shape[1])[..., : x.shape[-1]]
        return self.out_proj((c * y).transpose(1, 2))


def rotate(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


class Attention(nn.Module):
    """Bidirectional grouped-query attention with RoPE and per-head RMSNorm on queries and keys."""

    def __init__(self, dim: int, heads: int, kv_heads: int, eps: float):
        super().__init__()
        self.heads, self.kv_heads, hd = heads, kv_heads, dim // heads
        self.q_proj, self.out_proj = nn.Linear(dim, heads * hd, bias=False), nn.Linear(heads * hd, dim, bias=False)
        self.k_proj, self.v_proj = nn.Linear(dim, kv_heads * hd, bias=False), nn.Linear(dim, kv_heads * hd, bias=False)
        self.q_layernorm, self.k_layernorm = RMSNorm(hd, eps), RMSNorm(hd, eps)

    def forward(self, h, mask, cos, sin):
        w, t, _ = h.shape
        q = self.q_layernorm(self.q_proj(h).view(w, t, self.heads, -1)).transpose(1, 2)
        k = self.k_layernorm(self.k_proj(h).view(w, t, self.kv_heads, -1)).transpose(1, 2)
        v = self.v_proj(h).view(w, t, self.kv_heads, -1).transpose(1, 2)
        q, k = rotate(q, cos, sin), rotate(k, cos, sin)
        rep = self.heads // self.kv_heads
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :])  # padding keys only: bidirectional
        return self.out_proj(out.transpose(1, 2).reshape(w, t, -1))


class Block(nn.Module):
    def __init__(self, cfg: dict, kind: str):
        super().__init__()
        dim, eps = cfg["hidden_size"], cfg["norm_eps"]
        self.kind = kind
        if kind == "conv":
            self.conv = ShortConv(dim, cfg["conv_L_cache"])
        else:
            self.self_attn = Attention(dim, cfg["num_attention_heads"], cfg["num_key_value_heads"], eps)
        self.feed_forward = SwiGLU(dim, ffn_dim(cfg))
        self.operator_norm, self.ffn_norm = RMSNorm(dim, eps), RMSNorm(dim, eps)

    def forward(self, h, mask, cos, sin):
        op = self.conv if self.kind == "conv" else self.self_attn
        h = h + op(self.operator_norm(h), mask, cos, sin)
        return h + self.feed_forward(self.ffn_norm(h))


def ffn_dim(cfg: dict) -> int:
    d = cfg["intermediate_size"]
    if cfg.get("block_auto_adjust_ff_dim"):  # false for this checkpoint; kept so the code follows the config
        d = int(2 * d / 3)
        if cfg.get("block_ffn_dim_multiplier") is not None:
            d = int(cfg["block_ffn_dim_multiplier"] * d)
        m = cfg["block_multiple_of"]
        d = m * ((d + m - 1) // m)
    return d


class LFMEncoder(nn.Module):
    """input_ids, attention_mask [W, T] -> token states [W, T, 1024]."""

    spec = SPEC

    def __init__(self, cfg: dict):
        super().__init__()
        self.config, self.hidden_size = cfg, cfg["hidden_size"]
        self.embed_tokens = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"], padding_idx=cfg["pad_token_id"])
        self.layers = nn.ModuleList(Block(cfg, "attention" if t == "full_attention" else "conv") for t in cfg["layer_types"])
        self.embedding_norm = RMSNorm(cfg["hidden_size"], cfg["norm_eps"])
        hd = cfg["hidden_size"] // cfg["num_attention_heads"]
        theta = cfg["rope_parameters"]["rope_theta"]
        self.register_buffer("inv_freq", 1.0 / theta ** (torch.arange(0, hd, 2).float() / hd), persistent=False)

    def rope(self, t: int, device, dtype):
        f = torch.arange(t, device=device).float()[:, None] * self.inv_freq[None]
        emb = torch.cat((f, f), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def forward(self, input_ids, attention_mask, type_ids=None):
        h = self.embed_tokens(input_ids)
        cos, sin = self.rope(h.shape[1], h.device, h.dtype)
        mask = attention_mask.bool()
        for block in self.layers:
            h = block(h, mask, cos, sin)
        return self.embedding_norm(h)

    def mlm_logits(self, states):
        """Masked-LM scores (tied head); used only by the sanity check."""
        return states @ self.embed_tokens.weight.T

    def layer_groups(self) -> list[list[nn.Parameter]]:
        """Parameters from the bottom up (embeddings, each block, final norm): for layer-wise lr decay and freezing."""
        return [list(self.embed_tokens.parameters()), *[list(b.parameters()) for b in self.layers],
                list(self.embedding_norm.parameters())]


def load() -> LFMEncoder:
    """The pretrained encoder, every checkpoint tensor matched by name (plan M3 check (i): zero missing / unexpected)."""
    cfg = json.loads(Path(hub_file(REPO, "config.json")).read_text())
    model = LFMEncoder(cfg)
    state = {k.removeprefix("lfm2."): v for k, v in load_file(hub_file(REPO, "model.safetensors")).items()
             if k != "lm_head.weight"}  # the MLM head is tied to the embeddings
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"LFM2.5 weights do not match the re-implementation: missing {missing}, unexpected {unexpected}")
    return model.eval()


def save_config(model: LFMEncoder, folder: Path) -> None:
    (folder / "encoder_config.json").write_text(json.dumps(model.config, indent=1), encoding="utf-8")


def from_config(folder: Path) -> LFMEncoder:
    """The architecture from a checkpoint's encoder_config.json, without pretrained weights (the checkpoint has them)."""
    return LFMEncoder(json.loads((folder / "encoder_config.json").read_text())).eval()


def describe(model: LFMEncoder) -> list[str]:
    """Block-by-block architecture with parameter counts."""
    n = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    cfg = model.config
    out = [f"{REPO}: hidden {cfg['hidden_size']}, vocab {cfg['vocab_size']}, {len(model.layers)} blocks, "
           f"{cfg['num_attention_heads']} query / {cfg['num_key_value_heads']} kv heads, conv kernel {cfg['conv_L_cache']}, "
           f"feed-forward {ffn_dim(cfg)}, RoPE theta {cfg['rope_parameters']['rope_theta']:.0e}",
           f"  embed_tokens            {n(model.embed_tokens):>12,}"]
    for i, b in enumerate(model.layers):
        op = n(b.conv if b.kind == "conv" else b.self_attn)
        out.append(f"  block {i:>2}  {b.kind:<9}     {n(b):>12,}   (operator {op:,}, feed-forward {n(b.feed_forward):,})")
    out.append(f"  embedding_norm (final)  {n(model.embedding_norm):>12,}")
    out.append(f"  total                   {n(model):>12,}")
    return out


@torch.no_grad()
def checks(model: LFMEncoder, inputs: Inputs) -> list[str]:
    """Plan M3 model checks for A: MLM sanity, bidirectionality, padding invariance (strict loading is in load())."""
    model = model.eval()
    tok = inputs.tok
    ids = inputs.prefix + tok.encode("The capital of France is <|mask|>.", add_special_tokens=False).ids
    pos = ids.index(inputs.token_id("<|mask|>"))
    x = torch.tensor([ids])
    top = model.mlm_logits(model(x, torch.ones_like(x)))[0, pos].topk(5).indices.tolist()
    words = [tok.decode([t]).strip() for t in top]
    out = [f"MLM sanity  'The capital of France is <|mask|>.' -> {words}  {'ok' if 'Paris' in words else 'FAILED'}"]

    ids = inputs.prefix + tok.encode("The battery life is great but the screen is dim", add_special_tokens=False).ids
    a, b = torch.tensor([ids]), torch.tensor([ids[:-1] + [tok.token_to_id("Ġbright") or ids[-1] + 1]])
    m = torch.ones_like(a)
    diff = (model(a, m)[0, 1] - model(b, m)[0, 1]).abs().max().item()
    out.append(f"bidirectional  changing the last token changes the state of token 1 by {diff:.3f}  "
               f"{'ok' if diff > 1e-3 else 'FAILED'}")

    pad = torch.tensor([ids + [inputs.pad] * (64 - len(ids))])
    pm = (pad != inputs.pad).long()
    diff = (model(a, m)[0] - model(pad, pm)[0, : len(ids)]).abs().max().item()
    out.append(f"padding invariance  same sentence padded to 64: max state difference {diff:.2e}  "
               f"{'ok' if diff < 1e-3 else 'FAILED'}")
    return out


class TweakedLFM(nn.Module):
    def __init__(self, base: LFMEncoder, context: bool = False, neg_scope: bool = False, drop_blocks=(),
                 conv_kernel: int = 3, layer_mix: bool = False, grad_ckpt: bool = False):
        super().__init__()
        self.base, self.spec, self.hidden_size = base, base.spec, base.hidden_size
        self.kept = [i for i in range(len(base.layers)) if i not in set(drop_blocks)]
        base.layers = nn.ModuleList(base.layers[i] for i in self.kept)
        if conv_kernel != 3:
            if conv_kernel % 2 == 0 or conv_kernel < 3:
                raise ValueError("conv_kernel must be odd and >= 3 (the convolution is centred)")
            for block in base.layers:
                if block.kind == "conv":
                    w = block.conv.conv.weight.data  # [dim, 1, 3]
                    pad = (conv_kernel - w.shape[-1]) // 2
                    block.conv.conv = nn.Conv1d(w.shape[0], w.shape[0], conv_kernel, groups=w.shape[0], bias=False)
                    block.conv.conv.weight.data = F.pad(w, (pad, pad))
        self.roles = nn.Embedding(3, self.hidden_size) if context or neg_scope else None
        if self.roles is not None:
            nn.init.zeros_(self.roles.weight)
        self.mix = nn.Parameter(torch.zeros(len(base.layers))) if layer_mix else None
        self.grad_ckpt = grad_ckpt

    def forward(self, input_ids, attention_mask, type_ids=None):
        b = self.base
        h = b.embed_tokens(input_ids)
        if self.roles is not None and type_ids is not None:
            h = h + self.roles(type_ids)
        cos, sin = b.rope(h.shape[1], h.device, h.dtype)
        mask, outs = attention_mask.bool(), []
        for block in b.layers:
            if self.grad_ckpt and self.training:
                h = checkpoint(block, h, mask, cos, sin, use_reentrant=False)
            else:
                h = block(h, mask, cos, sin)
            if self.mix is not None:
                outs.append(h)
        if self.mix is not None:
            h = (torch.softmax(self.mix, 0)[:, None, None, None] * torch.stack(outs)).sum(0)
        return b.embedding_norm(h)

    def layer_groups(self) -> list[list[nn.Parameter]]:
        return self.base.layer_groups()

    def describe(self) -> list[str]:
        kinds = [f"{i}:{b.kind}" for i, b in zip(self.kept, self.base.layers)]
        k = next((b.conv.conv.weight.shape[-1] for b in self.base.layers if b.kind == "conv"), "-")
        return [f"tweaks: blocks kept {len(self.kept)}/14 [{' '.join(kinds)}], conv kernel {k}, "
                f"role embedding {'on' if self.roles is not None else 'off'}, layer mix {'on' if self.mix is not None else 'off'}, "
                f"gradient checkpointing {'on' if self.grad_ckpt else 'off'}"]
