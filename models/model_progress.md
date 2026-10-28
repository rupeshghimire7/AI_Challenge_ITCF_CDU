# Model phase: progress

Plan sections covered: M1–M3 (models A and B, base head, model checks), M4 ladder steps L1–L5 and L10 combined in one
`tweaked` variant (no per-step ablation), plus a DETR-style set-prediction model. Not yet: see the last section.

## Files

| File | One job |
|---|---|
| `data.py` | DATA/final → model tensors and back: final CSVs, token windows (long text), long-input training data, star summary, metrics |
| `lfm.py` | Model A, LFM2.5-Encoder-230M: our own implementation, weight loading, special tokens, plan M3 checks; backbone tweaks (`TweakedLFM`, not heads) |
| `bge.py` | Model B, bge-small-en-v1.5: loading, special tokens, plan M3 checks; backbone tweaks (`TweakedBGE`, not heads) |
| `heads.py` | decision heads: plan M2 base head and ladder heads / losses (L1, L2, L3, L5, L9); window merging |
| `set_detector.py` | DETR-style head: queries → (entity, attribute, star) sets |
| `train.py` | the CLI: one model × variant for N epochs, a self-contained checkpoint per epoch |
| `infer.py` | any checkpoint: analyse text, score the test set, compare every run |
| `custom_model.py`, `custom_train.py`, `custom_inference.py` | a separate three-head model (entity, attribute, polarity per pair) on any CSV / JSONL; uses neither `data.py` nor the checkpoints above |

## Data the models read: `DATA/final/{train,test}_final.csv`

Built by `python models/data.py` from `DATA/final/{train,test}_merged.csv` (also built automatically if missing). One
row per opinion; a sentence without opinions is one row with empty labels. Counts per split are in
`DATA/final/reports/final_data.txt`.

| Column | Why it is needed |
|---|---|
| `text` | the input (already normalised by `src.preprocess.normalise`) |
| `category`, `attribute` | Slot 1: the entity (LAPTOP, BATTERY, ...) and the attribute (GENERAL, QUALITY, ...); E#A = `category#attribute` |
| `polarity` | Slot 2, as a star: negative 0, neutral 2, positive 4 |
| `sentence_id`, `review_id` | one sentence has several tuples, and the same text occurs in several reviews, so text is no key |
| `split` | train / validation (15% of the train reviews, whole reviews, fixed by a hash of the review id) / test (the official SemEval-2016 gold set) |
| `prev_text`, `next_text` | the brief: "taking also into account the context of the review" (ladder step L4) |

Sentences without opinions stay in the data: the model must learn to output nothing (they are 18.5% of train, 29.1%
of test). The brief's `NULL#GENERAL` is a restaurant-domain target value; the laptop annotation of "Do not buy it!"
is `LAPTOP#GENERAL` (plan 5.2).

## How to run

```bash
python models/data.py                                       # DATA/final/{train,test}_final.csv + reports/final_data.txt

python models/train.py --model bge --variant base           # pretrained encoder + linear head (plan M2)
python models/train.py --model bge --variant tweaked        # L1 L2 L3 L4 L5 L10 + long-input training + llrd 0.9
python models/train.py --model bge --variant detr           # tweaked encoder + DETR-style set head
python models/train.py --model lfm --variant tweaked --grad-ckpt   # LFM on a small GPU
python models/train.py --model bge --variant base --limit 300 --epochs 2   # smoke test
python models/train.py -h                                   # every flag

python models/infer.py --checkpoint result/bge_tweaked/checkpoint_5 --text "Great screen, but the battery dies fast."
python models/infer.py --checkpoint result/bge_tweaked/checkpoint_5 --file reviews.txt   # one review per line
python models/infer.py --checkpoint result/bge_base/checkpoint_5 --test                  # DATA/final/test_final.csv
python models/infer.py --table                                                            # every run in result/

python models/custom_train.py --data DATA/final/train_final.csv --output-dir custom_results/run
python models/custom_inference.py --checkpoint custom_results/run/best.pt --text "The battery dies fast."
```

`train.py` flags: `--model {bge,lfm}`, `--variant {base,tweaked,detr}`, `--epochs` (5), `--batch-size` (32), `--lr`
(encoder; bge 5e-5, lfm 3e-5), `--grad-ckpt`, `--device` (auto), `--out` (result), `--seed` (2026), `--limit`. Everything
else is fixed per variant (`VARIANTS`) or for all runs (`FIXED`) in `train.py`.

Each epoch writes `result/{model}_{variant}/checkpoint_{epoch}/`: `model.pt`, `config.json`, `encoder_config.json`,
`tokenizer.json` (all `infer.py` needs, no download), `metrics.csv` (train / validation / test) and
`test_predictions.csv`. The run folder also gets `metrics.csv` (every epoch) and `log.txt`. After every epoch the
detection threshold τ is fitted on validation (best joint F1); the summary marks the best validation epoch. The test
set is scored every epoch but never used to choose anything. `model.pt` is about 130 MB for bge and about 920 MB for
LFM, so 5 LFM epochs take about 4.6 GB per run.

`custom_train.py` splits its input at random (15% validation, 10% test) and ignores the `split` column, so its test
score is not the official test set.

**Where to run.** `--device auto` uses a GPU only if a kernel actually runs on it. This laptop's MX110 is too old
for the installed torch build (cu130 needs sm_75+), so it falls back to the CPU. Measured on this CPU: bge-small
takes about 1.8 s per training step of 32 sentences. A full LFM run needs the plan's Colab or Kaggle T4. Mixed
precision is used automatically on a GPU.

## Long inputs (nothing is truncated, at any length)

Inference input can be a 100-word run-on or a whole review. Three things handle it:

1. **Windows, not truncation** (`data.Inputs`). A text longer than `max_len` (56 tokens, `train.FIXED`; EDA: S1 p99)
   is cut into windows that overlap by a quarter. Each window gets the model's own special tokens, so an opinion cut
   at one boundary is whole in the next window.
2. **Merging** (`heads.merge_windows`). A category's score is its best window's score, and its polarity comes from
   that window. An opinion anywhere in the text counts. The encoder runs over at most 64 windows at a time, so
   memory stays bounded. The set detector decodes groups of 4 windows and merges the groups the same way, so its 16
   queries never cap how many opinions a long text can have.
3. **Training for it** (variants `tweaked` and `detr`, `long_aug` 0.25):
   - each epoch adds 0.25 × (number of training sentences) inputs made of 2–6 consecutive sentences of one training
     review, with labels = the union of theirs.
   - Half of these lose their full stops, like an unsplit Amazon paragraph.
   - They go through the same windows and merging as at inference.

Edge-case probe (measured with an earlier `--inspect` mode of `train.py`, since removed; untrained head; checks that
every input runs and gives finite outputs):

| input | bge tokens / windows / ms | LFM tokens / windows / ms |
|---|---|---|
| empty, whitespace only | 0 / 1 / ~20 | 0 / 1 / ~60 |
| emoji only | 1 ([UNK]) / 1 | 9 / 1 |
| HTML, entities, "!!!!!!", "sooooo" | normalised first, 9 / 1 | 7 / 1 |
| one 5,000-character token (URL-like) | 9 / 1 (a word over 100 characters is one [UNK]) | 2,499 / 60 |
| 3,000-word review (15,817 characters) | 3,635 / 89 / 1,406 | 3,690 / 88 / 12,357 |

bge has a hard 512-position limit; `Inputs` refuses any setting that could exceed it. LFM was trained up to 8,192.

## Special tokens ("EOS and other tokens")

| | A: LFM2.5-Encoder | B: bge-small |
|---|---|---|
| sentence | `<|startoftext|> text` | `[CLS] text [SEP]` |
| with context (L4) | `<|startoftext|> prev <|startoftext|> target <|startoftext|> next` | `[CLS] prev [SEP] target [SEP] next [SEP]` |
| EOS | **none**: the tokenizer's own template is BOS-only; `config.json`'s `eos_token` `<|im_end|>` is a chat token the encoder never saw at the end of its inputs | `[SEP]` |
| padding | `<|pad|>` id 0, right side | `[PAD]` id 0, right side |
| sentence vector | mean of the target tokens (no CLS objective) | the `[CLS]` state (BGE documentation) |
| case / emoji | kept | lowercased; emoji → `[UNK]` |

These are encoders with classification heads, so they generate no text and there is no EOS to strip from an output.
Special tokens, padding and context are "removed" from the decision instead: every window carries a `target_mask`, and
pooling, the attention heads and the set decoder read only target tokens. Printed framing (`*` = target token,
`|1` = context, `|2` = negation scope; context and negation scope on, as in `tweaked` and `detr`):

```
bge  [CLS] i|1 bought|1 it|1 last|1 week|1 .|1 [SEP] it* is* not|2* bad|2* ,* but* the* fan* is* loud* .* [SEP] do|1 not|1 buy|1 it|1 !|1 [SEP]
LFM  <|startoftext|> I|1 Ġbought|1 ... <|startoftext|> It* Ġis* Ġnot|2* Ġbad|2* ,* Ġbut* Ġthe* Ġfan* Ġis* Ġloud* .* <|startoftext|> Do|1 ...
```

## Architectures

Both backbones are transformer-family encoders. bge-small is a classic 12-layer transformer (BERT). LFM2.5 is a
hybrid: 6 attention blocks and 8 convolution blocks.

### A: LFM2.5-Encoder-230M (`lfm.py`)

```
LiquidAI/LFM2.5-Encoder-230M: hidden 1024, vocab 65536, 14 blocks, 16 query / 8 kv heads, conv kernel 3, feed-forward 2560
  embed_tokens              67,108,864
  block  0  conv            12,063,744   (operator 4,197,376, feed-forward 7,864,320)
  block  1  conv            12,063,744
  block  2  attention       11,012,224   (operator 3,145,856, feed-forward 7,864,320)
  block  3  conv   / 4 attention / 5 conv / 6 attention / 7 conv / 8 attention / 9 conv / 10 attention / 11 conv
  block 12  attention       11,012,224
  block 13  conv            12,063,744
  embedding_norm (final)         1,024
  total                    229,693,184

every block:  h = h + Operator(RMSNorm(h))          Operator = ShortConv or Attention
              h = h + SwiGLU(RMSNorm(h))            1024 -> 2560 (x2, gated) -> 1024
ShortConv:    B, C, x = split(Linear 1024->3072 (h))
              y = Linear 1024->1024 ( C * depthwise_conv_k3_centred(B * x) )   sees the previous, current and next token
Attention:    q = RMSNorm(Wq h) 16 heads x 64; k, v = 8 heads (each shared by 2 query heads); RoPE theta 1e6;
              every token attends to every non-padding token (bidirectional)
```

29% of the parameters are the embedding table (65k multilingual vocabulary); compute is in the 14 blocks.

Checks (`lfm.checks()`; loading is strict in `lfm.load()`):
- **Parity:** our implementation matches transformers' LFM2 with LiquidAI's encoder patches to within 2.5e-5 (float
  rounding), and all 132 checkpoint tensors load with nothing missing or unexpected.
- **MLM:** "The capital of France is `<|mask|>`." → Paris, Strasbourg, Paris, Lyon, Versailles.
- **Bidirectional:** changing the last word moves token 1's state by 0.89.
- **Padding invariance:** max difference 2.8e-5.

### B: bge-small-en-v1.5 (`bge.py`)

```
BAAI/bge-small-en-v1.5: hidden 384, vocab 30522, 12 layers, 12 heads, feed-forward 1536, 512 positions, dropout 0.1
  embeddings (word 11,720,448, position 196,608, type 768, LayerNorm)   11,918,592
  layer 0..11  attention 592,128 + feed-forward 1,182,336     1,774,464 each
  total    33,212,160

every layer (post-norm):  h = LayerNorm(h + SelfAttention(h))     12 heads x 32
                          h = LayerNorm(h + GELU-FFN(h))          384 -> 1536 -> 384
```

Checks (`bge.checks()`):
- **CLS parity:** our CLS vector equals the sentence-transformers embedding (min cosine 1.00000).
- **Padding invariance:** max difference 1.1e-6.

## What can be tweaked (backbone, apart from the decision heads)

Every implemented tweak is the pretrained model at step 0: new weights start at zero, or added kernel taps are zero.
A ladder step therefore measures what training does with the tweak, not a damaged start.

| Tweak | Set by | A (LFM) | B (bge) | Idea |
|---|---|---|---|---|
| review context | `context`: `tweaked`, `detr` | new zero-init role embedding (LFM has no segment embeddings) | BERT's own token type 1 | entity carry-over, implicit references (plan L4) |
| negation scope | `neg_scope`: `tweaked`, `detr` | role 2 in the same table | new zero-init vector | "not bad", "never had a problem" (plan L10) |
| prune blocks | `drop_blocks`: class argument only | conv 0 1 3 5 7 9 11 13, attention 2 4 6 8 10 12 | layers 0–11 | speed / memory (plan C2); which block types survive is itself a result |
| wider convolution | `conv_kernel`: class argument only | kernel 3 → 5/7, pretrained taps in the middle | – | more local context per conv block at almost no cost |
| layer mix | `layer_mix`: class argument only | weights over 14 block outputs | weights over 12 layer outputs | lower layers keep word-level detail (aspect words) |
| layer-wise lr decay | `llrd` 0.9: `tweaked`, `detr` | per block | per layer | keep lower (general) layers closer to pretraining |
| window length | `max_len` 56, `ctx_len` 32 in `train.FIXED` | up to 8,192 | ≤ 512 total | cost vs context |
| gradient checkpointing | `--grad-ckpt` | per block | BERT's own | memory |

Not implemented (possible later):
- **A:** local or windowed attention in the 6 attention blocks (cheaper on long text); converting attention blocks to
  conv blocks; fewer kv heads.
- **B:** mapping emoji to words before tokenising (bge sees them as `[UNK]`); a case feature (bge lowercases
  "AMAZING"); dropout rate.
- **Both:** freezing the embeddings or lower layers; LoRA adapters instead of full fine-tuning; int8 / ONNX (plan C1,
  needs `onnxruntime`).

## Decision heads and the ladder

| Plan step | Set by | Where |
|---|---|---|
| M2 base: linear → C sigmoid + C × 5 star logits (star loss on gold categories only, smoothed star targets), BCE + CE | `base` | `heads.py` |
| L1 category attention (query per category, initialised from the encoder's embedding of its name) | `tweaked` | `heads.py` |
| L2 E × A factorisation | `tweaked` | `heads.py` |
| L3 opinion gate | `tweaked` | `heads.py` |
| L4 review context + context dropout 0.3 | `tweaked`, `detr` | `data.py`, `lfm.py`, `bge.py` |
| L5 asymmetric loss + focal + class-balanced star loss | `tweaked` | `heads.py` |
| L9 supervised contrastive | `ABSAClassifier(supcon=...)` only, no variant | `heads.py` |
| L10 negation scope | `tweaked`, `detr` | `data.py`, `lfm.py`, `bge.py` |
| long-input training | `tweaked`, `detr` | `data.py` |

A category with two polarities in one sentence gets a half-half star target. After every epoch the threshold τ is
fitted on validation and saved in the checkpoint's `config.json`.

## The set detector (`--variant detr`): DETR for opinions

DETR detects objects with N learned queries. Each query reads the image features through a transformer decoder and
outputs one (class, box) or "no object". Training matches queries to gold objects one-to-one (Hungarian matching),
so the output is a set without duplicates.

| DETR | here |
|---|---|
| image backbone features | encoder token states (bge or LFM), projected to 256 |
| 100 object queries | 16 opinion queries (train has at most 5 opinions per sentence, test 7) |
| class (+ no-object) | entity (22 + "no opinion") × attribute (9) × star (0–4) |
| box | where the query looks: its cross-attention over the words (free evidence highlight, no box labels needed) |
| Hungarian matching on class + box cost | on −log p(entity) − log p(attribute) − log-likelihood of the smoothed star target |

Architecture: a 2-layer transformer decoder (d = 256, 8 heads, feed-forward 1024, pre-norm) and three linear outputs.
It has 2.2M parameters on top of the encoder. "No opinion" has class weight 0.1, as in DETR.

What it can do that the classifier cannot:
- name any of the 198 E#A pairs, including pairs never seen in training;
- emit one category twice with different polarities;
- handle empty sentences without a separate gate.

For decoding and metrics, its output is reduced to the classifier's form (per category: the best query's
p(entity) × p(attribute), and that query's stars). Every model is therefore scored by the same code.

## Not done yet (by plan)

- L6 ACOS auxiliary tasks (the ACOS rows were removed from the data). L7 augmentation and L8 distillation: they need
  D7 data and the R2 teacher. L11 needs the D6 lexicons and S7.
- Per-step ladder ablation, cross-validation and refit: `tweaked` turns L1–L5 and L10 on together in one run.
- L9 and the backbone options `drop_blocks`, `conv_kernel`, `layer_mix` are in the code, but no variant uses them.
- The M3 checks (`lfm.checks`, `bge.checks`) and `describe()` are in the code, but no CLI calls them.
- References R1–R3.
- Three seeds per configuration and the MDE.
- Compression C1–C3 (ONNX int8, pruning sweep).
- M6 scale-transfer check.
- Temperature scaling and post-processing (Phase P).

**Verified (2026-10-28):** CPU smoke runs (`--limit 24`) of bge `base`, `tweaked`, `detr` and `base --grad-ckpt`; every
`infer.py` mode on their checkpoints; `custom_train.py` → `custom_inference.py`; `data.py` rebuilds both final CSVs
byte-identical. LFM loads, passes `lfm.checks()`, and the tweaked LFM equals the pretrained one at step 0. LFM training
was not run (CPU only). The smoke outputs were deleted. No accuracy numbers exist until real runs.
