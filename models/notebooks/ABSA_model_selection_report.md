# Choosing the ABSA model: the Colab custom model against the bge / LFM experiments

## 1. Decision

I am going forward with **`bge_detr` at epoch 9** (bge-small-en-v1.5 encoder, review context and negation-scope tweaks, DETR-style set-prediction head), from `ABSA_experiments_base_tweak.ipynb`.

My case rests on four points:

1. **Best test joint F1 of every completed run.** On the 808-sentence held-out test set it scores 47.7 joint F1, against 46.1 for the best LFM run and about 36 for both other bge runs.
2. **The lead is not one lucky number.** It also leads on detection AUC, detection AP, star QWK, star RMSE, calibration (ECE) and negative / neutral F1.
3. **It is not overfit in the sense that matters.** Test F1 is flat from epoch 7 to 10 (48.4, 48.2, 48.5, 48.9) and does not decay. The train-to-test gap is large (83 vs 48), but it is no larger than that of the other model I could inspect (`lfm_base`: 83 vs 46).
4. **The Colab custom model gives me no reason to switch.** Its 57.4 triple-F1 is on an easier test set. On the metrics that do transfer (sentiment accuracy, star error, QWK) `bge_detr` is equal or better.

Two runs, `lfm_tweaked` and `lfm_detr`, never finished (CUDA out of memory), so the LFM tweaks remain untested. I treat that as an open item, not a rejection (section 9).

---

## 2. What I compared

### 2.1 The Colab custom model (`ABSA_colab.ipynb`)

A single, deliberately simple model:

| Part | Detail |
|---|---|
| Encoder | `BAAI/bge-small-en-v1.5`, 33.36M parameters, mean pooling |
| Input | the sentence only (`use_context=False`), max 128 tokens |
| Heads | `category_head` [B,22], `attribute_head` [B,9], `pair_head` [B,22,9] (MLP), `rating_head` [B,22,9,5] (MLP) |
| Head size | 0.76M parameters (total 34.12M) |
| Losses | BCE for category, attribute and pair; soft-ordinal cross-entropy for the rating (sigma 0.5) |
| Training | 15 epochs, minimum 10, patience 4, lr 3e-5 (encoder), 1e-3 (heads), batch 32, fp16 |
| Decoding | pair probability above `pair_thr` gives an opinion; expected rating r = sum(k·p_k); r below `lo` is negative, r above `hi` is positive, otherwise neutral; confidence below `min_conf` forces neutral |
| Threshold tuning | `pair`, `cat`, `attr`, `lo`, `hi`, `min_conf` are all grid-searched on the validation set at every epoch |

Its data and split:

- 3,345 raw rows were cleaned to 2,886 opinion rows. 459 rows without an opinion were dropped, so the model only ever sees sentences that contain an opinion.
- That leaves 2,016 sentences from 449 reviews.
- The split is fold 0: train 1,409, validation 217, test 390 sentences.
- The test set has 543 gold (category, attribute, rating) triples.

### 2.2 The experiments notebook (`ABSA_experiments_base_tweak.ipynb`)

Six experiments were planned: bge and LFM2.5-Encoder-230M, each as `base`, `tweaked` and `detr`.

| Variant | What it is |
|---|---|
| `base` | linear head: C sigmoid logits plus C×5 star logits, BCE + smoothed-star cross-entropy |
| `tweaked` | category attention, E×A factorised head, opinion gate, review context, asymmetric class-balanced loss, negation scope, long-input training, layer-wise lr decay |
| `detr` | the `tweaked` backbone with a set-prediction head: 16 queries, Hungarian matching, entity + attribute + star |

Common settings:

- 10 epochs, evaluation and a checkpoint at every epoch, no early stopping.
- A detection threshold fitted on validation at every epoch.
- Best epoch = best validation joint F1.

The data differs from the Colab run:

| Split | Reviews | Sentences | Opinions | Without opinion |
|---|---|---|---|---|
| train | 374 | 2,109 | 2,452 | 390 (18.5%) |
| validation | 75 | 366 | 434 | 69 (18.9%) |
| test (`test_final.csv`) | 80 | 808 | 801 | 235 (29.1%) |

Sentences without an opinion are kept. The label space is 22 entities × 9 attributes = 198 E#A categories, 80 of which occur in train.

---

## 3. The two evaluations are not the same test

Both setups predict the same kind of triple, (entity, attribute, star 0–4). But their test conditions differ in ways that favour the Colab number:

| | Colab custom model | Experiments notebook |
|---|---|---|
| Test sentences | 390, all with at least one opinion | 808, of which 29.1% have no opinion |
| Gold opinions per sentence | 1.39 | 0.99 |
| False positives on empty sentences | impossible to make | each one lowers precision |
| Training sentences | 1,409 | 2,109 |
| Threshold tuning | 6 parameters on 217 val sentences | detection tau on 366 val sentences |

So Colab's 57.4 vs 47.7 does not mean the Colab model is 10 points better. Its test set cannot punish the main failure mode of the experiments setting, firing on sentences with no opinion.

The experiments notebook shows how much that matters: `bge_detr` has an empty-sentence accuracy of only 43%, and it is the single largest cost to its joint F1.

For the comparison I therefore use only the metrics that are defined the same way in both setups: sentiment accuracy and macro-F1, star MAE / RMSE / QWK, and calibration.

---

## 4. Results of the experiments notebook

### 4.1 Test set, at each run's best validation epoch (all values in %, star error in stars)

| Run | Best epoch | Val joint F1 | Det F1 | Entity F1 | Attr F1 | Sent acc | Sent mF1 | Joint P | Joint R | **Joint F1** | Empty acc | Star MAE | cover90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bge_detr** | 9 | **61.4** | 52.7 | 63.9 | 61.3 | 88.6 | 71.2 | 45.7 | 49.9 | **47.7** | 43.0 | **0.35** | 93.8 |
| lfm_base | 10 | 59.7 | 50.6 | 60.4 | 58.0 | 84.0 | 62.9 | 54.9 | 39.7 | 46.1 | 69.4 | 0.52 | 91.1 |
| bge_tweaked | 9 | 45.0 | 40.5 | 54.3 | 48.0 | 86.0 | 72.2 | 40.5 | 33.6 | 36.7 | 83.0 | 0.37 | 93.9 |
| bge_base | 9 | 46.1 | 42.1 | 56.4 | 53.8 | 82.8 | 65.8 | 37.3 | 34.8 | 36.0 | 52.3 | 0.53 | 91.6 |
| lfm_tweaked | — | out of memory, no result | | | | | | | | | | | |
| lfm_detr | — | out of memory, no result | | | | | | | | | | | |

### 4.2 Extended metrics at the same epochs (test set)

| Metric | bge_base | **bge_detr** | bge_tweaked | lfm_base |
|---|---|---|---|---|
| tau (fitted on validation) | 0.20 | 0.85 | 0.60 | 0.40 |
| joint F1, negative | 0.246 | **0.397** | 0.281 | 0.336 |
| joint F1, neutral | 0.076 | **0.140** | 0.091 | 0.074 |
| joint F1, positive | 0.435 | **0.563** | 0.440 | 0.543 |
| joint macro-F1 over sentiments | 0.252 | **0.367** | 0.271 | 0.318 |
| detection AP (micro) | 0.386 | 0.532 | 0.350 | **0.533** |
| detection AUC (micro) | 0.932 | **0.984** | 0.895 | 0.962 |
| detection mAP (macro) | 0.215 | **0.387** | 0.222 | 0.331 |
| star exact accuracy | 0.716 | 0.792 | **0.798** | 0.699 |
| star within ±1 | 0.863 | **0.909** | 0.885 | 0.883 |
| star RMSE of expected star | 1.107 | **0.882** | 0.899 | 1.063 |
| star Spearman | 0.759 | **0.811** | 0.808 | 0.767 |
| star QWK | 0.797 | **0.879** | 0.872 | 0.806 |
| confidence ECE (lower is better) | 0.073 | 0.054 | **0.049** | 0.086 |
| sentence exact match | 0.314 | 0.380 | 0.345 | **0.416** |
| avg predicted opinions / sentence (gold 0.99) | 0.93 | **1.02** | 0.82 | 0.72 |

What this table tells me:

- **`bge_detr` wins 10 of the 15 comparable rows** (tau is not a score), including every sentiment-resolved F1. It is the only model whose number of predicted opinions per sentence (1.02) matches the gold (0.99).
- **`lfm_base`** is the strongest of the non-DETR models. It is precise (joint P 54.9, the best of the four) but under-predicts (0.72 opinions per sentence) and has weak sentiment macro-F1 (62.9).
- **`bge_tweaked`** has the best star accuracy and calibration and the best empty-sentence accuracy (83%), but it detects too little: det recall 37.0 and joint F1 only 36.7. Its backbone tweaks help the star head; they do not make detection better.
- **The set head is what lifts detection.** `bge_tweaked` and `bge_detr` share the same backbone tweaks. Swapping the classifier head for the set head moves test joint F1 from 36.7 to 47.7 (+11.0) and detection AUC from 0.895 to 0.984.

---

## 5. Is `bge_detr` overfit?

I looked at three things: the gap between train, validation and test, the shape of the learning curve, and whether picking an epoch by validation is reliable.

### 5.1 Learning curve of `bge_detr` (joint F1 %, tau refitted on validation each epoch)

| Epoch | Train | Validation | Test | Train − Test | Test sent acc | Test star MAE |
|---|---|---|---|---|---|---|
| 1 | 33.6 | 32.2 | 26.1 | 7.5 | 86.3 | 0.63 |
| 2 | 48.5 | 39.9 | 34.2 | 14.3 | 84.8 | 0.57 |
| 3 | 56.5 | 47.4 | 38.7 | 17.8 | 87.6 | 0.46 |
| 4 | 64.4 | 52.5 | 43.6 | 20.8 | 86.9 | 0.42 |
| 5 | 71.9 | 57.6 | 45.3 | 26.6 | 88.4 | 0.39 |
| 6 | 74.5 | 56.6 | 45.3 | 29.2 | 88.3 | 0.38 |
| 7 | 77.2 | 58.5 | 48.4 | 28.8 | 89.0 | 0.35 |
| 8 | 81.2 | 60.3 | 48.2 | 33.0 | 88.5 | 0.36 |
| **9** | **83.3** | **61.3** | **48.5** | 34.8 | 88.8 | 0.36 |
| 10 | 84.6 | 60.2 | 48.9 | 35.7 | 89.4 | 0.35 |

Reading it:

- **A real generalisation gap exists.** Train rises to 84.6 while test stalls near 48. The model has memorised part of the 2,109 training sentences (train sentiment accuracy is 99.6% from epoch 9).
- **It is a plateau, not a collapse.** Validation peaks at epoch 9 (61.3) and drops one point at epoch 10; test does not drop at all. From epoch 7 on, test joint F1 moves within 0.7 points. This is the signature of a model that has stopped learning generalisable signal, not of one that is getting worse.
- **Overfitting is not distorting the choice of epoch.** Validation-selected epoch 9 and the test-flat region agree. Epoch 7 reaches the same test F1 (48.4) with a smaller train-test gap (28.8 vs 34.8). If I wanted the least memorised weights I would take epoch 7, but I stay with epoch 9 because I selected it by the validation rule, not by looking at test.

### 5.2 The gap is shared by the other model I can inspect

The log has full train / validation / test rows for `bge_detr` and `lfm_base` only. The two other bge runs were skipped in the final session because they had already finished earlier, so I do not have their train numbers here.

| Model, epoch | Train | Validation | Test | Train − Test |
|---|---|---|---|---|
| bge_detr, 9 | 83.3 | 61.3 | 48.5 | 34.8 |
| lfm_base, 10 | 82.6 | 59.7 | 46.1 | 36.5 |
| lfm_base, 7 | 79.1 | 59.2 | 45.9 | 33.2 |

`bge_detr` is not worse than the alternative on this measure, and it ends higher on test.

### 5.3 Part of the validation-to-test drop is the data, not memorisation

Validation scores about 13 points higher than test (61.3 vs 48.5). Some of that is distribution shift: the test set has 29.1% sentences without an opinion against 18.9% in validation, and these are exactly what the detector gets wrong. That is why I read the gap as a data-shift effect on top of a moderate overfit, not as overfitting alone.

---

## 6. The Colab custom model: results and what they show

### 6.1 Training curve (validation, tuned thresholds)

| Epoch | Train loss | Val triple-F1 | pair-F1 | cat-F1 | attr-F1 | pol-acc | rating MAE |
|---|---|---|---|---|---|---|---|
| 1 | — | — | — | — | — | — | 1.47 |
| 2 | 1.599 | 0.302 | 0.317 | 0.664 | 0.469 | 0.927 | 0.522 |
| 4 | 1.219 | 0.419 | 0.443 | 0.679 | 0.603 | 0.901 | 0.460 |
| 6 | 1.002 | 0.501 | 0.547 | 0.726 | 0.703 | 0.911 | 0.442 |
| 8 | 0.871 | 0.539 | 0.592 | 0.750 | 0.727 | 0.914 | 0.390 |
| 10 | 0.770 | 0.564 | 0.616 | 0.787 | 0.742 | 0.908 | 0.391 |
| 12 | 0.717 | 0.574 | 0.632 | 0.804 | 0.748 | 0.905 | 0.369 |
| **14 (best)** | 0.687 | **0.582** | 0.638 | 0.807 | 0.748 | 0.911 | 0.371 |
| 15 | 0.692 | 0.580 | 0.639 | 0.815 | 0.749 | 0.911 | 0.371 |

The model was still improving at epoch 14 and flattened right at the end of its 15-epoch budget. It set a new best on 13 of 14 consecutive checks, so more epochs might have helped a little.

### 6.2 Test result (fold 0, 390 sentences, 543 gold triples)

| Metric | Value |
|---|---|
| triple P / R / F1 (tuned) | 0.607 / 0.545 / **0.574** |
| triple F1 with default thresholds | 0.507 |
| triple F1, negative / neutral / positive | 0.503 / **0.214** / 0.637 |
| pair F1 | 0.629 |
| category micro-F1 / macro-F1 | 0.781 / **0.256** |
| attribute micro-F1 / macro-F1 | 0.777 / 0.652 |
| polarity accuracy / macro-F1 | 0.873 / 0.715 |
| rating accuracy / within ±1 | 0.829 / 0.864 |
| rating MAE / RMSE | 0.452 / 0.928 |
| rating Spearman / QWK | 0.794 / 0.860 |
| ECE | 0.046 |
| sentence exact match | 0.415 |

### 6.3 What the Colab results show, and what worries me

- **Tuning is doing real work.** Default thresholds give 50.7; tuned thresholds give 57.4. The validation set has only 217 sentences, so six tuned parameters are a lot to fit. The close agreement between val (58.2) and test (57.4) suggests the tuning did not overfit badly, but the 6.7-point swing shows how threshold-dependent the model is.
- **It is carried by one category.** LAPTOP is 354 of 543 gold triples (65%). Category macro-F1 is 0.256: 14 of the 22 categories score exactly 0 (SOFTWARE, POWER_SUPPLY, MULTIMEDIA_DEVICES, GRAPHICS, MEMORY, CPU, HARDWARE, HARD_DISC, WARRANTY, SHIPPING, PORTS and others), and KEYBOARD gets 0.09 (16 gold, 1 hit).
- **Qualitative demo agrees.** For "The battery lasts 10 hours but the screen is dim" it returns BATTERY negative and LAPTOP negative, and misses DISPLAY. For "Super fast processor and a really nice graphics card" it returns LAPTOP ×2 and CPU, and misses GRAPHICS.
- **Neutral is poor** (F1 0.214 on 31 triples).
- **No context.** It reads one sentence alone, so it cannot use neighbouring sentences.

---

## 7. Putting the two side by side on metrics that transfer

These are the metrics that mean the same thing in both setups, regardless of the share of no-opinion sentences. I use the Colab model as the reference (its closest relative in the experiments is `bge_base`).

| Metric | Colab custom (fold 0) | bge_base | **bge_detr** |
|---|---|---|---|
| Sentiment accuracy | 87.3 | 82.8 | **88.6** |
| Sentiment macro-F1 | **71.5** | 65.8 | 71.2 |
| Star MAE | 0.452 | 0.53 | **0.35** |
| Star within ±1 | 86.4 | 86.3 | **90.9** |
| Star QWK | 0.860 | 0.797 | **0.879** |
| Star Spearman | 0.794 | 0.759 | **0.811** |
| ECE | **0.046** | 0.073 | 0.054 |
| Test sentences / gold triples | 390 / 543 | 808 / 801 | 808 / 801 |
| Share of test sentences without opinion | 0% | 29.1% | 29.1% |

My reading:

- **`bge_detr` ties or beats the Colab custom model on star quality.** It is ahead on sentiment accuracy, star MAE, within ±1, QWK and Spearman, and a hair behind on sentiment macro-F1 (71.2 vs 71.5) and calibration (0.054 vs 0.046). It achieves that on a test set that includes the harder empty sentences.
- **Both beat their respective plain baselines.** `bge_base` is the apples-to-apples relative of the Colab model (same encoder, flat heads, no context), and it is clearly worse than both, which supports the argument that the set head and backbone tweaks are what add value.
- **The Colab model's headline triple-F1 lead comes from the test set**, not from the model. It is evaluated on sentences that all contain opinions.

---

## 8. Why `bge_detr` and not the others

| Candidate | Why not |
|---|---|
| Colab custom model | Strong but evaluated on an easier test set; 14 of 22 categories at F1 0; six thresholds tuned on 217 sentences; no context. Star quality on the harder test is matched or beaten by `bge_detr`. |
| lfm_base | Closest competitor (46.1 vs 47.7 joint F1) but with weaker sentiment macro-F1 (62.9 vs 71.2), worse star RMSE (1.06 vs 0.88), 7× the parameters in the encoder (229.7M vs 33.2M), 7× the disk per epoch, and a train-test gap just as big. |
| bge_tweaked | Best star accuracy and empty-sentence accuracy, but joint F1 of only 36.7 because it misses too many opinions (det recall 37.0). |
| bge_base | Lowest on almost everything. Useful only as the reference point. |
| lfm_tweaked / lfm_detr | No result; the runs ran out of GPU memory. |

On size and cost, `bge_detr` is also the cheapest of the strong candidates: 35.4M parameters in total (33.2M encoder + 2.2M head), about 45–65 s of training per epoch on the T4.

---

## 9. Limitations and what I would do next

1. **One seed, one split.** Differences of 1–2 joint-F1 points (for example `bge_detr` vs `lfm_base`) are inside the noise of 801 gold pairs. The case rests on the combined pattern across metrics, not on that single number.
2. **Empty-sentence accuracy is low (43%).** `bge_detr` over-predicts opinions on sentences with none. A calibrated "no opinion" threshold or a larger `no_opinion_weight` in the set loss is the first thing I would try.
3. **Neutral is weak everywhere** (joint F1 0.14 for `bge_detr`, 0.21 for the Colab model). Neutral is only about 6% of the gold labels, and stars 1 and 3 never occur in the gold data, so the 0–4 scale is effectively three classes.
4. **Rare categories.** Only 80 of 198 E#A categories occur in training. Per-category F1 for the rare ones will be near zero for any of these models.
5. **Checkpoint selection in the demo cell.** The inference demo printed `result/bge_detr/checkpoint_8` (val joint F1 60.3), while the log and summary pick epoch 9 (61.3). The difference is within noise, but I will use epoch 9 consistently.
6. **LFM tweaks are untested.** I will restart the Colab runtime (the earlier OOM came from another process still holding about 8 GB on the GPU), set `CFG.grad_ckpt=True` and run only `lfm_tweaked` and `lfm_detr`. If either beats 47.7 test joint F1 with a smaller train-test gap, it replaces `bge_detr`.
7. **Further checks on `bge_detr`.** Run two more seeds; add stronger regularisation (higher dropout or fewer epochs, since epoch 7 matches epoch 9 on test); and read per-category results from the saved `extended_metrics.json` to see which of the 80 seen categories actually carry the score.

---

## 10. Summary table

| Question | Answer |
|---|---|
| Best model | `bge_detr`, epoch 9 |
| Test joint F1 | 47.7 (val 61.4) |
| Test sentiment accuracy / macro-F1 | 88.6 / 71.2 |
| Test star MAE / QWK | 0.35 / 0.879 |
| Detection AUC | 0.984 |
| Train / val / test joint F1 | 83.3 / 61.3 / 48.5 |
| Does test degrade with more epochs? | No: flat from epoch 7 to 10 |
| Better than the Colab custom model? | Equal or better on every metric that transfers; headline triple-F1 not directly comparable |
| Open item | `lfm_tweaked` and `lfm_detr` need a rerun |
