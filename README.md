# Laptop Review Analyser: Aspect-Based Sentiment Analysis

A system that reads a laptop review and returns, for every sentence, the `{ENTITY#ATTRIBUTE, sentiment}` opinions it holds (SemEval-2016 Task 5), with a 1 to 5 rating, a confidence and a per-product aspect table.

![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Streamlit 1.66](https://img.shields.io/badge/streamlit-1.66-red)
![Weights: Git LFS](https://img.shields.io/badge/weights-Git%20LFS-lightgrey)

![The Analysis page with a review, its breakdown and the Verbose trace panel](docs/screenshots/08_input_mixed_verbose.png)

## Quickstart

```bash
git lfs install
git clone https://github.com/rupeshghimire7/AI_Challenge_ITCF_CDU.git && cd AI_Challenge_ITCF_CDU
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run src/ui_streamlit.py
```

The app opens at http://localhost:8501. The first start downloads `BAAI/bge-small-en-v1.5` (about 130 MB) for the domain gates. Loading the model and the gates took 22 s on my CPU; after that a short review is analysed in well under a second (0.21 s for two sentences in the command-line run below).

---

## Contents

1. [What it does](#what-it-does)
2. [Features](#features)
3. [Screenshots](#screenshots)
4. [How it works](#how-it-works)
5. [Models and results](#models-and-results)
6. [Repository structure](#repository-structure)
7. [Installation](#installation)
8. [Usage](#usage)
9. [Training and experiments](#training-and-experiments)
10. [Configuration](#configuration)
11. [Output format](#output-format)
12. [Limitations](#limitations)
13. [FAQ and troubleshooting](#faq-and-troubleshooting)
14. [Citation and acknowledgements](#citation-and-acknowledgements)
15. [License](#license)

---

## What it does

The challenge brief asks for every opinion in a review as an entity and attribute pair with a polarity, where the entity or the attribute is `NULL` when the text does not name it. My system follows that brief. These are real outputs of `python -m src.main` and `scripts/evaluate_system.py`:

```text
"Great screen, but the battery dies fast."
    -> {BATTERY#OPERATION_PERFORMANCE, negative}, {DISPLAY#QUALITY, positive}
"Do not buy it!"                      -> {NULL#GENERAL, negative}        (nothing is named, so the entity is NULL)
"It is the worst laptop ever."        -> {LAPTOP#GENERAL, negative}
"Apple mac mini"                      -> {LAPTOP#NULL}, {COMPANY#NULL}   (named, but no opinion)
"The pizza was cold."                 -> {NULL#NULL}                     (stopped by a domain gate)
```

For a product, the opinions of all its reviews are summarised in an aspect-sentiment table. This is the table for the two reviews above (`docs/artifacts/usage/aspect_table.csv`):

| category | mentions | negative | neutral | positive | reviews | rating | stars |
|---|---|---|---|---|---|---|---|
| DISPLAY#QUALITY | 1 | 0 | 0 | 1 | 1 | 3.4 | ★★★☆☆ |
| LAPTOP#GENERAL | 1 | 1 | 0 | 0 | 1 | 1.3 | ★☆☆☆☆ |
| BATTERY#OPERATION_PERFORMANCE | 1 | 1 | 0 | 0 | 1 | 1.2 | ★☆☆☆☆ |

## Features

- **Sentence-level ABSA with review context.** Each sentence is read together with its previous and next sentence, as the brief asks.
- **198 categories.** The model can output any of the 22 × 9 entity and attribute pairs of the SemEval-2016 laptop guidelines.
- **Five-star sentiment.** Every opinion gets a star distribution from 0 to 4; the system turns it into a sentiment, a 1 to 5 rating, a 90% star interval and a confidence label.
- **Seven domain gates.** Sentences that are not about a laptop (a pizza, a phone, a greeting) are stopped before the model and returned as `{NULL#NULL}` with the reason.
- **The brief's NULL rule.** The entity becomes `NULL` when the sentence does not name it.
- **Long text is never cut.** Text longer than one window is split into overlapping windows and merged.
- **Streamlit app.** Single review, CSV batch with one aspect table per product, CSV and JSON downloads, and a Verbose panel that shows every decision.
- **Command line and Python API** for the same pipeline.

## Screenshots

All screenshots were taken from the running app with Playwright (`docs/tools/capture_screenshots.py`); `docs/screenshots/index.md` lists every image.

| | |
|---|---|
| ![Analysis page on first load](docs/screenshots/01_home.png) | ![Breakdown of the example review](docs/screenshots/02_results_viewport.png) |
| **Analysis page.** Example pills, the review box and the product name. | **Breakdown.** An overall verdict, then each sentence with its tuples, stars and confidence. |
| ![Output tab](docs/screenshots/04_output_tab.png) | ![Aspect table tab](docs/screenshots/05_aspect_table.png) |
| **Output tab.** The brief's format, with CSV and JSON downloads. | **Aspect table.** Mentions by sentiment and a rating per category. |
| ![Off-domain example](docs/screenshots/10_off_domain_gates.png) | ![Batch CSV](docs/screenshots/14_batch_sample_results.png) |
| **Gates at work.** Both sentences are stopped, and the trace names the gate. | **Batch CSV.** The built-in sample of six reviews of two laptops. |

The Verbose panel for one sentence (`docs/screenshots/23_trace_panel_mixed.png`) shows the preprocessing, every gate with its score, the raw model output and the post-processing checks:

<img src="docs/screenshots/23_trace_panel_mixed.png" alt="Verbose trace panel" width="360">

---

## How it works

![System flow](docs/diagrams/system_flow.png)

A review passes through seven stages. All of them live in `src/`, and the Streamlit app, the command line and the evaluation script use the same `Pipeline` class (`src/main.py`).

| Stage | Module | What happens | What it passes on |
|---|---|---|---|
| 1 Preprocess | `src/preprocess.py` | HTML entities and tags, Unicode NFKC, curly quotes, invisible characters, letter runs capped at 3 ("soooo" becomes "sooo"). Split into sentences without breaking after abbreviations. Tags: nuances (negation, contrast, comparison, ...), the entities a sentence names, off-topic objects, filler. | one record per sentence, with its previous and next sentence |
| 2 Domain gates | `src/domain_gate.py` | Seven checks decide whether the sentence is about a laptop (below). | `gate` dict with every gate's result and the sentence's mentions |
| 3 Model | `src/model_inference.py` | bge_detr, epoch 9, reads `[CLS] previous [SEP] sentence [SEP] next [SEP]`. | `det` [n, 198] scores and `pol` [n, 198, 5] star probabilities |
| 4 Decode | `src/post_process.py` | Every category scoring 0.3 or more, with its star summary. | opinions, highest score first |
| 5 Post-process | `src/post_process.py` | Threshold τ = 0.85, a second threshold 0.80, the entity rule, no-opinion tuples. | tuples |
| 6 Merge | `src/main.py` | A `{NULL#NULL}` sentence that no hard gate stopped is joined with a neighbour and tried again. | tuples |
| 7 Output | `src/post_process.py`, `src/ui_streamlit.py` | Rating, stars, confidence label, the brief's format, the aspect table. | screen, CSV, JSON |

### The domain gates

![Domain gates](docs/diagrams/gates.png)

The model was trained only on laptop reviews, so it reads any opinion as a laptop opinion. The gates run before it, cheapest first:

| # | Gate | Kind | Passes when | Example |
|---|---|---|---|---|
| 1 | tokens | hard | the bge tokenizer finds a word piece with a letter | stops "!!!", "12345" |
| 2 | filler | hard | at least one word is not a greeting, thanks or stop word | stops "Hello.", "Thanks everyone!" |
| 3 | off-topic | hard | no other object (phone, pizza, hotel, ...) is the subject or what is loved or bought; skipped when the laptop is named or the sentence compares | stops "The pizza was cold." |
| 4 | laptop named | strong pass | laptop, computer, MacBook, a brand, ... is named: the sentence goes to the model | "It is the worst laptop ever." |
| 5 | aspect terms | evidence | a mention of one of the 22 entities, by word list or by a content word with cosine ≥ 0.74 to an entity label | "build" is HARDWARE at 0.85 |
| 6 | domain | evidence | best cosine to the 22 laptop labels minus best cosine to 22 other topics ≥ −0.11 | |
| 7 | category | evidence | best cosine to a laptop label ≥ 0.43 | |

A sentence goes to the model when gates 1 to 3 pass and either gate 4 passes or gate 6 passes together with gate 5 or 7. The similarities use the pretrained `bge-small-en-v1.5` [CLS] vector. The three thresholds were fitted on laptop training sentences only (`python -m src.domain_gate`).

### The model

![bge_detr](docs/diagrams/arch_detr.png)

`bge_detr` is a bge-small-en-v1.5 encoder (12 layers, hidden size 384) with two input tweaks and a DETR-style set head:

- **Review context.** The previous and next sentence (up to 32 tokens each) get token type 1.
- **Negation scope.** A negation cue and up to five following words get a learned vector added to their embedding (`negation_spans` in `src/model_inference.py`).
- **Set head.** Sixteen learned queries read the sentence tokens through a two-layer transformer decoder. Each query predicts an entity (22 or "no opinion"), an attribute (9) and a star (0 to 4). The score of a category is `max over queries of p(E) · p(A)`, and its star distribution comes from that query.

**Long input.** A sentence longer than 54 tokens is cut into 54-token chunks with a step of 41 (13 tokens of overlap). Up to four chunks are read by the decoder together; longer text is split into groups of four, and the groups are merged by taking each category's maximum score. In a real run (`docs/artifacts/predictions/window_demo.json`), a 243-token run-on text became 6 windows and 2 groups, and `SOFTWARE#MISCELLANEOUS` scored 0.0369 in the first group and 0.9342 in the second, so it kept 0.9342.

![Window merging](docs/diagrams/window_merging.png)

### Post-processing

![Post-processing](docs/diagrams/post_processing.png)

`star_summary` in `src/post_process.py` turns the five star probabilities p(0..4) into the output fields:

| Field | Rule |
|---|---|
| expected star | E = Σ k · p(k), k = 0 to 4 |
| 90% interval | the narrowest run of stars holding at least 90% of the probability |
| brackets | negative = p(0) + p(1), neutral = p(2), positive = p(3) + p(4) |
| certain | the 90% interval sits inside one bracket |
| sentiment | the interval's bracket when certain; otherwise the largest bracket, except that an interval from negative to positive stars with no bracket above 0.5 is neutral (a mixed opinion) |
| confidence | the probability of the chosen bracket |
| star | E rounded and kept inside the chosen bracket; the rating is star + 1 |
| confidence label | high: confidence ≥ 0.8 and certain; medium: ≥ 0.6; otherwise low |

Then `finalise` applies the rules of the brief:

| Rule | What it does | Example (real output) |
|---|---|---|
| threshold τ = 0.85 | categories scoring τ or more are the opinions (τ was fitted on validation during training) | |
| second threshold 0.80 | a sentence that mentions a part but has nothing at τ keeps the categories scoring 0.80 or more | "The hinge broke after two weeks." → `{LAPTOP#QUALITY, negative}` |
| entity rule | the entity stays when it is named, or when the sentence mentions some part and one more check agrees (named in a neighbouring sentence, the whole laptop, or among the three closest labels); otherwise it is NULL | "Do not buy it!" → `{NULL#GENERAL, negative}` |
| no opinion | each mentioned entity with attribute NULL, or `{NULL#NULL}` | "Apple mac mini" → `{LAPTOP#NULL}, {COMPANY#NULL}` |
| merge | a `{NULL#NULL}` sentence that no hard gate stopped is joined with its previous and next sentence; new opinions of the joined text become its own | "Called tech support about the screen. Gave me the runaround ..." → the second sentence gets `{NULL#OPERATION_PERFORMANCE, negative}` |
| part re-map | off by default (`REMAP = False`): it costs about one joint-F1 point | |

**Worked example.** For "Great screen, but the battery dies fast.", `DISPLAY#QUALITY` scores 0.9630. Its star probabilities are 0.2835, 0.1038, 0.0055, 0.1645, 0.4428, so E = 2.379, the 90% interval is [0, 4] and the prediction is not certain. The positive bracket holds 0.607, which is above 0.5, so the sentiment is positive with confidence 0.607 ("medium"). The star is 2 after rounding and becomes 3 inside the positive bracket, so the rating is 4. "screen" is in the DISPLAY word list, so the entity stays: `{DISPLAY#QUALITY, positive}`, 4 of 5 stars, 61% confident.

---

## Models and results

### Data

I used the SemEval-2016 Task 5 English laptop data (subtask 1 sentences and subtask 2 review labels, `DATA/HF/`). `scripts/prepare_data.py` decodes the double-escaped HTML of the test XML, normalises the text, removes 19 exact duplicates and 6 training sentences that also appear in the test set, and flags polarity conflicts. I also removed the auxiliary Laptop-ACOS data (commit 4fef809) because its labels conflicted. Validation is 15% of the training reviews, chosen by a hash of the review id, so no review is split.

| Split | Reviews | Sentences | Opinions | Without opinion | Stars 0 / 2 / 4 |
|---|---|---|---|---|---|
| train | 374 | 2,109 | 2,452 | 390 (18.5%) | 928 / 150 / 1,374 |
| validation | 75 | 366 | 434 | 69 (18.9%) | 153 / 38 / 243 |
| test (official gold) | 80 | 808 | 801 | 235 (29.1%) | 274 / 46 / 481 |

Source: `DATA/final/reports/final_data.txt`. Only 80 of the 198 categories occur in training, and stars 1 and 3 never occur in the gold data.

### Experiments

I trained two encoders (bge-small-en-v1.5 and LFM2.5-Encoder-230M), each with three head families, in `models/notebooks/ABSA_experiments_base_tweak.ipynb`:

| Variant | What it is |
|---|---|
| base | the encoder's sentence vector with a linear detector and a linear star head |
| tweaked | review context, negation scope, category attention, a factorised E × A detector, an opinion gate, asymmetric and focal losses, long-input training, layer-wise lr decay |
| detr | the tweaked encoder with the set head above |

Every run used 10 epochs, batch 32, seed 2026, max_len 56 and no early stopping. After each epoch, τ is fitted on validation, and the best epoch is the one with the highest validation joint F1. The test set is only reported.

![Architecture comparison](docs/diagrams/arch_comparison.png)

### Results on the test set (each run at its best validation epoch, %)

| Run | Best epoch | τ | Val joint F1 | Test det F1 | Test joint P / R / F1 | Sentiment acc / macro-F1 | Empty acc | Star MAE |
|---|---|---|---|---|---|---|---|---|
| **bge_detr** (selected, deployed) | 9 | 0.85 | 61.35 | 54.0 | 47.8 / 49.2 / **48.5** | 88.8 / 71.1 | 46.4 | 0.36 |
| lfm_base | 10 | 0.40 | 59.71 | 50.6 | 54.9 / 39.7 / 46.1 | 84.0 / 62.9 | 69.4 | 0.52 |
| bge_tweaked | 9 | 0.60 | 44.99 | 40.5 | 40.5 / 33.6 / 36.7 | 86.0 / 72.2 | 83.0 | 0.37 |
| bge_base | 9 | 0.20 | 46.14 | 42.1 | 37.3 / 34.8 / 36.0 | 82.8 / 65.8 | 52.3 | 0.53 |
| lfm_tweaked | - | - | - | CUDA out of memory, no result | | | | |
| lfm_detr | - | - | - | CUDA out of memory, no result | | | | |

Sources: `model_checkpoint/bge_detr/checkpoint_9/metrics.csv` for bge_detr, and the notebook's `summary.csv` (in `result/notebook_runs/absa_experiments.zip`) for the others. Joint F1 counts a prediction as correct only when the sentence, the E#A category and the sentiment are all right.

> **A note on 47.7 and 48.5.** The notebook summary, `models/notebooks/ABSA_model_selection_report.md` and the app's Model Experiments page give 47.7 for bge_detr. While checking the files, I found that two training sessions of bge_detr wrote into the same run folder, and the summary row takes its validation number from one session and its test number from the other. The deployed checkpoint is the epoch-9 model of the later session (τ = 0.85, validation 61.35), and its own `metrics.csv` and the training log both give a test joint F1 of 48.5. I report 48.5 for the weights in this repository. As an independent check, I re-scored the deployed checkpoint with `python models/infer.py --checkpoint model_checkpoint/bge_detr/checkpoint_9 --test` on a CPU and got 48.6 (`docs/artifacts/usage/models_infer_test.txt`); the training run scored it in bfloat16 on a GPU, which probably explains the 0.1 difference.

Extended test metrics at the same epochs (`summary_extended.csv`; the bge_detr column equals `checkpoint_9/extended_metrics.json`), best in bold:

| Metric (test) | bge_detr | lfm_base | bge_tweaked | bge_base |
|---|---|---|---|---|
| Joint F1, negative | **0.397** | 0.336 | 0.281 | 0.246 |
| Joint F1, neutral | **0.140** | 0.074 | 0.091 | 0.076 |
| Joint F1, positive | **0.563** | 0.543 | 0.440 | 0.435 |
| Detection AP (micro) | 0.532 | **0.533** | 0.350 | 0.386 |
| Detection AUC (micro) | **0.984** | 0.962 | 0.895 | 0.932 |
| Detection mAP (macro) | **0.387** | 0.331 | 0.222 | 0.215 |
| Star exact accuracy | 0.792 | 0.699 | **0.798** | 0.716 |
| Star within ±1 | **0.909** | 0.883 | 0.885 | 0.863 |
| Star QWK | **0.879** | 0.806 | 0.872 | 0.797 |
| Star Spearman | **0.811** | 0.766 | 0.808 | 0.759 |
| Star RMSE (lower is better) | **0.882** | 1.063 | 0.899 | 1.107 |
| Confidence ECE (lower is better) | 0.054 | 0.086 | **0.049** | 0.073 |
| Sentence exact match | 0.380 | **0.416** | 0.345 | 0.314 |
| Predicted opinions per sentence (gold 0.99) | 1.020 | 0.717 | 0.822 | 0.926 |

![Extended metrics](docs/diagrams/chart_extended.png)

### Is bge_detr overfit?

![Learning curve](docs/diagrams/chart_learning_curve_bge_detr.png)

| Epoch | Train | Validation | Test | Train − test |
|---|---|---|---|---|
| 1 | 33.6 | 32.20 | 26.1 | 7.5 |
| 2 | 48.5 | 39.85 | 34.2 | 14.3 |
| 3 | 56.5 | 47.45 | 38.7 | 17.8 |
| 4 | 64.4 | 52.51 | 43.6 | 20.8 |
| 5 | 71.9 | 57.56 | 45.3 | 26.6 |
| 6 | 74.5 | 56.64 | 45.3 | 29.2 |
| 7 | 77.2 | 58.52 | 48.4 | 28.9 |
| 8 | 81.2 | 60.32 | 48.2 | 33.0 |
| **9** | 83.3 | 61.35 | 48.5 | 34.8 |
| 10 | 84.6 | 60.18 | 48.9 | 35.8 |

Train joint F1 keeps rising while test stays near 48, so the model memorises part of the training set. But test does not fall: from epoch 7 to 10 it moves within 0.7 points, which is a plateau and not a collapse. Validation chose epoch 9 by rule, and lfm_base shows a gap of the same size (82.6 train, 46.1 test at epoch 10). Part of the validation-to-test drop is the data: 29.1% of test sentences have no opinion against 18.9% in validation, and those sentences are the model's main error. Source: `docs/_data/epoch_metrics_final_session.csv`, built from each run's `metrics.csv`.

### The custom Colab model

`models/notebooks/ABSA_colab.ipynb` trains a separate model, the custom 3 head transformer classifier: bge-small with mean pooling, no context, and category, attribute and pair heads (34.12M parameters). On its own held-out fold it reaches 57.4 triple F1 (P 60.7, R 54.5; 50.7 with untuned thresholds). That number is not comparable with ours: its 390 test sentences all contain an opinion, so it can never be punished for false opinions on empty sentences, and six thresholds were tuned on 217 validation sentences. On the metrics defined the same way in both setups, bge_detr wins five of seven:

| Metric | Custom 3 head classifier (its fold 0) | bge_base | bge_detr |
|---|---|---|---|
| Sentiment accuracy | 87.3 | 82.8 | 88.8 |
| Sentiment macro-F1 | 71.5 | 65.8 | 71.1 |
| Star MAE (lower is better) | 0.451 | 0.534 | 0.356 |
| Star within ±1 | 86.4 | 86.3 | 90.9 |
| Star QWK | 0.860 | 0.797 | 0.879 |
| Star Spearman | 0.794 | 0.759 | 0.811 |
| Confidence ECE (lower is better) | 0.046 | 0.073 | 0.054 |
| Test sentences / gold opinions | 390 / 543 | 808 / 801 | 808 / 801 |

### Why bge_detr

- It has the best test joint F1 (48.5) and the best joint F1 for each sentiment.
- The set head is what lifts it: bge_tweaked shares its encoder tweaks, and swapping in the set head moves test joint F1 from 36.7 to 48.5 and detection AUC from 0.895 to 0.984.
- It does not degrade with more epochs, and its overfitting gap matches the other model I could inspect.
- It is small: 35,432,613 parameters, against 229,693,184 in the LFM encoder alone.

The full reasoning is in [`models/notebooks/ABSA_model_selection_report.md`](models/notebooks/ABSA_model_selection_report.md); the experiments are in [`ABSA_experiments_base_tweak.ipynb`](models/notebooks/ABSA_experiments_base_tweak.ipynb) and [`ABSA_colab.ipynb`](models/notebooks/ABSA_colab.ipynb).

### The whole system

`scripts/evaluate_system.py` runs the full pipeline on the gold sentences in review order. These numbers are from my run on 2026-10-29 (`docs/artifacts/evaluate_system.txt`), joint F1 with the system's E#A:

| Setting | Validation | Test |
|---|---|---|
| Model only, τ = 0.85, with the entity rule | 58.9 | 48.4 |
| + second threshold 0.80 | 59.3 | 48.4 |
| + merge of `{NULL#NULL}` sentences (default) | **59.8** | **48.2** |
| Default with the gates off | 60.0 | 48.4 |
| Default + part re-map (off) | 58.5 | 47.3 |

The rules change the form of the answer more than the score: on test the default is within 0.2 points of the bare model, and on validation it is 0.9 points better. The gates stop 15 of the 808 laptop test sentences, 7 of which the model had an opinion for. The gate check against restaurant sentences (98.6% of laptop opinion sentences kept, 59.5% of restaurant sentences rejected, from the `src/domain_gate.py` docstring) needs the SemEval restaurant XML, which is not in the repository, so I could not re-run it.

---

## Repository structure

| Path | Purpose |
|---|---|
| `src/` | **The system, one task per file** |
| `src/ui_streamlit.py` | Streamlit app: Analysis, Batch CSV, About System, Model Experiments; Verbose trace panel |
| `src/main.py` | `Pipeline` (preprocess → gates → model → post-process → merge) and the command line |
| `src/preprocess.py`, `src/nuance_checker.py` | normalisation, sentence splitting, artefact and nuance patterns |
| `src/aspect_terms.py`, `src/domain_gate.py` | entity word lists; the seven gates and the mentions |
| `src/model_inference.py` | the bge_detr checkpoint: input windows, set head, `predict` |
| `src/post_process.py` | star summary, rules, rating, confidence, brief format, aspect table |
| `src/config.py`, `src/data_maker.py` | paths, the 22 × 9 label space; loading of `DATA/final` |
| `models/` | **Model phase** |
| `models/bge.py`, `models/lfm.py` | the two encoders and their tweak wrappers |
| `models/heads.py`, `models/set_detector.py` | base and tweaked heads; the DETR-style set head |
| `models/train.py`, `models/infer.py`, `models/data.py` | training, checkpoint inference, data to tensors and metrics |
| `models/custom_*.py` | a standalone three-head model with its own trainer and inference |
| `models/notebooks/` | the experiments notebook, the Colab notebook and the model selection report |
| `scripts/` | data preparation and checks: `prepare_data.py`, `eda.py`, `preprocess.py`, `evaluate_system.py` |
| `DATA/HF/`, `DATA/final/` | SemEval-2016 laptop XML; prepared CSVs and reports |
| `model_checkpoint/bge_detr/checkpoint_9/` | the selected checkpoint (model.pt in Git LFS, config with τ, tokenizer, metrics, test predictions) |
| `docs/` | this documentation: screenshots, diagrams, artifacts, the presentation and the tools that build them (currently git-ignored, see `GAPS.md`) |
| `result/` | local only (git-ignored): `notebook_runs/absa_experiments.zip` with every run's metrics and best weights |

---

## Installation

### Prerequisites

- Python 3.12 (the version the repository was developed and tested with).
- Git and [Git LFS](https://git-lfs.com) for the model weights.
- Disk space for PyTorch and the other packages, plus 142 MB for `model.pt`. The CUDA build of torch is much larger than the CPU build.
- A GPU is optional. The app runs on a CPU; `--device auto` uses a GPU only if a kernel really runs on it.

### Clone with the weights

```bash
git lfs install                       # once per machine
git clone https://github.com/rupeshghimire7/AI_Challenge_ITCF_CDU.git
cd AI_Challenge_ITCF_CDU
```

If you cloned before installing Git LFS, the weights are only small pointer files. Fetch them with:

```bash
git lfs install
git lfs pull                                                     # every LFS file
git lfs pull --include="model_checkpoint/bge_detr/checkpoint_9/model.pt"   # only the model
```

Check that the weights really downloaded:

```bash
git lfs ls-files                      # "*" after the id means the file is present, "-" means only a pointer
ls -l model_checkpoint/bge_detr/checkpoint_9/model.pt   # should be 141,818,515 bytes
```

A `model.pt` of about 130 bytes is only an LFS pointer, and the app will stop with "has no weights: run `git lfs pull`".

To clone quickly without the weights and fetch them later:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/rupeshghimire7/AI_Challenge_ITCF_CDU.git
cd AI_Challenge_ITCF_CDU && git lfs pull
```

### Python environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# smaller CPU-only torch:
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

`requirements.txt` lists the version each package was tested with.

### Where the weights live

`src/config.py` points to `model_checkpoint/bge_detr/checkpoint_9/`. That folder holds `model.pt`, `config.json` (τ, the 198 categories, the settings), `encoder_config.json` and `tokenizer.json`, so the model needs no download. If `model.pt` is missing or a pointer and `result/notebook_runs/absa_experiments.zip` exists, `ABSAModel` unpacks the weights from that zip instead. The domain gate downloads `BAAI/bge-small-en-v1.5` from Hugging Face on its first run and then uses the local cache.

---

## Usage

### Streamlit app

```bash
streamlit run src/ui_streamlit.py     # from the repository root, so .streamlit/config.toml applies
```

| Page | What you do | What you get |
|---|---|---|
| **Analysis** (default) | type a review or pick an example, name the product, press "Analyse review" | the overall verdict, each sentence with its tuples, stars and confidence, the brief's format with CSV and JSON downloads, and the aspect table |
| **Batch CSV** | upload a CSV (or use the sample), pick the text and product columns | the same views for every review, with a product selector |
| **About System** | read | the four steps, the seven gates, the entity rule, a legend and an in-depth flow chart |
| **Model Experiments** | read | architecture cards, the candidate table, the custom model and the system ablation |

The **Verbose** switch opens the trace panel. It shows, for each sentence, the normalised text and what was repaired, the nuances and mentions, every gate with its score and threshold, the raw model categories with scores, the merge and the entity checks behind each tuple. I added it so that anyone grading the system can see why each answer was given without reading the code.

![Verbose trace for a multi-sentence review](docs/screenshots/09_input_negated_multi_verbose.png)

### Command line

```bash
python -m src.main --text "Great screen, but the battery dies fast." --text "Do not buy it!"
python -m src.main --file reviews.txt --product "Apple Mac mini"          # one review per line
python -m src.main --csv reviews.csv --text-column review --product-column asin --out output/
python -m src.main --csv reviews.csv --text-column review --product-column asin --product B0074703CM
```

| Option | Meaning |
|---|---|
| `--text` | a review; repeat for several |
| `--file` | a text file with one review per line |
| `--csv` | a CSV with one review per row |
| `--text-column` | the CSV column with the review (default `text`) |
| `--product-column` | the CSV column with the product name or id (one aspect table per product) |
| `--product` | the product name for `--text` / `--file`; with `--product-column`, keep only this product |
| `--checkpoint` | another bge_detr checkpoint folder |
| `--device` | `auto`, `cpu` or `cuda` |
| `--out` | folder for `opinions.csv` and `aspect_table.csv` |

Real output of the first command with `--product "Demo laptop"` (`docs/artifacts/usage/src_main_text.txt`):

```text
bge_detr epoch 9, threshold 0.85, cpu: loaded in 22.1 s; 2 reviews, 2 sentences (2 passed the gates) in 0.21 s

[Demo laptop, review 0]
  Great screen, but the battery dies fast.
    -> {BATTERY#OPERATION_PERFORMANCE, negative}, {DISPLAY#QUALITY, positive}

[Demo laptop, review 1]
  Do not buy it!
    -> {NULL#GENERAL, negative}
```

### Python

```python
import pandas as pd
from src.main import Pipeline
from src.post_process import brief, opinion_rows, aspect_table

pipe = Pipeline()                                       # loads the model and the gates once
records = pipe.run(pd.DataFrame({"product": ["x"], "review_id": [0], "text": ["Great screen, awful battery."]}))
for r in records:
    print(r["sentence"], brief(r["tuples"]))
table = aspect_table(opinion_rows(records))
```

### Any experiment checkpoint: `models/infer.py`

```bash
python models/infer.py --checkpoint model_checkpoint/bge_detr/checkpoint_9 --text "Great screen, but the battery dies fast."
python models/infer.py --checkpoint model_checkpoint/bge_detr/checkpoint_9 --file reviews.txt
python models/infer.py --checkpoint model_checkpoint/bge_detr/checkpoint_9 --test     # score DATA/final/test_final.csv
python models/infer.py --table --out result                                          # compare every run in result/
```

This script runs the bare model with its own τ, without the gates and post-processing rules. Its options are `--checkpoint`, `--text`, `--file`, `--test`, `--table`, `--out` and `--device`. Real output:

```text
Great screen, but the battery dies fast.
  -> {BATTERY#OPERATION_PERFORMANCE, negative}, {DISPLAY#QUALITY, positive}
     BATTERY#OPERATION_PERFORMANCE        star 0 (expected 0.22, 90% interval 0-1)  negative 100%  detection 0.99
     DISPLAY#QUALITY                      star 3 (expected 2.38, 90% interval 0-4)  positive 61% (uncertain)  detection 0.96
```

### Checks and reports

| Command | What it does | Report |
|---|---|---|
| `python scripts/prepare_data.py` | parse the SemEval-2016 XML, merge subtasks, clean | `DATA/final/reports/prepare_data.txt` |
| `python models/data.py` | write `{train,test}_final.csv` | `DATA/final/reports/final_data.txt` |
| `python scripts/eda.py` | exploratory data analysis with figures | `DATA/final/reports/eda.txt` |
| `python scripts/preprocess.py` | check preprocessing and the sentence splitter | `DATA/final/reports/preprocess.txt` |
| `python -m src.domain_gate` | fit the gate thresholds | printed |
| `python scripts/evaluate_system.py` | end-to-end system evaluation, ablation and edge cases | `DATA/final/reports/evaluate_system.txt` |

---

## Training and experiments

### Retrain in Colab (how the results were made)

1. Open `models/notebooks/ABSA_experiments_base_tweak.ipynb` in Google Colab with a GPU (a T4 is enough; LFM is the slow one).
2. Upload `DATA/final/train_final.csv` and `DATA/final/test_final.csv` when the data cell asks for them.
3. Choose the runs in `CFG.runs` (`bge_base`, `bge_tweaked`, `bge_detr`, `lfm_base`, `lfm_tweaked`, `lfm_detr`) and run all cells. For the LFM tweaked runs, set `CFG.grad_ckpt = True` and restart the runtime first; without it, both crashed with CUDA out of memory.
4. The last cell packs `absa_experiments.zip` with every metric file and the best epoch's weights of each run.
5. To deploy a new bge_detr checkpoint, copy its `checkpoint_<epoch>/` folder to `model_checkpoint/bge_detr/` and point `CHECKPOINT` in `src/config.py` to it.

The custom 3 head classifier is trained in `models/notebooks/ABSA_colab.ipynb` (it writes `absa_results.zip`).

### Train locally

```bash
python models/train.py --model bge --variant detr --epochs 10        # --variant base | tweaked | detr, --model bge | lfm
python models/train.py --model lfm --variant tweaked --grad-ckpt     # less GPU memory
```

Other options: `--batch-size`, `--lr`, `--device`, `--out`, `--seed`, `--limit` (smoke test). The default is 5 epochs; the notebook runs used 10.

### Output layout of the experiments

```text
result/
  <model>_<variant>/
    checkpoint_<epoch>/   model.pt, config.json (τ, settings, categories), encoder_config.json, tokenizer.json,
                          metrics.csv (train / validation / test), test_predictions.csv, extended_metrics.json
    metrics.csv           every epoch          extended_metrics.csv
    log.txt               training log, with the printed architecture
    best_epoch_report/    per-category, per-entity and per-attribute tables, tau sweeps, confusion matrices
  summary.csv, summary_extended.csv, compare_runs.png, data_report.txt
```

The printed architectures and parameter counts of all six variants can be regenerated with `python docs/tools/print_architectures.py` (see `docs/artifacts/architectures/`).

---

## Configuration

| Setting | Value | Where |
|---|---|---|
| checkpoint folder | `model_checkpoint/bge_detr/checkpoint_9` | `src/config.py` (`CHECKPOINT`) |
| detection threshold τ | 0.85 | the checkpoint's `config.json` (fitted on validation) |
| second threshold | 0.80 | `src/post_process.py` (`TAU_LOW`) |
| decode floor | 0.3 | `src/main.py` (`FLOOR`) |
| part re-map | off | `src/post_process.py` (`REMAP`) |
| confidence labels | high ≥ 0.8 and certain; medium ≥ 0.6 | `src/post_process.py` (`HIGH`, `MEDIUM`) |
| star interval | 90% | `src/post_process.py` (`LEVEL`) |
| gate thresholds | margin −0.11, category 0.43, word 0.74 | `src/domain_gate.py` (`MARGIN`, `CATEGORY_FLOOR`, `WORD_FLOOR`) |
| window | 56 tokens (54 target + [CLS] + [SEP]), 32 context tokens per side, step 41 | checkpoint `config.json` (`max_len`, `ctx_len`), `src/model_inference.py` |
| batching | 64 windows per encoder pass, 4 windows per decoder segment | `src/model_inference.py` (`CHUNK`, `GROUP`) |
| theme | colours, DM Sans / Fraunces / DM Mono, 8 px widget radius | `.streamlit/config.toml` (git-ignored) and the CSS in `src/ui_streamlit.py` |

---

## Output format

Every sentence gets at least one tuple. `opinions.csv` (from `--out` or the Download CSV button) has one row per tuple:

| Column | Meaning |
|---|---|
| `product`, `review_id`, `sentence_index`, `sentence` | where the tuple comes from |
| `aspect` | the final E#A in the brief's format; `NULL#A` when the entity is not explicit; `E#NULL` or `NULL#NULL` without an opinion |
| `category` | the system's E#A (used by the aspect table and the metrics) |
| `model_category` | what the model said before the entity rule |
| `sentiment` | negative, neutral or positive; empty without an opinion |
| `star` | 0 to 4, kept inside the sentiment's bracket; the rating shown is star + 1 |
| `expected_star` | E = Σ k · p(k) |
| `star_low`, `star_high` | the 90% star interval |
| `confidence` | probability of the sentiment's bracket |
| `certain` | True when the 90% interval sits inside one bracket |
| `score` | the model's detection score for the category |
| `source` | `model`, `second threshold 0.80`, or `merged with the previous / next sentence` |
| `note` | why there is no opinion (the gate's reason, or "no opinion") |

`output.json` (Download JSON) holds, per sentence, `product`, `review`, `sentence` and `tuples` with `aspect`, `category`, `model_category`, `sentiment`, `confidence`, `source`, `note`, and for opinions `rating` and `stars`.

`aspect_table.csv` has one row per product and category: `mentions` (opinions), `negative`, `neutral`, `positive`, `reviews` (reviews that mention it), `rating` (1 + the mean over reviews of each review's mean expected star, so one review counts once) and `stars`.

---

## Limitations

- **Empty sentences.** The model gives an opinion to many sentences that have none: empty-sentence accuracy is 46.4% on test. The gates remove off-domain sentences, but an in-domain sentence without an opinion can still get one.
- **Neutral is weak.** Joint F1 for neutral is 0.140. Neutral is about 6% of the gold opinions, and stars 1 and 3 never occur in training.
- **Rare categories.** Only 80 of the 198 categories occur in training.
- **One seed, one split.** Differences of one or two joint-F1 points between models are within noise.
- **Two runs missing.** lfm_tweaked and lfm_detr ran out of GPU memory and were never compared.
- **Known misses** (from `scripts/evaluate_system.py`): "The fan is noisy." gives `{LAPTOP#QUALITY, negative}` rather than a FANS_COOLING category; "This laptop is way better than my old phone." gives only `{LAPTOP#NULL}`.
- **English laptop reviews only.**

---

## FAQ and troubleshooting

**The app stops with "has no weights: run `git lfs pull`".** `model.pt` is an LFS pointer. Run `git lfs install` and `git lfs pull`, then check the file size (141,818,515 bytes).

**The first start is slow or fails offline.** The domain gate downloads `BAAI/bge-small-en-v1.5` once. Run the app once with internet access; later starts use the Hugging Face cache.

**The theme or fonts look different.** Run `streamlit run` from the repository root so `.streamlit/config.toml` is found. That file is git-ignored, so a fresh clone uses Streamlit's default theme; the layout CSS in `src/ui_streamlit.py` still applies.

**`git status` shows `model.pt` as modified although I changed nothing.** Run `git lfs install` in the clone and then `git status` again; if the file's SHA-256 is `1bf39fb1...` it is the committed checkpoint.

**pip downloads a very large torch.** Use the CPU index: `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu`.

**CUDA errors on an old GPU.** Use `--device cpu`; `auto` already falls back to the CPU when a test kernel fails.

**Why do the app and the notebook report 47.7 while this README says 48.5?** See the note under [Results](#results-on-the-test-set-each-run-at-its-best-validation-epoch-).

---

## Citation and acknowledgements

- Data: M. Pontiki et al., "SemEval-2016 Task 5: Aspect Based Sentiment Analysis", Proceedings of SemEval-2016. The challenge brief points to the Hugging Face copy `alexcadillon/SemEval2016Task5`.
- Encoder: `BAAI/bge-small-en-v1.5` (FlagEmbedding, BAAI).
- Encoder: `LiquidAI/LFM2.5-Encoder-230M` (Liquid AI).
- Set prediction: N. Carion et al., "End-to-End Object Detection with Transformers" (DETR), 2020, which `models/set_detector.py` adapts to opinions.
- Built for the CDU IT Code Fair Sydney 2026 AI Challenge.

## License

The code and documentation in this repository are released under the [MIT License](LICENSE) (copyright 2026 Rupesh Ghimire).

### Third-party licences

| Component | Licence | Note |
|---|---|---|
| `BAAI/bge-small-en-v1.5` | MIT, as stated on its Hugging Face model card ("The released models can be used for commercial purposes free of charge") | used by the domain gate and as the encoder of the selected model |
| `LiquidAI/LFM2.5-Encoder-230M` | LFM Open License v1.0, as stated on its model card | allows redistribution under conditions; commercial use is limited for entities above USD 10 million annual revenue. Only used in the experiments |
| SemEval-2016 Task 5 data (`DATA/HF/`) | not stated in this repository, and the Hugging Face dataset card has no licence entry | verify before redistribution |
| `model_checkpoint/bge_detr/checkpoint_9/model.pt` | fine-tuned from bge-small-en-v1.5 on the SemEval-2016 data | verify the data terms before redistributing the weights |
