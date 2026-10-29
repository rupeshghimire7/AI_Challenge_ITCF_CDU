# Laptop Review Analyser: Aspect-Based Sentiment Analysis

This project reads a laptop review and finds every opinion in it. For each sentence it returns the opinions as `{ENTITY#ATTRIBUTE, sentiment}` pairs, following SemEval-2016 Task 5. Each opinion also gets a 1 to 5 star rating and a confidence score, and all reviews of a product are summarised in one aspect table.

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

The app opens at http://localhost:8501. On the first start it downloads `BAAI/bge-small-en-v1.5` (about 130 MB), which the domain gates use. On my CPU the model and gates load in about 22 seconds. After that, a short review is analysed in under a second.

---

## Contents

1. [Overview](#overview)
2. [Key terms](#key-terms)
3. [Features](#features)
4. [Screenshots](#screenshots)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Output format](#output-format)
8. [How it works](#how-it-works)
9. [Data](#data)
10. [Models and results](#models-and-results)
11. [Training](#training)
12. [Configuration](#configuration)
13. [Repository structure](#repository-structure)
14. [Limitations](#limitations)
15. [Troubleshooting](#troubleshooting)
16. [Acknowledgements](#acknowledgements)
17. [License](#license)

---

## Overview

The challenge brief asks for every opinion in a review as an entity and attribute pair with a sentiment. When the text does not name the entity or the attribute, that part is `NULL`. My system follows this brief.

Some example outputs:

```text
"Great screen, but the battery dies fast."
    -> {BATTERY#OPERATION_PERFORMANCE, negative}, {DISPLAY#QUALITY, positive}
"Do not buy it!"                      -> {NULL#GENERAL, negative}        (nothing is named, so the entity is NULL)
"It is the worst laptop ever."        -> {LAPTOP#GENERAL, negative}
"Apple mac mini"                      -> {LAPTOP#NULL}, {COMPANY#NULL}   (named, but no opinion)
"The pizza was cold."                 -> {NULL#NULL}                     (not about a laptop, stopped by a domain gate)
```

For each product, the opinions from all of its reviews are combined into an aspect table. This is the table for two reviews of one laptop, "Great screen, but the battery dies fast." and "Do not buy it!". The second review is shown as `NULL#GENERAL` in the brief's format, but it is counted under `LAPTOP#GENERAL` in the table.

| category | mentions | negative | neutral | positive | reviews | rating | stars |
|---|---|---|---|---|---|---|---|
| DISPLAY#QUALITY | 1 | 0 | 0 | 1 | 1 | 3.4 | ★★★☆☆ |
| LAPTOP#GENERAL | 1 | 1 | 0 | 0 | 1 | 1.3 | ★☆☆☆☆ |
| BATTERY#OPERATION_PERFORMANCE | 1 | 1 | 0 | 0 | 1 | 1.2 | ★☆☆☆☆ |

## Key terms

| Term | Meaning |
|---|---|
| Entity (E) | what the opinion is about, for example `BATTERY`, `DISPLAY` or `LAPTOP` (22 in total) |
| Attribute (A) | which side of the entity, for example `QUALITY`, `PRICE` or `OPERATION_PERFORMANCE` (9 in total) |
| Category (E#A) | an entity and attribute pair, for example `BATTERY#OPERATION_PERFORMANCE` (22 × 9 = 198 in total) |
| Sentiment | negative, neutral or positive |
| Star | the model's sentiment score from 0 (very negative) to 4 (very positive); the rating shown is star + 1 |
| τ (tau) | the detection threshold: a category counts as an opinion when its score is τ or more |
| Joint F1 | the main metric: a prediction is correct only when the sentence, the category and the sentiment are all right |

## Features

- **Sentence-level analysis with context.** Each sentence is read together with the sentence before and after it, as the brief asks.
- **198 categories.** The model can output any of the 22 × 9 entity and attribute pairs from the SemEval-2016 laptop guidelines.
- **Star ratings.** Every opinion gets a probability for each star from 0 to 4. The system turns this into a sentiment, a 1 to 5 rating, a 90% star interval and a confidence label.
- **Seven domain gates.** Sentences that are not about a laptop (a pizza, a phone, a greeting) are stopped before the model and returned as `{NULL#NULL}` with the reason.
- **The brief's NULL rule.** The entity becomes `NULL` when the sentence does not name it.
- **Long text is never cut off.** Text longer than the model's input size is split into overlapping windows and the results are merged.
- **Streamlit app.** Analyse a single review or a CSV of many reviews, see one aspect table per product, download CSV and JSON, and open a Verbose panel that shows every decision.
- **Command line and Python API** that run the same pipeline as the app.

## Screenshots

| | |
|---|---|
| ![Analysis page on first load](docs/screenshots/01_home.png) | ![Breakdown of the example review](docs/screenshots/02_results_viewport.png) |
| **Analysis page.** Example reviews, the review box and the product name. | **Breakdown.** An overall verdict, then each sentence with its opinions, stars and confidence. |
| ![Output tab](docs/screenshots/04_output_tab.png) | ![Aspect table tab](docs/screenshots/05_aspect_table.png) |
| **Output tab.** The brief's format, with CSV and JSON downloads. | **Aspect table.** Mentions by sentiment and a rating for each category. |
| ![Off-domain example](docs/screenshots/10_off_domain_gates.png) | ![Batch CSV](docs/screenshots/14_batch_sample_results.png) |
| **Gates at work.** Both sentences are stopped, and the trace names the gate. | **Batch CSV.** The built-in sample of six reviews of two laptops. |

The Verbose panel shows, for one sentence, the preprocessing, every gate with its score, the raw model output and the post-processing checks:

<img src="docs/screenshots/23_trace_panel_mixed.png" alt="Verbose trace panel" width="360">

---

## Installation

### Requirements

- Python 3.12 (the version I developed and tested with).
- Git and [Git LFS](https://git-lfs.com), which stores the model weights.
- Enough disk space for PyTorch and the other packages, plus 142 MB for `model.pt`. The CUDA build of PyTorch is much larger than the CPU build.
- A GPU is optional. Everything runs on a CPU. With `--device auto`, a GPU is used only if it actually works.

### 1. Clone the repository with the weights

```bash
git lfs install                       # once per machine
git clone https://github.com/rupeshghimire7/AI_Challenge_ITCF_CDU.git
cd AI_Challenge_ITCF_CDU
```

If you cloned before installing Git LFS, the weights are only small pointer files. Download them with:

```bash
git lfs install
git lfs pull                                                     # every LFS file
git lfs pull --include="model_checkpoint/bge_detr/checkpoint_9/model.pt"   # only the model
```

Check that the weights downloaded correctly:

```bash
git lfs ls-files                      # "*" after the id means the file is present, "-" means only a pointer
ls -l model_checkpoint/bge_detr/checkpoint_9/model.pt   # should be 141,818,515 bytes
```

A `model.pt` of about 130 bytes is only a pointer. In that case the app stops with "has no weights: run `git lfs pull`".

To clone quickly without the weights and download them later:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/rupeshghimire7/AI_Challenge_ITCF_CDU.git
cd AI_Challenge_ITCF_CDU && git lfs pull
```

### 2. Create the Python environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For a smaller, CPU-only PyTorch, use this install command instead:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

`requirements.txt` lists the version of each package I tested with.

### 3. Where the weights are loaded from

`src/config.py` points to `model_checkpoint/bge_detr/checkpoint_9/`. That folder holds everything the model needs, so nothing else is downloaded for it:

- `model.pt`: the weights
- `config.json`: τ, the 198 categories and the training settings
- `encoder_config.json` and `tokenizer.json`

If `model.pt` is missing and `result/notebook_runs/absa_experiments.zip` exists, the weights are unpacked from that zip instead.

The domain gates download `BAAI/bge-small-en-v1.5` from Hugging Face on the first run and use the local cache after that.

---

## Usage

There are four ways to use the system. All of them run the same pipeline.

### Streamlit app

Run this from the repository root, so that `.streamlit/config.toml` is found:

```bash
streamlit run src/ui_streamlit.py
```

The app has four pages in the top navigation bar:

| Page | What you do | What you get |
|---|---|---|
| **Analysis** (default) | type a review or pick an example, name the product, press "Analyse review" | an overall verdict; each sentence with its opinions, stars and confidence; the brief's format with CSV and JSON downloads; the aspect table |
| **Batch CSV** | upload a CSV (or use the sample) and pick the text and product columns | the same views for every review, with a product selector |
| **About System** | read | the four main steps, the seven gates, the entity rule, a legend and a detailed flow chart |
| **Model Experiments** | read | the architectures, the results of every model, the custom model and the system ablation |

The **Verbose** switch opens a trace panel on the right. For each sentence it shows the cleaned text and what was fixed, the nuances and mentions found, every gate with its score and threshold, the raw model categories and scores, any merging, and the entity checks behind each opinion. I added it so that anyone grading the system can see why each answer was given without reading the code.

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
| `--text` | a review; repeat it for several reviews |
| `--file` | a text file with one review per line |
| `--csv` | a CSV file with one review per row |
| `--text-column` | the CSV column that holds the review (default `text`) |
| `--product-column` | the CSV column with the product name or id (one aspect table per product) |
| `--product` | the product name for `--text` or `--file`; with `--product-column`, keep only this product |
| `--checkpoint` | use a different bge_detr checkpoint folder |
| `--device` | `auto`, `cpu` or `cuda` |
| `--out` | folder to save `opinions.csv` and `aspect_table.csv` |

Example output of the first command with `--product "Demo laptop"`:

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

### Model only: `models/infer.py`

This script runs only the model with its own τ. It skips the gates and the post-processing rules, and it works with any experiment checkpoint.

```bash
python models/infer.py --checkpoint model_checkpoint/bge_detr/checkpoint_9 --text "Great screen, but the battery dies fast."
python models/infer.py --checkpoint model_checkpoint/bge_detr/checkpoint_9 --file reviews.txt
python models/infer.py --checkpoint model_checkpoint/bge_detr/checkpoint_9 --test     # score DATA/final/test_final.csv
python models/infer.py --table --out result                                          # compare every run in result/
```

Options: `--checkpoint`, `--text`, `--file`, `--test`, `--table`, `--out` and `--device`.

Example output:

```text
Great screen, but the battery dies fast.
  -> {BATTERY#OPERATION_PERFORMANCE, negative}, {DISPLAY#QUALITY, positive}
     BATTERY#OPERATION_PERFORMANCE        star 0 (expected 0.22, 90% interval 0-1)  negative 100%  detection 0.99
     DISPLAY#QUALITY                      star 3 (expected 2.38, 90% interval 0-4)  positive 61% (uncertain)  detection 0.96
```

### Data scripts and checks

Each script prints its results and saves the same text as a report.

| Command | What it does | Report |
|---|---|---|
| `python scripts/prepare_data.py` | reads the SemEval-2016 XML, merges the subtasks and cleans the text | `DATA/final/reports/prepare_data.txt` |
| `python models/data.py` | writes `train_final.csv` and `test_final.csv` | `DATA/final/reports/final_data.txt` |
| `python scripts/eda.py` | exploratory data analysis with figures | `DATA/final/reports/eda.txt` |
| `python scripts/preprocess.py` | checks the preprocessing and the sentence splitter | `DATA/final/reports/preprocess.txt` |
| `python -m src.domain_gate` | fits the gate thresholds | printed only |
| `python scripts/evaluate_system.py` | evaluates the whole system, with an ablation and edge cases | `DATA/final/reports/evaluate_system.txt` |

---

## Output format

Every sentence gets at least one tuple.

### `opinions.csv`

Saved with `--out` or the Download CSV button. One row per tuple:

| Column | Meaning |
|---|---|
| `product`, `review_id`, `sentence_index`, `sentence` | where the tuple comes from |
| `aspect` | the final E#A in the brief's format: `NULL#A` when the entity is not named, `E#NULL` or `NULL#NULL` when there is no opinion |
| `category` | the system's E#A (used by the aspect table and the metrics) |
| `model_category` | what the model predicted before the entity rule |
| `sentiment` | negative, neutral or positive; empty when there is no opinion |
| `star` | 0 to 4, kept inside the sentiment's range; the rating shown is star + 1 |
| `expected_star` | the average star, E = Σ k · p(k) |
| `star_low`, `star_high` | the 90% star interval |
| `confidence` | the probability of the chosen sentiment |
| `certain` | True when the 90% interval falls inside one sentiment |
| `score` | the model's detection score for the category |
| `source` | `model`, `second threshold 0.80`, or `merged with the previous / next sentence` |
| `note` | why there is no opinion (the gate's reason, or "no opinion") |

### `output.json`

Saved with the Download JSON button. For each sentence it holds `product`, `review`, `sentence` and `tuples`. Each tuple has `aspect`, `category`, `model_category`, `sentiment`, `confidence`, `source`, `note`, and for opinions also `rating` and `stars`.

### `aspect_table.csv`

One row per product and category:

| Column | Meaning |
|---|---|
| `mentions` | number of opinions |
| `negative`, `neutral`, `positive` | opinions by sentiment |
| `reviews` | number of reviews that mention the category |
| `rating` | 1 + the average expected star, where each review counts once |
| `stars` | the rating as stars |

---

## How it works

![System flow](docs/diagrams/system_flow.png)

A review goes through seven stages. All of them are in `src/`. The Streamlit app, the command line and the evaluation script all use the same `Pipeline` class from `src/main.py`.

| Stage | Module | What happens | Output |
|---|---|---|---|
| 1. Preprocess | `src/preprocess.py` | Cleans the text (HTML, Unicode, curly quotes, invisible characters, "soooo" becomes "sooo"). Splits it into sentences without breaking after abbreviations. Tags nuances (negation, contrast, comparison, ...), the entities named, off-topic objects and filler words. | one record per sentence, with the sentence before and after |
| 2. Domain gates | `src/domain_gate.py` | Seven checks decide whether the sentence is about a laptop (see below). | each gate's result and the mentions found |
| 3. Model | `src/model_inference.py` | bge_detr (epoch 9) reads `[CLS] previous [SEP] sentence [SEP] next [SEP]`. | a score for each of the 198 categories and star probabilities for each |
| 4. Decode | `src/post_process.py` | Keeps every category scoring 0.3 or more, with its star summary. | candidate opinions, highest score first |
| 5. Post-process | `src/post_process.py` | Applies τ = 0.85, a second threshold of 0.80, the entity rule and the no-opinion tuples. | tuples |
| 6. Merge | `src/main.py` | A `{NULL#NULL}` sentence that no hard gate stopped is joined with a neighbour and checked again. | tuples |
| 7. Output | `src/post_process.py`, `src/ui_streamlit.py` | Rating, stars, confidence label, the brief's format and the aspect table. | screen, CSV, JSON |

### The domain gates

![Domain gates](docs/diagrams/gates.png)

The model was trained only on laptop reviews, so it treats any opinion as a laptop opinion. The gates run before the model, cheapest first, to stop sentences that are not about laptops.

| # | Gate | Type | Passes when | Example |
|---|---|---|---|---|
| 1 | tokens | hard | the text has at least one word piece with a letter | stops "!!!", "12345" |
| 2 | filler | hard | at least one word is not a greeting, a thank-you or a stop word | stops "Hello.", "Thanks everyone!" |
| 3 | off-topic | hard | no other object (phone, pizza, hotel, ...) is the subject, or what is loved or bought; skipped when a laptop is named or the sentence is a comparison | stops "The pizza was cold." |
| 4 | laptop named | strong pass | a laptop, computer, MacBook, brand, ... is named, so the sentence goes straight to the model | "It is the worst laptop ever." |
| 5 | aspect terms | evidence | one of the 22 entities is mentioned, found by word list or by a word with similarity ≥ 0.74 to an entity name | "build" matches HARDWARE at 0.85 |
| 6 | domain | evidence | the sentence is closer to the 22 laptop labels than to 22 other topics (margin ≥ −0.11) | |
| 7 | category | evidence | the best similarity to a laptop label is ≥ 0.43 | |

A sentence goes to the model when gates 1 to 3 pass and then either gate 4 passes, or gate 6 passes together with gate 5 or gate 7. Hard gates stop a sentence on their own; evidence gates only count together.

The similarities use the `[CLS]` vector of the pretrained `bge-small-en-v1.5`. The three thresholds were fitted on laptop training sentences only, with `python -m src.domain_gate`.

### The model: bge_detr

![bge_detr](docs/diagrams/arch_detr.png)

`bge_detr` is a bge-small-en-v1.5 encoder (12 layers, hidden size 384) with two input changes and a set prediction head inspired by DETR:

- **Review context.** The sentence before and after (up to 32 tokens each) are added as context, marked with token type 1.
- **Negation scope.** A negation word and up to five words after it get an extra learned vector added to their embeddings, so the model can tell "good" from "not good".
- **Set head.** Sixteen learned queries read the sentence through a two-layer transformer decoder. Each query predicts an entity (22 or "no opinion"), an attribute (9) and a star (0 to 4). A category's score is the highest `p(E) · p(A)` over all queries, and its star probabilities come from that query.

**Long input.** A sentence longer than 54 tokens is cut into 54-token windows that overlap by 13 tokens. Up to four windows are read by the decoder together. Longer text is split into groups of four windows, and the groups are merged by keeping each category's highest score. For example, a 243-token run-on text became 6 windows in 2 groups. `SOFTWARE#MISCELLANEOUS` scored 0.0369 in the first group and 0.9342 in the second, so it kept 0.9342.

![Window merging](docs/diagrams/window_merging.png)

### Post-processing

![Post-processing](docs/diagrams/post_processing.png)

**From star probabilities to a sentiment.** `star_summary` in `src/post_process.py` turns the five star probabilities p(0) to p(4) into these fields:

| Field | Rule |
|---|---|
| expected star | E = Σ k · p(k), for k = 0 to 4 |
| 90% interval | the narrowest run of stars that holds at least 90% of the probability |
| sentiment ranges | negative = p(0) + p(1), neutral = p(2), positive = p(3) + p(4) |
| certain | the 90% interval falls inside one sentiment range |
| sentiment | if certain, that range; otherwise the largest range. An interval that spans negative to positive with no range above 0.5 is neutral (a mixed opinion) |
| confidence | the probability of the chosen sentiment |
| star | E rounded and kept inside the chosen sentiment's range; the rating is star + 1 |
| confidence label | high: confidence ≥ 0.8 and certain; medium: ≥ 0.6; otherwise low |

**The brief's rules.** `finalise` then applies these rules:

| Rule | What it does | Example |
|---|---|---|
| threshold τ = 0.85 | categories scoring τ or more are the opinions (τ was fitted on validation during training) | |
| second threshold 0.80 | if a sentence mentions a part but nothing reaches τ, categories scoring 0.80 or more are kept | "The hinge broke after two weeks." → `{LAPTOP#QUALITY, negative}` |
| entity rule | the entity is kept when it is named, or when the sentence mentions some part and one more check agrees (named in a neighbouring sentence, the whole laptop, or among the three closest labels); otherwise it becomes NULL | "Do not buy it!" → `{NULL#GENERAL, negative}` |
| no opinion | each mentioned entity with attribute NULL, or `{NULL#NULL}` | "Apple mac mini" → `{LAPTOP#NULL}, {COMPANY#NULL}` |
| merge | a `{NULL#NULL}` sentence that no hard gate stopped is joined with the sentence before and after; any new opinions found become its own | "Called tech support about the screen. Gave me the runaround ..." → the second sentence gets `{NULL#OPERATION_PERFORMANCE, negative}` |
| part re-map | off by default (`REMAP = False`), because it costs about one joint F1 point | |

**Worked example.** For "Great screen, but the battery dies fast.":

1. `DISPLAY#QUALITY` scores 0.9630, which is above τ, so it is an opinion.
2. Its star probabilities are 0.2835, 0.1038, 0.0055, 0.1645, 0.4428, so the expected star is 2.379.
3. The 90% interval is [0, 4], so the prediction is not certain.
4. The positive range holds 0.607, which is above 0.5, so the sentiment is positive with confidence 0.607 (medium).
5. The star rounds to 2 and is moved to 3 to stay inside the positive range, so the rating is 4.
6. "screen" is in the DISPLAY word list, so the entity is kept.

Result: `{DISPLAY#QUALITY, positive}`, 4 of 5 stars, 61% confident.

---

## Data

I used the SemEval-2016 Task 5 English laptop data (subtask 1 sentences and subtask 2 review labels), stored in `DATA/HF/`.

`scripts/prepare_data.py` prepares it:

- decodes the double-escaped HTML in the test XML and normalises the text
- removes 19 exact duplicates and 6 training sentences that also appear in the test set
- flags sentences with conflicting polarities

I had also added the Laptop-ACOS data as extra training data, but I removed it because its labels conflicted. Validation is 15% of the training reviews, chosen by a hash of the review id, so a review is never split between training and validation.

| Split | Reviews | Sentences | Opinions | Sentences without an opinion | Stars 0 / 2 / 4 |
|---|---|---|---|---|---|
| train | 374 | 2,109 | 2,452 | 390 (18.5%) | 928 / 150 / 1,374 |
| validation | 75 | 366 | 434 | 69 (18.9%) | 153 / 38 / 243 |
| test (official gold) | 80 | 808 | 801 | 235 (29.1%) | 274 / 46 / 481 |

Source: `DATA/final/reports/final_data.txt`. Only 80 of the 198 categories appear in training, and stars 1 and 3 never appear in the gold data.

---

## Models and results

### Experiments

I trained two encoders, bge-small-en-v1.5 and LFM2.5-Encoder-230M, each with three types of head, in `models/notebooks/ABSA_experiments_base_tweak.ipynb`:

| Variant | What it is |
|---|---|
| base | the encoder's sentence vector with a linear detector and a linear star head |
| tweaked | review context, negation scope, category attention, a factorised E × A detector, an opinion gate, asymmetric and focal losses, long-input training and layer-wise learning rate decay |
| detr | the tweaked encoder with the set head described above |

Every run used 10 epochs, batch size 32, seed 2026, a maximum length of 56 and no early stopping. After each epoch, τ was fitted on validation, and the best epoch was the one with the highest validation joint F1. The test set was used only for reporting.

![Architecture comparison](docs/diagrams/arch_comparison.png)

### Test results

Each run at its best validation epoch, in %:

| Run | Best epoch | τ | Val joint F1 | Test detection F1 | Test joint P / R / F1 | Sentiment acc / macro-F1 | Empty-sentence acc | Star MAE |
|---|---|---|---|---|---|---|---|---|
| **bge_detr** (selected) | 9 | 0.85 | 61.35 | 54.0 | 47.8 / 49.2 / **48.5** | 88.8 / 71.1 | 46.4 | 0.36 |
| lfm_base | 10 | 0.40 | 59.71 | 50.6 | 54.9 / 39.7 / 46.1 | 84.0 / 62.9 | 69.4 | 0.52 |
| bge_tweaked | 9 | 0.60 | 44.99 | 40.5 | 40.5 / 33.6 / 36.7 | 86.0 / 72.2 | 83.0 | 0.37 |
| bge_base | 9 | 0.20 | 46.14 | 42.1 | 37.3 / 34.8 / 36.0 | 82.8 / 65.8 | 52.3 | 0.53 |
| lfm_tweaked | - | - | - | ran out of GPU memory, no result | | | | |
| lfm_detr | - | - | - | ran out of GPU memory, no result | | | | |

Sources: `model_checkpoint/bge_detr/checkpoint_9/metrics.csv` for bge_detr, and the notebook's `summary.csv` for the others.

> **Why 48.5 and not 47.7?** The notebook summary, `models/notebooks/ABSA_model_selection_report.md` and the app's Model Experiments page show 47.7 for bge_detr. That number comes from a mixed row: two training sessions of bge_detr wrote into the same run folder, and the row takes its validation number from one session and its test number from the other. The checkpoint in this repository is the epoch 9 model of the later session (τ = 0.85, validation 61.35). Its own `metrics.csv` and training log both give a test joint F1 of 48.5. Re-scoring it on a CPU with `python models/infer.py --checkpoint model_checkpoint/bge_detr/checkpoint_9 --test` gives 48.6. The small difference most likely comes from training scoring it in bfloat16 on a GPU.

### Extended test metrics

Same epochs as above. The bge_detr column matches `checkpoint_9/extended_metrics.json`. Best in bold.

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

### Is bge_detr overfitting?

![Learning curve](docs/diagrams/chart_learning_curve_bge_detr.png)

Joint F1 (%) per epoch:

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

- Train joint F1 keeps rising while test stays near 48, so the model memorises part of the training data.
- Test does not drop, though. From epoch 7 to 10 it stays within 0.7 points, which is a plateau, not a collapse.
- Epoch 9 was chosen by the validation rule, not by looking at test.
- lfm_base has a gap of the same size (82.6 train, 46.1 test at epoch 10).
- Part of the drop from validation to test comes from the data: 29.1% of test sentences have no opinion, against 18.9% in validation, and these sentences are where the model makes most of its mistakes.

### The custom 3 head transformer classifier

`models/notebooks/ABSA_colab.ipynb` trains a separate model: bge-small with mean pooling, no context, and separate category, attribute and pair heads (34.12M parameters). On its own held-out fold it reaches 57.4 triple F1 (P 60.7, R 54.5; 50.7 with untuned thresholds).

This number cannot be compared directly with the results above, for two reasons:

- all 390 of its test sentences contain an opinion, so it is never penalised for predicting opinions on empty sentences
- six thresholds were tuned on 217 validation sentences

On the metrics that are defined the same way in both setups, bge_detr wins five of seven:

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

### Why I chose bge_detr

- It has the best test joint F1 (48.5), and the best joint F1 for each sentiment.
- The set head is what makes the difference. bge_tweaked has the same encoder changes, and adding the set head raises test joint F1 from 36.7 to 48.5 and detection AUC from 0.895 to 0.984.
- It does not get worse with more epochs, and its overfitting gap is the same as the other model I could check.
- It is small: 35,432,613 parameters, against 229,693,184 in the LFM encoder alone.

The full reasoning is in [`models/notebooks/ABSA_model_selection_report.md`](models/notebooks/ABSA_model_selection_report.md). The experiments are in [`ABSA_experiments_base_tweak.ipynb`](models/notebooks/ABSA_experiments_base_tweak.ipynb) and [`ABSA_colab.ipynb`](models/notebooks/ABSA_colab.ipynb).

### The whole system

`python scripts/evaluate_system.py` runs the full pipeline (gates, model and rules) on the gold sentences in review order. Joint F1 (%) with the system's E#A:

| Setting | Validation | Test |
|---|---|---|
| Model only, τ = 0.85, with the entity rule | 58.9 | 48.4 |
| + second threshold 0.80 | 59.3 | 48.4 |
| + merge of `{NULL#NULL}` sentences (the default) | **59.8** | **48.2** |
| Default with the gates turned off | 60.0 | 48.4 |
| Default + part re-map (off by default) | 58.5 | 47.3 |

- The rules mainly change the form of the answer, not the score. On test, the default is within 0.2 points of the model alone, and on validation it is 0.9 points better.
- The gates stop 15 of the 808 laptop test sentences. The model had predicted an opinion for 7 of them.
- When I fitted the gates, they kept 98.6% of laptop opinion sentences and rejected 59.5% of restaurant sentences. This check needs the SemEval restaurant XML, which is not included in this repository.

---

## Training

### Retrain in Colab (how the results above were made)

1. Open `models/notebooks/ABSA_experiments_base_tweak.ipynb` in Google Colab with a GPU. A T4 is enough; the LFM runs are the slowest.
2. Upload `DATA/final/train_final.csv` and `DATA/final/test_final.csv` when the data cell asks for them.
3. Choose the runs in `CFG.runs` (`bge_base`, `bge_tweaked`, `bge_detr`, `lfm_base`, `lfm_tweaked`, `lfm_detr`) and run all cells. For the LFM tweaked and detr runs, set `CFG.grad_ckpt = True` and restart the runtime first, otherwise they run out of GPU memory.
4. The last cell packs `absa_experiments.zip` with every metric file and the best epoch's weights for each run.
5. To use a new bge_detr checkpoint, copy its `checkpoint_<epoch>/` folder into `model_checkpoint/bge_detr/` and point `CHECKPOINT` in `src/config.py` to it.

The custom 3 head transformer classifier is trained in `models/notebooks/ABSA_colab.ipynb`, which saves `absa_results.zip`.

### Train locally

```bash
python models/train.py --model bge --variant detr --epochs 10        # --model bge | lfm, --variant base | tweaked | detr
python models/train.py --model lfm --variant tweaked --grad-ckpt     # uses less GPU memory
```

Other options: `--batch-size`, `--lr`, `--device`, `--out`, `--seed` and `--limit` (train on only the first N sentences, for a quick test). The default is 5 epochs; the results above used 10.

### Training output

```text
result/
  <model>_<variant>/
    checkpoint_<epoch>/   model.pt, config.json (τ, settings, categories), encoder_config.json, tokenizer.json,
                          metrics.csv (train / validation / test), test_predictions.csv, extended_metrics.json
    metrics.csv           every epoch          extended_metrics.csv
    log.txt               training log, including the printed architecture
    best_epoch_report/    per-category, per-entity and per-attribute tables, τ sweeps, confusion matrices
  summary.csv, summary_extended.csv, compare_runs.png, data_report.txt
```

---

## Configuration

| Setting | Value | Where to change it |
|---|---|---|
| checkpoint folder | `model_checkpoint/bge_detr/checkpoint_9` | `src/config.py` (`CHECKPOINT`) |
| detection threshold τ | 0.85 | the checkpoint's `config.json` (fitted on validation) |
| second threshold | 0.80 | `src/post_process.py` (`TAU_LOW`) |
| decode floor | 0.3 | `src/main.py` (`FLOOR`) |
| part re-map | off | `src/post_process.py` (`REMAP`) |
| confidence labels | high ≥ 0.8 and certain; medium ≥ 0.6 | `src/post_process.py` (`HIGH`, `MEDIUM`) |
| star interval | 90% | `src/post_process.py` (`LEVEL`) |
| gate thresholds | margin −0.11, category 0.43, word 0.74 | `src/domain_gate.py` (`MARGIN`, `CATEGORY_FLOOR`, `WORD_FLOOR`) |
| input window | 56 tokens (54 for the sentence + [CLS] + [SEP]), 32 context tokens per side, step 41 | the checkpoint's `config.json` (`max_len`, `ctx_len`), `src/model_inference.py` |
| batching | 64 windows per encoder pass, 4 windows per decoder group | `src/model_inference.py` (`CHUNK`, `GROUP`) |
| app theme | colours, DM Sans / Fraunces / DM Mono fonts, 8 px corner radius | `.streamlit/config.toml` and the CSS in `src/ui_streamlit.py` |

---

## Repository structure

| Path | Contents |
|---|---|
| `src/` | **The system, one task per file** |
| `src/ui_streamlit.py` | the Streamlit app: Analysis, Batch CSV, About System, Model Experiments and the Verbose panel |
| `src/main.py` | the `Pipeline` class (preprocess → gates → model → post-process → merge) and the command line |
| `src/preprocess.py`, `src/nuance_checker.py` | text cleaning, sentence splitting, nuance patterns |
| `src/aspect_terms.py`, `src/domain_gate.py` | entity word lists; the seven gates and mention detection |
| `src/model_inference.py` | loads the bge_detr checkpoint and runs it: input windows, set head, `predict` |
| `src/post_process.py` | star summary, rules, rating, confidence, the brief's format, aspect table |
| `src/config.py`, `src/data_maker.py` | paths and the 22 × 9 label space; loading `DATA/final` |
| `models/` | **Model training and experiments** |
| `models/bge.py`, `models/lfm.py` | the two encoders and their tweaks |
| `models/heads.py`, `models/set_detector.py` | the base and tweaked heads; the DETR-style set head |
| `models/train.py`, `models/infer.py`, `models/data.py` | training, checkpoint inference, data loading and metrics |
| `models/custom_*.py` | the custom 3 head transformer classifier with its own training and inference |
| `models/notebooks/` | the experiments notebook, the Colab notebook and the model selection report |
| `scripts/` | data preparation and checks: `prepare_data.py`, `eda.py`, `preprocess.py`, `evaluate_system.py` |
| `DATA/HF/` | the SemEval-2016 laptop XML files |
| `DATA/final/` | the prepared CSV files and the script reports |
| `model_checkpoint/bge_detr/checkpoint_9/` | the selected model: `model.pt` (Git LFS), config with τ, tokenizer, metrics and test predictions |
| `docs/` | the screenshots and diagrams used in this README |
| `requirements.txt` | Python packages and the versions I tested with |
| `LICENSE` | MIT License |

---

## Limitations

- **Sentences without an opinion.** The model often predicts an opinion for sentences that have none. Only 46.4% of empty test sentences are left empty. The gates stop sentences that are not about laptops, but a laptop sentence with no opinion can still get one.
- **Neutral is weak.** Joint F1 for neutral is only 0.140. Neutral is about 6% of the gold opinions, and stars 1 and 3 never appear in training.
- **Rare categories.** Only 80 of the 198 categories appear in training, so the model has never seen the other 118.
- **One seed, one split.** A difference of one or two joint F1 points between models is within noise.
- **Two runs missing.** lfm_tweaked and lfm_detr ran out of GPU memory, so they were not compared.
- **Known mistakes.** "The fan is noisy." gives `{LAPTOP#QUALITY, negative}` instead of a FANS_COOLING category. "This laptop is way better than my old phone." gives only `{LAPTOP#NULL}`.
- **English laptop reviews only.**

---

## Troubleshooting

**The app stops with "has no weights: run `git lfs pull`".**
`model.pt` is only an LFS pointer. Run `git lfs install` and `git lfs pull`, then check that the file is 141,818,515 bytes.

**The first start is slow, or fails without internet.**
The domain gates download `BAAI/bge-small-en-v1.5` once. Start the app once with internet access; later starts use the Hugging Face cache.

**The theme or fonts look different.**
Run `streamlit run` from the repository root so that `.streamlit/config.toml` is found. Without it, Streamlit's default theme is used, but the layout still works.

**`git status` shows `model.pt` as modified even though I did not change it.**
Run `git lfs install` inside the clone, then run `git status` again.

**pip downloads a very large PyTorch.**
Use the CPU-only build: `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu`.

**CUDA errors on an older GPU.**
Use `--device cpu`. The default `--device auto` already falls back to the CPU when the GPU test fails.

**The app and the notebook say 47.7, but this README says 48.5.**
See [Why 48.5 and not 47.7?](#test-results) under Test results.

---

## Acknowledgements

- **Data:** M. Pontiki et al., "SemEval-2016 Task 5: Aspect Based Sentiment Analysis", Proceedings of SemEval-2016. The challenge brief points to the Hugging Face copy `alexcadillon/SemEval2016Task5`.
- **Encoder:** `BAAI/bge-small-en-v1.5` (FlagEmbedding, BAAI).
- **Encoder:** `LiquidAI/LFM2.5-Encoder-230M` (Liquid AI).
- **Set prediction:** N. Carion et al., "End-to-End Object Detection with Transformers" (DETR), 2020. `models/set_detector.py` adapts this idea to opinions.

Built for the CDU IT Code Fair Sydney 2026 AI Challenge.

---

## License

The code and documentation in this repository are released under the [MIT License](LICENSE). Copyright (c) 2026 Rupesh Ghimire.

The MIT License covers only my own code and documentation. The models and data below have their own terms:

| Component | License | Notes |
|---|---|---|
| `BAAI/bge-small-en-v1.5` | MIT (from its Hugging Face model card) | used by the domain gates and as the encoder of the selected model |
| `LiquidAI/LFM2.5-Encoder-230M` | LFM Open License v1.0 (from its model card) | redistribution is allowed under conditions; commercial use is limited for companies with more than USD 10 million annual revenue. Used only in the experiments |
| SemEval-2016 Task 5 data (`DATA/HF/`) | not stated; the Hugging Face dataset card has no license entry | check the terms before redistributing |
| `model_checkpoint/bge_detr/checkpoint_9/model.pt` | fine-tuned from bge-small-en-v1.5 on the SemEval-2016 data | check the data terms before redistributing the weights |
