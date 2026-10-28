# Laptop Review Analyser: Aspect-Based Sentiment Analysis

Give the system a laptop review and it returns, for each sentence, which part of the laptop the sentence is about and how the reviewer feels about it. Each opinion is an `{ENTITY#ATTRIBUTE, polarity}` tuple in the SemEval-2016 Task 5 format, with a 1-5 star rating, a confidence and a face (🙂 / 😐 / 🙁).

```
"The battery easily lasts 9 hours and the screen is gorgeous."
  -> {BATTERY#OPERATION_PERFORMANCE, positive} ★★★★★ 🙂   {DISPLAY#QUALITY, positive} ★★★★★ 🙂
"Do not buy it!"                          -> {NULL#GENERAL, negative} ★☆☆☆☆ 🙁   (entity not named: NULL)
"The restaurant is good and has good food" -> {NULL#NULL}   (rejected by the domain gates: not about a laptop)
```

---

## Contents

1. [How it works](#how-it-works)
2. [Quick start](#quick-start)
3. [Usage](#usage)
4. [Output format](#output-format)
5. [Domain gates](#domain-gates)
6. [Results](#results)
7. [Project structure](#project-structure)
8. [Limitations](#limitations)

---

## How it works

```
 review text
     │
     ▼
 1  PREPROCESS     src/preprocess.py      HTML entities and tags, Unicode, quotes, invisible characters, elongation;
                                          split into sentences; nuance badges (negation, contrast, sarcasm, ...);
                                          aspect cues (which of the 22 laptop entities each sentence names)
     │
     ▼
 2  DOMAIN GATES   src/domain_gate.py     tokens → filler → off-topic → laptop named → domain and (aspect terms or
                                          category). Only a sentence about the laptop goes on. The rest get {NULL#NULL}.
     │
     ▼
 3  MODEL          src/model_inference.py bge_detr, epoch 9: bge-small-en-v1.5 encoder with review context and negation
                                          scopes, and a DETR-style set head (16 queries → entity, attribute, star 0-4)
     │
     ▼
 4  POST-PROCESS   src/post_process.py    threshold τ, 90% star interval → sentiment, star, confidence; NULL entity when
                                          the sentence does not name it; rating ★, face, confidence label;
                                          brief-format tuples; aspect-sentiment table per product
```

The model was fine-tuned on laptop reviews only, so it reads any opinion as a laptop opinion: without the gates, "restaurant is good and has good food" becomes `{LAPTOP#GENERAL, positive}`. The gates run before the model, so a sentence that is not about a laptop never reaches it.

---

## Quick start

Requires Python 3.12.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # CPU torch: --extra-index-url https://download.pytorch.org/whl/cpu
streamlit run src/ui_streamlit.py      # opens http://localhost:8501
```

On the first run:

- **Checkpoint.** The selected checkpoint is in `model_checkpoint/bge_detr/checkpoint_9/`, stored with Git LFS. Run `git lfs pull` to fetch `model.pt` (about 140 MB). If `model.pt` is still only an LFS pointer and `result/notebook_runs/absa_experiments.zip` exists, the weights are unpacked from that zip instead.
- **Gate encoder.** The domain gate downloads `BAAI/bge-small-en-v1.5` from Hugging Face once (about 130 MB).

After that the model and gates load in about 10 s, and a review of a few sentences takes well under a second on a CPU.

---

## Usage

### Streamlit app

```bash
streamlit run src/ui_streamlit.py
```

| Tab | What you do | What you get |
|---|---|---|
| **Analyse a review** | Type a review or click an example: mixed, implicit, off-domain, greeting + spec, sarcasm / negation | KPI tiles; an overall verdict (face + stars); per-sentence cards with the gate trace (hover a chip for its score), nuance badges and each opinion's face, stars, rating and confidence bar; the brief-format output; the aspect-sentiment table; CSV and JSON downloads |
| **Batch (CSV)** | Upload a CSV, then pick the text column and, optionally, a product column | The same views for every review; one aspect table per product |

The sidebar shows the model (epoch, τ, device), the gate thresholds and a legend.

### Command line

```bash
python -m src.main --text "The So called laptop Runs to Slow and I hate it!" --text "Do not buy it!"
python -m src.main --file reviews.txt --product "Apple Mac mini"       # one review per line
python -m src.main --csv reviews.csv --text-column review --product-column asin --out output/
```

`--out DIR` writes `DIR/opinions.csv` (one row per tuple) and `DIR/aspect_table.csv`. `--device auto|cpu|cuda` picks the device.

### In Python

```python
import pandas as pd
from src.main import Pipeline
from src.post_process import brief, stars, face

pipe = Pipeline()                                   # loads the model and the gates once
records = pipe.run(pd.DataFrame({"product": ["x"], "review_id": [0], "text": ["Great screen, awful battery."]}))
for r in records:
    print(r["sentence"], brief(r["tuples"]), [stars(t) + face(t) for t in r["tuples"]])
```

### Scripts (checks and reports)

| Command | What it does | Report |
|---|---|---|
| `python scripts/prepare_data.py` | Parse SemEval-2016 laptops (DATA/HF), merge subtasks, clean | `DATA/final/reports/prepare_data.txt` |
| `python scripts/eda.py` | Exploratory data analysis | `DATA/final/reports/eda.txt` |
| `python scripts/preprocess.py` | Check preprocessing on fixed cases and the corpus | `DATA/final/reports/preprocess.txt` |
| `python -m src.domain_gate` | Fit the gate thresholds; check them on laptop vs restaurant sentences | printed |
| `python scripts/evaluate_system.py` | End to end: joint F1 with and without gates, restaurant leakage, edge cases | `DATA/final/reports/evaluate_system.txt` |

---

## Output format

Each sentence gets at least one tuple.

| Case | Example | Output |
|---|---|---|
| Opinion, entity named | "The screen is gorgeous." | `{DISPLAY#QUALITY, positive}` ★★★★★ 🙂 |
| Opinion, entity not named | "Do not buy it!" | `{NULL#GENERAL, negative}` ★☆☆☆☆ 🙁 |
| No opinion, entity named | "Apple mac mini" | `{LAPTOP#NULL}, {COMPANY#NULL}` (no polarity, no stars) |
| No opinion, nothing named | "I bought it last week." | `{NULL#NULL}` |
| Rejected by a gate | "The pizza was cold." | `{NULL#NULL}` with the gate's reason |

**Rating and confidence.** The model predicts a distribution over stars 0-4.

| Field | How it is computed |
|---|---|
| Sentiment | The bracket (0-1 negative, 2 neutral, 3-4 positive) that holds the 90% star interval. If the interval spans brackets, the most probable bracket wins; a negative-to-positive spread with no bracket above 50% is neutral (a mixed opinion). |
| Rating (1-5) | Expected star, rounded and kept inside the sentiment's bracket, plus 1. Stars are shown only for opinions with a rating. |
| Face | 🙂 positive, 😐 neutral, 🙁 negative |
| Confidence | Probability of the sentiment's bracket. **high**: at least 0.8 and the 90% interval sits in one bracket; **medium**: at least 0.6; **low**: below 0.6. |
| Aspect table rating | Per product and category: 1 + the mean over reviews of each review's mean expected star, so one review counts once. |

---

## Domain gates

Gates run cheapest first. Every gate's result is kept and shown in the app.

| # | Gate | Kind | Passes when |
|---|---|---|---|
| 1 | tokens | hard | The bge tokenizer finds at least one word piece with a letter ("!!!", "123" fail) |
| 2 | filler | hard | The sentence is not only greetings or thanks ("Hello.", "Thanks everyone!") |
| 3 | off-topic | hard | No non-laptop object is the subject or the thing loved or bought ("The pizza was cold", "I love my new phone"). Comparisons and sentences that name the laptop are exempt. |
| 4 | laptop named | strong pass | The laptop itself is named (laptop, computer, MacBook, a brand, ...): accepted without the evidence gates |
| 5 | aspect terms | evidence | The sentence names one of the 22 entities (screen, battery, keyboard, ...) |
| 6 | domain | evidence, required | Similarity margin ≥ −0.11: best cosine to the 22 laptop labels minus best cosine to 22 other-topic labels (restaurant, phone, car, ...) |
| 7 | category | evidence | Best cosine to a laptop label ≥ 0.43, so the sentence falls inside one of our categories |

A sentence is about the laptop if gates 1-3 pass and either gate 4 passes or gate 6 passes together with gate 5 or 7. Similarity uses the pretrained `bge-small-en-v1.5` ([CLS], normalised). The pretrained encoder separates laptop from restaurant sentences better than the fine-tuned one (test AUC 0.957 vs 0.925).

Both thresholds are the 1st percentile on laptop **train** opinion sentences; nothing is fitted on test. The table below was measured on the SemEval-2016 test sets: laptops, plus restaurants SB1 as the out-of-domain set. The restaurant XML is not in the repo; to rerun the check, put `ABSA16_Restaurants_Train_SB1_v2.xml` and `EN_REST_SB1_TEST.xml.gold` in `DATA/HF/`.

| Rule (same thresholds) | Laptop opinion sentences kept | Restaurant sentences rejected |
|---|---|---|
| margin only | 99.5% | 54.4% |
| 2 of 3 votes (terms, margin, category) | 98.6% | 55.9% |
| **domain and (terms or category)** (used) | **98.6%** | **59.5%** |

The restaurant sentences that pass are mostly ones that could just as well describe a laptop purchase ("The staff was friendly", "Great service", "Highly recommended").

---

## Results

### Model selection

Four runs completed, all on the same 808-sentence SemEval-2016 laptop test set; 29% of its sentences have no opinion. Numbers are at each run's best validation epoch.

| Run | Test joint F1 | Sentiment acc | Sentiment macro-F1 | Star MAE | Star QWK | Detection AUC |
|---|---|---|---|---|---|---|
| **bge_detr (epoch 9, selected)** | **47.7** | **88.6** | **71.2** | **0.35** | **0.879** | **0.984** |
| lfm_base | 46.1 | 84.0 | 62.9 | 0.52 | 0.806 | 0.962 |
| bge_tweaked | 36.7 | 86.0 | 72.2 | 0.37 | 0.872 | 0.895 |
| bge_base | 36.0 | 82.8 | 65.8 | 0.53 | 0.797 | 0.932 |

Why `bge_detr`:

- It has the best test joint F1 and wins 10 of 15 extended metrics, including every per-sentiment F1.
- The set head is what lifts it. `bge_tweaked` and `bge_detr` share a backbone, and swapping the head moves joint F1 from 36.7 to 47.7.
- It is not overfit in a way that hurts: test F1 is flat from epoch 7 to 10 (48.4 to 48.9).
- It is small: 35.4M parameters, against 229.7M in the LFM encoder.
- Against the Colab custom model (57.4 triple-F1, measured on an easier test set with no empty sentences), it is equal or better on every metric that transfers: sentiment accuracy, star MAE, QWK and Spearman.

The full analysis is in these files:

- [`models/notebooks/ABSA_model_selection_report.md`](models/notebooks/ABSA_model_selection_report.md): decision, learning curves, overfitting, and the comparison with the Colab model
- [`models/notebooks/ABSA_experiments_base_tweak.ipynb`](models/notebooks/ABSA_experiments_base_tweak.ipynb): all six experiments (bge / LFM × base / tweaked / detr)
- [`models/notebooks/ABSA_colab.ipynb`](models/notebooks/ABSA_colab.ipynb): the Colab custom three-head model
- [`models/model_progress.md`](models/model_progress.md): model-phase progress log

### System (gates + model + post-processing)

From `python scripts/evaluate_system.py` (full output in `DATA/final/reports/evaluate_system.txt`):

| Check | Result |
|---|---|
| Laptop test joint F1 of (E#A, sentiment), 808 gold sentences, **gates on** | **48.4** (P 47.9, R 48.9) |
| The same with every sentence sent to the model (gates off) | 48.6 (P 47.8, R 49.3) |
| Laptop test sentences rejected by the gates | 15 / 808 (1.9%). In 7 of them the model saw an opinion; most are about store staff and do not name the laptop ("Gave me the runaround for an entire day ...") |
| Restaurant test sentences rejected by the gates | 59.5% (needs the restaurant XML in `DATA/HF/`, see [Domain gates](#domain-gates)) |
| Speed (CPU) | Model and gates load in about 10 s; 808 sentences take 46 s (about 0.06 s per sentence) |

The gates cost 0.2 joint-F1 points on laptop text. In exchange, the model no longer turns pizza, phones and waiters into laptop opinions. The score matches the notebook's epoch-9 test score (48.5), so serving reproduces training.

**Edge cases** (section 3 of the report):

| Input | Output | Handled by |
|---|---|---|
| `""`, `"   "` | no sentence (the app asks for text) | preprocessing |
| `"!!!"`, `"????"`, `"12345"` | `{NULL#NULL}` | gate 1 tokens |
| `"Hello."`, `"Thanks everyone!"`, `"ok"` | `{NULL#NULL}` | gate 2 filler |
| `"The pizza was cold."`, `"I love my new phone."`, `"The waiter was rude."`, `"My printer keeps jamming."` | `{NULL#NULL}` | gate 3 off-topic |
| `"restaurant is good and has good food"` | `{NULL#NULL}` | gate 3 off-topic (the similarity gate catches the unseen wordings) |
| `"Do not buy it!"`, `"It works."`, `"Great."` | `{NULL#GENERAL, negative}`, `{NULL#OPERATION_PERFORMANCE, positive}`, `{NULL#GENERAL, positive}` | implicit-entity rule |
| `"Apple mac mini"` | `{LAPTOP#NULL}, {COMPANY#NULL}` | no-opinion rule |
| `"Battery life is terrible but the screen is gorgeous."` | `{BATTERY#OPERATION_PERFORMANCE, negative}, {DISPLAY#QUALITY, positive}` | model |
| `"&lt;br&gt;The fan is noisy&lt;br&gt;"`, `"The   screen\u200b is   amazing 😍"`, `"SOOOOO SLOOOOW!!!!!"` | entities, spaces and elongation repaired before the model | preprocessing |
| 2,000-character inputs | windowed by the model, no crash | model inputs |

**Known misses**, all inside the model rather than the gates:

| Input | Output |
|---|---|
| `"Yeah right, \"premium\" build."` | positive (sarcasm missed when read alone; with review context it is negative) |
| `"This laptop is way better than my old phone."` | `{LAPTOP#NULL}` (comparative opinion missed) |
| `"I watch movies on it all the time and the speakers are loud."` | negative ("loud speakers" read as a complaint) |
| `"Great screen, awful battery."` | both negative (the second opinion's polarity spills over) |
| `"The fan is noisy"` | no opinion above τ |

---

## Project structure

| Path | Purpose |
|---|---|
| **`src/`** | **The system: one task per file** |
| `src/ui_streamlit.py` | Streamlit app: single review and CSV batch, gate trace, stars and faces, downloads |
| `src/main.py` | `Pipeline` (preprocess → gates → model → post-process) and the command line |
| `src/preprocess.py` | Normalisation, abbreviation-safe sentence splitting, per-sentence record with cues |
| `src/nuance_checker.py` | Patterns for formatting artefacts (repaired) and linguistic nuances (badges) |
| `src/aspect_terms.py` | Words naming each of the 22 entities, non-laptop objects, filler words |
| `src/domain_gate.py` | The 7 domain gates and the bge-small label similarity; `python -m src.domain_gate` fits and checks the thresholds |
| `src/model_inference.py` | The bge_detr checkpoint: input windows with context and negation, set head, `predict` |
| `src/post_process.py` | Decoding, star interval, sentiment, NULL rules, rating / stars / face / confidence, brief format, aspect table |
| `src/config.py` | Paths, the checkpoint location, the 22 entities × 9 attributes |
| `src/data_maker.py` | Loads `DATA/final` CSVs as opinion rows or sentence records |
| **`scripts/`** | **Data preparation and checks; each prints a report and saves it to `DATA/final/reports/`** |
| `scripts/prepare_data.py` | SemEval-2016 XML → `DATA/final/{train,test}_*.csv` |
| `scripts/eda.py` | Exploratory data analysis, figures and tables |
| `scripts/preprocess.py` | Preprocessing checks |
| `scripts/evaluate_system.py` | End-to-end system evaluation and edge cases |
| **`models/`** | **Model phase: training code and experiment notebooks** |
| `models/notebooks/` | Experiment notebooks and the model selection report |
| `models/bge.py`, `models/lfm.py` | The two encoders (bge-small-en-v1.5, LFM2.5-Encoder-230M) |
| `models/heads.py`, `models/set_detector.py` | Base / tweaked heads and the DETR-style set head |
| `models/train.py`, `models/infer.py`, `models/data.py` | Training, checkpoint inference, data to tensors |
| `models/custom_*.py` | The Colab three-head model: model, training, inference |
| **`model_checkpoint/`** | The selected checkpoint `bge_detr/checkpoint_9/` (Git LFS): weights, config with τ and categories, tokenizer, test metrics |
| **`DATA/`** | `HF/` SemEval-2016 laptop XML (restaurant XML optional, for the gate check); `final/` prepared CSVs and reports |
| **`result/`** | Local only (git-ignored): `notebook_runs/absa_experiments.zip` (every Colab checkpoint), `local_runs/` |
| `requirements.txt` | Python dependencies |
| `.streamlit/config.toml` | App theme colour (run Streamlit from the repo root) |

---

## Limitations

- **Empty sentences.** The model fires on 57% of sentences that carry no opinion (empty-sentence accuracy 43%). The gates remove off-domain ones, but an in-domain sentence without an opinion can still get one.
- **Neutral is weak.** Joint F1 for neutral is 0.14: neutral is about 6% of the gold labels, and stars 1 and 3 never occur in training.
- **Rare categories.** Only 80 of the 198 E#A categories occur in training.
- **Gate trade-off.** About 40% of restaurant sentences still pass, mostly generic praise or service sentences that would fit a laptop review too. Tightening the thresholds would drop more real laptop opinions.
- **English only, one seed, one split.** Differences of 1-2 joint-F1 points between models are within noise.
