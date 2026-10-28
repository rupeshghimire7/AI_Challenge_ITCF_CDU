"""Project-wide constants: paths and the SemEval-2016 laptop label inventory."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "DATA"
FINAL_DIR = DATA_DIR / "final"
REPORT_DIR = FINAL_DIR / "reports"

<<<<<<< HEAD
# The selected model (models/notebooks/ABSA_model_selection_report.md): bge_detr at epoch 9, stored with Git LFS. Without
# `git lfs pull`, model.pt is a pointer file and the weights are unpacked from the notebook's zip instead.
CHECKPOINT = ROOT / "model_checkpoint" / "bge_detr" / "checkpoint_9"
CHECKPOINT_ZIP = ROOT / "result" / "notebook_runs" / "absa_experiments.zip"

=======
>>>>>>> origin/main
# SemEval-2016 Task 5 laptop annotation guidelines: 22 entities x 9 attributes (198 possible E#A categories).
ENTITIES = ("LAPTOP", "DISPLAY", "KEYBOARD", "MOUSE", "MOTHERBOARD", "CPU", "FANS_COOLING", "PORTS", "MEMORY", "POWER_SUPPLY",
            "OPTICAL_DRIVES", "BATTERY", "GRAPHICS", "HARD_DISC", "MULTIMEDIA_DEVICES", "HARDWARE", "SOFTWARE", "OS", "WARRANTY",
            "SHIPPING", "SUPPORT", "COMPANY")
ATTRIBUTES = ("GENERAL", "PRICE", "QUALITY", "OPERATION_PERFORMANCE", "USABILITY", "DESIGN_FEATURES", "PORTABILITY",
              "CONNECTIVITY", "MISCELLANEOUS")
POLARITIES = ("negative", "neutral", "positive")
