"""
config.py
---------
Centralised configuration for the Multimodal ASD Trait Detection AI Agent.

Local usage   : copy .env.example to .env and add GROQ_API_KEY=your_key
Streamlit Cloud: add GROQ_API_KEY in App Settings → Secrets
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API Key ───────────────────────────────────────────────────────────────────
groq_api_key = os.getenv("GROQ_API_KEY", "")

# ── Directory layout ──────────────────────────────────────────────────────────
#   project_root/
#   ├── models/
#   │   ├── xgboost_asd_model.pkl
#   │   ├── asd_classifier_model/       ← fine-tuned BERT folder
#   │   └── feature_cols.json
#   ├── data/
#   │   └── asd_materials.json          ← social stories + schedules library
#   ├── config.py
#   ├── agent.py
#   ├── app.py
#   └── requirements.txt

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR   = os.path.join(BASE_DIR, "data")

# ── ML model paths ────────────────────────────────────────────────────────────
XGBOOST_MODEL_PATH = os.path.join(MODELS_DIR, "xgboost_asd_model.pkl")
BERT_MODEL_PATH    = os.path.join(MODELS_DIR, "asd_classifier_model")

# ── BERT inference config ─────────────────────────────────────────────────────
BERT_MAX_LENGTH = 64  # Must match training MAX_LENGTH

# ── Feature column order (must match XGBoost training) ────────────────────────
FEATURE_COLS_DEFAULT = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","Sex"]
FEATURE_COLS_PATH    = os.path.join(MODELS_DIR, "feature_cols.json")

# ── Companion library path ────────────────────────────────────────────────────
# JSON file containing pre-built social stories and visual schedules.
# The retrieval engine embeds these for semantic search.
MATERIALS_PATH = os.path.join(DATA_DIR, "asd_materials.json")

# ── Sentence-transformer model for semantic retrieval ─────────────────────────
# Using a lightweight model that runs well on CPU.
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
