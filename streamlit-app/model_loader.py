"""
model_loader.py
Loads both models for SmartSense (Streamlit Cloud ready).

- Models are downloaded dynamically (Google Drive)
- No local file paths
- Works in cloud deployment
"""

import os
import torch
import torch.nn as nn
import streamlit as st
import gdown
from transformers import BertTokenizer, BertForSequenceClassification


# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
BERT_BASE = "bert-base-uncased"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create local model directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Local paths (after download)
BERT_PT  = os.path.join(MODEL_DIR, "dreaddit_final.pt")
VOICE_PT = os.path.join(MODEL_DIR, "voice_stress_cnn.pt")

# 🔴 REPLACE THESE WITH YOUR GOOGLE DRIVE FILE IDs
BERT_URL  = "https://drive.google.com/uc?id=1Pju3a27MEv2HIRyoxRRKWyxSNyuDPSlw"
VOICE_URL = "https://drive.google.com/uc?id=1s5iNw_qvuHw5YlR7lFoZ4iOW5KAr4cUh"


# ─────────────────────────────────────────
# DOWNLOAD HELPER
# ─────────────────────────────────────────
def download_model(url, output_path):
    """Download model only if not already present."""
    if not os.path.exists(output_path):
        with st.spinner(f"Downloading {os.path.basename(output_path)}..."):
            gdown.download(url, output_path, quiet=False, fuzzy=True)


# ─────────────────────────────────────────
# VoiceStressCNN (same as training)
# ─────────────────────────────────────────
class VoiceStressCNN(nn.Module):
    def __init__(self, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()

        def block(in_ch, out_ch, drop):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(drop),
            )

        self.encoder = nn.Sequential(
            block(1,   32,  dropout * 0.5),
            block(32,  64,  dropout * 0.5),
            block(64,  128, dropout),
        )

        self.gap  = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.head(self.gap(self.encoder(x)))


# ─────────────────────────────────────────
# TEXT MODEL LOADER
# ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_text_model():
    # Download model if needed
    download_model(BERT_URL, BERT_PT)

    tokenizer = BertTokenizer.from_pretrained(BERT_BASE)

    model = BertForSequenceClassification.from_pretrained(
        BERT_BASE,
        num_labels=2,
        ignore_mismatched_sizes=True,
        attn_implementation="eager"
    )

    model.load_state_dict(
        torch.load(BERT_PT, map_location=DEVICE, weights_only=False)
    )

    model.to(DEVICE)
    model.eval()

    return tokenizer, model


# ─────────────────────────────────────────
# VOICE MODEL LOADER
# ─────────────────────────────────────────
@st.cache_resource
def load_voice_model():
    try:
        # Download model if needed
        download_model(VOICE_URL, VOICE_PT)

        model = VoiceStressCNN(num_classes=3, dropout=0.3).to(DEVICE)

        model.load_state_dict(
            torch.load(VOICE_PT, map_location=DEVICE, weights_only=False)
        )

        model.eval()
        return model

    except Exception as e:
        st.warning(f"Voice model failed to load ({e}) — running text-only mode.")
        return None
