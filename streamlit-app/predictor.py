"""
predictor.py
Handles stress prediction and 3-level mapping.
"""

import torch

MAX_LEN = 128

# ─────────────────────────────────────────
# 3-LEVEL MAPPING
# ─────────────────────────────────────────
def map_stress_level(stress_prob: float) -> tuple:
    """
    Maps stress probability (0-100) to level + color.
    Returns: (level, color)
    """
    if stress_prob < 35:
        return "Low",      "#2ecc71"
    elif stress_prob < 65:
        return "Moderate", "#f39c12"
    else:
        return "High",     "#e74c3c"

# ─────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────
def predict(text: str, tokenizer, model) -> dict:
    """
    Runs inference on input text.
    Returns dict with stress_prob, no_stress_prob, level, color.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs   = torch.softmax(outputs.logits, dim=-1).cpu()

    stress_prob    = round(probs[0][1].item() * 100, 1)
    no_stress_prob = round(probs[0][0].item() * 100, 1)
    level, color   = map_stress_level(stress_prob)

    return {
    "stress_prob"   : stress_prob,
    "no_stress_prob": no_stress_prob,
    "level"         : level,
    "color"         : color
    }
