"""
fusion.py — Multimodal Stress Fusion for SmartSense
=====================================================
Combines text model (BERT) + voice model (CNN) outputs
into a single final stress prediction.

Fusion strategy: weighted average of probabilities
  - Text weight : 0.6  (higher — trained on real Reddit stress data, 80% F1)
  - Voice weight: 0.4  (lower  — trained on acted speech, RAVDESS)

Three modes:
  - Text only  : weight = (1.0, 0.0)  → text result passed through directly
  - Voice only : weight = (0.0, 1.0)  → voice result passed through directly
  - Both        : weight = (0.6, 0.4) → weighted average

Usage in app.py:
    from fusion import fuse_predictions

    # text only
    result = fuse_predictions(text_result=predict(...), voice_result=None)

    # text + voice (mic was used)
    result = fuse_predictions(text_result=predict(...), voice_result=predict_stress(...))
"""

# ─────────────────────────────────────────
# WEIGHTS
# ─────────────────────────────────────────
TEXT_WEIGHT  = 0.6
VOICE_WEIGHT = 0.4

# ─────────────────────────────────────────
# LEVEL MAPPING  (same thresholds as predictor.py)
# ─────────────────────────────────────────
def _map_level(stress_prob_100: float) -> tuple[str, str]:
    """
    stress_prob_100: 0–100 float
    Returns: (level, color)
    """
    if stress_prob_100 < 35:
        return "Low",      "#2ecc71"
    elif stress_prob_100 < 65:
        return "Moderate", "#f39c12"
    else:
        return "High",     "#e74c3c"


# ─────────────────────────────────────────
# MAIN FUSION FUNCTION
# ─────────────────────────────────────────
def fuse_predictions(text_result: dict, voice_result: dict | None) -> dict:
    """
    Fuses text and voice model outputs into one result dict.

    Parameters
    ----------
    text_result : dict
        Output of predictor.predict(). Must have keys:
            stress_prob    : float (0–100)
            no_stress_prob : float (0–100)
            level          : str
            color          : str

    voice_result : dict or None
        Output of voice_stress_model.predict_stress(). Must have keys:
            label          : str
            stress_level   : int
            probabilities  : {"Low": float, "Moderate": float, "High": float}
                             Values are 0–1 floats.
        Pass None if user only typed (no mic recording).

    Returns
    -------
    dict with keys:
        stress_prob    : float (0–100) — final fused stress probability
        no_stress_prob : float (0–100)
        level          : str  — "Low" | "Moderate" | "High"
        color          : str  — hex color for UI
        fusion_mode    : str  — "text_only" | "voice_only" | "multimodal"
        text_prob      : float (0–100) — text model stress prob (for UI display)
        voice_prob     : float | None  — voice model stress prob (for UI display)
    """

    # ── Text-only mode ────────────────────────────────────────────────────
    # No audio was recorded — pass text result through unchanged
    if voice_result is None:
        return {
            **text_result,
            "fusion_mode": "text_only",
            "text_prob"  : text_result["stress_prob"],
            "voice_prob" : None,
        }

    # ── Convert text model output to 3-class probabilities (0–1) ─────────
    # text model is binary (stress / no_stress), so we map its stress_prob
    # onto the 3-class scale using the same thresholds as map_stress_level:
    #   0–35% stress_prob  → mostly Low
    #   35–65%             → mostly Moderate
    #   65–100%            → mostly High
    # We do this by distributing the stress_prob across Moderate+High,
    # and no_stress_prob into Low, proportionally.

    t_stress     = text_result["stress_prob"] / 100.0       # 0–1
    t_no_stress  = text_result["no_stress_prob"] / 100.0    # 0–1

    # Distribute stress probability across Moderate and High
    # Below 50% stress → more Moderate, above 50% → more High
    if t_stress <= 0.5:
        # Scale 0–0.5 stress → Moderate gets more
        t_moderate = t_stress * 1.2
        t_high     = t_stress * 0.8
    else:
        # Scale 0.5–1.0 stress → High gets more
        t_moderate = (1 - t_stress) * 0.8
        t_high     = t_stress * 1.2

    # Normalise so they sum to 1
    total = t_no_stress + t_moderate + t_high
    text_probs = {
        "Low"     : t_no_stress / total,
        "Moderate": t_moderate  / total,
        "High"    : t_high      / total,
    }

    # ── Voice model probabilities (already 0–1) ───────────────────────────
    voice_probs = voice_result["probabilities"]

    # ── Weighted average ──────────────────────────────────────────────────
    fused = {
        cls: TEXT_WEIGHT * text_probs[cls] + VOICE_WEIGHT * voice_probs[cls]
        for cls in ("Low", "Moderate", "High")
    }

    # ── Map fused probs back to stress_prob (0–100) for existing UI ───────
    # stress_prob = Moderate + High probability combined (anything above Low)
    fused_stress_prob    = round((fused["Moderate"] + fused["High"]) * 100, 1)
    fused_no_stress_prob = round(fused["Low"] * 100, 1)
    level, color         = _map_level(fused_stress_prob)

    # Voice stress prob for UI (High + Moderate combined)
    voice_stress_prob = round(
        (voice_probs["Moderate"] + voice_probs["High"]) * 100, 1
    )

    return {
        "stress_prob"   : fused_stress_prob,
        "no_stress_prob": fused_no_stress_prob,
        "level"         : level,
        "color"         : color,
        "fusion_mode"   : "multimodal",
        "text_prob"     : text_result["stress_prob"],
        "voice_prob"    : voice_stress_prob,
    }