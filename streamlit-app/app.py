"""
app.py — SmartSense Main Entry Point
Run: streamlit run app.py

Project structure:
    app.py               ← this file
    model_loader.py      ← loads BERT + CNN voice model
    predictor.py         ← text stress predict()
    fusion.py            ← multimodal fusion
    charts.py            ← Plotly charts
    voice_component.py   ← unified mic (speech-to-text + WAV capture)
    coping_suggestions.py← Gemini API suggestions
"""

import os
import base64
import tempfile
import streamlit as st
from datetime import datetime

from model_loader       import load_text_model, load_voice_model
from predictor          import predict
from fusion             import fuse_predictions
from charts             import gauge_chart, confidence_chart, trend_chart, word_contribution_chart
from voice_component    import voice_input_component
from coping_suggestions import get_coping_suggestions

import librosa
import numpy as np
import torch

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
STRESS_LABELS = {0: "Low", 1: "Moderate", 2: "High"}

def _extract_mfcc(file_path: str) -> np.ndarray:
    """Identical to notebook extract_mfcc() — must match training exactly."""
    sr      = 22050
    max_len = sr * 3
    y, _ = librosa.load(file_path, sr=sr, mono=True)
    if len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)))
    else:
        y = y[:max_len]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=2048, hop_length=512)
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
    return mfcc[np.newaxis, :, :].astype(np.float32)

def predict_voice_stress(audio_path: str, model) -> dict:
    feats  = _extract_mfcc(audio_path)
    tensor = torch.tensor(feats).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    stress_level = int(probs.argmax())
    return {
        "label"        : STRESS_LABELS[stress_level],
        "stress_level" : stress_level,
        "probabilities": {
            "Low"     : float(probs[0]),
            "Moderate": float(probs[1]),
            "High"    : float(probs[2]),
        }
    }

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(page_title="SmartSense", layout="wide")

# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
if "history"     not in st.session_state:
    st.session_state.history = []
if "input_mode"  not in st.session_state:
    st.session_state.input_mode = "Voice"         # "Text" or "Voice"

# ─────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────
with st.spinner(" Loading SmartSense AI models"):
    try:
        tokenizer, bert_model = load_text_model()
        voice_model = load_voice_model()
    except Exception as e:
        st.error(f"Could not load models: {e}")
        st.stop()

voice_model_available = voice_model is not None

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.title("SmartSense: For Mental Well-Being")
st.markdown("---")

# ─────────────────────────────────────────
# MODE TOGGLE  — two side-by-side buttons
# ─────────────────────────────────────────
st.subheader("How are you feeling?")

# Custom CSS for the toggle buttons
st.markdown("""
<style>
div[data-testid="column"] button {
    width: 100%;
    border-radius: 8px;
    font-size: 15px;
    font-weight: 600;
    padding: 10px 0;
    transition: all 0.2s ease;
}
</style>
""", unsafe_allow_html=True)

col_voice, col_text = st.columns(2)

with col_voice:
    if st.button(
        "🎙️  Voice + Text",
        use_container_width=True,
        type="primary" if st.session_state.input_mode == "Voice" else "secondary",
        disabled=not voice_model_available,
        help=None if voice_model_available else "Voice model not loaded",
    ):
        st.session_state.input_mode = "Voice"
        st.rerun()

with col_text:
    # Solid style when active, outline when inactive
    if st.button(
        "✍️  Text",
        use_container_width=True,
        type="primary" if st.session_state.input_mode == "Text" else "secondary",
    ):
        st.session_state.input_mode = "Text"
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# INPUT  — conditional on mode
# ─────────────────────────────────────────
mode = st.session_state.input_mode
text       = ""
audio_b64  = ""
transcript = ""

if mode == "Text":
    # ── Text mode: just a text area, no mic ───────────────────────────────
    text = st.text_area(
        label="Input text",
        placeholder="Type how you're feeling...",
        height=130,
        label_visibility="collapsed"
    )

else:
    # ── Voice mode: mic component, read-only transcript preview ───────────
    transcript, audio_b64 = voice_input_component()

    if transcript:
        # Read-only preview — user sees what was captured
        st.markdown("**Transcript** (read-only — this is what will be analysed)")
        st.markdown(f"""
        <div style='
            background: #1c2128;
            border: 1px solid #388bfd;
            border-radius: 6px;
            padding: 10px 14px;
            color: #c9d1d9;
            font-family: sans-serif;
            font-size: 14px;
            line-height: 1.5;
            margin-bottom: 8px;
        '>{transcript}</div>
        """, unsafe_allow_html=True)

        if audio_b64:
            st.caption("🎵 Audio captured · Multimodal analysis (text + voice) will run")
        else:
            st.caption("📝 No audio captured · Text-only analysis will run")
    else:
        st.caption("Speak into the mic, then click Analyze.")

    # text for BERT is the transcript
    text = transcript

# ─────────────────────────────────────────
# ANALYZE BUTTON
# ─────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
col_btn, col_note = st.columns([1, 3])
with col_btn:
    analyze = st.button("🔍 Analyze Stress", use_container_width=True)
with col_note:
    if mode == "Voice":
        st.caption("🧠 Multimodal analysis · ⚠️ Not a medical diagnosis.")
    if mode == "Text":
        st.caption("📝 Text analysis · ⚠️ Not a medical diagnosis.")

# ─────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────
if analyze:
    if not text.strip():
        if mode == "Voice":
            st.warning("Please use the mic to speak first.")
        else:
            st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing..."):

            # Step 1: Text model (always runs)
            text_result = predict(text.strip(), tokenizer, bert_model)

            # Step 2: Voice model (only in Voice mode with captured audio)
            voice_result = None
            if mode == "Voice" and audio_b64 and voice_model_available:
                try:
                    audio_bytes = base64.b64decode(audio_b64)
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        tmp.write(audio_bytes)
                        tmp_path = tmp.name
                    voice_result = predict_voice_stress(tmp_path, voice_model)
                    os.unlink(tmp_path)
                except Exception as e:
                    st.warning(f"Voice analysis failed — using text only. ({e})")
                    voice_result = None

            # Step 3: Fuse
            result = fuse_predictions(text_result, voice_result)

        # Unpack
        stress_prob    = result["stress_prob"]
        no_stress_prob = result["no_stress_prob"]
        level          = result["level"]
        color          = result["color"]
        fusion_mode    = result["fusion_mode"]

        # Log to history
        st.session_state.history.append({
            "time"       : datetime.now().strftime("%H:%M:%S"),
            "stress_prob": stress_prob,
            "level"      : level
        })

        st.markdown("---")

        # Fusion mode info bar
        if fusion_mode == "multimodal":
            st.info(
                f"🧠 **Multimodal result** — "
                f"Text: **{result['text_prob']}%** · "
                f"Voice: **{result['voice_prob']}%** · "
                f"Fused: **{stress_prob}%**"
            )
        else:
            st.caption("📝 Text-only analysis")

        st.subheader(f"Result: {level} Stress")

        # Row 1: Gauge + Confidence
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(gauge_chart(stress_prob, color), use_container_width=True)
        with c2:
            st.plotly_chart(confidence_chart(stress_prob, no_stress_prob), use_container_width=True)
            badge_colors = {
                "Low"     : ("#d5f5e3", "#1e8449"),
                "Moderate": ("#fdebd0", "#b7770d"),
                "High"    : ("#fadbd8", "#c0392b")
            }
            bg, fg = badge_colors[level]
            st.markdown(f"""
            <div style='background:{bg}; color:{fg}; padding:12px 20px;
                        border-radius:8px; text-align:center;
                        font-size:18px; font-weight:600; margin-top:10px;'>
                {level} Stress Detected
            </div>
            """, unsafe_allow_html=True)

        # Row 2: Word contribution
        st.markdown("---")
        st.subheader("📝 Word Contribution Analysis")
        st.caption("Which words in your text drove the stress prediction")
        with st.spinner("Generating word analysis..."):
            try:
                fig_words = word_contribution_chart(text.strip(), tokenizer, bert_model)
                if fig_words:
                    st.plotly_chart(fig_words, width='stretch')
            except Exception as e:
                st.warning(f"Word analysis unavailable: {e}")

        # Row 3: Coping suggestions
        st.markdown("---")
        st.subheader("💡 Coping Suggestions")
        with st.spinner("Generating personalised suggestions..."):
            suggestions, used_ai = get_coping_suggestions(text.strip(), level)
        if used_ai:
            st.caption("✨ Personalised suggestions powered by Gemini AI. Not medical advice.")
        else:
            st.caption("General wellness tips — not medical advice.")
        cols = st.columns(len(suggestions))
        for i, tip in enumerate(suggestions):
            with cols[i]:
                st.info(tip)

# ─────────────────────────────────────────
# SESSION TREND
# ─────────────────────────────────────────
if len(st.session_state.history) >= 2:
    st.markdown("---")
    st.subheader("📈 Session Stress Trend")
    st.plotly_chart(trend_chart(st.session_state.history), use_container_width=True)

# ─────────────────────────────────────────
# CLEAR HISTORY
# ─────────────────────────────────────────
if st.session_state.history:
    if st.button("🗑️ Clear Session History"):
        st.session_state.history = []
        st.rerun()