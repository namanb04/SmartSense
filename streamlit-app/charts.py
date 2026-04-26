"""
charts.py
All Plotly chart functions for SmartSense.
"""

import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go

MAX_LEN = 128

# ─────────────────────────────────────────
# CHART 1: GAUGE
# ─────────────────────────────────────────
def gauge_chart(stress_prob: float, color: str):
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = stress_prob,
        delta = {"reference": 50, "valueformat": ".1f"},
        title = {"text": "Stress Probability (%)", "font": {"size": 16}},
        number= {"suffix": "%", "font": {"size": 40, "color": color}},
        gauge = {
            "axis" : {"range": [0, 100], "tickwidth": 1},
            "bar"  : {"color": color, "thickness": 0.3},
            "steps": [
                {"range": [0,  35], "color": "#d5f5e3"},
                {"range": [35, 65], "color": "#fdebd0"},
                {"range": [65,100], "color": "#fadbd8"},
            ],
            "threshold": {
                "line"     : {"color": color, "width": 4},
                "thickness": 0.8,
                "value"    : stress_prob
            }
        }
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
    return fig

# ─────────────────────────────────────────
# CHART 2: CONFIDENCE BARS
# ─────────────────────────────────────────
def confidence_chart(stress_prob: float, no_stress_prob: float):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=["No Stress", "Stress"],
        x=[no_stress_prob, stress_prob],
        orientation="h",
        marker_color=["#2ecc71", "#e74c3c"],
        text=[f"{no_stress_prob}%", f"{stress_prob}%"],
        textposition="outside",
        textfont=dict(size=14, color="white"),
    ))
    fig.update_layout(
        title="Model Confidence Breakdown",
        xaxis=dict(range=[0, 115], showgrid=False,
                   zeroline=False, showticklabels=False),
        yaxis=dict(tickfont=dict(size=14, color="white")),
        height=200,
        margin=dict(l=20, r=40, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
        font_color   ="white",
        showlegend   =False
    )
    return fig

# ─────────────────────────────────────────
# CHART 3: SESSION TREND LINE
# ─────────────────────────────────────────
def trend_chart(history: list):
    df        = pd.DataFrame(history)
    color_map = {"Low": "#2ecc71", "Moderate": "#f39c12", "High": "#e74c3c"}
    colors    = [color_map[l] for l in df["level"]]

    fig = go.Figure()
    fig.add_hrect(y0=0,  y1=35,  fillcolor="#d5f5e3", opacity=0.15, line_width=0)
    fig.add_hrect(y0=35, y1=65,  fillcolor="#fdebd0", opacity=0.15, line_width=0)
    fig.add_hrect(y0=65, y1=100, fillcolor="#fadbd8", opacity=0.15, line_width=0)

    fig.add_trace(go.Scatter(
        x=df["time"],
        y=df["stress_prob"],
        mode="lines+markers+text",
        line=dict(color="#5dade2", width=2.5),
        marker=dict(color=colors, size=12,
                    line=dict(color="white", width=2)),
        text=[f"{p}%" for p in df["stress_prob"]],
        textposition="top center",
        textfont=dict(size=11, color="white"),
        hovertemplate="<b>%{x}</b><br>Stress: %{y}%<extra></extra>"
    ))

    fig.add_annotation(x=df["time"].iloc[-1], y=17,
                       text="LOW", font=dict(color="#2ecc71", size=10),
                       showarrow=False, xanchor="right")
    fig.add_annotation(x=df["time"].iloc[-1], y=50,
                       text="MODERATE", font=dict(color="#f39c12", size=10),
                       showarrow=False, xanchor="right")
    fig.add_annotation(x=df["time"].iloc[-1], y=82,
                       text="HIGH", font=dict(color="#e74c3c", size=10),
                       showarrow=False, xanchor="right")

    fig.update_layout(
        title="Stress Level Trend (This Session)",
        xaxis=dict(showgrid=False, color="white", title="Time"),
        yaxis=dict(showgrid=True, gridcolor="#333", range=[0, 105],
                   color="white", title="Stress %"),
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
        font_color   ="white",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

# ─────────────────────────────────────────
# CHART 4: WORD CONTRIBUTION (Attention-based)
# ─────────────────────────────────────────
def word_contribution_chart(text: str, tokenizer, model):
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
        outputs    = model(**inputs, output_attentions=True)
        probs      = torch.softmax(outputs.logits, dim=-1).cpu()
        attentions = outputs.attentions

    # Guard against empty attentions
    if not attentions or len(attentions) == 0:
        return None

    stress_prob   = probs[0][1].item()
    avg_attention = torch.stack(list(attentions)).squeeze(1)
    avg_attention = avg_attention.mean(dim=(0, 1))
    avg_attention = avg_attention.mean(dim=0).cpu().numpy()

    tokens      = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu())
    skip_tokens = {"[CLS]", "[SEP]", "[PAD]"}

    word_scores = []
    for token, score in zip(tokens, avg_attention):
        if token not in skip_tokens:
            if token.startswith("##") and word_scores:
                word_scores[-1] = (
                    word_scores[-1][0] + token[2:],
                    max(word_scores[-1][1], float(score))
                )
            else:
                word_scores.append((token, float(score)))

    if not word_scores:
        return None

    scores     = np.array([s for _, s in word_scores])
    mean_score = scores.mean()
    direction  = 1 if stress_prob > 0.5 else -1
    normalized = [(s - mean_score) * direction for s in scores]

    pairs = sorted(zip([w for w, _ in word_scores], normalized),
                   key=lambda x: abs(x[1]), reverse=True)[:10]
    pairs = sorted(pairs, key=lambda x: x[1])

    words   = [p[0] for p in pairs]
    weights = [p[1] for p in pairs]
    colors  = ["#e74c3c" if w > 0 else "#2ecc71" for w in weights]

    fig = go.Figure(go.Bar(
        x=weights, y=words,
        orientation="h",
        marker_color=colors,
        text=[f"{w:+.4f}" for w in weights],
        textposition="outside",
        textfont=dict(color="white", size=11)
    ))
    fig.update_layout(
        title="Word Contribution to Stress Prediction (Attention-based)",
        xaxis=dict(title="Attention Score", showgrid=False,
                   zeroline=True, zerolinecolor="#555", color="white"),
        yaxis=dict(color="white", tickfont=dict(size=12)),
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
        font_color   ="white",
        margin=dict(l=20, r=60, t=50, b=40),
        annotations=[
            dict(x=max(weights) * 0.5 if max(weights) > 0 else 0,
                 y=-0.15, xref="x", yref="paper",
                 text="🔴 Stress-contributing words",
                 showarrow=False, font=dict(color="#e74c3c", size=11)),
            dict(x=min(weights) * 0.5 if min(weights) < 0 else 0,
                 y=-0.15, xref="x", yref="paper",
                 text="🟢 Calm-contributing words",
                 showarrow=False, font=dict(color="#2ecc71", size=11))
        ]
    )
    return fig