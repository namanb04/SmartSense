"""
coping_suggestions.py
Generates personalized coping suggestions using Gemini API.
Falls back to static suggestions if API fails.
"""

import google.generativeai as genai
import streamlit as st

# ─────────────────────────────────────────
# CONFIG — paste your Gemini API key here
# Get free key at: https://aistudio.google.com
# ─────────────────────────────────────────
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

# ─────────────────────────────────────────
# STATIC FALLBACK SUGGESTIONS
# Used if Gemini API fails or is unavailable
# ─────────────────────────────────────────
STATIC_COPING = {
    "Low": [
        "✅ Keep up your current routine — consistency is key.",
        "🧘 Try 5 minutes of mindful breathing.",
        "📓 Journaling daily helps maintain low stress.",
    ],
    "Moderate": [
        "🌬️ Box breathing: inhale 4s, hold 4s, exhale 4s, hold 4s.",
        "🚶 A 15-minute walk can significantly reduce cortisol.",
        "📵 Take a short digital detox — step away from screens.",
        "💬 Talk to a friend about what's on your mind.",
    ],
    "High": [
        "😴 Prioritise sleep — aim for 7-8 hours tonight.",
        "✍️ Write down your top 3 stressors and one action for each.",
        "🫁 Try progressive muscle relaxation before bed.",
        "📵 Avoid caffeine and screens 2 hours before sleep.",
        "🆘 Consider speaking with a counsellor or mental health professional.",
    ]
}

# ─────────────────────────────────────────
# GEMINI PROMPT TEMPLATE
# ─────────────────────────────────────────
def build_prompt(user_text: str, stress_level: str) -> str:
    return f"""
You are a supportive wellness assistant. A user has shared how they are feeling, 
and an AI system has detected {stress_level} stress in their message.

User's message: "{user_text}"

Based on what they shared, provide exactly 4 brief, specific, and empathetic 
wellness suggestions to help them manage their stress.

Important rules:
- Do NOT give any medical advice or diagnoses
- Do NOT suggest medications or clinical treatments  
- Keep each suggestion to 1-2 sentences maximum
- Be warm, practical, and actionable
- Start each suggestion with a relevant emoji
- Base suggestions on what the user actually said, not generic advice
- Do not number the suggestions
- Return only the suggestions, no intro or outro text

Respond with exactly 4 suggestions, one per line.
""".strip()

# ─────────────────────────────────────────
# GET SUGGESTIONS
# ─────────────────────────────────────────
def get_coping_suggestions(user_text: str, stress_level: str) -> tuple[list, bool]:
    """
    Returns (suggestions_list, used_ai).
    used_ai = True if Gemini responded, False if fallback was used.
    
    Only calls Gemini for Moderate and High stress.
    Low stress uses static suggestions (no need to call API).
    """

    # For Low stress — static is fine, save API calls
    if stress_level == "Low":
        return STATIC_COPING["Low"], False

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-2.5-flash")  # free tier model

        prompt   = build_prompt(user_text, stress_level)
        response = gemini_model.generate_content(prompt)
        raw_text = response.text.strip()

        # Parse response — split by newline, clean empty lines
        suggestions = [
            line.strip()
            for line in raw_text.split("\n")
            if line.strip()
        ]

        # Validate — must have at least 2 suggestions
        if len(suggestions) < 2:
            raise ValueError("Too few suggestions returned")

        return suggestions[:5], True  # cap at 5

    except Exception as e:
        print(f"Gemini API error: {e} — using fallback suggestions")
        return STATIC_COPING[stress_level], False
