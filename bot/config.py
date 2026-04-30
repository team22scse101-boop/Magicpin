"""
Configuration for the Vera Message Engine bot.
"""
import os
from pathlib import Path

# ─── LLM Configuration ───────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 1500
LLM_TIMEOUT = 25  # seconds — must fit within 30s judge timeout

# ─── Team Metadata ────────────────────────────────────────────────────────────
TEAM_NAME = "Vera Engine"
TEAM_MEMBERS = ["Aditya"]
MODEL_NAME = "gemini-2.0-flash"
APPROACH = (
    "Signal-routing composer: picks the single best signal from trigger + merchant state + "
    "category context before composing. 20+ trigger-specific prompt variants. "
    "Auto-reply detection with 3-strike ladder. Intent transition detection for instant "
    "action-mode switching. Zero fabrication — every number grounded in context."
)
CONTACT_EMAIL = "aditya@example.com"
BOT_VERSION = "1.0.0"

# ─── Rate Limits & Suppression ────────────────────────────────────────────────
MAX_ACTIONS_PER_TICK = 20
MAX_ACTIONS_PER_MERCHANT_PER_TICK = 1
SUPPRESSION_TTL_SECONDS = 86400  # 24 hours

# ─── Auto-reply Detection ────────────────────────────────────────────────────
AUTO_REPLY_PATTERNS = [
    "thank you for contacting",
    "thanks for contacting",
    "our team will respond",
    "we will get back to you",
    "automated reply",
    "auto-reply",
    "automated assistant",
    "automated response",
    "i am an automated",
    "aapki jaankari ke liye",
    "hamari team tak pahuncha",
    "shukriya. main aapki",
]

# ─── Intent Commitment Phrases ───────────────────────────────────────────────
COMMITMENT_PHRASES = [
    "let's do it", "lets do it", "go ahead", "proceed",
    "yes please", "yes do it", "ok do it", "ok lets do",
    "haan", "haan karo", "kar do", "chalega", "chalo",
    "ok done", "sure", "yes sure", "definitely",
    "i want to join", "mujhe join karna", "join karna hai",
    "whats next", "what's next", "what next",
    "ok", "yes", "ya", "yep", "yup", "confirm", "approved",
]

# ─── Hostile / Opt-out Phrases ────────────────────────────────────────────────
HOSTILE_PHRASES = [
    "stop messaging", "stop sending", "don't message",
    "not interested", "unsubscribe", "spam",
    "useless", "bothering me", "leave me alone",
    "block", "report", "harassment",
]

# ─── Off-topic Phrases ────────────────────────────────────────────────────────
OFF_TOPIC_PHRASES = [
    "gst", "tax filing", "income tax", "loan", "insurance",
    "legal", "lawyer", "court", "police", "passport",
]

# ─── Dataset Path ─────────────────────────────────────────────────────────────
DATASET_DIR = Path(__file__).parent.parent / "dataset"
