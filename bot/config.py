"""
Configuration for the Vera Message Engine bot.
"""
import os
from pathlib import Path

try:
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

# ─── LLM Configuration ───────────────────────────────────────────────────────
# Primary: Groq GPT-OSS 120B via OpenAI-compatible Responses API
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_URL = f"{GROQ_BASE_URL}/responses"

# Fallback: NVIDIA (Gemma 3n)
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
NVIDIA_MODEL = "google/gemma-3n-e4b-it"
NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

LLM_TEMPERATURE = 0.1  # Slight randomness for natural language, near-deterministic
LLM_MAX_TOKENS = 1200
LLM_TIMEOUT = 25  # seconds — must fit within 30s judge timeout

# ─── Team Metadata ────────────────────────────────────────────────────────────
TEAM_NAME = "Vera Engine"
TEAM_MEMBERS = ["Aditya"]
MODEL_NAME = "openai/gpt-oss-120b (Groq Responses API)"
APPROACH = (
    "Signal-routing composer with 5-dimension scoring awareness. "
    "Groq GPT-OSS 120B for high-quality composition with explicit "
    "specificity anchoring, category voice enforcement, and compulsion lever "
    "tagging. 20+ trigger-specific prompt variants with deep payload extraction. "
    "Auto-reply detection with 3-strike ladder. Instant intent-to-action "
    "transition. Zero fabrication — every number grounded in provided context. "
    "Natural Hindi-English code-mix based on merchant language preference."
)
CONTACT_EMAIL = "aditya@example.com"
BOT_VERSION = "2.0.0"

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
    "we'll get back",
    "automated reply",
    "auto-reply",
    "auto reply",
    "automated assistant",
    "automated response",
    "i am an automated",
    "i'm an automated",
    "this is an automated",
    "this is an auto",
    "aapki jaankari ke liye",
    "hamari team tak pahuncha",
    "shukriya. main aapki",
    "apki jaankari ke liye",
    "aapke message ke liye dhanyavad",
    "hum jaldi se aapko",
    "hum aapko jald",
    "please wait while",
    "your message is important",
    "currently unavailable",
    "out of office",
    "will respond shortly",
    "will reply soon",
    "team will get back",
    "will revert",
    "noted your message",
    "message has been received",
]

# ─── Intent Commitment Phrases ───────────────────────────────────────────────
COMMITMENT_PHRASES = [
    "let's do it", "lets do it", "go ahead", "proceed",
    "yes please", "yes do it", "ok do it", "ok lets do",
    "haan", "haan karo", "kar do", "karo", "chalega", "chalo",
    "ok done", "sure", "yes sure", "definitely", "absolutely",
    "i want to join", "mujhe join karna", "join karna hai",
    "whats next", "what's next", "what next",
    "ok", "yes", "ya", "yep", "yup", "confirm", "approved",
    "do it", "go", "start", "launch", "activate",
    "theek hai", "thik hai", "bilkul", "zaroor",
    "send it", "share it", "draft it", "create it",
    "sounds good", "looks good", "perfect", "great",
    "i'm in", "im in", "count me in",
    "renew", "subscribe", "sign up", "sign me up",
    "ban jao", "bana do", "bhej do", "ship it",
]

# ─── Hostile / Opt-out Phrases ────────────────────────────────────────────────
HOSTILE_PHRASES = [
    "stop messaging", "stop sending", "don't message",
    "not interested", "unsubscribe", "spam",
    "useless", "bothering me", "leave me alone",
    "block", "report", "harassment",
    "don't contact", "do not contact",
    "remove me", "take me off",
    "waste of time", "waste of my time",
    "stop it", "enough", "no more",
    "band karo", "mat bhejo", "pareshan mat karo",
]

# ─── Off-topic Phrases ────────────────────────────────────────────────────────
OFF_TOPIC_PHRASES = [
    "gst", "tax filing", "income tax", "loan", "insurance",
    "legal", "lawyer", "court", "police", "passport",
    "aadhaar", "pan card", "bank account",
]

# ─── Dataset Path ─────────────────────────────────────────────────────────────
DATASET_DIR = Path(__file__).parent.parent / "dataset"
