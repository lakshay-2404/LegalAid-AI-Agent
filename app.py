import html
import json
import os
from pathlib import Path
import re
import logging
from logging.handlers import RotatingFileHandler
import time
import uuid
from urllib import error as urllib_error
from urllib import request as urllib_request
from datetime import datetime
import streamlit as st
from langchain_ollama.llms import OllamaLLM
import streamlit.components.v1 as components
from markdown_it import MarkdownIt

from rag_core import answer_query
import vector

try:
    from groq import Groq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False

# =========================
# STREAMLIT CONFIG
# =========================

st.set_page_config(
    page_title="NyayGram — Legal Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="⚖️",
)

CHAT_INPUT_KEY = "chat_draft"
CHAT_INPUT_PLACEHOLDER = "Ask about an Act, section, or case..."
MARKDOWN_RENDERER = MarkdownIt("commonmark", {"html": False, "linkify": True, "typographer": False})
ANSWER_LANGUAGE_CODES = {
    "english": "en", "hindi": "hi", "bengali": "bn", "marathi": "mr",
    "tamil": "ta", "telugu": "te", "kannada": "kn", "malayalam": "ml",
    "gujarati": "gu", "punjabi": "pa", "urdu": "ur",
}
DEFAULT_MODEL_NAME = os.environ.get("DEFAULT_LLM_MODEL", "qwen2.5:3b")
DEFAULT_GROQ_MODEL = os.environ.get("DEFAULT_GROQ_MODEL", "qwen/qwen3-32b")

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"
RESPONSES_LOG = LOG_DIR / "responses.jsonl"
LOG_MODE = (os.environ.get("APP_LOG_MODE") or "stdout").lower()
LOG_JSON = (os.environ.get("APP_LOG_JSON") or "false").lower() == "true"
LOG_MAX_BYTES = int(os.environ.get("APP_LOG_MAX_BYTES", "1000000"))
LOG_BACKUP_COUNT = int(os.environ.get("APP_LOG_BACKUP_COUNT", "3"))


def _configure_file_logging() -> None:
    level_name = (os.environ.get("APP_LOG_LEVEL") or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    already = {type(h) for h in logging.getLogger().handlers}
    if LOG_MODE in {"stdout", "both"} and logging.StreamHandler not in already:
        h = logging.StreamHandler()
        h.setFormatter(formatter)
        logger.addHandler(h)
    logging.captureWarnings(True)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


_configure_file_logging()
logger = logging.getLogger(__name__)


def _persist_response(entry: dict) -> None:
    try:
        RESPONSES_LOG.parent.mkdir(exist_ok=True)
        with RESPONSES_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("Failed to persist response: %s", e)


def _ollama_base_url() -> str | None:
    return (os.environ.get("OLLAMA_BASE_URL") or os.environ.get("OLLAMA_HOST") or "").strip() or None


def _normalized_ollama_base_url() -> str:
    return (_ollama_base_url() or "http://127.0.0.1:11434").rstrip("/")


@st.cache_data(ttl=30, show_spinner=False)
def list_ollama_models(base_url: str) -> list[str]:
    try:
        with urllib_request.urlopen(f"{base_url}/api/tags", timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return []
    models = [(m.get("name") or "").strip() for m in payload.get("models", []) if m.get("name")]
    chat_models = [m for m in models if not _is_embedding_model(m)]
    return chat_models or models


def pick_preferred_model(model_options: list[str]) -> str:
    if not model_options:
        return DEFAULT_MODEL_NAME
    if DEFAULT_MODEL_NAME in model_options:
        return DEFAULT_MODEL_NAME
    for name in model_options:
        if name.lower().startswith("qwen"):
            return name
    return model_options[0]


def _is_embedding_model(name: str | None) -> bool:
    if not name:
        return False
    return any(t in name.lower() for t in ("embed", "embedding", "nomic-embed", "all-minilm", "bge"))


# =========================
# CSS — DARK JUDICIAL AESTHETIC
# =========================

def inject_css():
    st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&family=EB+Garamond:ital,wght@0,400;0,500;1,400&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
  /* ── TOKENS ─────────────────────────────────────────── */
  :root {
    --ink:        #0d0d0d;
    --ink-soft:   #1a1a1a;
    --parchment:  #f5f0e8;
    --cream:      #faf7f2;
    --gold:       #c9a84c;
    --gold-dim:   #8a6f2e;
    --gold-glow:  rgba(201,168,76,0.18);
    --rust:       #8b3a2a;
    --sage:       #3a5c4a;
    --text:       #e8e0d0;
    --text-dim:   #9a8f7e;
    --text-muted: #5c534a;
    --border:     rgba(201,168,76,0.15);
    --border-mid: rgba(201,168,76,0.28);
    --panel:      rgba(20,17,13,0.82);
    --shadow-deep: 0 32px 64px rgba(0,0,0,0.55);
    --shadow-lift: 0 8px 32px rgba(0,0,0,0.40);
    --radius:     4px;
  }

  /* ── GLOBAL ─────────────────────────────────────────── */
  html, body, .stApp {
    font-family: 'EB Garamond', Georgia, serif;
    background: var(--ink) !important;
    color: var(--text) !important;
  }

  .stApp {
    background:
      radial-gradient(ellipse 80% 50% at 50% -10%, rgba(201,168,76,0.07) 0%, transparent 60%),
      radial-gradient(ellipse 40% 40% at 90% 90%, rgba(139,58,42,0.06) 0%, transparent 50%),
      linear-gradient(180deg, #100e0a 0%, #0a0908 50%, #0d0b09 100%) !important;
    min-height: 100vh;
  }

  div[data-testid="stAppViewContainer"],
  section[data-testid="stMain"],
  div[data-testid="stMainBlockContainer"] {
    background: transparent !important;
  }

  /* ── HEADER ─────────────────────────────────────────── */
  header[data-testid="stHeader"] {
    background: rgba(10,9,8,0.92) !important;
    border-bottom: 1px solid var(--border) !important;
    backdrop-filter: blur(20px);
  }
  div[data-testid="stToolbar"],
  div[data-testid="stDecoration"] { background: transparent !important; }

  /* ── MAIN BLOCK ─────────────────────────────────────── */
  .block-container {
    max-width: 880px !important;
    padding-top: 2rem !important;
    padding-bottom: 7rem !important;
  }

  /* ── SIDEBAR ─────────────────────────────────────────── */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111009 0%, #0d0b09 100%) !important;
    border-right: 1px solid var(--border) !important;
  }
  section[data-testid="stSidebar"] * { color: var(--text) !important; }
  section[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Playfair Display', serif !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: var(--gold) !important;
    margin-bottom: 0.6rem !important;
  }
  section[data-testid="stSidebar"] hr {
    border-color: var(--border) !important;
    margin: 1rem 0 !important;
  }
  section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
  }
  section[data-testid="stSidebar"] input,
  section[data-testid="stSidebar"] textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid var(--border-mid) !important;
    color: var(--text) !important;
    border-radius: var(--radius) !important;
  }
  section[data-testid="stSidebar"] .stSlider > div > div > div {
    background: var(--gold-dim) !important;
  }
  section[data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"] {
    color: var(--gold) !important;
  }

  /* ── BUTTONS ─────────────────────────────────────────── */
  div[data-testid="stButton"] button {
    background: transparent !important;
    border: 1px solid var(--border-mid) !important;
    color: var(--gold) !important;
    border-radius: var(--radius) !important;
    font-family: 'EB Garamond', serif !important;
    font-size: 0.92rem !important;
    letter-spacing: 0.04em !important;
    transition: all 160ms ease !important;
    box-shadow: none !important;
  }
  div[data-testid="stButton"] button * {
    color: var(--gold) !important;
    -webkit-text-fill-color: var(--gold) !important;
  }
  div[data-testid="stButton"] button:hover {
    background: var(--gold-glow) !important;
    border-color: var(--gold) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(201,168,76,0.2) !important;
  }
  div[data-testid="stButton"] button:hover * {
    color: var(--gold) !important;
    -webkit-text-fill-color: var(--gold) !important;
  }

  /* Starter prompt buttons — taller */
  div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button {
    min-height: 5.5rem !important;
    text-align: center !important;
    padding: 0.85rem 1rem !important;
    font-size: 0.88rem !important;
    line-height: 1.5 !important;
    white-space: normal !important;
  }

  /* ── CHAT MESSAGES ───────────────────────────────────── */
  div[data-testid="stChatMessage"] {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 0.6rem 0.3rem !important;
    box-shadow: var(--shadow-lift) !important;
    backdrop-filter: blur(12px) !important;
    animation: fadeUp 220ms ease both !important;
    max-width: 840px !important;
    margin-left: auto !important;
    margin-right: auto !important;
  }
  div[data-testid="stChatMessage"] * { color: var(--text) !important; }
  div[data-testid="stChatMessage"] p,
  div[data-testid="stChatMessage"] li { line-height: 1.75 !important; font-size: 1rem !important; }
  div[data-testid="stChatMessage"] strong { color: var(--gold) !important; -webkit-text-fill-color: var(--gold) !important; }
  div[data-testid="stChatMessage"] code {
    font-family: 'JetBrains Mono', monospace !important;
    background: rgba(201,168,76,0.08) !important;
    border: 1px solid rgba(201,168,76,0.15) !important;
    padding: 0.1em 0.4em !important;
    border-radius: 2px !important;
    font-size: 0.88em !important;
    color: var(--gold) !important;
    -webkit-text-fill-color: var(--gold) !important;
  }
  div[data-testid="stChatMessage"] pre {
    background: rgba(201,168,76,0.05) !important;
    border: 1px solid rgba(201,168,76,0.12) !important;
    border-left: 3px solid var(--gold-dim) !important;
    border-radius: var(--radius) !important;
    padding: 1rem !important;
    font-family: 'JetBrains Mono', monospace !important;
  }

  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  /* ── CHAT INPUT ──────────────────────────────────────── */
  /* Nuke every possible Streamlit bottom-bar element */
  [data-testid="stBottom"],
  [data-testid="stBottomBlockContainer"],
  [data-testid="stBottomBlockContainer"] > *,
  [data-testid="stBottomBlockContainer"] > * > *,
  [data-testid="stVerticalBlock"],
  .st-emotion-cache-1dp5vir,
  .st-emotion-cache-h5rgaw,
  .st-emotion-cache-1wbqy5l,
  .st-emotion-cache-ue6h4q,
  .st-emotion-cache-0,
  [class*="st-emotion-cache"] + [class*="st-emotion-cache"] {
    background: transparent !important;
    background-color: transparent !important;
    box-shadow: none !important;
    border: none !important;
  }
  [data-testid="stBottom"] {
    background: linear-gradient(180deg, transparent 0%, rgba(10,9,8,0.92) 35%, #0a0908 100%) !important;
  }
  div[data-testid="stChatInput"] { background: transparent !important; }
  div[data-testid="stChatInput"] > div {
    max-width: 840px !important;
    margin: 0 auto !important;
    background: rgba(18,15,11,0.96) !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: 2px !important;
    padding: 0.15rem 0.3rem !important;
    box-shadow: 0 0 0 1px rgba(201,168,76,0.06), var(--shadow-deep) !important;
  }
  div[data-testid="stChatInput"] form,
  div[data-testid="stChatInput"] form > div,
  div[data-testid="stChatInput"] [data-baseweb="textarea"],
  div[data-testid="stChatInput"] [data-baseweb="textarea"] > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
  }
  div[data-testid="stChatInput"] textarea {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: var(--text) !important;
    font-family: 'EB Garamond', serif !important;
    font-size: 1rem !important;
    caret-color: var(--gold) !important;
  }
  div[data-testid="stChatInput"] textarea::placeholder {
    color: var(--text-muted) !important;
    opacity: 1 !important;
    font-style: italic !important;
  }
  div[data-testid="stChatInput"] button {
    background: var(--gold) !important;
    color: var(--ink) !important;
    border: none !important;
    border-radius: 2px !important;
    box-shadow: none !important;
  }
  div[data-testid="stChatInput"] button * { color: var(--ink) !important; -webkit-text-fill-color: var(--ink) !important; }
  div[data-testid="stChatInput"] button svg { fill: var(--ink) !important; }

  /* ── EXPANDERS ───────────────────────────────────────── */
  div[data-testid="stExpander"] {
    background: rgba(201,168,76,0.04) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
  }
  div[data-testid="stExpander"] summary { color: var(--gold) !important; }
  div[data-testid="stExpander"] summary * { color: var(--gold) !important; -webkit-text-fill-color: var(--gold) !important; }

  /* ── CHECKBOXES ──────────────────────────────────────── */
  div[role="checkbox"] {
    border: 1px solid var(--border-mid) !important;
    background: rgba(201,168,76,0.04) !important;
    border-radius: 2px !important;
  }
  div[role="checkbox"][aria-checked="true"] {
    background: var(--gold-dim) !important;
    border-color: var(--gold) !important;
  }

  /* ── RADIO ───────────────────────────────────────────── */
  div[role="radiogroup"] label { color: var(--text) !important; }
  div[data-baseweb="radio"] div { border-color: var(--border-mid) !important; }
  div[data-baseweb="radio"][aria-checked="true"] div { background: var(--gold-dim) !important; border-color: var(--gold) !important; }

  /* ── SCROLLBAR ───────────────────────────────────────── */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-thumb { background: var(--gold-dim); border-radius: 0; }
  ::-webkit-scrollbar-track { background: rgba(201,168,76,0.03); }

  /* ── HERO ────────────────────────────────────────────── */
  .nyay-hero {
    padding: 2.5rem 0 1.8rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
    position: relative;
  }
  .nyay-hero::before {
    content: "⚖";
    position: absolute;
    right: 0;
    top: 50%;
    transform: translateY(-50%);
    font-size: 6rem;
    opacity: 0.04;
    pointer-events: none;
    user-select: none;
  }
  .nyay-eyebrow {
    font-family: 'EB Garamond', serif;
    font-size: 0.78rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 0.6rem;
  }
  .nyay-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.8rem, 5vw, 4.2rem);
    font-weight: 700;
    line-height: 0.98;
    letter-spacing: -0.02em;
    color: var(--parchment);
    margin: 0 0 0.9rem 0;
  }
  .nyay-title em {
    font-style: italic;
    color: var(--gold);
  }
  .nyay-subtitle {
    font-family: 'EB Garamond', serif;
    font-size: 1.08rem;
    color: var(--text-dim);
    max-width: 580px;
    line-height: 1.7;
    font-style: italic;
  }

  /* ── STATUS STRIP ────────────────────────────────────── */
  .nyay-status-strip {
    display: flex;
    gap: 1.5rem;
    padding: 0.85rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.8rem;
    flex-wrap: wrap;
    align-items: center;
  }
  .nyay-status-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    color: var(--text-muted);
    letter-spacing: 0.03em;
  }
  .nyay-status-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .nyay-status-dot.ok  { background: #4a9b6f; box-shadow: 0 0 6px rgba(74,155,111,0.5); }
  .nyay-status-dot.warn{ background: var(--gold); box-shadow: 0 0 6px rgba(201,168,76,0.4); }
  .nyay-status-dot.bad { background: var(--rust); box-shadow: 0 0 6px rgba(139,58,42,0.5); }
  .nyay-status-label { color: var(--text-dim); }

  /* ── STARTER PROMPTS LABEL ───────────────────────────── */
  .nyay-prompts-label {
    font-family: 'EB Garamond', serif;
    font-size: 0.8rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
  }

  /* ── TOOLBAR NOTE ────────────────────────────────────── */
  .toolbar-note {
    color: var(--text-muted);
    font-family: 'EB Garamond', serif;
    font-size: 0.88rem;
    font-style: italic;
    margin: 0.3rem 0 0.2rem 0;
  }

  /* ── SPINNER ─────────────────────────────────────────── */
  div[data-testid="stSpinner"] * { color: var(--gold) !important; }

  /* ── ALERTS ──────────────────────────────────────────── */
  div[data-testid="stAlert"] {
    background: rgba(139,58,42,0.12) !important;
    border: 1px solid rgba(139,58,42,0.3) !important;
    border-radius: var(--radius) !important;
  }
  div[data-testid="stAlert"] * { color: #e8a090 !important; -webkit-text-fill-color: #e8a090 !important; }
  div[data-testid="stInfo"] {
    background: rgba(201,168,76,0.07) !important;
    border: 1px solid rgba(201,168,76,0.2) !important;
  }
  div[data-testid="stInfo"] * { color: var(--text-dim) !important; -webkit-text-fill-color: var(--text-dim) !important; }

  /* ── CAPTIONS ────────────────────────────────────────── */
  .stCaption, small { color: var(--text-muted) !important; font-style: italic !important; }

  /* ── SELECT / INPUT ──────────────────────────────────── */
  div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
  }
  div[data-baseweb="popover"] { background: #18140f !important; border: 1px solid var(--border-mid) !important; }
  div[data-baseweb="option"] { background: #18140f !important; color: var(--text) !important; }
  div[data-baseweb="option"]:hover { background: var(--gold-glow) !important; }

  /* ── DIVIDER ─────────────────────────────────────────── */
  hr { border-color: var(--border) !important; }

</style>
""", unsafe_allow_html=True)


inject_css()

# Strip blue bottom bar background at runtime via JS
st.components.v1.html("""<script>
(function() {
  function strip() {
    var bottom = window.parent.document.querySelector('[data-testid="stBottom"]');
    if (!bottom) return;
    var all = [bottom].concat(Array.from(bottom.querySelectorAll('*')));
    all.forEach(function(el) {
      var s = window.getComputedStyle(el);
      var bg = s.backgroundColor;
      if (!bg || bg === 'rgba(0, 0, 0, 0)' || bg === 'transparent') return;
      var m = bg.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
      if (m && +m[3] > +m[1] && +m[3] > +m[2]) {
        el.style.setProperty('background','transparent','important');
        el.style.setProperty('background-color','transparent','important');
        el.style.setProperty('box-shadow','none','important');
      }
    });
  }
  strip();
  new MutationObserver(strip).observe(
    window.parent.document.body, {childList:true,subtree:true}
  );
})();
</script>""", height=0)

# JS: strip any blue background Streamlit runtime-injects into the bottom bar
st.components.v1.html("""<script>
(function nukeBlueBg() {
  function strip() {
    var bottom = window.parent.document.querySelector('[data-testid="stBottom"]');
    if (!bottom) return;
    var els = [bottom].concat(Array.from(bottom.querySelectorAll('*')));
    els.forEach(function(el) {
      var bg = window.getComputedStyle(el).backgroundColor;
      if (!bg || bg === 'rgba(0, 0, 0, 0)' || bg === 'transparent') return;
      var m = bg.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
      if (m && parseInt(m[3]) > parseInt(m[1]) && parseInt(m[3]) > parseInt(m[2])) {
        el.style.setProperty('background', 'transparent', 'important');
        el.style.setProperty('background-color', 'transparent', 'important');
        el.style.setProperty('box-shadow', 'none', 'important');
      }
    });
  }
  strip();
  new MutationObserver(strip).observe(window.parent.document.body, {childList:true, subtree:true});
})();
</script>""", height=0)


# =========================
# HELPERS
# =========================

def get_browser_language_code(language: str) -> str:
    cleaned = " ".join((language or "").strip().split())
    if not cleaned:
        return "en"
    lowered = cleaned.lower()
    if lowered in ANSWER_LANGUAGE_CODES:
        return ANSWER_LANGUAGE_CODES[lowered]
    if re.fullmatch(r"[a-z]{2,3}(?:-[a-z]{2,4})?", lowered):
        return lowered
    return ANSWER_LANGUAGE_CODES.get(lowered.split()[0], lowered)


def _milvus_settings() -> dict[str, str]:
    return {
        "host": os.environ.get("MILVUS_HOST", "localhost"),
        "port": os.environ.get("MILVUS_PORT", "19530"),
        "collection": os.environ.get("MILVUS_COLLECTION", "indian_legal_rag"),
    }


def _is_milvus_issue(message: str) -> bool:
    lowered = (message or "").lower()
    if "milvus" not in lowered and "pymilvus" not in lowered:
        return False
    return any(t in lowered for t in ("server unavailable","fail connecting","failed to initialize","19530","illegal connection params","missing dependency pymilvus"))


def _render_milvus_help(message: str) -> None:
    cfg = _milvus_settings()
    st.info(f"Milvus is configured for `{cfg['host']}:{cfg['port']}` but is unreachable.")
    with st.expander("Milvus Troubleshooting", expanded=True):
        st.markdown("Ensure Docker Desktop is running and the Milvus containers are up:")
        st.code("docker compose -f infra/docker-compose.yml up -d", language="powershell")
        st.caption(f"Error: {message}")


def _is_ollama_issue(message: str) -> bool:
    lowered = (message or "").lower()
    return any(t in lowered for t in ("unable to allocate cuda_host buffer","error loading model","model requires more system memory","ollama._types.responseerror","runner process has terminated","does not support generate"))


def _render_ollama_help(message: str, model_name: str) -> None:
    st.info(f"Host Ollama could not load `{model_name}`.")
    with st.expander("Ollama Troubleshooting", expanded=True):
        st.code(f'ollama run {model_name} "Reply with exactly: ok"', language="powershell")
        st.caption(f"Error: {message}")


def _status_snapshot(available_models: list[str], selected_model: str | None) -> dict:
    snap = {"index_count": None, "index_error": None, "manifest_exists": None,
            "models_available": len(available_models), "selected_model": selected_model}
    try:
        info = vector.index_status()
        snap["index_count"] = info.get("count")
        snap["manifest_exists"] = info.get("manifest_exists")
    except Exception as e:
        snap["index_error"] = str(e)
    return snap


def render_status_strip(status: dict) -> None:
    docs = status.get("index_count")
    docs_text = f"{docs:,} docs" if isinstance(docs, (int, float)) else "index unavailable"
    docs_cls = "ok" if docs and docs > 0 else ("bad" if status.get("index_error") else "warn")
    model_cls = "ok" if status.get("selected_model") and status.get("models_available") else "warn"
    model_text = status.get("selected_model") or "no model"
    st.markdown(f"""
<div class="nyay-status-strip">
  <div class="nyay-status-item">
    <div class="nyay-status-dot {docs_cls}"></div>
    <span class="nyay-status-label">Corpus</span>
    <span>{docs_text}</span>
  </div>
  <div class="nyay-status-item">
    <div class="nyay-status-dot {model_cls}"></div>
    <span class="nyay-status-label">Model</span>
    <span>{model_text}</span>
  </div>
  <div class="nyay-status-item">
    <div class="nyay-status-dot ok"></div>
    <span class="nyay-status-label">Vector DB</span>
    <span>{os.environ.get("MILVUS_HOST","localhost")}:{os.environ.get("MILVUS_PORT","19530")}</span>
  </div>
</div>
""", unsafe_allow_html=True)


def render_translated_answer(answer_markdown: str, target_language: str) -> None:
    target_language = " ".join((target_language or "").strip().split()) or "English"
    target_code = get_browser_language_code(target_language)
    if target_code == "en":
        st.markdown(answer_markdown)
        return
    rendered_html = MARKDOWN_RENDERER.render(answer_markdown)
    answer_html_js = json.dumps(rendered_html)
    target_code_js = json.dumps(target_code)
    target_language_js = json.dumps(target_language)
    component_html = f"""
    <html><head><style>
      :root {{ color-scheme: dark; }}
      body {{ margin:0; padding:0; background:transparent; color:#e8e0d0; font-family:'EB Garamond',Georgia,serif; }}
      #translation-status {{ margin:0 0 0.7rem 0; color:#5c534a; font-size:0.82rem; font-style:italic; }}
      #translated-answer {{ font-size:1rem; line-height:1.75; color:#e8e0d0; overflow-wrap:anywhere; }}
      #translated-answer p {{ margin:0.7rem 0; }}
      #translated-answer h1,#translated-answer h2,#translated-answer h3,#translated-answer h4 {{ margin:1rem 0 0.4rem 0; color:#c9a84c; }}
      #translated-answer code {{ font-family:'JetBrains Mono',monospace; background:rgba(201,168,76,0.08); padding:0.1em 0.35em; border-radius:2px; color:#c9a84c; font-size:0.88em; }}
      #translated-answer pre {{ padding:0.85rem; border-radius:2px; background:rgba(201,168,76,0.05); border-left:3px solid #8a6f2e; }}
      #google_translate_element,.goog-te-banner-frame,.skiptranslate {{ display:none!important; }}
      body {{ top:0!important; }}
    </style></head><body>
    <div id="translation-status">Translating to {html.escape(target_language)}...</div>
    <div id="translated-answer">{rendered_html}</div>
    <div id="google_translate_element"></div>
    <script>
      const englishHtml={answer_html_js};
      const targetLanguage={target_code_js};
      const targetLanguageLabel={target_language_js};
      const answer=document.getElementById("translated-answer");
      const status=document.getElementById("translation-status");
      function setFrameHeight(){{const h=Math.max(document.body.scrollHeight,document.documentElement.scrollHeight,answer.scrollHeight+24);window.parent.postMessage({{isStreamlitMessage:true,type:"streamlit:setFrameHeight",height:h+16}},"*");}}
      function candidateTextNodes(root){{const walker=document.createTreeWalker(root,NodeFilter.SHOW_TEXT,{{acceptNode(node){{if(!node.textContent||!node.textContent.trim())return NodeFilter.FILTER_REJECT;const p=node.parentElement;if(!p||p.closest("code,pre,script,style"))return NodeFilter.FILTER_REJECT;return NodeFilter.FILTER_ACCEPT;}}}});const out=[];while(walker.nextNode())out.push(walker.currentNode);return out;}}
      async function translateWithBrowserApi(){{if(!("Translator" in self)||!targetLanguage||targetLanguage==="en")return false;try{{const av=await Translator.availability({{sourceLanguage:"en",targetLanguage}});if(!av||av==="unavailable")return false;const t=await Translator.create({{sourceLanguage:"en",targetLanguage}});if(t.ready)try{{await t.ready;}}catch(e){{}}for(const node of candidateTextNodes(answer)){{const s=node.textContent||"";const pre=(s.match(/^\s*/)||[""])[0];const suf=(s.match(/\s*$/)||[""])[0];const core=s.trim();if(!core||/^\[Source\s+\d+\]$/.test(core))continue;const tr=await t.translate(core);node.textContent=`${{pre}}${{tr}}${{suf}}`;}}status.textContent=`Translated in-browser to ${{targetLanguageLabel}}.`;setFrameHeight();return true;}}catch(e){{return false;}}}}
      function loadGoogleWidget(){{return new Promise(resolve=>{{if(typeof google!=="undefined"&&google.translate&&google.translate.TranslateElement){{resolve(true);return;}}window.googleTranslateElementInit=function(){{new google.translate.TranslateElement({{pageLanguage:"en",autoDisplay:false,includedLanguages:targetLanguage,multilanguagePage:true}},"google_translate_element");resolve(true);}};const s=document.createElement("script");s.src="https://translate.google.com/translate_a/element.js?cb=googleTranslateElementInit";s.async=true;s.onerror=()=>resolve(false);document.head.appendChild(s);}});}}
      async function translateWithGoogleWidget(){{if(!targetLanguage||targetLanguage==="en")return false;try{{const ready=await loadGoogleWidget();if(!ready)return false;let attempts=0;return await new Promise(resolve=>{{const timer=setInterval(()=>{{const combo=document.querySelector(".goog-te-combo");if(combo){{combo.value=targetLanguage;combo.dispatchEvent(new Event("change"));clearInterval(timer);status.textContent=`Translated via Google to ${{targetLanguageLabel}}.`;setTimeout(()=>{{setFrameHeight();resolve(true);}},1200);return;}}attempts++;if(attempts>32){{clearInterval(timer);resolve(false);}}}},250);}});}}catch(e){{return false;}}}}
      (async function boot(){{if(!targetLanguage||targetLanguage==="en"){{status.textContent="Showing English answer.";setFrameHeight();return;}}const ok=await translateWithBrowserApi();if(ok)return;answer.innerHTML=englishHtml;const ok2=await translateWithGoogleWidget();if(!ok2){{answer.innerHTML=englishHtml;status.textContent="Translation unavailable. Showing English.";setFrameHeight();}}}})();
      window.addEventListener("load",setFrameHeight);new ResizeObserver(setFrameHeight).observe(document.body);
    </script></body></html>
    """
    components.html(component_html, height=360)


# =========================
# MODEL
# =========================

@st.cache_resource
def load_model(model_name: str, temperature: float, num_ctx: int, num_thread: int, top_p: float):
    return OllamaLLM(
        model=model_name, temperature=temperature, num_ctx=num_ctx,
        num_thread=num_thread, top_p=top_p, base_url=_ollama_base_url(),
    )


def _groq_available() -> bool:
    if not _GROQ_AVAILABLE:
        return False
    return bool(
        os.environ.get("GROQ_API_KEY") or
        (hasattr(st, "secrets") and st.secrets.get("GROQ_API_KEY"))
    )


def groq_answer(query: str, model_name: str, temperature: float) -> str:
    api_key = os.environ.get("GROQ_API_KEY") or (st.secrets.get("GROQ_API_KEY") if hasattr(st, "secrets") else None)
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")
    client = Groq(api_key=api_key)
    system = ("You are a grounded legal assistant. Use only the supplied context. "
              "Do not include hidden reasoning like <think>. "
              "If unsure, reply exactly: Corpus unavailable.")
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": query}],
        temperature=float(temperature),
    )
    raw = (resp.choices[0].message.content or "").strip()
    return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()


class GroqLLMAdapter:
    def __init__(self, model_name: str, temperature: float):
        self.model_name = model_name
        self.temperature = temperature
    def invoke(self, prompt: str) -> str:
        return groq_answer(prompt, self.model_name, self.temperature)


# =========================
# VOICE INPUT
# =========================

def voice_input_component(target_placeholder: str, lang: str = "en-IN") -> None:
    placeholder_js = json.dumps(target_placeholder)
    lang_js = json.dumps(lang)
    component_html = f"""
    <div style="font-family:'EB Garamond',Georgia,serif; display:flex; align-items:center; gap:10px;">
      <button id="btn" type="button" style="padding:7px 18px; border-radius:2px; border:1px solid rgba(201,168,76,0.35); background:transparent; cursor:pointer; font-weight:600; color:#c9a84c; font-family:inherit; font-size:0.9rem; letter-spacing:0.06em; transition:all 140ms ease;">
        ◉ Mic
      </button>
      <span id="status" style="color:#5c534a; font-size:0.82rem; font-style:italic;">Ready</span>
    </div>
    <script>
      const btn=document.getElementById("btn");
      const status=document.getElementById("status");
      const targetPlaceholder={placeholder_js};
      const targetLang={lang_js};
      function supported(){{return ('webkitSpeechRecognition' in window)||('SpeechRecognition' in window);}}
      function findTargetBox(){{const parentDoc=window.parent.document;const textareas=Array.from(parentDoc.querySelectorAll("textarea"));return(textareas.find(el=>(el.getAttribute("placeholder")||"")===targetPlaceholder)||parentDoc.querySelector('div[data-testid="stChatInput"] textarea')||textareas.find(el=>!el.disabled));}}
      function setControlledValue(el,value){{const p=Object.getPrototypeOf(el);const d=Object.getOwnPropertyDescriptor(p,"value")||Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype,"value");if(d&&d.set)d.set.call(el,value);else el.value=value;el.dispatchEvent(new Event("input",{{bubbles:true}}));el.dispatchEvent(new Event("change",{{bubbles:true}}));el.focus();}}
      function appendTranscript(text){{const box=findTargetBox();if(!box){{status.textContent="No box";return;}}const existing=(box.value||"").trim();setControlledValue(box,existing?`${{existing}} ${{text}}`:text);status.textContent="Added ✓";}}
      btn.onclick=()=>{{if(!supported()){{status.textContent="Unsupported";return;}}const SR=window.SpeechRecognition||window.webkitSpeechRecognition;const rec=new SR();rec.lang=targetLang;rec.interimResults=true;rec.maxAlternatives=1;rec.continuous=false;btn.style.borderColor="rgba(201,168,76,0.8)";btn.style.color="#e8c96a";status.textContent="Listening…";rec.onresult=e=>{{const r=e.results[e.resultIndex];if(!r||!r[0])return;if(r.isFinal)appendTranscript(r[0].transcript);}};rec.onerror=()=>{{status.textContent="Error";btn.style.borderColor="rgba(201,168,76,0.35)";btn.style.color="#c9a84c";}};rec.onend=()=>{{btn.style.borderColor="rgba(201,168,76,0.35)";btn.style.color="#c9a84c";if(status.textContent==="Listening…")status.textContent="Ready";}};rec.start();}};
    </script>
    """
    components.html(component_html, height=52)


def transcribe_audio_openai(audio_bytes: bytes) -> str:
    try:
        from openai import OpenAI
    except Exception:
        return ""
    api_key = st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None
    if not api_key:
        return ""
    import io
    client = OpenAI(api_key=api_key)
    f = io.BytesIO(audio_bytes)
    f.name = "question.wav"
    try:
        resp = client.audio.transcriptions.create(model="whisper-1", file=f)
        return (resp.text or "").strip()
    except Exception:
        return ""


# =========================
# SIDEBAR
# =========================

with st.sidebar:
    st.markdown('<h3>NyayGram</h3>', unsafe_allow_html=True)

    ollama_base_url = _normalized_ollama_base_url()
    available_models = list_ollama_models(ollama_base_url)
    preferred_model = pick_preferred_model(available_models)

    st.markdown('<h3>Session</h3>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("New Chat", use_container_width=True):
            st.session_state.chat = []
            st.session_state.last_audio_hash = ""
    with col_b:
        if st.button("Rebuild Index", help="Clears and rebuilds the vector DB.", use_container_width=True):
            with st.spinner("Rebuilding…"):
                try:
                    vector.rebuild_index()
                    st.success("Index rebuilt.")
                except Exception as e:
                    st.error(f"Rebuild failed: {e}")

    st.divider()

    st.markdown('<h3>Provider</h3>', unsafe_allow_html=True)
    provider = st.radio("Model provider", options=["Ollama", "Groq"], horizontal=True, index=0, label_visibility="collapsed")
    show_sources = st.checkbox("Show sources", value=False)

    st.markdown('<h3>Answer</h3>', unsafe_allow_html=True)
    answer_language_mode = st.selectbox("Language", options=["English", "Hindi", "Custom"], index=0, label_visibility="collapsed")
    custom_answer_language = ""
    if answer_language_mode == "Custom":
        custom_answer_language = st.text_input("Language code", placeholder="e.g. Bengali or bn", label_visibility="collapsed")
    st.caption("Generated in English, translated in-browser.")

    st.markdown('<h3>Index</h3>', unsafe_allow_html=True)
    if st.checkbox("Show index status", value=False):
        try:
            status_info = vector.index_status()
            if status_info.get("count_error"):
                err = str(status_info["count_error"])
                if _is_milvus_issue(err):
                    st.error("Milvus unreachable.")
                    _render_milvus_help(err)
                else:
                    st.error(f"Index error: {err}")
            else:
                st.write(f"Documents: `{status_info.get('count')}`")
        except Exception as e:
            st.error(f"Status unavailable: {e}")

    st.markdown('<h3>Voice</h3>', unsafe_allow_html=True)
    speech_label = st.selectbox("Speech language", options=["English (India)", "Hindi (India)"], index=0, label_visibility="collapsed")
    enable_voice = st.checkbox("Browser mic", value=True)
    enable_audio_transcribe = st.checkbox("Server transcription (OpenAI)", value=False)

    with st.expander("Model settings", expanded=False):
        if provider == "Ollama" and available_models:
            current_model = st.session_state.get("selected_model")
            if current_model and current_model not in available_models:
                st.session_state["selected_model"] = preferred_model
            model_name = st.selectbox("Model", options=available_models, key="selected_model")
            st.caption(f"Ollama: `{ollama_base_url}`")
        elif provider == "Ollama":
            model_name = st.text_input("Model", value=DEFAULT_MODEL_NAME)
            st.caption(f"`{ollama_base_url}` — model list unavailable")
        else:
            model_name = st.text_input("Groq model", value=DEFAULT_GROQ_MODEL, key="groq_model")
            st.caption("Requires GROQ_API_KEY")
        temperature = st.slider("Temperature", 0.0, 0.8, 0.1, 0.05)
        if provider == "Ollama":
            num_ctx = st.select_slider("Context window", options=[2048, 4096, 8192], value=2048)
            num_thread = st.slider("Threads", 1, 12, 4, 1)
            top_p = st.slider("Top-p", 0.1, 1.0, 0.85, 0.05)


# Fallback locals
model_name      = locals().get("model_name", DEFAULT_MODEL_NAME if locals().get("provider","Ollama")=="Ollama" else DEFAULT_GROQ_MODEL)
temperature     = locals().get("temperature", 0.1)
num_ctx         = locals().get("num_ctx", 2048)
num_thread      = locals().get("num_thread", 4)
top_p           = locals().get("top_p", 0.85)
enable_audio_transcribe = locals().get("enable_audio_transcribe", False)
enable_voice    = locals().get("enable_voice", False)
speech_label    = locals().get("speech_label", "English (India)")
answer_language_mode = locals().get("answer_language_mode", "English")
custom_answer_language = locals().get("custom_answer_language", "")

selected_answer_language = custom_answer_language.strip() if answer_language_mode == "Custom" else answer_language_mode
if not selected_answer_language:
    selected_answer_language = "English"

speech_lang_map = {"English (India)": "en-IN", "Hindi (India)": "hi-IN"}
selected_speech_lang = speech_lang_map.get(speech_label, "en-IN")

model = None
if provider == "Ollama":
    try:
        model = load_model(model_name, temperature, num_ctx, num_thread, top_p)
    except Exception as e:
        st.error(f"Could not load model `{model_name}`: {e}")
        st.info("Pick another model in the sidebar or ensure host Ollama is running.")


# =========================
# STATUS
# =========================

status_snapshot = _status_snapshot(available_models, model_name)
render_status_strip(status_snapshot)


# =========================
# SESSION STATE
# =========================

if "chat" not in st.session_state:
    st.session_state.chat = []
if "last_audio_hash" not in st.session_state:
    st.session_state.last_audio_hash = ""
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
if "processing" not in st.session_state:
    st.session_state.processing = False


def _queue_question(user_query: str) -> None:
    user_query = (user_query or "").strip()
    if not user_query or st.session_state.pending_question:
        return
    st.session_state.pending_question = user_query
    st.session_state.chat.append({"question": user_query, "answer": None, "docs": [], "pending": True})
    st.rerun()


def _process_pending_question() -> None:
    if st.session_state.pending_question is None or st.session_state.processing:
        return

    user_query = st.session_state.pending_question
    st.session_state.processing = True
    request_id = uuid.uuid4().hex[:12]
    logger.info("Query received request_id=%s", request_id)

    try:
        vector.ensure_ingested()
    except Exception as e:
        error_text = str(e)
        if _is_milvus_issue(error_text):
            st.error("Milvus unavailable — cannot process query.")
            _render_milvus_help(error_text)
        else:
            st.error(f"Index initialization failed: {error_text}")
        st.session_state.processing = False
        return

    try:
        start = time.perf_counter()
        if provider == "Groq":
            groq_llm = GroqLLMAdapter(model_name, temperature)
            answer, docs = answer_query(user_query, groq_llm)
        else:
            if model is None:
                st.error("No model loaded. Pick a model in the sidebar.")
                st.session_state.processing = False
                return
            answer, docs = answer_query(user_query, model)
        duration_ms = int((time.perf_counter() - start) * 1000)
        docs_to_store = docs or []

        pending_idx = next((i for i in reversed(range(len(st.session_state.chat))) if st.session_state.chat[i].get("pending")), None)
        payload = {"question": user_query, "answer": answer, "docs": docs_to_store, "pending": False}
        if pending_idx is not None:
            st.session_state.chat[pending_idx] = payload
        else:
            st.session_state.chat.append(payload)

        _persist_response({"ts": datetime.utcnow().isoformat()+"Z", "request_id": request_id,
                           "provider": provider, "model": model_name, "question": user_query,
                           "answer": answer, "duration_ms": duration_ms, "docs_count": len(docs_to_store)})
        logger.info("Answer generated request_id=%s duration_ms=%s", request_id, duration_ms)

    except Exception as e:
        error_text = str(e)
        if provider == "Ollama" and _is_ollama_issue(error_text) and _groq_available():
            st.warning("Ollama failed — trying Groq fallback…")
            try:
                answer = groq_answer(user_query, DEFAULT_GROQ_MODEL, temperature)
                pending_idx = next((i for i in reversed(range(len(st.session_state.chat))) if st.session_state.chat[i].get("pending")), None)
                payload = {"question": user_query, "answer": answer, "docs": [], "pending": False}
                if pending_idx is not None:
                    st.session_state.chat[pending_idx] = payload
                else:
                    st.session_state.chat.append(payload)
            except Exception as e2:
                st.error(f"Both Ollama and Groq failed: {e2}")
        elif _is_ollama_issue(error_text):
            st.error(f"Ollama error loading `{model_name}`.")
            _render_ollama_help(error_text, model_name)
        elif _is_milvus_issue(error_text):
            st.error("Milvus unavailable.")
            _render_milvus_help(error_text)
        else:
            st.error(f"Query failed: {error_text}")
        # Remove stuck pending bubble so it doesn't linger on error
        _pending_idx = next(
            (i for i in reversed(range(len(st.session_state.chat)))
             if st.session_state.chat[i].get("pending")), None
        )
        if _pending_idx is not None:
            st.session_state.chat.pop(_pending_idx)
        logger.exception("Query failed request_id=%s", request_id)
    finally:
        _did_resolve = st.session_state.pending_question is not None
        st.session_state.pending_question = None
        st.session_state.processing = False
        # Only rerun if we actually resolved a pending question — avoids extra flicker on no-op calls
        if _did_resolve:
            st.rerun()


# =========================
# HERO + STARTERS
# =========================

starter_prompts = [
    "Give me an overview of the Indian Contract Act, 1872.",
    "What does Section 27 of the Indian Contract Act say about restraint of trade?",
    "Summarize the main ideas in the Bharatiya Sakshya Adhiniyam.",
]

if not st.session_state.chat and not st.session_state.get('pending_question'):
    st.markdown("""
<div class="nyay-hero">
  <div class="nyay-eyebrow">Indian Legal Intelligence</div>
  <div class="nyay-title">Nyay<em>Gram</em></div>
  <div class="nyay-subtitle">
    Interrogate the legal corpus. Retrieve statutes, sections, and case doctrine
    with precision — in the language of your choice.
  </div>
</div>
""", unsafe_allow_html=True)
    st.markdown('<div class="nyay-prompts-label">Begin with a question</div>', unsafe_allow_html=True)
    starter_cols = st.columns(len(starter_prompts))
    for idx, (col, prompt) in enumerate(zip(starter_cols, starter_prompts), start=1):
        with col:
            if st.button(prompt, key=f"starter_{idx}", use_container_width=True):
                _queue_question(prompt)


# =========================
# RENDER CHAT
# =========================

for idx, turn in enumerate(st.session_state.chat, start=1):
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        if turn.get("pending"):
            st.markdown(
                '<p style="color:var(--text-muted);font-style:italic;font-family:EB Garamond,serif;margin:0.4rem 0;">'
                '&#x27F3; Consulting the corpus…</p>',
                unsafe_allow_html=True,
            )
        else:
            render_translated_answer(turn["answer"], selected_answer_language)
        if show_sources and turn.get("docs") and not turn.get("pending"):
            docs = turn["docs"]
            with st.expander(f"Sources ({len(docs)})", expanded=False):
                for i, d in enumerate(docs, 1):
                    meta = d.metadata or {}
                    parts = []
                    if meta.get("act"):     parts.append(f"Act: {meta['act']}")
                    if meta.get("section"): parts.append(f"§ {meta['section']}")
                    if meta.get("source"):  parts.append(f"Source: {meta['source']}")
                    st.markdown(f"**{i}. {' · '.join(parts) if parts else 'Unknown Source'}**")
                    with st.expander("Show excerpt", expanded=False):
                        st.code(d.page_content[:2000])


# =========================
# INPUT ROW — hidden while a query is in-flight to avoid double rendering
# =========================

if (enable_voice or enable_audio_transcribe) and not st.session_state.get("pending_question"):
    toolbar_cols = st.columns([1.2, 2.8, 2.0])
    audio = None
    with toolbar_cols[0]:
        if enable_voice:
            voice_input_component(CHAT_INPUT_PLACEHOLDER, lang=selected_speech_lang)
    with toolbar_cols[1]:
        if enable_voice:
            st.markdown('<div class="toolbar-note">Mic transcribes locally — no audio sent to server.</div>', unsafe_allow_html=True)
    with toolbar_cols[2]:
        if enable_audio_transcribe:
            audio = st.audio_input("Record", key="audio_input", label_visibility="collapsed")
    if enable_audio_transcribe and audio is not None:
        import hashlib
        audio_bytes = audio.getvalue()
        h = hashlib.sha256(audio_bytes).hexdigest()
        if h != st.session_state.last_audio_hash:
            st.session_state.last_audio_hash = h
            with st.spinner("Transcribing…"):
                transcript = transcribe_audio_openai(audio_bytes)
            if transcript:
                _queue_question(transcript)
            else:
                st.warning("Transcription unavailable — set OPENAI_API_KEY in secrets.")

user_query = st.chat_input(CHAT_INPUT_PLACEHOLDER, key=CHAT_INPUT_KEY)
if user_query:
    _queue_question(user_query)

_process_pending_question()