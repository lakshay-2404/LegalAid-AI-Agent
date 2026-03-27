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

# =========================
# CSS — NATIVE STREAMLIT AESTHETIC
# =========================

def inject_css() -> None:
    # A tiny amount of CSS to keep the chat max-width clean
    st.markdown("""
<style>
  .block-container {
    max-width: 880px !important;
    padding-top: 2rem !important;
    padding-bottom: 7rem !important;
  }
  div[data-testid="stChatMessage"] {
    max-width: 840px !important;
    margin-left: auto !important;
    margin-right: auto !important;
  }
</style>
""", unsafe_allow_html=True)


inject_css()


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


# Status strip removed for minimalism

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
    client = Groq(api_key=api_key, max_retries=1)
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
    <style>
      body {{ margin: 0; padding: 0; background: transparent; font-family: sans-serif; color-scheme: light dark; }}
      #mic-container {{ display: flex; align-items: center; justify-content: flex-start; gap: 12px; }}
      button {{ width: 38px; height: 38px; padding: 0; border-radius: 50%; border: 1px solid gray; background: var(--background-color, transparent); cursor: pointer; color: inherit; font-size: 1.1rem; transition: background 0.2s; display: flex; align-items: center; justify-content: center; }}
      button:hover {{ background: rgba(128,128,128,0.2); }}
      span {{ color: gray; font-size: 0.85rem; white-space: nowrap; }}
    </style>
    <div id="mic-container">
      <button id="btn" type="button" title="Use Microphone">🎤</button>
      <span id="status">Click to speak</span>
    </div>
    <script>
      const btn=document.getElementById("btn");
      const status=document.getElementById("status");
      const targetPlaceholder={placeholder_js};
      const targetLang={lang_js};
      function supported(){{return ('webkitSpeechRecognition' in window)||('SpeechRecognition' in window);}}
      function findTargetBox(){{const parentDoc=window.parent.document;const textareas=Array.from(parentDoc.querySelectorAll("textarea"));return(textareas.find(el=>(el.getAttribute("placeholder")||"")===targetPlaceholder)||parentDoc.querySelector('div[data-testid="stChatInput"] textarea')||textareas.find(el=>!el.disabled));}}
      function setControlledValue(el,value){{const p=Object.getPrototypeOf(el);const d=Object.getOwnPropertyDescriptor(p,"value")||Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype,"value");if(d&&d.set)d.set.call(el,value);else el.value=value;el.dispatchEvent(new Event("input",{{bubbles:true}}));el.dispatchEvent(new Event("change",{{bubbles:true}}));el.focus();}}
      function appendTranscript(text){{const box=findTargetBox();if(!box){{status.textContent="No input box found";return;}}const existing=(box.value||"").trim();setControlledValue(box,existing?`${{existing}} ${{text}}`:text);status.textContent="Added ✓"; setTimeout(()=>{{if(status.textContent==="Added ✓")status.textContent="";}},2000);}}
      btn.onclick=()=>{{if(!supported()){{status.textContent="Unsupported";return;}}const SR=window.SpeechRecognition||window.webkitSpeechRecognition;const rec=new SR();rec.lang=targetLang;rec.interimResults=true;rec.maxAlternatives=1;rec.continuous=false;btn.style.borderColor="currentColor";status.textContent="Listening…";rec.onresult=e=>{{const r=e.results[e.resultIndex];if(!r||!r[0])return;if(r.isFinal)appendTranscript(r[0].transcript);}};rec.onerror=()=>{{status.textContent="Error";btn.style.borderColor="gray";setTimeout(()=>status.textContent="",2000);}};rec.onend=()=>{{btn.style.borderColor="gray";if(status.textContent==="Listening…")status.textContent="";}};rec.start();}};
    </script>
    """
    components.html(component_html, height=45)


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
    st.title("NyayGram")
    st.caption("Legal Intelligence")

    ollama_base_url = _normalized_ollama_base_url()
    available_models = list_ollama_models(ollama_base_url)
    preferred_model = pick_preferred_model(available_models)

    if st.button("New Chat", type="primary", use_container_width=True):
        st.session_state.chat = []
        st.session_state.last_audio_hash = ""

    st.divider()

    provider = st.radio("Model Provider", options=["Ollama", "Groq"], horizontal=True, label_visibility="collapsed")
    
    if provider == "Ollama" and available_models:
        current_model = st.session_state.get("selected_model")
        if current_model and current_model not in available_models:
            st.session_state["selected_model"] = preferred_model
        model_name = st.selectbox("Model", options=available_models, key="selected_model", label_visibility="collapsed")
    elif provider == "Ollama":
        model_name = st.text_input("Model", value=DEFAULT_MODEL_NAME, label_visibility="collapsed")
    else:
        model_name = st.text_input("Groq model", value=DEFAULT_GROQ_MODEL, key="groq_model", label_visibility="collapsed")

    with st.expander("Advanced Settings"):
        st.markdown("**Configuration**")
        show_sources = st.checkbox("Show sources inline", value=False)
        answer_language_mode = st.selectbox("Response Auto-Translate", options=["English", "Hindi", "Custom"], index=0)
        custom_answer_language = st.text_input("Custom Language Code", placeholder="bn") if answer_language_mode == "Custom" else ""
        
        st.markdown("**Model Hypertuning**")
        temperature = st.slider("Temperature", 0.0, 0.8, 0.1, 0.05)
        num_ctx = st.select_slider("Context window", options=[2048, 4096, 8192], value=2048) if provider == "Ollama" else 2048
        num_thread = st.slider("Threads", 1, 12, 4, 1) if provider == "Ollama" else 4
        top_p = st.slider("Top-p", 0.1, 1.0, 0.85, 0.05) if provider == "Ollama" else 0.85

        st.markdown("**Voice Native Sync**")
        speech_label = st.selectbox("Speech Input", options=["English (India)", "Hindi (India)"], index=0)
        enable_voice = st.checkbox("Enable Browser Mic", value=True)
        enable_audio_transcribe = st.checkbox("Server Transcription (OpenAI API)", value=False)

        st.markdown("**Index & DB**")
        if st.button("Check Index Sync", use_container_width=True):
            try:
                info = vector.index_status()
                st.write(f"{info.get('count', 0):,} chunks in DB." if not info.get("count_error") else f"Error: {info['count_error']}")
            except Exception as e:
                st.error(str(e))
        if st.button("Rebuild Index", use_container_width=True):
            try:
                vector.rebuild_index()
                st.success("Rebuilt")
            except Exception as e:
                st.error(str(e))

    st.divider()
    st.markdown("**Voice Area**")
    
    selected_speech_lang = {"English (India)": "en-IN", "Hindi (India)": "hi-IN"}.get(speech_label, "en-IN")
    if enable_voice and not st.session_state.get("pending_question"):
        voice_input_component(CHAT_INPUT_PLACEHOLDER, lang=selected_speech_lang)

    if enable_audio_transcribe and not st.session_state.get("pending_question"):
        audio = st.audio_input("Record", key="audio_input", label_visibility="collapsed")
        if audio is not None:
            import hashlib
            audio_bytes = audio.getvalue()
            h = hashlib.sha256(audio_bytes).hexdigest()
            if h != st.session_state.last_audio_hash:
                st.session_state.last_audio_hash = h
                with st.spinner("Transcribing…"):
                    transcript = transcribe_audio_openai(audio_bytes)
                if transcript:
                    _queue_question(transcript)
                    st.rerun()
                else:
                    st.warning("Transcription unavailable.")


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

# Status strip disabled


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

def _on_chat_submit() -> None:
    val = st.session_state.get(CHAT_INPUT_KEY)
    if val:
        _queue_question(val)


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
    st.title("NyayGram")
    st.caption("Interrogate the legal corpus. Retrieve statutes, sections, and case doctrine with precision.")
    st.divider()
    
    st.markdown("**Starter Questions:**")
    for idx, prompt in enumerate(starter_prompts, start=1):
        if st.button(prompt, key=f"starter_{idx}"):
            _queue_question(prompt)
            st.rerun()



# =========================
# RENDER CHAT
# =========================

for idx, turn in enumerate(st.session_state.chat, start=1):
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        if turn.get("pending"):
            with st.spinner("Consulting the corpus..."):
                st.empty()
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

# Voice input moved to sidebar
st.chat_input(CHAT_INPUT_PLACEHOLDER, key=CHAT_INPUT_KEY, on_submit=_on_chat_submit)

_process_pending_question()
