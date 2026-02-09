import json
import streamlit as st
from langchain_ollama.llms import OllamaLLM
import streamlit.components.v1 as components

from rag_core import answer_query
import vector

# =========================
# STREAMLIT CONFIG
# =========================

st.set_page_config(page_title="NyayGram", layout="wide", initial_sidebar_state="expanded")


# NOTE: Color/theme overrides (CSS) were causing readability issues in the sidebar
# on some Streamlit versions. Per your request, we keep the same UI/layout but
# revert to Streamlit's default color scheme.
#
# If you want to re-enable theming later, uncomment this block and tune the CSS
# carefully (especially sidebar label opacity/contrast).
#
# def inject_minimal_css() -> None:
#     st.markdown(
#         """
# <style>
#   :root{
#     --bg: #f7f7f9;
#     --panel: #ffffff;
#     --text: #111827;
#     --muted: #4b5563;
#     --border: #e5e7eb;
#     --shadow: rgba(17,24,39,0.06);
#     --accent: #111827;
#     --accentSoft: rgba(17,24,39,0.06);
#   }
#
#   html, body, .stApp {
#     background: var(--bg) !important;
#     color: var(--text) !important;
#   }
#   /* ... CSS intentionally kept for reference ... */
# </style>
# """,
#         unsafe_allow_html=True,
#     )
#
# inject_minimal_css()

st.markdown("## Nyaygram")
st.caption("Precision-first answers from ingested legal documents, with clear provenance/confidence.")

# =========================
# MODEL
# =========================

@st.cache_resource
def load_model(model_name: str, temperature: float, num_ctx: int, num_thread: int, top_p: float):
    return OllamaLLM(
        model=model_name,
        temperature=temperature,
        num_ctx=num_ctx,
        num_thread=num_thread,
        top_p=top_p,
    )

# =========================
# VOICE INPUT (BROWSER SPEECH API)
# =========================

def voice_input_component(key: str = "voice") -> str:
    """
    Client-side speech-to-text using the browser's Web Speech API.
    No additional Python deps; works best in Chrome/Edge.
    Returns transcript string (or empty).
    """
    html = f"""
    <div style="font-family: system-ui, -apple-system, Segoe UI, sans-serif;">
      <button id="btn" style="padding:8px 12px; border-radius:10px; border:1px solid #ccc; background:#fff; cursor:pointer;">
        Speak
      </button>
      <span id="status" style="margin-left:10px; color:#444;">Idle</span>
      <div id="out" style="margin-top:6px; color:#111; font-size:14px;"></div>
    </div>
    <script>
      const btn = document.getElementById("btn");
      const status = document.getElementById("status");
      const out = document.getElementById("out");

      function sendValue(v) {{
        // Streamlit component protocol (works inside components.html)
        window.parent.postMessage({{
          isStreamlitMessage: true,
          type: "streamlit:setComponentValue",
          value: v
        }}, "*");
      }}

      function supported() {{
        return ('webkitSpeechRecognition' in window) || ('SpeechRecognition' in window);
      }}

      btn.onclick = () => {{
        if (!supported()) {{
          status.textContent = "SpeechRecognition not supported in this browser.";
          return;
        }}

        const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
        const rec = new SR();
        rec.lang = "en-IN";
        rec.interimResults = false;
        rec.maxAlternatives = 1;

        status.textContent = "Listening...";
        rec.onresult = (e) => {{
          const t = e.results[0][0].transcript;
          out.textContent = t;
          status.textContent = "Captured";
          sendValue(t);
        }};
        rec.onerror = (e) => {{
          status.textContent = "Error: " + e.error;
        }};
        rec.onend = () => {{
          if (status.textContent === "Listening...") {{
            status.textContent = "Stopped";
          }}
        }};
        rec.start();
      }};
    </script>
    """
    # streamlit.components.v1.html does NOT support returning values to Python.
    # This UI is kept as a convenience for users to see the transcript and manually paste it.
    # Previous attempt (kept for reference) expected a return value and used `key=...`:
    # value = components.html(html, height=80, key=key)
    components.html(html, height=80)
    return ""


def transcribe_audio_openai(audio_bytes: bytes) -> str:
    """
    Optional server-side transcription using OpenAI Whisper API.
    Enabled only if the `openai` package is installed and OPENAI_API_KEY is set.
    """
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return ""

    api_key = st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None
    if not api_key:
        api_key = st.session_state.get("OPENAI_API_KEY") or None
    if not api_key:
        return ""

    client = OpenAI(api_key=api_key)
    # Streamlit provides bytes; OpenAI expects a file-like object with a name.
    import io
    f = io.BytesIO(audio_bytes)
    f.name = "question.wav"
    try:
        resp = client.audio.transcriptions.create(model="whisper-1", file=f)
        return (resp.text or "").strip()
    except Exception:
        return ""

# =========================
# SIDEBAR SETTINGS
# =========================

with st.sidebar:
    st.subheader("Controls")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("New Chat", use_container_width=True):
            st.session_state.chat = []
    with col_b:
        if st.button(
            "Rebuild Index",
            help="Clears and rebuilds the vector DB. This can take a while.",
            use_container_width=True,
        ):
            with st.spinner("Rebuilding index..."):
                try:
                    vector.rebuild_index()
                    st.success("Index rebuild complete.")
                except Exception as e:
                    st.error(f"Rebuild failed: {e}")

    st.divider()
    show_sources = st.checkbox("Show Sources", value=False, help="Show the retrieved snippets used for the answer.")

    st.subheader("Index")
    try:
        status = vector.index_status()
        if status.get("count_error"):
            st.error(f"Index error: {status['count_error']}")
        else:
            st.write(f"Documents: `{status.get('count')}`")
        st.caption(f"DB: {status.get('db_dir')}")
    except Exception as e:
        st.error(f"Status unavailable: {e}")

    st.subheader("Voice")
    enable_audio_transcribe = st.checkbox("Audio transcription (server-side)", value=False)
    if enable_audio_transcribe:
        st.caption("Set OPENAI_API_KEY in Streamlit secrets to enable transcription.")

    # Browser Speech API: kept available, but it's manual (Streamlit can't receive the transcript).
    enable_voice = st.checkbox("Browser speech (manual transcript)", value=False)
    if enable_voice:
        voice_input_component(key="voice_input")
        st.caption("Copy the transcript into the chat box below.")

    with st.expander("Model (Advanced)", expanded=False):
        model_name = st.text_input("Model", value="llama3.1:8b")
        temperature = st.slider("Temperature", 0.0, 0.8, 0.1, 0.05)
        num_ctx = st.select_slider("Context Window", options=[2048, 4096, 8192], value=4096)
        num_thread = st.slider("Threads", 1, 12, 4, 1)
        top_p = st.slider("Top-p", 0.1, 1.0, 0.85, 0.05)

# Backward compatibility for variables if users collapse advanced sections.
model_name = locals().get("model_name", "llama3.1:8b")
temperature = locals().get("temperature", 0.1)
num_ctx = locals().get("num_ctx", 4096)
num_thread = locals().get("num_thread", 4)
top_p = locals().get("top_p", 0.85)
enable_audio_transcribe = locals().get("enable_audio_transcribe", False)
enable_voice = locals().get("enable_voice", False)

model = load_model(model_name, temperature, num_ctx, num_thread, top_p)

# =========================
# SESSION STATE
# =========================

if "chat" not in st.session_state:
    st.session_state.chat = []
if "queued_query" not in st.session_state:
    st.session_state.queued_query = ""
if "last_audio_hash" not in st.session_state:
    st.session_state.last_audio_hash = ""

def _append_and_answer(user_query: str) -> None:
    user_query = (user_query or "").strip()
    if not user_query:
        return

    with st.spinner("Thinking..."):
        try:
            answer, docs = answer_query(user_query, model)
            st.session_state.chat.append({"question": user_query, "answer": answer, "docs": docs})
            # Force a rerun so the newly appended messages render immediately.
            st.rerun()
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")


# Optional: audio transcription triggers a queued query automatically
if enable_audio_transcribe:
    audio = st.audio_input("Voice question (record)", key="audio_input")
    if audio is not None:
        audio_bytes = audio.getvalue()
        import hashlib

        h = hashlib.sha256(audio_bytes).hexdigest()
        if h != st.session_state.last_audio_hash:
            st.session_state.last_audio_hash = h
            with st.spinner("Transcribing audio..."):
                transcript = transcribe_audio_openai(audio_bytes)
            if transcript:
                st.session_state.queued_query = transcript
            else:
                st.warning("Audio transcription unavailable (missing OPENAI_API_KEY or openai package).")

# =========================
# RENDER CHAT
# =========================

for turn in st.session_state.chat:
    with st.chat_message("user"):
        st.write(turn["question"])

    with st.chat_message("assistant"):
        st.markdown(turn["answer"])

        if show_sources and turn.get("docs"):
            docs = turn["docs"]
            with st.expander(f"Sources ({len(docs)})", expanded=False):
                for i, d in enumerate(docs, 1):
                    meta = d.metadata or {}
                    title_parts = []
                    if meta.get("act"):
                        title_parts.append(f"Act: {meta['act']}")
                    if meta.get("section"):
                        title_parts.append(f"Section: {meta['section']}")
                    if meta.get("source"):
                        title_parts.append(f"Source: {meta['source']}")

                    st.markdown(f"**{i}. {' | '.join(title_parts) if title_parts else 'Unknown Source'}**")
                    with st.expander("Show text", expanded=False):
                        st.code(d.page_content[:2000])

# =========================
# INPUT
# =========================

# If audio queued a transcript, send it automatically once.
if st.session_state.queued_query:
    q = st.session_state.queued_query
    st.session_state.queued_query = ""
    _append_and_answer(q)

user_query = st.chat_input("Ask about an Act, section, or case…")
if user_query:
    _append_and_answer(user_query)
 
# =========================
# Previous UI (kept for reference)
# =========================
# The original minimal UI code is intentionally kept (commented out) per your request
# to avoid removing lines you may want to revert to.
#
# Previous input UI (text area + Ask button) – kept for reference:
# cols = st.columns([3, 1])
# with cols[0]:
#     st.session_state.draft = st.text_area(
#         "Ask a legal question",
#         value=st.session_state.draft,
#         placeholder="e.g. What is the Indian Contract Act, 1872?",
#         height=80,
#     )
#
# with cols[1]:
#     if enable_voice:
#         voice_input_component(key="voice_input")
#         st.caption("Voice transcript appears above. Copy/paste into the question box.")
#
#     if enable_audio_transcribe:
#         audio = st.audio_input("Record Audio", key="audio_input")
#         if audio is not None:
#             audio_bytes = audio.getvalue()
#             with st.spinner("Transcribing audio..."):
#                 transcript = transcribe_audio_openai(audio_bytes)
#             if transcript:
#                 st.session_state.draft = transcript
#             else:
#                 st.warning("Audio transcription unavailable (missing OPENAI_API_KEY or openai package).")
#
# submitted = st.button("Ask", type="primary")
#
# query = st.session_state.draft.strip()
# if submitted and query:
#     with st.spinner("Thinking..."):
#         answer, docs = answer_query(query, model)
#         st.session_state.chat.append({"question": query, "answer": answer, "docs": docs})
#
# import streamlit as st
# from langchain_ollama.llms import OllamaLLM
#
# from rag_core import answer_query
#
# # =========================
# # STREAMLIT CONFIG
# # =========================
#
# st.set_page_config(page_title="Indian Legal RAG", layout="wide")
#
# st.title("⚖️ Indian Legal RAG Assistant")
# st.caption("Answers strictly from ingested legal documents (Acts, Sections, Judgments)")
#
# # =========================
# # MODEL
# # =========================
#
# @st.cache_resource
# def load_model():
#     return OllamaLLM(
#         model="llama3.1:8b",
#         temperature=0.1,  # Lower for more consistent legal answers
#         num_ctx=4096,  # Larger context window for longer legal documents
#         num_thread=4,
#         top_p=0.85,  # More focused on likely tokens
#     )
#
# model = load_model()
#
# # =========================
# # SESSION STATE
# # =========================
#
# if "chat" not in st.session_state:
#     st.session_state.chat = []
#
# # =========================
# # INPUT
# # =========================
#
# query = st.text_input(
#     "Ask a legal question",
#     placeholder="e.g. What is the punishment for culpable homicide under BNS?",
# )
#
# # =========================
# # MAIN FLOW
# # =========================
#
# if query:
#     # Prevent duplicate submissions
#     if not st.session_state.chat or st.session_state.chat[-1]["question"] != query:
#         with st.spinner("🧠 Thinking..."):
#             try:
#                 answer, docs = answer_query(query, model)
#                 st.session_state.chat.append(
#                     {
#                         "question": query,
#                         "answer": answer,
#                         "docs": docs,
#                     }
#                 )
#             except Exception as e:
#                 st.error(f"Error processing query: {str(e)}")
#
# # =========================
# # RENDER CHAT
# # =========================
#
# for turn in reversed(st.session_state.chat):
#     st.markdown("### ❓ Question")
#     st.write(turn["question"])
#
#     st.markdown("### ✅ Answer")
#     st.write(turn["answer"])
#
#     if turn["docs"]:
#         with st.expander("📚 Sources"):
#             for i, d in enumerate(turn["docs"], 1):
#                 meta = d.metadata
#                 source_text = []
#                 if meta.get("act"):
#                     source_text.append(f"**Act**: {meta['act']}")
#                 if meta.get("section"):
#                     source_text.append(f"**Section**: {meta['section']}")
#                 if meta.get("source"):
#                     source_text.append(f"**Source**: {meta['source']}")
#                 st.markdown(f"**{i}. {' | '.join(source_text) if source_text else 'Unknown Source'}**")
#                 st.write(d.page_content[:300] + "...")  # Show excerpt
#
#     st.markdown("---")
