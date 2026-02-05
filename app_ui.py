# # import streamlit as st
# # import time
# # import json
# # from datetime import datetime
# # from pathlib import Path
# # from typing import Dict, Optional
# # from vector import retriever
# # from langchain_ollama.llms import OllamaLLM
# # from langchain_core.documents import Document
# # import multiprocessing

# # # Configuration
# # MODEL_NAME = "gemma3:1b"
# # TIMEOUT = 300
# # MAX_CHARS = 300
# # K = 5
# # CACHE_FILE = Path("query_cache.json")

# # # Initialize session state
# # if 'chat_history' not in st.session_state:
# #     st.session_state.chat_history = []

# # # Initialize model
# # @st.cache_resource
# # def load_model():
# #     return OllamaLLM(
# #         model=MODEL_NAME,
# #         temperature=0.6,  # Slightly higher for more creative but still focused responses
# #         top_p=0.9,       # Controls diversity of responses
# #         num_ctx=4096,    # Larger context window
# #         num_thread=4,    # Number of CPU threads
# #         repeat_penalty=1.1,  # Slightly penalize repetition
# #         top_k=40,        # Consider top 40 tokens for sampling
# #         stop=["<|im_end|>", "<|endoftext|>"]  # Stop sequences
# #     )

# # model = load_model()

# # # Load cache
# # @st.cache_data(ttl=3600)
# # def load_cache():
# #     if CACHE_FILE.exists():
# #         try:
# #             with open(CACHE_FILE, 'r', encoding='utf-8') as f:
# #                 return json.load(f)
# #         except Exception as e:
# #             st.warning(f"Could not load cache: {e}")
# #     return {}

# # query_cache = load_cache()

# # def save_cache():
# #     try:
# #         with open(CACHE_FILE, 'w', encoding='utf-8') as f:
# #             json.dump(query_cache, f, indent=2)
# #     except Exception as e:
# #         st.warning(f"Could not save cache: {e}")

# # def get_cached_response(query: str) -> Optional[str]:
# #     if query not in query_cache:
# #         return None
    
# #     cached = query_cache[query]
# #     cache_time = datetime.fromisoformat(cached['timestamp'])
# #     if (datetime.now() - cache_time).days < 7:  # Cache valid for 7 days
# #         return cached['response']
# #     return None

# # def cache_response(query: str, response: str):
# #     query_cache[query] = {
# #         'response': response,
# #         'timestamp': datetime.now().isoformat()
# #     }
# #     save_cache()

# # def _shorten_text(s: str, max_chars: int = MAX_CHARS) -> str:
# #     return s if len(s) <= max_chars else s[:max_chars] + "..."

# # def format_document(doc: Document, max_length: int = 300) -> str:
# #     content = _shorten_text(doc.page_content, max_length)
# #     source = doc.metadata.get('source', 'Unknown source')
# #     return f"{content}\n\n**Source:** {source}"

# # def _call_retriever(query: str):
# #     try:
# #         return retriever.invoke(query)
# #     except Exception as e:
# #         st.error(f"Error retrieving documents: {str(e)}")
# #         return []

# # def _model_worker(q, prompt):
# #     try:
# #         # Use generate and extract text robustly
# #         try:
# #             raw = model.generate([prompt])
# #         except Exception:
# #             # Fallback to other call styles
# #             try:
# #                 raw = model.invoke(prompt)
# #             except Exception:
# #                 raw = model(prompt) if callable(model) else None
# #         text = _extract_text_from_model_result(raw)
# #         q.put(("ok", text))
# #     except Exception as e:
# #         q.put(("err", str(e)))

# # def call_model_with_timeout(model_name, prompt, timeout=30):
# #     q = multiprocessing.Queue()
# #     p = multiprocessing.Process(target=_model_worker, args=(q, prompt), daemon=True)
# #     p.start()
# #     start = time.time()
# #     while time.time() - start < timeout:
# #         if not q.empty():
# #             status, payload = q.get()
# #             p.join(0.1)
# #             if status == "ok": 
# #                 return payload
# #             raise RuntimeError(payload)
# #         time.sleep(0.05)
# #     p.terminate()
# #     raise TimeoutError(f"Model call timed out after {timeout} seconds")

# # def _extract_text_from_model_result(res) -> str:
# #     """Extract text from various model response formats with improved error handling."""
# #     if isinstance(res, str):
# #         return res.strip()

# #     # Handle LangChain LLMResult
# #     if hasattr(res, 'generations') and res.generations:
# #         try:
# #             for gen_list in res.generations:
# #                 for gen in gen_list:
# #                     text = getattr(gen, 'text', None) or getattr(gen, 'message', '')
# #                     if isinstance(text, str) and text.strip():
# #                         return text.strip()
# #         except Exception as e:
# #             st.warning(f"Warning: Error extracting from generations: {e}")

# #     # Handle common response patterns
# #     for attr in ('response', 'text', 'output', 'content', 'result'):
# #         val = getattr(res, attr, None)
# #         if isinstance(val, str) and val.strip():
# #             return val.strip()

# #     # Handle dictionary-like objects
# #     if hasattr(res, 'get'):
# #         for key in ('response', 'text', 'output', 'content', 'result'):
# #             val = res.get(key)
# #             if isinstance(val, str) and val.strip():
# #                 return val.strip()

# #     # Handle OpenAI-style response
# #     if hasattr(res, 'choices') and isinstance(res.choices, list) and res.choices:
# #         choice = res.choices[0]
# #         if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
# #             return choice.message.content.strip()
# #         if hasattr(choice, 'text'):
# #             return choice.text.strip()

# #     # Last resort: string representation
# #     return str(res).strip()

# # def get_ai_response(query: str) -> str:
# #     # Check cache first
# #     if cached_response := get_cached_response(query):
# #         return cached_response

# #     # Retrieve relevant documents
# #     with st.spinner("Searching for relevant legal information..."):
# #         docs = _call_retriever(query)
    
# #     if not docs:
# #         return "I couldn't find any relevant legal information. Please try rephrasing your question."

# #     # Prepare context
# #     context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
    
# #     prompt = f"""[INST] You are a legal information assistant.

# # You MUST answer ONLY using the text in CONTEXT.
# # Do NOT use outside knowledge.
# # Do NOT give advice, steps, or procedures unless explicitly stated in the context.

# # If the answer is not fully supported by the context, reply exactly:
# # "Insufficient information in the provided documents."

# # Answer format:
# # - Answer: (1–3 sentences)
# # - Citation: (Act/Section or Case name)

# # CONTEXT: {context}

# # QUESTION: {question}[/INST]"""

# #     # Generate response with timeout
# #     with st.spinner("Analyzing legal information..."):
# #         try:
# #             # Use the same model calling logic as main.py
# #             response = call_model_with_timeout(MODEL_NAME, prompt)
            
# #             # Cache the response
# #             cache_response(query, response)
# #             return response
# #         except TimeoutError:
# #             return "The request timed out. Please try again with a more specific question."
# #         except Exception as e:
# #             return f"Error generating response: {str(e)}"

# # def main():
# #     st.set_page_config(
# #         page_title="LegalAID AI Assistant",
# #         page_icon="⚖️",
# #         layout="wide"
# #     )

# #     st.title("⚖️ LegalAID AI Assistant")
# #     st.caption("Ask me anything about Indian law and legal procedures")

# #     # Sidebar
# #     with st.sidebar:
# #         st.header("About")
# #         st.markdown("""
# #         This AI assistant provides legal information based on Indian law.
        
# #         **Disclaimer:** This is for informational purposes only and does not constitute legal advice.
        
# #         For specific legal advice, please consult a qualified attorney.
# #         """)
        
# #         st.divider()
# #         st.markdown("### Chat History")
# #         if st.button("Clear Chat History"):
# #             st.session_state.chat_history = []
# #             st.rerun()

# #     # Display chat history
# #     for message in st.session_state.chat_history:
# #         with st.chat_message(message["role"]):
# #             st.markdown(message["content"])

# #     # Chat input
# #     if prompt := st.chat_input("Ask a legal question..."):
# #         # Add user message to chat history
# #         st.session_state.chat_history.append({"role": "user", "content": prompt})
        
# #         # Display user message
# #         with st.chat_message("user"):
# #             st.markdown(prompt)
        
# #         # Get and display AI response
# #         with st.chat_message("assistant"):
# #             response = get_ai_response(prompt)
# #             st.markdown(response)
            
# #             # Add AI response to chat history
# #             st.session_state.chat_history.append({"role": "assistant", "content": response})

# # if __name__ == "__main__":
# #     main()


# import streamlit as st
# import json
# from datetime import datetime
# from pathlib import Path
# from typing import Optional

# from vector import retrieve_strict
# from langchain_ollama.llms import OllamaLLM
# from langchain_core.documents import Document

# # =========================
# # CONFIG
# # =========================

# MODEL_NAME = "gemma3:1b"
# CACHE_FILE = Path("query_cache.json")
# MAX_CHARS = 300

# # =========================
# # SESSION STATE
# # =========================

# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # =========================
# # MODEL (STRICT + SAFE)
# # =========================

# @st.cache_resource
# def load_model():
#     return OllamaLLM(
#         model=MODEL_NAME,
#         temperature=0.1,   # 🔒 critical
#         num_ctx=2048,      # keep small
#         num_thread=4
#     )

# model = load_model()

# # =========================
# # CACHE
# # =========================

# @st.cache_data(ttl=3600)
# def load_cache():
#     if CACHE_FILE.exists():
#         with open(CACHE_FILE, "r", encoding="utf-8") as f:
#             return json.load(f)
#     return {}

# query_cache = load_cache()

# def save_cache():
#     with open(CACHE_FILE, "w", encoding="utf-8") as f:
#         json.dump(query_cache, f, indent=2)

# def get_cached_response(query: str) -> Optional[str]:
#     if query not in query_cache:
#         return None
#     cached = query_cache[query]
#     ts = datetime.fromisoformat(cached["timestamp"])
#     if (datetime.now() - ts).days < 7:
#         return cached["response"]
#     return None

# def cache_response(query: str, response: str):
#     query_cache[query] = {
#         "response": response,
#         "timestamp": datetime.now().isoformat()
#     }
#     save_cache()

# # =========================
# # HELPERS
# # =========================

# def shorten(text: str, max_chars=MAX_CHARS):
#     return text if len(text) <= max_chars else text[:max_chars] + "..."

# def format_citation(doc: Document):
#     case = doc.metadata.get("case_name", "Unknown case")
#     date = doc.metadata.get("judgment_date", "")
#     ref = doc.metadata.get("reference", "")
#     return f"{case} ({date}) {ref}".strip()

# # =========================
# # CORE LOGIC
# # =========================

# def get_ai_response(query: str) -> str:
#     if cached := get_cached_response(query):
#         return cached

#     docs, score = retrieve_strict(query)

#     # 🚨 HARD REFUSAL
#     if not docs:
#         return "Insufficient information in the provided documents."

#     context = "\n\n".join(doc.page_content for doc in docs)

#     prompt = f"""
# You are a legal information assistant.

# You MUST answer ONLY using the text in CONTEXT.
# Do NOT use outside knowledge.
# Do NOT give advice or steps.

# If the answer is not fully supported by the context, reply exactly:
# "Insufficient information in the provided documents."

# Answer format:
# Answer:
# Citation:

# CONTEXT:
# {context}

# QUESTION:
# {query}
# """

#     response = model.invoke(prompt).strip()
#     cache_response(query, response)
#     return response

# # =========================
# # STREAMLIT UI
# # =========================

# def main():
#     st.set_page_config(
#         page_title="LegalAID AI Assistant",
#         page_icon="⚖️",
#         layout="wide"
#     )

#     st.title("⚖️ LegalAID AI Assistant")
#     st.caption("Strict citation-based Indian legal information")

#     with st.sidebar:
#         st.markdown("""
# **Rules**
# - Answers only from dataset
# - Automatic refusal on weak context
# - Not legal advice
# """)
#         if st.button("Clear Chat"):
#             st.session_state.chat_history = []
#             st.rerun()

#     for msg in st.session_state.chat_history:
#         with st.chat_message(msg["role"]):
#             st.markdown(msg["content"])

#     if query := st.chat_input("Ask a legal question"):
#         st.session_state.chat_history.append({"role": "user", "content": query})
#         with st.chat_message("user"):
#             st.markdown(query)

#         with st.chat_message("assistant"):
#             answer = get_ai_response(query)
#             st.markdown(answer)
#             st.session_state.chat_history.append(
#                 {"role": "assistant", "content": answer}
#             )

# if __name__ == "__main__":
#     main()


import streamlit as st, json
from datetime import datetime
from pathlib import Path

from vector import retrieve_strict
from langchain_ollama.llms import OllamaLLM

MODEL = "gemma3:1b"
CACHE_FILE = Path("query_cache.json")

@st.cache_resource
def load_model():
    return OllamaLLM(
        model=MODEL,
        temperature=0.1,
        num_ctx=2048,
        num_thread=4,
        # Hard cap on output length to avoid very long generations.
        num_predict=500,
        # Prevent indefinite hangs when the Ollama server/model is stalled.
        client_kwargs={"timeout": 120},
    )

model = load_model()

def audit_log(query, docs, score, reason):
    rec = {
        "query": query,
        "score": score,
        "reason": reason,
        "docs": [d.metadata for d in docs],
        "time": datetime.now().isoformat()
    }
    with open("audit_log.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")

def validate_citation(answer, docs):
    for d in docs:
        if d.metadata.get("doc_type") == "statute":
            if d.metadata.get("section") in answer:
                return True
        else:
            if d.metadata.get("case_name", "").lower() in answer.lower():
                return True
    return False

def confidence(score, reason):
    base = int(score * 100)
    penalties = {"low_similarity": 40, "contradiction": 50, "no_match": 100}
    return max(0, base - penalties.get(reason, 0))

def answer_query(q):
    docs, score, reason = retrieve_strict(q)

    if not docs:
        return f"❌ {reason.replace('_',' ').title()}."

    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
Answer ONLY using the context.
If unsupported, reply exactly:
"Insufficient information in the provided documents."

CONTEXT:
{context}

QUESTION:
{q}
"""

    ans = model.invoke(prompt).strip()

    if not validate_citation(ans, docs):
        return "Insufficient information in the provided documents."

    audit_log(q, docs, score, reason)
    return ans + f"\n\nConfidence: {confidence(score, reason)}%"

st.title("⚖️ Legal RAG Assistant")

q = st.chat_input("Ask a legal question")
if q:
    with st.spinner("Analyzing..."):
        st.markdown(answer_query(q))
