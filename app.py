"""
StudyBuddy - AI-powered 1:1 tutoring assistant
Main Streamlit app entry point
"""

import base64
import io
import json
from datetime import datetime
from pathlib import Path
import secrets
import tempfile
import os
import re

import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from streamlit_tldraw import st_tldraw
from streamlit_ketcher import st_ketcher
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from google.api_core.exceptions import (
    InvalidArgument,
    PermissionDenied,
    ResourceExhausted,
    ServiceUnavailable,
    DeadlineExceeded,
)
from requests.exceptions import ConnectionError as RequestsConnectionError

from utils import (
    MODEL_NAME,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_NVIDIA_MODEL,
    DEFAULT_NVIDIA_BASE_URL,
    get_mime_type,
    compress_image,
    validate_file_size,
    parse_tutor_log,
    parse_log_fields,
    normalize_markdown_newlines,
)
from tts import generate_tts
from subjects import SUBJECT_NAMES, DEFAULT_SUBJECT, get_subject_config

# --- Feature Flags ---
KG_ENABLED = True


# --- Session and Path Management ---

def get_session_id() -> str:
    """Get or create a persistent session ID via URL query param."""
    params = st.query_params
    sid = params.get("s")

    # 1. Valid ?s= in URL — use it
    if sid and re.match(r'^[0-9a-f]{8}$', sid):
        return sid

    # 2. New visitor — generate fresh ID
    sid = secrets.token_hex(4)
    st.query_params["s"] = sid
    return sid




def get_data_dir() -> Path:
    """Return the per-session data directory, creating it if needed."""
    sid = get_session_id()
    data_dir = Path(__file__).parent / "data" / sid
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_chat_file_path() -> Path:
    """Return the path to the local chat history JSON file."""
    return get_data_dir() / "chat_history.json"


def get_graph_file_path() -> Path:
    """Return the path to the knowledge graph JSON file."""
    return get_data_dir() / "knowledge_graph.json"


def save_chat(messages: list):
    """Serialize messages to JSON and write to chat_history.json."""
    serialized = []
    for msg in messages:
        entry = {k: v for k, v in msg.items() if k not in ("files", "audio")}
        if "files" in msg:
            entry["files"] = []
            for f in msg["files"]:
                serialized_file = {k: v for k, v in f.items() if k not in ("data", "data_b64_cache")}
                serialized_file["data_b64"] = base64.b64encode(f["data"]).decode()
                entry["files"].append(serialized_file)
        if "audio" in msg:
            entry["audio_b64"] = base64.b64encode(msg["audio"]).decode()
        serialized.append(entry)
    path = get_chat_file_path()
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(serialized, f, indent=2)
        os.replace(tmp, path)
    except:
        os.unlink(tmp)
        raise


def load_chat() -> list | None:
    """Read chat_history.json and deserialize messages. Returns None if no file."""
    path = get_chat_file_path()
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, ValueError):
        st.warning("Chat history was corrupted and has been reset.")
        return None
    messages = []
    for entry in data:
        msg = {k: v for k, v in entry.items() if k not in ("files", "audio_b64")}
        if "files" in entry:
            msg["files"] = []
            for f in entry["files"]:
                deserialized_file = {k: v for k, v in f.items() if k != "data_b64"}
                deserialized_file["data"] = base64.b64decode(f["data_b64"])
                msg["files"].append(deserialized_file)
        if "audio_b64" in entry:
            msg["audio"] = base64.b64decode(entry["audio_b64"])
        messages.append(msg)
    return messages


# --- Model Management ---

@st.cache_resource
def get_model(provider: str, model_name: str, api_key: str = "", base_url: str = ""):
    """Create and cache a chat model instance for the given provider."""
    if provider == "ollama":
        return ChatOllama(model=model_name, base_url=base_url, temperature=0.5, num_ctx=16384)
    elif provider == "nvidia":
        return ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url, temperature=0.5)
    return ChatGoogleGenerativeAI(model=model_name, temperature=0.5, google_api_key=api_key)


def stream_response(
    user_query: str,
    provider: str,
    model_name: str,
    system_prompt: str,
    api_key: str = "",
    base_url: str = "",
    file_attachments: list | None = None,
    audio_data: bytes | None = None,
    chat_history: list | None = None,
    graph_context: str = "",
):
    """
    Stream a response from the configured model provider.

    Yields response chunks for use with st.write_stream().
    Returns None (via st.error) if an error occurs.
    """
    if provider == "gemini" and not api_key:
        st.error("Please provide your Google API Key in the sidebar.")
        return None

    if provider == "nvidia" and not api_key:
        st.error("No NVIDIA API key found. Add NVIDIA_API_KEY to secrets.toml or enter one in the sidebar.")
        return None

    def _image_part(mime_type, b64_data):
        url = f"data:{mime_type};base64,{b64_data}"
        if provider == "ollama":
            return {"type": "image_url", "image_url": url}
        return {"type": "image_url", "image_url": {"url": url}}

    try:
        model = get_model(provider, model_name, api_key=api_key, base_url=base_url)

        # Inject graph context if available
        full_system_prompt = system_prompt
        if graph_context:
            full_system_prompt = f"{system_prompt}\n\n{graph_context}"

        messages = [SystemMessage(content=full_system_prompt)]

        # Add prior conversation turns
        for msg in (chat_history or []):
            if msg["role"] == "user":
                hist_content = [{"type": "text", "text": msg.get("content", "")}]
                for f in msg.get("files", []):
                    if "data_b64_cache" not in f:
                        f["data_b64_cache"] = base64.b64encode(f["data"]).decode()
                    hist_content.append(_image_part(f["mime_type"], f["data_b64_cache"]))
                if msg.get("audio") and provider == "gemini":
                    hist_content.append({
                        "type": "media",
                        "mime_type": "audio/wav",
                        "data": msg["audio"],
                    })
                messages.append(HumanMessage(content=hist_content))
            elif msg["role"] == "assistant" and msg.get("content"):
                messages.append(AIMessage(content=msg["content"]))

        # Add current turn
        content = [{"type": "text", "text": user_query}]
        for f in file_attachments or []:
            if "data_b64_cache" not in f:
                f["data_b64_cache"] = base64.b64encode(f["data"]).decode()
            content.append(_image_part(f["mime_type"], f["data_b64_cache"]))
        if audio_data:
            if provider == "gemini":
                content.append(
                    {
                        "type": "media",
                        "mime_type": "audio/wav",
                        "data": audio_data,
                    }
                )
            else:
                st.info("Audio input is only supported with Gemini API.")
        messages.append(HumanMessage(content=content))

        response_stream = model.stream(messages)
        buffer = ""
        log_done = False

        for chunk in response_stream:
            # Extract text from chunk (same logic as before)
            chunk_text = ""
            if isinstance(chunk.content, str):
                chunk_text = chunk.content
            elif isinstance(chunk.content, list):
                for block in chunk.content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        chunk_text += block["text"]
                    elif isinstance(block, str):
                        chunk_text += block
            if not chunk_text:
                continue

            if not log_done:
                buffer += chunk_text
                if "[/TUTOR_LOG]" in buffer:
                    # Log complete — extract and start streaming
                    _, after = buffer.split("[/TUTOR_LOG]", 1)
                    log_match = re.search(
                        r'\[TUTOR_LOG\](.*?)\[/TUTOR_LOG\]', buffer, re.DOTALL
                    )
                    if log_match:
                        log_text = log_match.group(1).strip()
                        log = parse_log_fields(log_text)
                        st.session_state._pending_tutor_log = log
                    remainder = after.lstrip()
                    if remainder:
                        yield remainder
                    log_done = True
            else:
                yield chunk_text

        if not log_done and buffer:
            # Model didn't output tags — yield entire buffer as-is
            yield buffer

    except (PermissionDenied, InvalidArgument) as e:
        if "API key" in str(e).lower() or "api_key" in str(e).lower() or "permission" in str(e).lower():
            st.session_state._stream_error = "Invalid API key. Please check your key and try again."
        else:
            st.session_state._stream_error = f"Request error: {e}"
        return
    except ResourceExhausted:
        st.session_state._stream_error = "Rate limited. Please wait a moment and try again."
        return
    except DeadlineExceeded:
        st.session_state._stream_error = "Request timed out. Try a shorter question or smaller file."
        return
    except (ServiceUnavailable, RequestsConnectionError):
        st.session_state._stream_error = "Connection failed. Check your internet connection and try again."
        return
    except Exception as e:
        error_msg = str(e).lower()
        if "api key" in error_msg or "api_key" in error_msg or "unauthorized" in error_msg:
            st.session_state._stream_error = "Invalid API key. Please check your key and try again."
        elif "rate" in error_msg and "limit" in error_msg:
            st.session_state._stream_error = "Rate limited. Please wait a moment and try again."
        elif "timeout" in error_msg:
            st.session_state._stream_error = "Request timed out. Try a shorter question or smaller file."
        elif provider == "ollama" and ("refused" in error_msg or "connect" in error_msg):
            st.session_state._stream_error = "Cannot connect to Ollama. Make sure it's running (`ollama serve`)."
        elif provider == "ollama" and "not found" in error_msg:
            st.session_state._stream_error = f"Model '{model_name}' not found. Run: `ollama pull {model_name}`"
        elif provider == "nvidia" and ("401" in error_msg or "unauthorized" in error_msg):
            st.session_state._stream_error = "Invalid NVIDIA API key. Get one at https://build.nvidia.com/"
        elif provider == "nvidia" and ("429" in error_msg or "rate" in error_msg):
            st.session_state._stream_error = "⚠️ NVIDIA rate limit hit. Free-tier keys allow ~5 requests/min. Wait 60 seconds and try again."
        elif provider == "nvidia" and ("402" in error_msg or "payment" in error_msg or "credit" in error_msg):
            st.session_state._stream_error = "NVIDIA API credits exhausted. Check your account at https://build.nvidia.com/"
        elif "connect" in error_msg or "network" in error_msg:
            st.session_state._stream_error = "Connection failed. Check your internet connection and try again."
        else:
            st.session_state._stream_error = f"An error occurred: {e}"
        return


# --- Streamlit Page Configuration ---
st.set_page_config(initial_sidebar_state="collapsed", page_title="StudyBuddy", page_icon="📚")

st.markdown("""
<style>
/* Prevent image overflow in chat messages */
[data-testid="stChatMessage"] img {
    max-width: 100%;
    height: auto;
}

/* --- Chat bubble styling --- */
[data-testid="stChatMessage"] {
    border-radius: 12px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
}
/* Assistant bubble — visible blue tint + left accent */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: rgba(37, 99, 235, 0.07) !important;
    border-left: 3px solid rgba(37, 99, 235, 0.35);
}
/* User bubble — visible gray tint + right accent */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: rgba(100, 116, 139, 0.07) !important;
    border-right: 3px solid rgba(100, 116, 139, 0.3);
}
/* Turn separator — faint line + spacing before each user message */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    margin-top: 1rem;
    border-top: 1px solid rgba(100, 116, 139, 0.1);
    padding-top: 1rem;
}

/* --- Typography --- */
[data-testid="stChatMessage"] p {
    line-height: 1.75;
    margin-bottom: 0.5em;
}
[data-testid="stChatMessage"] ul,
[data-testid="stChatMessage"] ol {
    margin-top: 0.25em;
    margin-bottom: 0.75em;
    padding-left: 1.5em;
}
[data-testid="stChatMessage"] li {
    line-height: 1.65;
    margin-bottom: 0.25em;
}
[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3,
[data-testid="stChatMessage"] h4 {
    margin-top: 1em;
    margin-bottom: 0.4em;
}

/* --- Blockquotes --- */
[data-testid="stChatMessage"] blockquote {
    border-left: 3px solid rgba(37, 99, 235, 0.5);
    background: rgba(37, 99, 235, 0.04);
    margin: 0.75em 0;
    padding: 0.5em 1em;
    border-radius: 0 8px 8px 0;
}

/* --- Code blocks --- */
[data-testid="stChatMessage"] pre {
    background: #1e1e2e;
    color: #cdd6f4;
    border-radius: 8px;
    padding: 1em;
    overflow-x: auto;
    font-family: "SF Mono", "Fira Code", "JetBrains Mono", "Cascadia Code", monospace;
    font-size: 0.875em;
    margin: 0.75em 0;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    border: 1px solid rgba(0, 0, 0, 0.06);
}

/* --- Inline code --- */
[data-testid="stChatMessage"] :not(pre) > code {
    background: rgba(37, 99, 235, 0.08);
    padding: 0.15em 0.4em;
    border-radius: 4px;
    font-size: 0.875em;
    font-family: "SF Mono", "Fira Code", "JetBrains Mono", "Cascadia Code", monospace;
}

/* --- KaTeX display math card --- */
[data-testid="stChatMessage"] .katex-display {
    border-left: 3px solid rgba(37, 99, 235, 0.45);
    background: rgba(37, 99, 235, 0.03);
    border-radius: 0 8px 8px 0;
    padding: 0.75em 1em;
    margin: 0.75em 0;
    overflow-x: auto;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
}

/* --- KaTeX inline math --- */
[data-testid="stChatMessage"] .katex {
    font-size: 1.1em;
}

/* --- Sidebar refinements --- */
[data-testid="stSidebar"] .stButton button {
    width: 100%;
}
[data-testid="stSidebar"] hr {
    opacity: 0.35;
}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding-top: 1.5rem;
}

/* --- Mobile --- */
@media (max-width: 767px) {
    .stMainBlockContainer {
        padding-left: 0.5rem;
        padding-right: 0.5rem;
    }
    [data-testid="stCustomComponentV1"] iframe {
        max-height: 250px !important;
    }
    [data-testid="stChatMessage"] {
        padding: 0.5rem 0.65rem;
        border-radius: 8px;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        border-left-width: 2px;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        border-right-width: 2px;
    }
    [data-testid="stChatMessage"] pre {
        padding: 0.65em;
        font-size: 0.8em;
    }
    [data-testid="stChatMessage"] .katex-display {
        padding: 0.5em 0.65em;
    }
    h1 {
        font-size: 1.5rem !important;
    }
}

/* --- Tablet --- */
@media (min-width: 768px) and (max-width: 1024px) {
    .stMainBlockContainer {
        padding-left: 1rem;
        padding-right: 1rem;
    }
}
</style>
""", unsafe_allow_html=True)

st.title("📚 StudyBuddy")
st.caption("Your AI-powered tutoring assistant")

# --- Load System Prompt Template from Secrets ---
try:
    _system_prompt_template = st.secrets["SYSTEM_PROMPT"]
except (KeyError, FileNotFoundError):
    st.warning("SYSTEM_PROMPT not found in secrets.toml. Using a default prompt.")
    _system_prompt_template = (
        "You are a helpful and patient tutor. Help the student understand concepts, "
        "work through problems step-by-step, and encourage their learning progress."
    )

def build_system_prompt(subject_name: str) -> str:
    """Build final system prompt by filling in subject-specific content."""
    config = get_subject_config(subject_name)
    prompt = _system_prompt_template
    for key, value in config.items():
        prompt = prompt.replace(f"{{{key}}}", str(value))
    return prompt

# --- Chat History Initialization ---
def get_default_greeting(has_graph: bool) -> dict:
    """Generate appropriate greeting based on whether knowledge graph exists."""
    subject = st.session_state.get("active_subject", DEFAULT_SUBJECT)
    config = get_subject_config(subject)
    icon = config["icon"]
    examples = config["greeting_examples"]

    if KG_ENABLED and has_graph:
        return {
            "role": "assistant",
            "content": (
                f"**Welcome back!** {icon} I've loaded your {subject} knowledge map. "
                "Ready to continue where we left off? Just say 'yes' or ask a question to begin."
            ),
        }
    else:
        kg_hint = "\n\n**Or generate a Knowledge Map** from the sidebar to start structured learning!" if KG_ENABLED else ""
        return {
            "role": "assistant",
            "content": (
                f"**Welcome to StudyBuddy!** {icon} I'm your AI tutor — here to help you learn, "
                "not just give answers.\n\n"
                "Here's what I can do:\n"
                "- **Explain concepts** step-by-step with guided questions\n"
                "- **Work through problems** together (show me a photo of your homework!)\n"
                "- **Analyze images & diagrams** — just attach a file or use the drawing tools\n"
                "- **Listen to your voice** — tap the mic to ask a question out loud\n\n"
                f"**Try one of these {subject} questions to get started:**\n"
                f"- {examples}\n"
                '- Attach a photo of a problem you\'re stuck on' + kg_hint
            ),
        }

# --- Session State Initialization ---
# Must initialize before sidebar accesses these values
if "canvas_version" not in st.session_state:
    st.session_state.canvas_version = 0
if "tldraw_version" not in st.session_state:
    st.session_state.tldraw_version = 0

# KG integration point 1: Session init
if KG_ENABLED:
    from knowledge_graph import load_graph, get_next_node
    if "knowledge_graph" not in st.session_state:
        st.session_state.knowledge_graph = load_graph(get_graph_file_path())
    if "active_node" not in st.session_state:
        # Auto-select first node if graph exists
        if st.session_state.knowledge_graph:
            st.session_state.active_node = get_next_node(st.session_state.knowledge_graph)
        else:
            st.session_state.active_node = None
else:
    if "knowledge_graph" not in st.session_state:
        st.session_state.knowledge_graph = None
    if "active_node" not in st.session_state:
        st.session_state.active_node = None

# Subject initialization (after KG load, so we can infer from saved graph)
if "active_subject" not in st.session_state:
    if st.session_state.knowledge_graph and st.session_state.knowledge_graph.get("subject"):
        kg_sub = st.session_state.knowledge_graph["subject"]
        # Fuzzy match: if graph says "Chemistry", "chemistry", or "Chem", select Chemistry
        st.session_state.active_subject = next(
            (s for s in SUBJECT_NAMES if s.lower() in kg_sub.lower()), DEFAULT_SUBJECT
        )
    else:
        st.session_state.active_subject = DEFAULT_SUBJECT

# --- Sidebar: Configuration ---
with st.sidebar:
    st.markdown("#### :blue[Model]")
    provider = st.selectbox("Provider", ["NVIDIA API", "Gemini API", "Local (Ollama)"],
                            key="provider", label_visibility="collapsed")

    if provider == "NVIDIA API":
        # Load from secrets, allow user override
        secret_nvidia_key = ""
        try:
            secret_nvidia_key = st.secrets["NVIDIA_API_KEY"]
        except (KeyError, FileNotFoundError):
            pass

        nvidia_api_key = st.text_input(
            "NVIDIA API Key (optional — built-in key available)",
            type="password", key="nvidia_api_key",
            placeholder="Override with your own key"
        )
        # Use user's key if provided, else fall back to secret
        nvidia_api_key = nvidia_api_key or secret_nvidia_key

        if nvidia_api_key:
            st.caption("✅ API key active")
        else:
            st.caption("⚠️ No API key. [Get one at nvidia.com](https://build.nvidia.com/)")
        provider_key = "nvidia"
        model_name = DEFAULT_NVIDIA_MODEL
        google_api_key = ""
    elif provider == "Gemini API":
        # Load from secrets, allow user override
        secret_google_key = ""
        try:
            secret_google_key = st.secrets["GOOGLE_API_KEY"]
        except (KeyError, FileNotFoundError):
            pass

        google_api_key = st.text_input(
            "Google API Key (optional — built-in key available)",
            type="password", key="google_api_key",
            placeholder="Override with your own key"
        )
        # Use user's key if provided, else fall back to secret
        google_api_key = google_api_key or secret_google_key

        if google_api_key:
            st.caption("✅ API key active")
        else:
            st.caption("⚠️ No API key. [Get one at aistudio.google.com](https://aistudio.google.com/app/apikey)")
        provider_key = "gemini"
        model_name = MODEL_NAME
        nvidia_api_key = ""
    else:
        ollama_model = st.text_input("Model name:", value=DEFAULT_OLLAMA_MODEL,
                                      key="ollama_model")
        ollama_url = st.text_input("Ollama URL:", value=DEFAULT_OLLAMA_BASE_URL,
                                    key="ollama_base_url")
        st.caption(
            "Running remotely? Expose Ollama with "
            "[ngrok](https://ngrok.com) (`ngrok http 11434`) "
            "and paste the URL above."
        )
        google_api_key = ""
        nvidia_api_key = ""
        provider_key = "ollama"
        model_name = ollama_model

    st.divider()
    st.markdown("#### :blue[Options]")
    tts_enabled = st.toggle("Read aloud", key="tts_enabled")
    st.divider()

    # --- Subject Selector ---
    st.markdown("#### :blue[Subject]")
    selected_subject = st.selectbox(
        "Subject", SUBJECT_NAMES,
        index=SUBJECT_NAMES.index(st.session_state.active_subject),
        format_func=lambda s: f"{get_subject_config(s)['icon']} {s}",
        key="subject_selector", label_visibility="collapsed",
    )
    if selected_subject != st.session_state.active_subject:
        st.session_state.active_subject = selected_subject
    st.divider()

    # --- Knowledge Map Section ---
    # KG integration point 2: Sidebar
    if KG_ENABLED:
        from knowledge_graph import render_sidebar
        active_api_key = nvidia_api_key if provider_key == "nvidia" else google_api_key
        if provider_key == "nvidia":
            base_url_param = DEFAULT_NVIDIA_BASE_URL
        elif provider_key == "ollama":
            base_url_param = st.session_state.get("ollama_base_url", DEFAULT_OLLAMA_BASE_URL)
        else:
            base_url_param = ""
        render_sidebar(provider_key, model_name, active_api_key, base_url_param, get_graph_file_path(), get_model)
        st.divider()


    if st.button("New Chat", use_container_width=True):
        has_graph = st.session_state.knowledge_graph is not None
        st.session_state.messages = [get_default_greeting(has_graph)]
        chat_file = get_chat_file_path()
        if chat_file.exists():
            chat_file.unlink()
        keys_to_delete = [k for k in st.session_state.keys() if k.startswith("tts_cache_")]
        for k in keys_to_delete:
            del st.session_state[k]
        st.rerun()

if "messages" not in st.session_state:
    saved = load_chat()
    if saved:
        st.session_state.messages = saved
    else:
        has_graph = st.session_state.knowledge_graph is not None
        st.session_state.messages = [get_default_greeting(has_graph)]

# Flag to check if we need to auto-start
if "auto_start_needed" not in st.session_state:
    st.session_state.auto_start_needed = (
        st.session_state.knowledge_graph is not None and
        st.session_state.active_node is not None and
        len(st.session_state.messages) == 1
    )

# --- Display Chat History ---
for idx, message in enumerate(st.session_state.messages):
    if message.get("content", "").startswith("[NAVIGATE TO NODE:"):
        continue

    with st.chat_message(message["role"]):
        for f in message.get("files", []):
            if f.get("type", "image") == "image":
                st.image(f["data"], width=200)
            else:
                st.info(f"📄 PDF: {f['name']}")
        if message.get("audio"):
            st.audio(message["audio"], format="audio/wav")
        if message.get("content"):
            st.markdown(message["content"])
        if message.get("timestamp"):
            st.caption(message["timestamp"][:16].replace("T", " "))
        if message["role"] == "assistant" and message.get("tutor_log"):
            with st.expander("Tutor reasoning", icon="🧠"):
                for k, v in message["tutor_log"].items():
                    st.markdown(f"**{k}:** {v}")
        if message["role"] == "assistant" and tts_enabled and message.get("content"):
            cache_key = f"tts_cache_{idx}"
            if st.button("🔊 Read aloud", key=f"tts_btn_{idx}"):
                if cache_key not in st.session_state:
                    with st.spinner("Generating speech..."):
                        st.session_state[cache_key] = generate_tts(message["content"])
                audio_bytes = st.session_state.get(cache_key)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/wav", autoplay=True)

# --- Auto-start with first problem if needed ---
if st.session_state.auto_start_needed:
    if st.button("🚀 Start Learning!", type="primary", use_container_width=True):
        st.session_state.auto_start_needed = False
        # Trigger first problem
        prompt = "I'm ready to start!"
        user_message = {"role": "user", "content": prompt, "timestamp": datetime.now().isoformat()}
        st.session_state.messages.append(user_message)
        st.rerun()

# --- Drawing Canvas ---
st.caption("TOOLS")
toggle_cols = st.columns(3)
with toggle_cols[0]:
    show_canvas = st.toggle("✏️ Draw", key="show_canvas")
with toggle_cols[1]:
    show_tldraw = st.toggle("🖊️ Whiteboard", key="show_tldraw")
with toggle_cols[2]:
    if st.session_state.get("active_subject") == "Chemistry":
        show_ketcher = st.toggle("⚗️ Molecules", key="show_ketcher")
    else:
        show_ketcher = False

canvas_result = None
if show_canvas:
    canvas_result = st_canvas(
        stroke_width=3,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=250,
        drawing_mode="freedraw",
        display_toolbar=True,
        update_streamlit=True,
        key=f"canvas_{st.session_state.canvas_version}",
    )

tldraw_result = None
if show_tldraw:
    tldraw_result = st_tldraw(
        height=300,
        key=f"tldraw_{st.session_state.tldraw_version}",
    )

ketcher_smiles = None
if show_ketcher:
    ketcher_smiles = st_ketcher("")

# --- User Input and Response Handling ---
result = st.chat_input(
    "Ask a question or describe what you'd like help with...",
    accept_file="multiple",
    file_type=["png", "jpg", "jpeg", "gif", "webp", "pdf"],
    accept_audio=True,
)
if result:
    if '_pending_tutor_log' in st.session_state:
        del st.session_state._pending_tutor_log

    prompt = result.text or ""
    files = result.files or []
    audio = result.audio

    last_canvas_v = st.session_state.get("last_canvas_submitted", -1)

    # Process files
    file_attachments = []
    for f in files:
        file_bytes = f.getvalue()
        mime = get_mime_type(f.name)
        is_pdf = mime == "application/pdf"
        size_err = validate_file_size(file_bytes, mime, f.name)
        if size_err:
            st.error(size_err)
            continue
        if not is_pdf:
            file_bytes = compress_image(file_bytes)
        file_attachments.append({
            "data": file_bytes,
            "mime_type": mime,
            "name": f.name,
            "type": "pdf" if is_pdf else "image",
        })

    # Process canvas drawing
    if canvas_result is not None and canvas_result.json_data is not None:
        if (len(canvas_result.json_data.get("objects", [])) > 0
                and st.session_state.canvas_version != last_canvas_v):
            img = Image.fromarray(canvas_result.image_data.astype("uint8"), "RGBA")
            img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            file_attachments.append({
                "data": buf.getvalue(),
                "mime_type": "image/png",
                "name": "drawing.png",
                "type": "image",
            })

    # Process tldraw whiteboard
    display_prompt = prompt
    if tldraw_result is not None:
        snapshot = tldraw_result if isinstance(tldraw_result, dict) else {}
        shapes = snapshot.get("shapes", snapshot.get("objects", []))
        if shapes:
            tldraw_json = json.dumps(snapshot, default=str)
            prompt = f"[Whiteboard content: {tldraw_json}]\n\n{prompt}"
            display_prompt = f"[Whiteboard attached]\n\n{display_prompt}" if display_prompt else "[Whiteboard attached]"

    # Process ketcher molecule editor
    if ketcher_smiles and ketcher_smiles.strip():
        prompt = f"[Molecule SMILES: {ketcher_smiles.strip()}]\n\n{prompt}"
        display_prompt = f"[Molecule structure attached]\n\n{display_prompt}" if display_prompt else "[Molecule structure attached]"

    # Process audio
    audio_data = None
    if audio:
        audio_data = audio.getvalue()

    # Build user message for history
    user_message = {"role": "user", "content": prompt, "timestamp": datetime.now().isoformat()}
    if file_attachments:
        user_message["files"] = file_attachments
    if audio_data:
        user_message["audio"] = audio_data

    st.session_state.messages.append(user_message)

    # Display user message
    with st.chat_message("user"):
        for f in file_attachments:
            if f["type"] == "image":
                st.image(f["data"], width=200)
            else:
                st.info(f"📄 PDF: {f['name']}")
        if audio_data:
            st.audio(audio_data, format="audio/wav")
        if display_prompt:
            st.markdown(display_prompt)

    # Stream assistant response
    with st.chat_message("assistant"):
        # Build graph context if available
        graph_ctx = ""
        if KG_ENABLED and st.session_state.knowledge_graph and st.session_state.active_node:
            from knowledge_graph import build_graph_context
            graph_ctx = build_graph_context(st.session_state.knowledge_graph, st.session_state.active_node)

        active_api_key = nvidia_api_key if provider_key == "nvidia" else google_api_key
        if provider_key == "nvidia":
            base_url_param = DEFAULT_NVIDIA_BASE_URL
        elif provider_key == "ollama":
            base_url_param = st.session_state.get("ollama_base_url", DEFAULT_OLLAMA_BASE_URL)
        else:
            base_url_param = ""

        response = st.write_stream(
            stream_response(
                prompt,
                provider_key,
                model_name,
                build_system_prompt(st.session_state.active_subject),
                api_key=active_api_key,
                base_url=base_url_param,
                file_attachments=file_attachments or None,
                audio_data=audio_data,
                chat_history=st.session_state.messages[:-1],
                graph_context=graph_ctx,
            )
        )

    if "_stream_error" in st.session_state:
        st.error(st.session_state._stream_error)
        del st.session_state._stream_error
    elif response:
        # Safety net: strip any TUTOR_LOG that slipped through streaming
        clean_response, extracted_log = parse_tutor_log(response)
        clean_response = normalize_markdown_newlines(clean_response)
        msg = {"role": "assistant", "content": clean_response, "timestamp": datetime.now().isoformat()}

        if '_pending_tutor_log' in st.session_state:
            tutor_log = st.session_state._pending_tutor_log
            msg["tutor_log"] = tutor_log
            del st.session_state._pending_tutor_log
        elif extracted_log:
            # Fallback: use extracted log if streaming didn't capture it
            msg["tutor_log"] = extracted_log
            st.session_state._pending_tutor_log = extracted_log

            # KG integration point 3: Post-response
            if KG_ENABLED and extracted_log.get("node") and st.session_state.get("knowledge_graph"):
                from knowledge_graph import process_tutor_response
                process_tutor_response(extracted_log, get_graph_file_path())

        # Handle tutor_log for KG updates (if it was already in _pending_tutor_log)
        if msg.get("tutor_log"):
            tutor_log = msg["tutor_log"]
            # KG integration point 3: Post-response
            if KG_ENABLED and tutor_log.get("node") and st.session_state.get("knowledge_graph"):
                from knowledge_graph import process_tutor_response
                process_tutor_response(tutor_log, get_graph_file_path())

        st.session_state.messages.append(msg)
        save_chat(st.session_state.messages)
        if tts_enabled:
            with st.spinner("Generating speech..."):
                tts_audio = generate_tts(response)
            if tts_audio:
                tts_idx = len(st.session_state.messages) - 1
                st.session_state[f"tts_cache_{tts_idx}"] = tts_audio
                st.audio(tts_audio, format="audio/wav", autoplay=True)

    # Reset canvas and tldraw for next drawing
    st.session_state.canvas_version = st.session_state.canvas_version + 1
    st.session_state.tldraw_version = st.session_state.tldraw_version + 1
    st.session_state.last_canvas_submitted = st.session_state.canvas_version

    st.rerun()
