import base64
import io
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from streamlit_tldraw import st_tldraw
from streamlit_ketcher import st_ketcher
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from google.api_core.exceptions import (
    InvalidArgument,
    PermissionDenied,
    ResourceExhausted,
    ServiceUnavailable,
    DeadlineExceeded,
)
from requests.exceptions import ConnectionError as RequestsConnectionError
import re
import wave

import requests as _requests_lib
import soundfile as sf
from google import genai
from google.genai import types
from kokoro_onnx import Kokoro

# --- Model Configuration ---
MODEL_NAME = "gemini-3-flash-preview"
DEFAULT_OLLAMA_MODEL = "kimi-k2.5:cloud"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

# --- File Size Limits ---
MAX_IMAGE_SIZE_MB = 10
MAX_PDF_SIZE_MB = 50
MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024
MAX_PDF_SIZE_BYTES = MAX_PDF_SIZE_MB * 1024 * 1024

# --- Unicode Math / Chemistry Symbols to Spoken English ---
UNICODE_MATH_MAP = {
    "≤": "less than or equal to", "≥": "greater than or equal to",
    "≠": "not equal to", "≈": "approximately equal to",
    "±": "plus or minus", "∓": "minus or plus",
    "×": "times", "÷": "divided by", "·": "times",
    "→": "yields", "←": "yields", "⇌": "is in equilibrium with",
    "⇒": "implies", "⇔": "if and only if",
    "∞": "infinity", "∂": "partial",
    "∑": "sum of", "∏": "product of", "∫": "integral of",
    "√": "square root of", "∛": "cube root of",
    "°": "degrees", "′": "prime", "″": "double prime",
    "Δ": "delta", "δ": "delta",
    "α": "alpha", "β": "beta", "γ": "gamma", "ε": "epsilon",
    "θ": "theta", "λ": "lambda", "μ": "mu", "π": "pi",
    "ρ": "rho", "σ": "sigma", "τ": "tau", "φ": "phi",
    "ω": "omega", "η": "eta", "κ": "kappa", "ν": "nu",
    "ξ": "xi", "ψ": "psi", "χ": "chi", "ζ": "zeta",
    "Ω": "omega", "Φ": "phi", "Ψ": "psi", "Σ": "sigma",
    "Π": "pi", "Λ": "lambda", "Θ": "theta", "Γ": "gamma",
    "⁺": "plus", "⁻": "minus",
    "₂": "2", "₃": "3", "₄": "4",
    "⁰": "0", "¹": "1", "²": "squared", "³": "cubed",
    "℃": "degrees celsius", "℉": "degrees fahrenheit",
    "Å": "angstroms",
}

# Regex covering major emoji Unicode ranges
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # symbols extended-A
    "\U00002702-\U000027B0"  # dingbats
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0000200D"             # zero width joiner
    "\U000023CF-\U000023F3"  # misc technical
    "\U0000231A-\U0000231B"  # watch/hourglass
    "\U00002934-\U00002935"  # arrows
    "\U000025AA-\U000025FE"  # geometric shapes
    "\U00002600-\U000026FF"  # misc symbols
    "\U00002B05-\U00002B55"  # arrows/stars
    "]+",
    flags=re.UNICODE,
)

# --- LaTeX-to-Speech Helpers ---
_GREEK_LATEX = {
    r"\alpha": "alpha", r"\beta": "beta", r"\gamma": "gamma",
    r"\delta": "delta", r"\epsilon": "epsilon", r"\varepsilon": "epsilon",
    r"\zeta": "zeta", r"\eta": "eta", r"\theta": "theta",
    r"\iota": "iota", r"\kappa": "kappa", r"\lambda": "lambda",
    r"\mu": "mu", r"\nu": "nu", r"\xi": "xi", r"\pi": "pi",
    r"\rho": "rho", r"\sigma": "sigma", r"\tau": "tau",
    r"\upsilon": "upsilon", r"\phi": "phi", r"\varphi": "phi",
    r"\chi": "chi", r"\psi": "psi", r"\omega": "omega",
    r"\Gamma": "gamma", r"\Delta": "delta", r"\Theta": "theta",
    r"\Lambda": "lambda", r"\Xi": "xi", r"\Pi": "pi",
    r"\Sigma": "sigma", r"\Phi": "phi", r"\Psi": "psi",
    r"\Omega": "omega",
}


def _superscript_to_text(content: str) -> str:
    """Convert superscript content like '2', '3', '2+' to spoken form."""
    content = content.strip()
    if content == "2":
        return "squared"
    if content == "3":
        return "cubed"
    if re.match(r"^\d+[+\-]$", content):
        sign = "plus" if content[-1] == "+" else "minus"
        return f"to the {content[:-1]} {sign}"
    return f"to the {content}"


def _superscript_to_text_simple(char: str) -> str:
    """Convert a single-char superscript like ^2 to spoken form."""
    if char == "2":
        return "squared"
    if char == "3":
        return "cubed"
    return f"to the {char}"


def latex_to_speakable(text: str) -> str:
    """Convert LaTeX content (without $ delimiters) to spoken English."""
    # Phase 1: Greek letters (longest-first to avoid partial matches)
    for cmd in sorted(_GREEK_LATEX, key=len, reverse=True):
        text = text.replace(cmd, f" {_GREEK_LATEX[cmd]} ")

    # Phase 2: Structural commands (iterate for nested constructs)
    for _ in range(10):
        prev = text
        # \frac{a}{b} → "a over b"
        text = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r" \1 over \2 ", text)
        # \sqrt[n]{x} → "the nth root of x"
        text = re.sub(r"\\sqrt\[([^\]]+)\]\{([^{}]*)\}", r" the \1th root of \2 ", text)
        # \sqrt{x} → "square root of x"
        text = re.sub(r"\\sqrt\{([^{}]*)\}", r" square root of \1 ", text)
        # x^{content} → via helper
        text = re.sub(r"\^\{([^{}]*)\}", lambda m: f" {_superscript_to_text(m.group(1))} ", text)
        # x^N (single char) → via helper
        text = re.sub(r"\^([A-Za-z0-9])", lambda m: f" {_superscript_to_text_simple(m.group(1))} ", text)
        # x_{content} → subscript as space-separated
        text = re.sub(r"_\{([^{}]*)\}", r" \1 ", text)
        # x_N (single char) → space
        text = re.sub(r"_([A-Za-z0-9])", r" \1 ", text)
        # \text{...}, \mathrm{...}, \textbf{...}, \textit{...} → unwrap
        text = re.sub(r"\\(?:text|mathrm|textbf|textit|mathbf|mathit|operatorname)\{([^{}]*)\}", r" \1 ", text)
        if text == prev:
            break

    # Operators
    for cmd, spoken in [
        (r"\times", "times"), (r"\cdot", "times"), (r"\div", "divided by"),
        (r"\pm", "plus or minus"), (r"\mp", "minus or plus"),
        (r"\leq", "less than or equal to"), (r"\geq", "greater than or equal to"),
        (r"\neq", "not equal to"), (r"\approx", "approximately equal to"),
        (r"\lt", "less than"), (r"\gt", "greater than"),
        (r"\rightarrow", "yields"), (r"\to", "yields"),
        (r"\leftarrow", "yields"),
        (r"\rightleftharpoons", "is in equilibrium with"),
        (r"\leftrightarrow", "is in equilibrium with"),
        (r"\infty", "infinity"), (r"\partial", "partial"),
        (r"\nabla", "del"),
        (r"\int", "integral of"), (r"\sum", "sum of"),
        (r"\prod", "product of"), (r"\lim", "limit of"),
        (r"\log", "log"), (r"\ln", "natural log of"),
        (r"\sin", "sine"), (r"\cos", "cosine"), (r"\tan", "tangent"),
        (r"\sec", "secant"), (r"\csc", "cosecant"), (r"\cot", "cotangent"),
        (r"\arcsin", "arc sine"), (r"\arccos", "arc cosine"), (r"\arctan", "arc tangent"),
        (r"\overline", "bar"), (r"\vec", "vector"), (r"\hat", "hat"),
        (r"\dot", "dot"), (r"\ddot", "double dot"),
        (r"\left", ""), (r"\right", ""),
        (r"\big", ""), (r"\Big", ""), (r"\bigg", ""), (r"\Bigg", ""),
    ]:
        text = text.replace(cmd, f" {spoken} ")

    # Spacing commands → space
    for cmd in [r"\,", r"\;", r"\:", r"\!", r"\quad", r"\qquad", r"\ "]:
        text = text.replace(cmd, " ")

    # Phase 3: Unicode symbols
    for sym, spoken in UNICODE_MATH_MAP.items():
        text = text.replace(sym, f" {spoken} ")

    # Phase 4: Emoji stripping
    text = _EMOJI_RE.sub(" ", text)

    # Phase 5: Cleanup
    text = re.sub(r"\\[a-zA-Z]+", " ", text)  # remaining \commands
    text = re.sub(r"[{}$&\\]", " ", text)       # stray braces, $, &, backslash
    text = re.sub(r"\s+", " ", text)            # collapse whitespace
    return text.strip()


def get_chat_file_path() -> Path:
    """Return the path to the local chat history JSON file."""
    return Path(__file__).parent / "chat_history.json"


def save_chat(messages: list):
    """Serialize messages to JSON and write to chat_history.json."""
    serialized = []
    for msg in messages:
        entry = {k: v for k, v in msg.items() if k not in ("files", "audio")}
        if "files" in msg:
            entry["files"] = []
            for f in msg["files"]:
                serialized_file = {k: v for k, v in f.items() if k != "data"}
                serialized_file["data_b64"] = base64.b64encode(f["data"]).decode()
                entry["files"].append(serialized_file)
        if "audio" in msg:
            entry["audio_b64"] = base64.b64encode(msg["audio"]).decode()
        serialized.append(entry)
    with open(get_chat_file_path(), "w") as f:
        json.dump(serialized, f, indent=2)


def load_chat() -> list | None:
    """Read chat_history.json and deserialize messages. Returns None if no file."""
    path = get_chat_file_path()
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
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


def get_mime_type(filename: str) -> str:
    """Get MIME type based on file extension."""
    ext = filename.lower().split(".")[-1] if "." in filename else ""
    mime_types = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
        "pdf": "application/pdf",
    }
    return mime_types.get(ext, "image/jpeg")


def compress_image(image_bytes: bytes, max_dimension: int = 1024) -> bytes:
    """Resize image so its longest side is at most max_dimension pixels."""
    img = Image.open(io.BytesIO(image_bytes))
    if max(img.size) > max_dimension:
        img.thumbnail((max_dimension, max_dimension), Image.LANCZOS)
    buf = io.BytesIO()
    fmt = img.format or "PNG"
    img.save(buf, format=fmt)
    return buf.getvalue()


def validate_file_size(file_bytes: bytes, mime_type: str, filename: str) -> str | None:
    """Return an error message if the file exceeds size limits, else None."""
    size_mb = len(file_bytes) / (1024 * 1024)
    if mime_type == "application/pdf":
        if len(file_bytes) > MAX_PDF_SIZE_BYTES:
            return f"PDF '{filename}' is {size_mb:.1f} MB. Maximum allowed is {MAX_PDF_SIZE_MB} MB."
    else:
        if len(file_bytes) > MAX_IMAGE_SIZE_BYTES:
            return f"Image '{filename}' is {size_mb:.1f} MB. Maximum allowed is {MAX_IMAGE_SIZE_MB} MB."
    return None


@st.cache_resource
def get_model(provider: str, model_name: str, api_key: str = "", base_url: str = ""):
    """Create and cache a chat model instance for the given provider."""
    if provider == "ollama":
        return ChatOllama(model=model_name, base_url=base_url, temperature=0.5, num_ctx=16384)
    return ChatGoogleGenerativeAI(model=model_name, temperature=0.5, google_api_key=api_key)


KOKORO_MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx"
KOKORO_VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

KOKORO_EXPECTED_SIZES = {
    "kokoro-v1.0.int8.onnx": 92361271,
    "voices-v1.0.bin": 28214398,
}


def ensure_tts_files():
    """Download TTS model files if needed, with progress feedback and atomic writes."""
    cache_dir = Path.home() / ".cache" / "kokoro-onnx"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / "kokoro-v1.0.int8.onnx"
    voices_path = cache_dir / "voices-v1.0.bin"

    files_to_download = []
    for url, path in [(KOKORO_MODEL_URL, model_path), (KOKORO_VOICES_URL, voices_path)]:
        expected = KOKORO_EXPECTED_SIZES.get(path.name)
        if not path.exists() or (expected and path.stat().st_size != expected):
            if path.exists():
                path.unlink()
            files_to_download.append((url, path))
        # Clean up any leftover .partial files from interrupted downloads
        partial = path.with_suffix(path.suffix + ".partial")
        if partial.exists():
            partial.unlink()

    if not files_to_download:
        return

    status = st.status("Downloading voice model (first time only)...", expanded=True)
    for url, path in files_to_download:
        label = path.name
        status.update(label=f"Downloading {label}...")
        resp = _requests_lib.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        progress = status.progress(0, text=f"Downloading {label}...")
        downloaded = 0
        partial = path.with_suffix(path.suffix + ".partial")
        with open(partial, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    progress.progress(min(downloaded / total, 1.0),
                                      text=f"Downloading {label}...")
        partial.rename(path)
        progress.progress(1.0, text=f"{label} complete")
    status.update(label="Voice model ready!", state="complete", expanded=False)


@st.cache_resource
def get_tts_model():
    """Load and cache the Kokoro TTS model (files must already exist on disk)."""
    cache_dir = Path.home() / ".cache" / "kokoro-onnx"
    return Kokoro(
        str(cache_dir / "kokoro-v1.0.int8.onnx"),
        str(cache_dir / "voices-v1.0.bin"),
    )


def strip_markdown(text: str) -> str:
    """Remove markdown formatting so TTS reads clean prose."""
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`[^`]+`", "", text)
    # Convert LaTeX to spoken English instead of deleting it
    text = re.sub(r"\$\$([\s\S]*?)\$\$", lambda m: latex_to_speakable(m.group(1)), text)
    text = re.sub(r"\$([^$]+)\$", lambda m: latex_to_speakable(m.group(1)), text)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)
    text = re.sub(r"~~([^~]+)~~", r"\1", text)
    text = re.sub(r"^[>\-\*]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\d+\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{2,}", "\n", text)
    # Strip emoji from prose text
    text = _EMOJI_RE.sub(" ", text)
    # Convert Unicode math symbols outside LaTeX
    for sym, spoken in UNICODE_MATH_MAP.items():
        text = text.replace(sym, f" {spoken} ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def generate_tts(text: str) -> bytes | None:
    """Generate WAV audio bytes from text using Kokoro TTS. Returns None on error."""
    clean = strip_markdown(text)
    if not clean:
        return None
    try:
        ensure_tts_files()
        tts = get_tts_model()
        samples, sample_rate = tts.create(clean, voice="af_heart", speed=1.0, lang="en-us")
        buf = io.BytesIO()
        sf.write(buf, samples, sample_rate, format="WAV")
        return buf.getvalue()
    except Exception as e:
        st.toast(f"TTS error: {e}", icon="⚠️")
        return None


def parse_tutor_log(text: str) -> tuple[str, dict | None]:
    """Extract [TUTOR_LOG]...[/TUTOR_LOG] from response text.
    Returns (clean_text, tutor_log_dict or None)."""
    match = re.search(r'\[TUTOR_LOG\](.*?)\[/TUTOR_LOG\]', text, re.DOTALL)
    if not match:
        return text.strip(), None
    log_text = match.group(1).strip()
    log = {}
    for line in log_text.splitlines():
        if ':' in line:
            key, _, val = line.partition(':')
            log[key.strip()] = val.strip()
    clean = text[:match.start()] + text[match.end():]
    return clean.strip(), log


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
):
    """
    Stream a response from the configured model provider.

    Yields response chunks for use with st.write_stream().
    Returns None (via st.error) if an error occurs.
    """
    if provider == "gemini" and not api_key:
        st.error("Please provide your Google API Key in the sidebar.")
        return None

    def _image_part(mime_type, b64_data):
        url = f"data:{mime_type};base64,{b64_data}"
        if provider == "ollama":
            return {"type": "image_url", "image_url": url}
        return {"type": "image_url", "image_url": {"url": url}}

    try:
        model = get_model(provider, model_name, api_key=api_key, base_url=base_url)

        messages = [SystemMessage(content=system_prompt)]

        # Add prior conversation turns
        for msg in (chat_history or []):
            if msg["role"] == "user":
                hist_content = [{"type": "text", "text": msg.get("content", "")}]
                for f in msg.get("files", []):
                    f_b64 = base64.b64encode(f["data"]).decode()
                    hist_content.append(_image_part(f["mime_type"], f_b64))
                if msg.get("audio") and provider != "ollama":
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
            f_b64 = base64.b64encode(f["data"]).decode()
            content.append(_image_part(f["mime_type"], f_b64))
        if audio_data:
            if provider == "ollama":
                st.info("Audio input is not supported with local models.")
            else:
                content.append(
                    {
                        "type": "media",
                        "mime_type": "audio/wav",
                        "data": audio_data,
                    }
                )
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
                        log = {}
                        for line in log_text.splitlines():
                            if ':' in line:
                                k, _, v = line.partition(':')
                                log[k.strip()] = v.strip()
                        st.session_state._pending_tutor_log = log
                    remainder = after.lstrip()
                    if remainder:
                        yield remainder
                    log_done = True
            else:
                yield chunk_text

        if not log_done:
            # Model didn't output tags — yield entire buffer as-is
            yield buffer

    except (PermissionDenied, InvalidArgument) as e:
        if "API key" in str(e).lower() or "api_key" in str(e).lower() or "permission" in str(e).lower():
            st.error("Invalid API key. Please check your key and try again.")
        else:
            st.error(f"Request error: {e}")
    except ResourceExhausted:
        st.error("Rate limited. Please wait a moment and try again.")
    except DeadlineExceeded:
        st.error("Request timed out. Try a shorter question or smaller file.")
    except (ServiceUnavailable, RequestsConnectionError):
        st.error("Connection failed. Check your internet connection and try again.")
    except Exception as e:
        error_msg = str(e).lower()
        if "api key" in error_msg or "api_key" in error_msg:
            st.error("Invalid API key. Please check your key and try again.")
        elif "rate" in error_msg and "limit" in error_msg:
            st.error("Rate limited. Please wait a moment and try again.")
        elif "timeout" in error_msg:
            st.error("Request timed out. Try a shorter question or smaller file.")
        elif provider == "ollama" and ("refused" in error_msg or "connect" in error_msg):
            st.error("Cannot connect to Ollama. Make sure it's running (`ollama serve`).")
        elif provider == "ollama" and "not found" in error_msg:
            st.error(f"Model '{model_name}' not found. Run: `ollama pull {model_name}`")
        elif "connect" in error_msg or "network" in error_msg:
            st.error("Connection failed. Check your internet connection and try again.")
        else:
            st.error(f"An error occurred: {e}")


# --- Streamlit Page Configuration ---
st.set_page_config(page_title="StudyBuddy", page_icon="📚")

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

# --- Load System Prompt from Secrets ---
try:
    system_prompt = st.secrets["SYSTEM_PROMPT"]
except (KeyError, FileNotFoundError):
    st.warning("SYSTEM_PROMPT not found in secrets.toml. Using a default prompt.")
    system_prompt = (
        "You are a helpful and patient tutor. Help the student understand concepts, "
        "work through problems step-by-step, and encourage their learning progress."
    )

# --- Chat History Initialization ---
DEFAULT_GREETING = {
    "role": "assistant",
    "content": (
        "**Welcome to StudyBuddy!** I'm your AI tutor — here to help you learn, "
        "not just give answers.\n\n"
        "Here's what I can do:\n"
        "- **Explain concepts** step-by-step with guided questions\n"
        "- **Work through problems** together (show me a photo of your homework!)\n"
        "- **Analyze images & diagrams** — just attach a file or use the drawing tools\n"
        "- **Listen to your voice** — tap the mic to ask a question out loud\n\n"
        "**Try one of these to get started:**\n"
        '- "Explain the difference between ionic and covalent bonds"\n'
        '- "Help me balance this equation: Fe + O\u2082 \u2192 Fe\u2082O\u2083"\n'
        "- \"What does $E = mc^2$ really mean?\"\n"
        '- Attach a photo of a problem you\'re stuck on'
    ),
}

# --- Sidebar: Configuration ---
with st.sidebar:
    st.markdown("#### :blue[Model]")
    provider = st.radio("Provider", ["Gemini API", "Local (Ollama)"],
                        key="provider", horizontal=True, label_visibility="collapsed")

    if provider == "Gemini API":
        google_api_key = st.text_input(
            "Google API Key", type="password", key="google_api_key",
            placeholder="Paste your API key here"
        )
        st.caption("[Get your Google API key](https://aistudio.google.com/app/apikey)")
        provider_key = "gemini"
        model_name = MODEL_NAME
    else:
        ollama_model = st.text_input("Model name:", value=DEFAULT_OLLAMA_MODEL,
                                      key="ollama_model")
        ollama_url = st.text_input("Ollama URL:", value=DEFAULT_OLLAMA_BASE_URL,
                                    key="ollama_base_url")
        google_api_key = ""
        provider_key = "ollama"
        model_name = ollama_model

    st.divider()
    st.markdown("#### :blue[Options]")
    tts_enabled = st.toggle("Read aloud", key="tts_enabled")
    st.divider()
    if st.button("New Chat", use_container_width=True):
        st.session_state.messages = [DEFAULT_GREETING]
        chat_file = get_chat_file_path()
        if chat_file.exists():
            chat_file.unlink()
        st.rerun()

if "canvas_version" not in st.session_state:
    st.session_state.canvas_version = 0
if "tldraw_version" not in st.session_state:
    st.session_state.tldraw_version = 0

if "messages" not in st.session_state:
    saved = load_chat()
    if saved:
        st.session_state.messages = saved
    else:
        st.session_state.messages = [DEFAULT_GREETING]

# --- Display Chat History ---
for idx, message in enumerate(st.session_state.messages):
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

# --- Drawing Canvas ---
st.caption("TOOLS")
toggle_cols = st.columns(3)
with toggle_cols[0]:
    show_canvas = st.toggle("✏️ Draw", key="show_canvas")
with toggle_cols[1]:
    show_tldraw = st.toggle("🖊️ Whiteboard", key="show_tldraw")
with toggle_cols[2]:
    show_ketcher = st.toggle("⚗️ Molecules", key="show_ketcher")

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
    prompt = result.text or ""
    files = result.files or []
    audio = result.audio

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
        if len(canvas_result.json_data.get("objects", [])) > 0:
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
    if tldraw_result is not None:
        snapshot = tldraw_result if isinstance(tldraw_result, dict) else {}
        shapes = snapshot.get("shapes", snapshot.get("objects", []))
        if shapes:
            tldraw_json = json.dumps(snapshot, default=str)
            prompt = f"[Whiteboard content: {tldraw_json}]\n\n{prompt}"

    # Process ketcher molecule editor
    if ketcher_smiles and ketcher_smiles.strip():
        prompt = f"[Molecule SMILES: {ketcher_smiles.strip()}]\n\n{prompt}"

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
        if prompt:
            st.markdown(prompt)

    # Stream assistant response
    with st.chat_message("assistant"):
        response = st.write_stream(
            stream_response(
                prompt,
                provider_key,
                model_name,
                system_prompt,
                api_key=google_api_key,
                base_url=st.session_state.get("ollama_base_url", DEFAULT_OLLAMA_BASE_URL),
                file_attachments=file_attachments or None,
                audio_data=audio_data,
                chat_history=st.session_state.messages[:-1],
            )
        )
        if response:
            msg = {"role": "assistant", "content": response, "timestamp": datetime.now().isoformat()}
            if '_pending_tutor_log' in st.session_state:
                msg["tutor_log"] = st.session_state._pending_tutor_log
                del st.session_state._pending_tutor_log
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

    st.rerun()
