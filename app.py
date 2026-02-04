import base64
import io
import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from streamlit_tldraw import st_tldraw
from streamlit_ketcher import st_ketcher
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from google.api_core.exceptions import (
    InvalidArgument,
    PermissionDenied,
    ResourceExhausted,
    ServiceUnavailable,
    DeadlineExceeded,
)
from requests.exceptions import ConnectionError as RequestsConnectionError

# --- Model Configuration ---
MODEL_NAME = "gemini-3-flash-preview"

# --- File Size Limits ---
MAX_IMAGE_SIZE_MB = 10
MAX_PDF_SIZE_MB = 50
MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024
MAX_PDF_SIZE_BYTES = MAX_PDF_SIZE_MB * 1024 * 1024


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
def get_model(api_key: str):
    """Create and cache the Gemini model instance."""
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=0.5,
        google_api_key=api_key,
    )


def stream_gemini_response(
    user_query: str,
    api_key: str,
    system_prompt: str,
    file_attachments: list | None = None,
    audio_data: bytes | None = None,
):
    """
    Stream a response from the Gemini model.

    Yields response chunks for use with st.write_stream().
    Returns None (via st.error) if an error occurs.
    """
    if not api_key:
        st.error("Please provide your Google API Key in the sidebar.")
        return None

    try:
        model = get_model(api_key)

        content = [{"type": "text", "text": user_query}]
        for f in file_attachments or []:
            f_b64 = base64.b64encode(f["data"]).decode()
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{f['mime_type']};base64,{f_b64}"},
                }
            )
        if audio_data:
            content.append(
                {
                    "type": "media",
                    "mime_type": "audio/wav",
                    "data": audio_data,
                }
            )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content),
        ]

        response_stream = model.stream(messages)
        for chunk in response_stream:
            if not chunk.content:
                continue
            if isinstance(chunk.content, str):
                yield chunk.content
            elif isinstance(chunk.content, list):
                for block in chunk.content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        yield block["text"]
                    elif isinstance(block, str):
                        yield block

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
        elif "connect" in error_msg or "network" in error_msg:
            st.error("Connection failed. Check your internet connection and try again.")
        else:
            st.error(f"An error occurred: {e}")


# --- Streamlit Page Configuration ---
st.set_page_config(page_title="StudyBuddy", page_icon="📚")

st.markdown("""
<style>
/* Recover padding on phones */
@media (max-width: 767px) {
    .stMainBlockContainer {
        padding-left: 0.5rem;
        padding-right: 0.5rem;
    }
    [data-testid="stCustomComponentV1"] iframe {
        max-height: 250px !important;
    }
}

/* Moderate padding on tablets */
@media (min-width: 768px) and (max-width: 1024px) {
    .stMainBlockContainer {
        padding-left: 1rem;
        padding-right: 1rem;
    }
}

/* Prevent image overflow in chat messages */
[data-testid="stChatMessage"] img {
    max-width: 100%;
    height: auto;
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
    "content": "Hello! I'm your StudyBuddy. What would you like to learn or work on today? You can attach images, PDFs, or record voice messages right from the chat box below.",
}

# --- Sidebar: Configuration ---
with st.sidebar:
    st.header("Configuration")
    google_api_key = st.text_input(
        "Enter your Google API Key:", type="password", key="google_api_key"
    )
    st.markdown("[Get your Google API key](https://aistudio.google.com/app/apikey)")

    st.divider()
    if st.button("🗑️ New Chat"):
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
for message in st.session_state.messages:
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

# --- Drawing Canvas ---
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
    user_message = {"role": "user", "content": prompt}
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
            stream_gemini_response(
                prompt,
                google_api_key,
                system_prompt,
                file_attachments=file_attachments or None,
                audio_data=audio_data,
            )
        )
        if response:
            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )
            save_chat(st.session_state.messages)

    # Reset canvas and tldraw for next drawing
    st.session_state.canvas_version = st.session_state.canvas_version + 1
    st.session_state.tldraw_version = st.session_state.tldraw_version + 1

    st.rerun()
