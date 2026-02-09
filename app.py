import base64
import io
import json
from datetime import datetime
from pathlib import Path
import graphlib

import numpy as np
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
DEFAULT_NVIDIA_MODEL = "moonshotai/kimi-k2.5"
DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

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


def get_graph_file_path() -> Path:
    """Return the path to the knowledge graph JSON file."""
    return Path(__file__).parent / "knowledge_graph.json"


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


def save_graph(graph: dict):
    """Serialize knowledge graph to JSON and write to knowledge_graph.json."""
    with open(get_graph_file_path(), "w") as f:
        json.dump(graph, f, indent=2)


def load_graph() -> dict | None:
    """Read knowledge_graph.json. Returns None if no file."""
    path = get_graph_file_path()
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def validate_dag(nodes: dict) -> bool:
    """Validate that the knowledge graph is a DAG using topological sort. Returns True if valid."""
    try:
        ts = graphlib.TopologicalSorter()
        for node_id, node_data in nodes.items():
            deps = node_data.get("deps", [])
            ts.add(node_id, *deps)
        # This will raise CycleError if there's a cycle
        tuple(ts.static_order())
        return True
    except graphlib.CycleError:
        return False


def compute_node_state(node: dict) -> str:
    """Compute node state: mastered | failing | in_progress | untested."""
    mastery_level = node.get("mastery_level", 0)
    decay_score = node.get("decay_score", 1.0)
    times_tested = node.get("times_tested", 0)
    times_correct = node.get("times_correct", 0)

    # Green: mastered
    if mastery_level >= 3 and decay_score > 0.5:
        return "mastered"

    # Red: failing
    if times_tested > 0 and times_correct / times_tested < 0.5:
        return "failing"

    # Yellow: in progress or untested
    if times_tested > 0:
        return "in_progress"

    return "untested"


def get_next_node(graph: dict) -> str | None:
    """Return the first unmastered node with satisfied dependencies, in topological order."""
    nodes = graph.get("nodes", {})

    # Build topological order
    ts = graphlib.TopologicalSorter()
    for node_id, node_data in nodes.items():
        deps = node_data.get("deps", [])
        ts.add(node_id, *deps)

    topo_order = list(ts.static_order())

    # Find first unmastered node with all deps mastered
    for node_id in topo_order:
        node = nodes[node_id]
        state = compute_node_state(node)

        if state != "mastered":
            # Check if all dependencies are mastered
            deps_mastered = all(
                compute_node_state(nodes[dep]) == "mastered"
                for dep in node.get("deps", [])
                if dep in nodes
            )
            if deps_mastered:
                return node_id

    return None


def is_node_locked(node_id: str, graph: dict) -> bool:
    """Check if a node is locked (has unmastered dependencies)."""
    nodes = graph.get("nodes", {})
    node = nodes.get(node_id)
    if not node:
        return True

    for dep in node.get("deps", []):
        if dep in nodes and compute_node_state(nodes[dep]) != "mastered":
            return True

    return False


def update_node_from_log(graph: dict, tutor_log: dict):
    """Update node state based on TUTOR_LOG verdict."""
    node_id = tutor_log.get("node")
    if not node_id or node_id not in graph.get("nodes", {}):
        return

    node = graph["nodes"][node_id]
    verdict = tutor_log.get("node_verdict", "not_assessed")

    # Update times_tested
    if verdict != "not_assessed":
        node["times_tested"] = node.get("times_tested", 0) + 1
        node["last_tested"] = datetime.now().isoformat()

    # Update mastery based on verdict
    if verdict == "mastered":
        node["mastery_level"] = 3
        node["times_correct"] = node.get("times_correct", 0) + 1
        node["problem_step"] = 3  # Completed all steps
        node["decay_score"] = 1.0  # Reset decay on mastery
    elif verdict == "progressing":
        node["times_correct"] = node.get("times_correct", 0) + 1
        # Advance problem_step
        current_step = node.get("problem_step", 0)
        node["problem_step"] = min(current_step + 1, 3)
        if node["problem_step"] >= 2:
            node["mastery_level"] = min(node.get("mastery_level", 0) + 1, 3)
    elif verdict == "struggling":
        # Drop back to atomic (step 1) if not already there
        if node.get("problem_step", 0) > 1:
            node["problem_step"] = 1


def build_graph_context(graph: dict, active_node: str | None) -> str:
    """Build the <knowledge_graph_state> block for system prompt injection."""
    if not graph or not active_node:
        return ""

    nodes = graph.get("nodes", {})
    if active_node not in nodes:
        return ""

    active = nodes[active_node]
    step = active.get("problem_step", 0)
    step_names = {0: "Untested", 1: "Atomic Problem", 2: "Variation Problem", 3: "Boss Problem"}

    lines = [
        "<knowledge_graph_state>",
        f"Current focus: {active_node} (Step {step}: {step_names.get(step, 'Unknown')})",
        f"- Description: {active.get('description', 'N/A')}",
        f"- Progress: {active.get('times_correct', 0)}/{active.get('times_tested', 0)} correct",
        "",
        "Dependencies:",
    ]

    # Show dependency status
    for dep in active.get("deps", []):
        if dep in nodes:
            dep_node = nodes[dep]
            state = compute_node_state(dep_node)
            decay = dep_node.get("decay_score", 1.0)
            status = "MASTERED" if state == "mastered" else state.upper()
            lines.append(f"- {dep}: {status} (decay: {decay:.2f})")

    # Show locked nodes
    locked = [nid for nid in nodes if is_node_locked(nid, graph) and compute_node_state(nodes[nid]) != "mastered"]
    if locked:
        lines.append("")
        lines.append("Locked (needs prereqs): " + ", ".join(locked[:5]))

    # Show review-due nodes (decay < 0.5)
    review_due = [nid for nid in nodes if compute_node_state(nodes[nid]) == "mastered" and nodes[nid].get("decay_score", 1.0) < 0.5]
    if review_due:
        lines.append("")
        lines.append("Review due: " + ", ".join(review_due[:3]))

    lines.append("</knowledge_graph_state>")
    return "\n".join(lines)


def generate_graph_prompt(subject: str) -> str:
    """Generate a prompt for creating a knowledge graph."""
    return f"""Generate a knowledge graph for {subject} tutoring.

Create a JSON structure with 12-20 atomic, testable skills arranged as a dependency DAG. Each node should represent ONE specific skill that can be tested in isolation.

Requirements:
- Start with fundamental prerequisites (arithmetic, unit conversion)
- Build up to complex applications
- Each node needs exactly 0-3 dependencies
- Use snake_case IDs (e.g., "mole_concept", "stoichiometry")
- Labels should be concise (3-6 words)
- Descriptions should be specific and testable

Return ONLY valid JSON in this exact format:
{{
  "subject": "{subject}",
  "nodes": {{
    "arithmetic": {{
      "label": "Basic Arithmetic",
      "description": "Add, subtract, multiply, divide with decimals",
      "deps": []
    }},
    "unit_conversion": {{
      "label": "Unit Conversions",
      "description": "Convert between metric units using dimensional analysis",
      "deps": ["arithmetic"]
    }},
    "mole_concept": {{
      "label": "The Mole Concept",
      "description": "Avogadro's number and molar quantities",
      "deps": ["arithmetic", "unit_conversion"]
    }}
  }}
}}

Generate the complete graph now."""


def generate_graph(model, subject: str) -> dict | None:
    """Generate a knowledge graph using the LLM. Returns graph dict or None on error."""
    try:
        prompt = generate_graph_prompt(subject)
        messages = [
            SystemMessage(content="You are a curriculum design expert. Generate valid JSON only, no markdown formatting."),
            HumanMessage(content=prompt)
        ]

        response = model.invoke(messages)
        response_text = response.content if hasattr(response, 'content') else str(response)

        # Strip markdown code fences if present
        response_text = response_text.strip()
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1]) if len(lines) > 2 else response_text
            response_text = response_text.replace("```json", "").replace("```", "").strip()

        graph = json.loads(response_text)

        # Add default state fields to each node
        for node_id, node_data in graph.get("nodes", {}).items():
            node_data.setdefault("mastery_level", 0)
            node_data.setdefault("last_tested", None)
            node_data.setdefault("times_correct", 0)
            node_data.setdefault("times_tested", 0)
            node_data.setdefault("problem_step", 0)
            node_data.setdefault("decay_score", 1.0)

        # Validate DAG
        if not validate_dag(graph.get("nodes", {})):
            st.error("Generated graph contains cycles. Please try again.")
            return None

        return graph

    except json.JSONDecodeError as e:
        st.error(f"Failed to parse graph JSON: {e}")
        return None
    except Exception as e:
        st.error(f"Graph generation error: {e}")
        return None


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
    elif provider == "nvidia":
        return ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url, temperature=0.5)
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
                    f_b64 = base64.b64encode(f["data"]).decode()
                    hist_content.append(_image_part(f["mime_type"], f_b64))
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
            f_b64 = base64.b64encode(f["data"]).decode()
            content.append(_image_part(f["mime_type"], f_b64))
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
        if "api key" in error_msg or "api_key" in error_msg or "unauthorized" in error_msg:
            st.error("Invalid API key. Please check your key and try again.")
        elif "rate" in error_msg and "limit" in error_msg:
            st.error("Rate limited. Please wait a moment and try again.")
        elif "timeout" in error_msg:
            st.error("Request timed out. Try a shorter question or smaller file.")
        elif provider == "ollama" and ("refused" in error_msg or "connect" in error_msg):
            st.error("Cannot connect to Ollama. Make sure it's running (`ollama serve`).")
        elif provider == "ollama" and "not found" in error_msg:
            st.error(f"Model '{model_name}' not found. Run: `ollama pull {model_name}`")
        elif provider == "nvidia" and ("401" in error_msg or "unauthorized" in error_msg):
            st.error("Invalid NVIDIA API key. Get one at https://build.nvidia.com/")
        elif provider == "nvidia" and ("429" in error_msg or "rate" in error_msg):
            st.error("⚠️ NVIDIA rate limit hit. Free-tier keys allow ~5 requests/min. Wait 60 seconds and try again.")
        elif provider == "nvidia" and ("402" in error_msg or "payment" in error_msg or "credit" in error_msg):
            st.error("NVIDIA API credits exhausted. Check your account at https://build.nvidia.com/")
        elif "connect" in error_msg or "network" in error_msg:
            st.error("Connection failed. Check your internet connection and try again.")
        else:
            st.error(f"An error occurred: {e}")


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

# --- Load System Prompt from Secrets ---
try:
    base_system_prompt = st.secrets["SYSTEM_PROMPT"]
except (KeyError, FileNotFoundError):
    st.warning("SYSTEM_PROMPT not found in secrets.toml. Using a default prompt.")
    base_system_prompt = (
        "You are a helpful and patient tutor. Help the student understand concepts, "
        "work through problems step-by-step, and encourage their learning progress."
    )

# --- Chat History Initialization ---
def get_default_greeting(has_graph: bool) -> dict:
    """Generate appropriate greeting based on whether knowledge graph exists."""
    if has_graph:
        return {
            "role": "assistant",
            "content": (
                "**Welcome back!** I've loaded your knowledge map. "
                "Ready to continue where we left off? Just say 'yes' or ask a question to begin."
            ),
        }
    else:
        return {
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
                '- Attach a photo of a problem you\'re stuck on\n\n'
                "**Or generate a Knowledge Map** from the sidebar to start structured learning!"
            ),
        }

# --- Session State Initialization ---
# Must initialize before sidebar accesses these values
if "canvas_version" not in st.session_state:
    st.session_state.canvas_version = 0
if "tldraw_version" not in st.session_state:
    st.session_state.tldraw_version = 0

if "knowledge_graph" not in st.session_state:
    st.session_state.knowledge_graph = load_graph()

if "active_node" not in st.session_state:
    # Auto-select first node if graph exists
    if st.session_state.knowledge_graph:
        st.session_state.active_node = get_next_node(st.session_state.knowledge_graph)
    else:
        st.session_state.active_node = None

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

    # --- Knowledge Map Section ---
    st.markdown("#### :blue[Knowledge Map]")

    if st.session_state.knowledge_graph is None:
        subject_input = st.text_input("Subject:", value="Chemistry", key="subject_input")
        if st.button("Generate Map", use_container_width=True):
            active_api_key = nvidia_api_key if provider_key == "nvidia" else google_api_key
            if provider_key in ("nvidia", "gemini") and not active_api_key:
                st.error("Please provide your API Key first.")
            else:
                with st.spinner(f"Generating {subject_input} knowledge graph..."):
                    base_url_param = DEFAULT_NVIDIA_BASE_URL if provider_key == "nvidia" else st.session_state.get("ollama_base_url", DEFAULT_OLLAMA_BASE_URL)
                    model = get_model(provider_key, model_name, api_key=active_api_key,
                                    base_url=base_url_param)
                    graph = generate_graph(model, subject_input)
                    if graph:
                        st.session_state.knowledge_graph = graph
                        save_graph(graph)
                        # Auto-select first node
                        st.session_state.active_node = get_next_node(graph)
                        st.success(f"Created {len(graph['nodes'])} nodes!")
                        st.rerun()
    else:
        graph = st.session_state.knowledge_graph
        nodes = graph.get("nodes", {})

        # Progress bar
        mastered_count = sum(1 for n in nodes.values() if compute_node_state(n) == "mastered")
        total_count = len(nodes)
        progress = mastered_count / total_count if total_count > 0 else 0
        st.progress(progress, text=f"Progress: {mastered_count}/{total_count}")

        # Node list
        st.caption(f"**{graph.get('subject', 'Knowledge Graph')}**")

        # Build topological order
        ts = graphlib.TopologicalSorter()
        for node_id, node_data in nodes.items():
            deps = node_data.get("deps", [])
            ts.add(node_id, *deps)
        topo_order = list(ts.static_order())

        # Display nodes
        for node_id in topo_order:
            node = nodes[node_id]
            state = compute_node_state(node)
            locked = is_node_locked(node_id, graph)

            # Choose emoji
            if state == "mastered":
                emoji = "🟢"
            elif state == "failing":
                emoji = "🔴"
            elif locked:
                emoji = "🔒"
            else:
                emoji = "🟡"

            # Active indicator
            indicator = " ●" if node_id == st.session_state.active_node else ""

            # Button label
            label = f"{emoji} {node['label']}{indicator}"

            # Step indicator for active node
            if node_id == st.session_state.active_node:
                step = node.get("problem_step", 0)
                if step > 0:
                    label += f" [{step}/3]"

            if st.button(label, key=f"node_{node_id}", disabled=locked, use_container_width=True):
                st.session_state.active_node = node_id
                # Inject a navigation prompt
                nav_msg = {
                    "role": "user",
                    "content": f"[NAVIGATE TO NODE: {node_id}]",
                    "timestamp": datetime.now().isoformat()
                }
                st.session_state.messages.append(nav_msg)
                st.rerun()

        # Review-due section
        review_due = [nid for nid in nodes if compute_node_state(nodes[nid]) == "mastered" and nodes[nid].get("decay_score", 1.0) < 0.5]
        if review_due:
            st.divider()
            st.caption("🔄 **Review Needed**")
            for nid in review_due[:3]:
                node = nodes[nid]
                if st.button(f"🔄 {node['label']}", key=f"review_{nid}", use_container_width=True):
                    st.session_state.active_node = nid
                    st.rerun()

        st.divider()
        if st.button("Reset Map", use_container_width=True):
            graph_file = get_graph_file_path()
            if graph_file.exists():
                graph_file.unlink()
            st.session_state.knowledge_graph = None
            st.session_state.active_node = None
            st.rerun()

    st.divider()
    if st.button("New Chat", use_container_width=True):
        has_graph = st.session_state.knowledge_graph is not None
        st.session_state.messages = [get_default_greeting(has_graph)]
        chat_file = get_chat_file_path()
        if chat_file.exists():
            chat_file.unlink()
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
        # Build graph context if available
        graph_ctx = ""
        if st.session_state.knowledge_graph and st.session_state.active_node:
            graph_ctx = build_graph_context(st.session_state.knowledge_graph, st.session_state.active_node)

        active_api_key = nvidia_api_key if provider_key == "nvidia" else google_api_key
        base_url_param = DEFAULT_NVIDIA_BASE_URL if provider_key == "nvidia" else st.session_state.get("ollama_base_url", DEFAULT_OLLAMA_BASE_URL)

        response = st.write_stream(
            stream_response(
                prompt,
                provider_key,
                model_name,
                base_system_prompt,
                api_key=active_api_key,
                base_url=base_url_param,
                file_attachments=file_attachments or None,
                audio_data=audio_data,
                chat_history=st.session_state.messages[:-1],
                graph_context=graph_ctx,
            )
        )
        if response:
            msg = {"role": "assistant", "content": response, "timestamp": datetime.now().isoformat()}
            if '_pending_tutor_log' in st.session_state:
                tutor_log = st.session_state._pending_tutor_log
                msg["tutor_log"] = tutor_log
                del st.session_state._pending_tutor_log

                # Update knowledge graph if applicable
                if st.session_state.knowledge_graph and tutor_log.get("node"):
                    update_node_from_log(st.session_state.knowledge_graph, tutor_log)
                    save_graph(st.session_state.knowledge_graph)

                    # Auto-advance if mastered
                    if tutor_log.get("node_verdict") == "mastered":
                        next_node = get_next_node(st.session_state.knowledge_graph)
                        if next_node:
                            st.session_state.active_node = next_node
                            st.toast(f"✅ Mastered! Moving to: {st.session_state.knowledge_graph['nodes'][next_node]['label']}")

                    # Apply decay to all mastered nodes
                    for node in st.session_state.knowledge_graph.get("nodes", {}).values():
                        if compute_node_state(node) == "mastered":
                            node["decay_score"] = node.get("decay_score", 1.0) * 0.95
                    save_graph(st.session_state.knowledge_graph)

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
