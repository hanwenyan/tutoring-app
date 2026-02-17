"""
Pure utility functions for the tutoring app.
No Streamlit dependencies — only standard library and PIL.
"""

import io
import re

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
    from PIL import Image
    img = Image.open(io.BytesIO(image_bytes))
    # Early return if image already fits within max_dimension
    if max(img.size) <= max_dimension:
        return image_bytes
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


# Known TUTOR_LOG field names (longest first to avoid partial matches like node vs node_verdict)
_TUTOR_LOG_KEYS = sorted(
    ["step", "topic", "student_status", "misconception", "hint_level",
     "node_verdict", "node", "dependency_suspect", "problem_type"],
    key=len, reverse=True,
)
_LOG_SPLIT_RE = re.compile(
    r'(?:^|\s)(' + '|'.join(_TUTOR_LOG_KEYS) + r')\s*:', re.IGNORECASE
)


def parse_log_fields(log_text: str) -> dict:
    """Parse TUTOR_LOG body into a dict. Handles both newline-separated and single-line formats."""
    parts = _LOG_SPLIT_RE.split(log_text)
    # parts layout: [before, key1, val1, key2, val2, ...]
    log = {}
    for i in range(1, len(parts), 2):
        key = parts[i].strip()
        val = parts[i + 1].strip() if i + 1 < len(parts) else ""
        log[key] = val
    return log


def parse_tutor_log(text: str) -> tuple[str, dict | None]:
    """Extract [TUTOR_LOG]...[/TUTOR_LOG] from response text.
    Returns (clean_text, tutor_log_dict or None)."""
    match = re.search(r'\[TUTOR_LOG\](.*?)\[/TUTOR_LOG\]', text, re.DOTALL)
    if not match:
        return text.strip(), None
    log_text = match.group(1).strip()
    log = parse_log_fields(log_text)
    clean = text[:match.start()] + text[match.end():]
    return clean.strip(), log


_PROTECTED_BLOCK_RE = re.compile(
    r'(```[\s\S]*?```'
    r'|\$\$[\s\S]*?\$\$'
    r'|\$[^$\n]+\$)',
)
_INLINE_NUMBERED_RE = re.compile(r'(?<=\S)(?:[ \t]+|\n)(\d+\.\s)')
_INLINE_BULLET_RE = re.compile(r'(?<=\S)(?:[ \t]+|\n)([-*]\s)')
_INLINE_BOLD_HEADER_RE = re.compile(r'(?<=\S)(?:[ \t]+|\n)(\*\*[^*]+:\*\*)')
_INLINE_BOLD_STANDALONE_RE = re.compile(r'(?<=\S)(?:[ \t]+|\n)(\*\*[^*]+\*\*)\s*(?=\n|$)', re.MULTILINE)
_INLINE_HEADING_RE = re.compile(r'(?<=\S)(?:[ \t]+|\n)(#{1,4}\s)')


def normalize_markdown_newlines(text: str) -> str:
    """Ensure proper Markdown paragraph breaks in LLM response text.

    Protects code fences and display LaTeX from modification.
    Idempotent: running twice yields the same result.
    """
    # 1. Stash protected blocks
    blocks: list[str] = []

    def _stash(m: re.Match) -> str:
        blocks.append(m.group(0))
        return f'\x00BLOCK{len(blocks) - 1}\x00'

    work = _PROTECTED_BLOCK_RE.sub(_stash, text)

    # 2. Normalize prose
    work = re.sub(r'\n{2,}', '\n\n', work)                    # collapse excess blanks
    work = _INLINE_NUMBERED_RE.sub(r'\n\n\1', work)           # "text 1. ..." → break
    work = _INLINE_BULLET_RE.sub(r'\n\n\1', work)             # "text - ..." → break
    work = _INLINE_BOLD_HEADER_RE.sub(r'\n\n\1', work)        # "text **Hdr:**" → break
    work = _INLINE_BOLD_STANDALONE_RE.sub(r'\n\n\1', work)    # "text **Hdr**\n" → break
    work = _INLINE_HEADING_RE.sub(r'\n\n\1', work)            # "text ## Hdr" → break
    # Clamp h1/h2 to h3 (too large for chat bubbles)
    work = re.sub(r'^#{1,2}(\s)', r'###\1', work, flags=re.MULTILINE)
    # Repair orphaned list markers (broken by inline-break regexes above)
    work = re.sub(r'^(\s*[-*])[ \t]*\n\n+', r'\1 ', work, flags=re.MULTILINE)
    work = re.sub(r'^(\s*\d+\.)[ \t]*\n\n+', r'\1 ', work, flags=re.MULTILINE)
    work = re.sub(r'\n{3,}', '\n\n', work)                    # final cleanup

    # 3. Restore protected blocks
    for i, block in enumerate(blocks):
        work = work.replace(f'\x00BLOCK{i}\x00', block)

    return work


_CURRENCY_RE = re.compile(r'\$(\d[\d,]*(?:\.\d{1,2})?)(?=[\s,;:.!?)\]}\-—\'"]|$)')
_PROTECTED_CURRENCY_RE = re.compile(r'(```[\s\S]*?```|`[^`]+`|\$\$[\s\S]*?\$\$|\\\([\s\S]*?\\\))')


def escape_currency_dollars(text: str) -> str:
    """Escape dollar signs that are currency (e.g. $110, $5.99) to prevent LaTeX rendering."""
    blocks: list[str] = []

    def _stash(m: re.Match) -> str:
        blocks.append(m.group(0))
        return f'\x00CURR{len(blocks) - 1}\x00'

    work = _PROTECTED_CURRENCY_RE.sub(_stash, text)
    work = _CURRENCY_RE.sub(r'\\$\1', work)
    for i, b in enumerate(blocks):
        work = work.replace(f'\x00CURR{i}\x00', b)
    return work


def fix_multiline_inline_math(text: str) -> str:
    """Join newlines inside $...$ inline math so KaTeX doesn't break mid-expression."""
    blocks: list[str] = []

    def _stash(m: re.Match) -> str:
        blocks.append(m.group(0))
        return f'\x00FIX{len(blocks) - 1}\x00'

    # Stash code fences and display math ($$...$$) — don't touch these
    stash_re = re.compile(r'(```[\s\S]*?```|\$\$[\s\S]*?\$\$)')
    work = stash_re.sub(_stash, text)

    # Replace newlines inside $...$ with a space
    work = re.sub(r'\$([^$]+?)\$', lambda m: '$' + m.group(1).replace('\n', ' ') + '$', work)

    for i, block in enumerate(blocks):
        work = work.replace(f'\x00FIX{i}\x00', block)
    return work


def ensure_dollar_parity(text: str) -> str:
    """Truncate text at the last unpaired $ to prevent KaTeX from consuming trailing content.

    During streaming, a $..$ expression may be split across chunks. This prevents
    the opening $ from being matched with a later unrelated $ in subsequent chunks.
    No-ops on text where single-$ count is even (all pairs closed).
    """
    # Stash code fences and inline code to avoid counting their $ signs
    blocks: list[str] = []

    def _stash(m: re.Match) -> str:
        blocks.append(m.group(0))
        return f'\x00PAR{len(blocks) - 1}\x00'

    stash_re = re.compile(r'(```[\s\S]*?```|`[^`]+`|\$\$[\s\S]*?\$\$)')
    work = stash_re.sub(_stash, text)

    # Find positions of single unescaped $
    positions = []
    i = 0
    while i < len(work):
        if work[i] == '$':
            if i + 1 < len(work) and work[i + 1] == '$':
                i += 2  # skip $$
                continue
            if i > 0 and work[i - 1] == '\\':
                i += 1  # skip \$
                continue
            positions.append(i)
        i += 1

    if len(positions) % 2 == 1:
        # Odd count — truncate at the last (unpaired) $
        last_unpaired = positions[-1]
        work = work[:last_unpaired]

    for i, block in enumerate(blocks):
        work = work.replace(f'\x00PAR{i}\x00', block)
    return work


def process_for_display(text: str, final: bool = False) -> str:
    """Apply the full display processing pipeline to LLM response text.

    During streaming (final=False): escapes currency, converts delimiters,
    fixes multiline inline math, and truncates at unpaired $ to prevent
    KaTeX from consuming in-progress expressions.

    On final render (final=True): skips dollar-parity truncation and runs
    normalize_markdown_newlines for proper paragraph breaks.
    """
    text = escape_currency_dollars(text)
    text = convert_latex_delimiters(text)
    text = fix_multiline_inline_math(text)
    if not final:
        text = ensure_dollar_parity(text)
    if final:
        text = normalize_markdown_newlines(text)
    return text


def convert_latex_delimiters(text: str) -> str:
    """Convert LaTeX delimiters \(...\) → $...$ and \[...\] → $$...$$.

    Protects code fences and inline code from modification.
    Streamlit's KaTeX only recognizes $ delimiters, not \( \) \[ \].
    """
    # 1. Stash protected blocks
    blocks: list[str] = []

    def _stash(m: re.Match) -> str:
        blocks.append(m.group(0))
        return f'\x00BLOCK{len(blocks) - 1}\x00'

    # Protect code fences and inline code
    protected_re = re.compile(r'(```[\s\S]*?```|`[^`]+`)')
    work = protected_re.sub(_stash, text)

    # 2. Convert delimiters
    work = work.replace(r'\(', '$')
    work = work.replace(r'\)', '$')
    work = work.replace(r'\[', '$$')
    work = work.replace(r'\]', '$$')

    # 3. Restore protected blocks
    for i, block in enumerate(blocks):
        work = work.replace(f'\x00BLOCK{i}\x00', block)

    return work


def relative_time(iso_str: str) -> str:
    """Convert ISO timestamp to relative time string."""
    from datetime import datetime
    try:
        ts = datetime.fromisoformat(iso_str)
        delta = datetime.now() - ts
        seconds = delta.total_seconds()
        if seconds < 60:
            return "just now"
        elif seconds < 3600:
            mins = int(seconds // 60)
            return f"{mins} min ago"
        elif seconds < 86400:
            hours = int(seconds // 3600)
            return f"{hours} {'hour' if hours == 1 else 'hours'} ago"
        elif seconds < 172800:
            return "yesterday"
        else:
            return ts.strftime("%b %d") if ts.year == datetime.now().year else ts.strftime("%b %d, %Y")
    except (ValueError, TypeError):
        return iso_str[:16].replace("T", " ")


