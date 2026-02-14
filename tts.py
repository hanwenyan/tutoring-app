"""
Text-to-speech engine using Kokoro ONNX.
"""

import io
from pathlib import Path

import streamlit as st
import requests as _requests_lib
import soundfile as sf
from kokoro_onnx import Kokoro

from utils import strip_markdown


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
