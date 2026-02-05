# CLAUDE.md — 1:1 Tutoring Prototype

## What this is

AI-powered 1:1 tutoring assistant built with Streamlit + Google Gemini. Single-page app focused on scaffolded, Bloom's 2 Sigma-style tutoring. Currently configured as "ChemBuddy" (chemistry tutor) but the system prompt is pluggable.

## Tech stack

- **Python / Streamlit** — UI and session management
- **Google Gemini** (`gemini-3-flash-preview`) via LangChain — LLM backend
- **Pillow** — image compression
- **streamlit-drawable-canvas-fix** — in-app drawing tool
- **Local JSON** (`chat_history.json`) — chat persistence (no database)

## Project structure

```
app.py                  # Entire application (single file)
requirements.txt        # Python dependencies
.streamlit/secrets.toml # System prompt + secrets (not committed)
chat_history.json       # Persisted chat state (auto-generated)
```

## Running the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

API key is entered by the user in the sidebar at runtime (Google AI Studio key).

## Key architecture decisions

- **Single-file app**: Everything lives in `app.py`. Keep it that way unless there's a strong reason to split.
- **System prompt in secrets.toml**: The tutoring persona/pedagogy is defined in `.streamlit/secrets.toml` under `SYSTEM_PROMPT`. Falls back to a generic prompt if missing.
- **Multi-modal input**: Users can send text + images + audio + canvas drawings in a single message. Files are base64-encoded for JSON storage.
- **Streaming responses**: Uses `st.write_stream()` for real-time output.
- **Model caching**: `@st.cache_resource` keeps a single Gemini instance alive across requests.

## Important patterns

- **File validation**: Images capped at 10 MB, PDFs at 50 MB. Images auto-compressed to max 1024px before sending to API.
- **Error handling**: Specific error messages for auth failures, rate limits, timeouts, and network issues. Don't weaken this.
- **Chat persistence**: `save_chat()` / `load_chat()` serialize everything (including binary files as base64) to `chat_history.json`. "New Chat" button resets state and deletes the file.
- **Session state**: `st.session_state.messages`, `st.session_state.canvas_version`, `st.session_state.google_api_key`.

## When making changes

- The system prompt in `secrets.toml` is the main lever for tuning behavior — modify that before changing code.
- No tests exist yet. Test manually via the Streamlit UI.
- Don't add a database unless persistence needs outgrow a single JSON file.
- Keep dependencies minimal. This is a prototype.
