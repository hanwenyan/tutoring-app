# 1:1 Tutor

AI-powered tutoring assistant using Streamlit and Google Gemini.

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the app:
   ```bash
   streamlit run app.py
   ```

3. Enter your Google API key in the sidebar

4. Start chatting or upload images of worksheets/homework

## Custom System Prompt

Edit `.streamlit/secrets.toml` to customize the tutor's behavior:

```toml
SYSTEM_PROMPT = """
Your custom instructions here...
"""
```

## Features

- Text-based tutoring chat
- Image upload for worksheets, diagrams, and homework
- Powered by Gemini 3 Flash for fast responses
