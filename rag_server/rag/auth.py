"""Shared API key loader. Precedence: RAG_API_KEY env var > .rag_key file."""

import os
from pathlib import Path

KEY_FILE = Path(__file__).parent.parent / ".rag_key"


def load_api_key() -> str:
    key = os.environ.get("RAG_API_KEY")
    if key:
        return key.strip()
    if KEY_FILE.exists():
        perms = oct(KEY_FILE.stat().st_mode)[-3:]
        if perms != "600":
            print(f"[WARN] .rag_key permissions are {perms}, expected 600. Run: chmod 600 .rag_key")
        return KEY_FILE.read_text().strip()
    raise SystemExit(
        "Error: no API key found. Either set RAG_API_KEY env var or create .rag_key\n"
        "  python3 -c \"import secrets; print(secrets.token_hex(32))\" > .rag_key && chmod 600 .rag_key"
    )
