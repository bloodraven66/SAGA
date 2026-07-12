import os
from pathlib import Path

from unmute.websocket_utils import http_to_ws

HEADERS = {"kyutai-api-key": "public_token"}

# The defaults are already ws://, but make the env vars support http:// and https://
# STT_SERVER = http_to_ws(os.environ.get("KYUTAI_STT_URL", "ws://localhost:8090"))
# TTS_SERVER = http_to_ws(os.environ.get("KYUTAI_TTS_URL", "ws://localhost:8089"))
# LLM_SERVER = os.environ.get("KYUTAI_LLM_URL", "http://localhost:8091")

node = os.environ.get("KYUTAI_NODE", "localhost")
llm_node = os.environ.get("KYUTAI_LLM_NODE", node)

STT_SERVER = os.environ.get("KYUTAI_STT_URL", f"ws://{node}:8090")
TTS_SERVER = os.environ.get("KYUTAI_TTS_URL", f"ws://{node}:8089")
LLM_SERVER = os.environ.get("KYUTAI_LLM_URL", f"http://{llm_node}:8091")
ENDPOINTER_SERVER = os.environ.get("KYUTAI_ENDPOINTER_URL", f"ws://{node}:8093")

# Speech_Research_RAG server (agentic multi-collection RAG backend), used by
# unmute_handler_agentic_rag.py. Replaces the old FIT-course RAG backend for
# this line of work.
SPEECH_RAG_SERVER = os.environ.get("KYUTAI_SPEECH_RAG_URL", f"http://{node}:8096")
SPEECH_RAG_API_KEY = os.environ.get("RAG_API_KEY")

# Qwen3.5-9B vLLM server (dockerless/start_llm_qwen.sh), used by
# unmute_handler_agentic_rag.py for its tool-calling loop. Separate from
# LLM_SERVER (Gemma 3, used by the other handlers) so both can run at once.
AGENTIC_LLM_SERVER = os.environ.get("KYUTAI_AGENTIC_LLM_URL", f"http://{node}:8097")
AGENTIC_LLM_MODEL = os.environ.get("KYUTAI_AGENTIC_LLM_MODEL", "Qwen/Qwen3.5-9B")

# Standalone fast-ASR debug endpoint (dockerless/fd_asr_server.py), used by the
# eval scripts to get independent, ground-truth channel transcripts with
# word-level timestamps -- decoupled from the live STT pipeline's own event
# bookkeeping. Optional: eval scripts should degrade gracefully if unset/down.
FD_ASR_SERVER = os.environ.get("KYUTAI_FD_ASR_URL", f"http://{node}:8098")

KYUTAI_LLM_MODEL = os.environ.get("KYUTAI_LLM_MODEL")
KYUTAI_LLM_API_KEY = os.environ.get("KYUTAI_LLM_API_KEY")
VOICE_CLONING_SERVER = os.environ.get(
    "KYUTAI_VOICE_CLONING_URL", "http://localhost:8094"
)
# ENDPOINTER_SERVER = http_to_ws(os.environ.get("KYUTAI_ENDPOINTER_URL", "ws://localhost:8092"))

# If None, a dict-based cache will be used instead of Redis
REDIS_SERVER = os.environ.get("KYUTAI_REDIS_URL")

SPEECH_TO_TEXT_PATH = "/api/asr-streaming"
TEXT_TO_SPEECH_PATH = "/api/tts_streaming"
ENDPOINTER_PATH = "/api/endpointer_stream"

repo_root = Path(__file__).parents[1]
VOICE_DONATION_DIR = Path(
    os.environ.get("KYUTAI_VOICE_DONATION_DIR", repo_root / "voices" / "donation")
)

# If None, recordings will not be saved
_recordings_dir = os.environ.get("KYUTAI_RECORDINGS_DIR")
RECORDINGS_DIR = Path(_recordings_dir) if _recordings_dir else None

# Also checked on the frontend, see constant of the same name
MAX_VOICE_FILE_SIZE_MB = 4


SAMPLE_RATE = 24000
SAMPLES_PER_FRAME = 1920
FRAME_TIME_SEC = SAMPLES_PER_FRAME / SAMPLE_RATE  # 0.08
# TODO: make it so that we can read this from the ASR server?
STT_DELAY_SEC = 0.5
