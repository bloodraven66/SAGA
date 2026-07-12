#!/usr/bin/env python3
"""
Anticipator Inference Server v2

Key improvements over v1:
- Explicit frame contract in build info and ready message.
- Robust audio payload rechunking (supports 960 and 1920 sample client packets
  without truncating away half the samples).
- Cleaner session stats and lower-noise logging.
- Stable websocket serving defaults for evaluation/deployment.
"""

from __future__ import annotations

import argparse
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Optional

import msgpack
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from moshi.models import loaders
from moshi.modules.transformer import StreamingTransformer


# Offline mode for HuggingFace models.
os.environ["HUGGINGFACE_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
# Disable torch.compile / TorchInductor to avoid nvcc permission issues.
os.environ["TORCHDYNAMO_DISABLE"] = "1"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Audio/model constants.
SAMPLE_RATE = 24000
BASE_FRAME_SIZE = 960  # 40ms at 24kHz
MIMI_CHUNK_SIZE = 1920  # 80ms at 24kHz
ENDPOINTER_OUTPUT_RATE = 12.5
ACCEPTED_FRAME_SIZES = {960, 1920}
MAX_AUDIO_SAMPLES_PER_MESSAGE = SAMPLE_RATE * 5  # 5 seconds safety cap

MODEL_CHECKPOINT = "/mnt/matylda4/udupa/exps/endpointing/NAC-LD-Endpointer/checkpoints/data_mix_4__fc960_transformer_mimi_12.5hz_loss1-01_m3/best_val_acc.pt"
HF_MODEL = "kyutai/stt-1b-en_fr"


class FCTransformerModel(nn.Module):
    def __init__(self, **kwargs: Any):
        super().__init__()
        self.model1 = self._make_transformer(kwargs)
        self.model2 = self._make_transformer(kwargs)
        self.linear = nn.Linear(kwargs["hidden_size"] * 2, 1)

    @staticmethod
    def _make_transformer(kwargs: dict[str, Any]) -> StreamingTransformer:
        return StreamingTransformer(
            d_model=kwargs["hidden_size"],
            num_heads=kwargs["num_heads"],
            num_layers=kwargs["num_layers"],
            dim_feedforward=kwargs["dim_feedforward"],
            causal=True,
            context=kwargs["context"],
            positional_embedding=kwargs["positional_embedding"],
            max_period=kwargs["max_period"],
        )

    def forward(self, x: torch.Tensor):
        # x: (channels=2, features=512, T)
        x = x.reshape(1, -1, x.size(1), x.size(2))
        x1 = x[:, 0, :, :].permute(0, 2, 1)
        x2 = x[:, 1, :, :].permute(0, 2, 1)
        x1 = self.model1(x1)
        x2 = self.model2(x2)
        x = torch.cat([x1, x2], dim=2)
        return self.linear(x)


@dataclass
class SessionStats:
    messages_received: int = 0
    samples_received: int = 0
    frames_received: int = 0
    outputs_sent: int = 0
    mismatched_packets: int = 0


class AnticipatorSession:
    """Single websocket session state for anticipator streaming inference."""

    def __init__(
        self,
        mimi_model: Any,
        anticipator_model: Any,
        device: torch.device,
        max_context_steps: int = 240,
    ):
        self.mimi = mimi_model
        self.anticipator = anticipator_model
        self.device = device
        self.max_context_steps = max_context_steps

        self.base_frame_fifo = np.zeros(0, dtype=np.float32)
        self.mimi_chunk_fifo = np.zeros(0, dtype=np.float32)

        self.zero_stream_cache: torch.Tensor | None = None
        self.cached_embeds: torch.Tensor | None = None

        self.stats = SessionStats()

    def _coerce_audio_payload(self, pcm: np.ndarray) -> np.ndarray:
        if pcm.ndim != 1:
            pcm = pcm.reshape(-1)
        if pcm.dtype != np.float32:
            pcm = pcm.astype(np.float32)
        if len(pcm) == 0:
            return pcm
        if len(pcm) > MAX_AUDIO_SAMPLES_PER_MESSAGE:
            raise ValueError(
                f"Audio packet too large: {len(pcm)} samples; max={MAX_AUDIO_SAMPLES_PER_MESSAGE}"
            )

        # Replace NaN/Inf to avoid contaminating model state.
        if not np.isfinite(pcm).all():
            pcm = np.nan_to_num(pcm, nan=0.0, posinf=1.0, neginf=-1.0)

        if len(pcm) not in ACCEPTED_FRAME_SIZES:
            self.stats.mismatched_packets += 1
            logger.warning(
                "Unexpected anticipator packet size %d (accepted: %s); rechunking stream.",
                len(pcm),
                sorted(ACCEPTED_FRAME_SIZES),
            )

        return pcm

    async def process_audio_payload(self, pcm: np.ndarray) -> list[tuple[float, int]]:
        """
        Process arbitrary-length packet by rechunking into 960-sample frames,
        then into 1920-sample Mimi chunks.

        Returns a list of (probability, frame_count) predictions generated
        by this packet.
        """
        pcm = self._coerce_audio_payload(pcm)
        if len(pcm) == 0:
            return []

        self.stats.messages_received += 1
        self.stats.samples_received += len(pcm)

        self.base_frame_fifo = np.concatenate([self.base_frame_fifo, pcm])

        predictions: list[tuple[float, int]] = []
        while len(self.base_frame_fifo) >= BASE_FRAME_SIZE:
            frame = self.base_frame_fifo[:BASE_FRAME_SIZE]
            self.base_frame_fifo = self.base_frame_fifo[BASE_FRAME_SIZE:]
            self.stats.frames_received += 1

            prob = await self._process_base_frame(frame)
            if prob is not None:
                self.stats.outputs_sent += 1
                predictions.append((prob, self.stats.frames_received))

        return predictions

    async def _process_base_frame(self, frame: np.ndarray) -> Optional[float]:
        """Process one 960-sample frame; returns probability when enough audio accumulated."""
        self.mimi_chunk_fifo = np.concatenate([self.mimi_chunk_fifo, frame])
        if len(self.mimi_chunk_fifo) < MIMI_CHUNK_SIZE:
            return None

        chunk = self.mimi_chunk_fifo[:MIMI_CHUNK_SIZE]
        self.mimi_chunk_fifo = self.mimi_chunk_fifo[MIMI_CHUNK_SIZE:]

        audio_tensor = torch.from_numpy(chunk).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            codes = self.mimi.encode(audio_tensor.unsqueeze(0))
            embeddings = self.mimi.quantizer.decode(codes)

        current_chunk_steps = embeddings.shape[-1]
        if self.cached_embeds is not None:
            embeddings = torch.cat([self.cached_embeds, embeddings], dim=-1)

        if embeddings.shape[-1] > self.max_context_steps:
            embeddings = embeddings[..., -self.max_context_steps :]

        self.cached_embeds = embeddings.clone()

        if self.zero_stream_cache is None:
            zero_stream_audio = torch.zeros_like(audio_tensor)
            with torch.no_grad():
                zero_codes = self.mimi.encode(zero_stream_audio.unsqueeze(0))[:, :, :1]
                self.zero_stream_cache = self.mimi.quantizer.decode(zero_codes)

        assert self.zero_stream_cache is not None
        zero_stream = self.zero_stream_cache.repeat(1, 1, embeddings.shape[-1])
        model_input = torch.cat([embeddings, zero_stream], dim=0)

        with torch.no_grad():
            output = self.anticipator(model_input)
            logits = output.squeeze(0)
            probs = torch.sigmoid(logits)[-current_chunk_steps:]

        return float(probs[-1].item())

    def get_stats(self) -> dict:
        return {
            "messages_received": self.stats.messages_received,
            "samples_received": self.stats.samples_received,
            "frames_received": self.stats.frames_received,
            "outputs_sent": self.stats.outputs_sent,
            "mismatched_packets": self.stats.mismatched_packets,
            "expected_output_rate_hz": ENDPOINTER_OUTPUT_RATE,
            "accepted_frame_sizes": sorted(ACCEPTED_FRAME_SIZES),
            "base_frame_size": BASE_FRAME_SIZE,
            "mimi_chunk_size": MIMI_CHUNK_SIZE,
        }


mimi_model = None
anticipator_model = None
device = None


def load_models() -> None:
    global mimi_model, anticipator_model, device

    logger.info("Loading anticipator v2 models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    logger.info("Loading Mimi model from %s", HF_MODEL)
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(HF_MODEL)
    mimi_model = checkpoint_info.get_mimi(device=device)
    mimi_model.streaming_forever(batch_size=1)
    logger.info("Mimi model loaded")

    logger.info("Loading anticipator model from %s", MODEL_CHECKPOINT)
    model_kwargs = {
        "hidden_size": 512,
        "num_heads": 4,
        "num_layers": 6,
        "dim_feedforward": 1024,
        "context": 240,
        "positional_embedding": "rope",
        "max_period": 1000.0,
    }
    anticipator_model = FCTransformerModel(**model_kwargs)
    checkpoint = torch.load(MODEL_CHECKPOINT, map_location=device, weights_only=True)
    anticipator_model.load_state_dict(checkpoint["model_state_dict"])
    anticipator_model.to(device)
    anticipator_model.eval()
    logger.info("Anticipator model loaded")

    # Warm up GPU kernels with one dummy forward pass so the first real WebSocket
    # session gets predictions immediately instead of waiting 1-3s for CUDA init.
    logger.info("Running GPU warmup forward pass...")
    with torch.no_grad():
        dummy_audio = torch.zeros(1, 1, MIMI_CHUNK_SIZE, device=device)
        _codes = mimi_model.encode(dummy_audio)
        _embeds = mimi_model.quantizer.decode(_codes)
        _zero = _embeds.clone()
        _model_input = torch.cat([_embeds, _zero], dim=0)
        _ = anticipator_model(_model_input)
    logger.info("GPU warmup done.")


app = FastAPI(title="Anticipator Inference Server v2")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_models()
    yield


app.router.lifespan_context = lifespan


@app.get("/api/build_info")
async def build_info():
    return JSONResponse(
        {
            "service": "anticipator",
            "version": "2.0.0",
            "model_checkpoint": MODEL_CHECKPOINT,
            "sample_rate": SAMPLE_RATE,
            "base_frame_size": BASE_FRAME_SIZE,
            "mimi_chunk_size": MIMI_CHUNK_SIZE,
            "accepted_frame_sizes": sorted(ACCEPTED_FRAME_SIZES),
            "output_rate_hz": ENDPOINTER_OUTPUT_RATE,
            "device": str(device),
        }
    )


@app.websocket("/api/endpointer_stream")
async def endpointer_stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("Anticipator websocket accepted")

    session = AnticipatorSession(mimi_model, anticipator_model, device)

    ready_msg = msgpack.packb(
        {
            "type": "Ready",
            "service": "anticipator",
            "version": "2.0.0",
            "accepted_frame_sizes": sorted(ACCEPTED_FRAME_SIZES),
            "base_frame_size": BASE_FRAME_SIZE,
        },
        use_bin_type=True,
    )
    await websocket.send_bytes(ready_msg)

    try:
        while True:
            message_bytes = await websocket.receive_bytes()
            message = msgpack.unpackb(message_bytes, raw=False)
            msg_type = message.get("type")

            if msg_type == "Audio":
                pcm_list = message.get("pcm", [])
                pcm = np.array(pcm_list, dtype=np.float32)

                predictions = await session.process_audio_payload(pcm)
                for prob, frame_count in predictions:
                    response = msgpack.packb(
                        {
                            "type": "Prediction",
                            "user_end_probability": prob,
                            "frame_count": frame_count,
                        },
                        use_bin_type=True,
                        use_single_float=True,
                    )
                    await websocket.send_bytes(response)

            elif msg_type == "GetStats":
                response = msgpack.packb(
                    {"type": "Stats", **session.get_stats()},
                    use_bin_type=True,
                )
                await websocket.send_bytes(response)
            else:
                logger.warning("Unknown websocket message type: %r", msg_type)

    except WebSocketDisconnect:
        logger.info("Anticipator websocket disconnected")
    except Exception as exc:
        logger.error("Error in anticipator websocket handler: %r", exc, exc_info=True)
        err = msgpack.packb(
            {"type": "Error", "message": str(exc)},
            use_bin_type=True,
        )
        try:
            await websocket.send_bytes(err)
        except Exception:
            pass
    finally:
        logger.info("Anticipator session ended. Stats: %s", session.get_stats())


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8093)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    logger.info("Starting Anticipator Inference Server v2 on %s:%s", args.host, args.port)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        ws="websockets",
        log_level="info",
        timeout_keep_alive=300,
        ws_ping_interval=60,
        ws_ping_timeout=60,
    )
