#!/usr/bin/env python3
"""
Endpointer Inference Server
Streams audio and returns turn-end predictions synchronized with STT.
Based on the LSTM endpointer model from infer_ep.py
"""

import asyncio
import logging
import os
import sys
from typing import Optional

import msgpack
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from moshi.models import loaders

# Set offline mode for HuggingFace
os.environ["HUGGINGFACE_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

logging.basicConfig(level=logging.DEBUG)  # Changed to DEBUG
logger = logging.getLogger(__name__)

# Constants
SAMPLE_RATE = 24000
INCOMING_FRAME_SIZE = 960  # 40ms at 24kHz - actual frame size from frontend
FRAME_RATE = SAMPLE_RATE / INCOMING_FRAME_SIZE  # 25Hz
MIMI_CHUNK_SIZE = 1920  # Mimi requires 80ms chunks (1920 samples at 24kHz)
ENDPOINTER_OUTPUT_RATE = 12.5  # Hz (same as Mimi encoding rate)
FRAMES_PER_OUTPUT = int(FRAME_RATE / ENDPOINTER_OUTPUT_RATE)  # 2 frames of 960 = 1920 samples
MAX_AUDIO_SAMPLES_PER_MESSAGE = SAMPLE_RATE * 5  # 5-second safety cap

# Model configuration
MODEL_CHECKPOINT = "/mnt/matylda4/udupa/exps/endpointing/smart-endpointing/checkpoints/humdial_lstm_mimi-12.5hz-nq8_delay2f_load_spokenwoz/best_val_acc.pt"
HF_MODEL = "kyutai/stt-1b-en_fr"

# Label indices
LABELS = ["bos", "system_end", "user_end", "system", "user"]
USER_END_IDX = 2


class LSTM_Model(torch.nn.Module):
    """LSTM-based endpointer model."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float,
        bidirectional: bool = False,
    ):
        super(LSTM_Model, self).__init__()
        self.model = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.linear = nn.Linear(hidden_size, output_size)
        self.system_embed = nn.Embedding(2, input_size)
        
    def init_hidden(self, batch_size: int, device: torch.device):
        """Initialize hidden states for LSTM."""
        h_0 = torch.zeros(
            self.model.num_layers, batch_size, self.model.hidden_size, device=device
        )
        c_0 = torch.zeros(
            self.model.num_layers, batch_size, self.model.hidden_size, device=device
        )
        return h_0, c_0
    
    def forward(
        self, 
        x: torch.Tensor, 
        h: Optional[torch.Tensor] = None, 
        c: Optional[torch.Tensor] = None,
        system_ids: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for streaming inference.
        
        Args:
            x: Input tensor of shape (batch, channels, sequence)
            h: Hidden state from previous step
            c: Cell state from previous step
            system_ids: System embedding indices
            
        Returns:
            output: Predictions of shape (batch, sequence, num_classes)
            h: Updated hidden state
            c: Updated cell state
        """
        batch_size = x.size(0)
        
        # Initialize hidden states if not provided
        if h is None or c is None:
            h, c = self.init_hidden(batch_size, x.device)
        
        # Initialize system IDs if not provided
        if system_ids is None:
            system_ids = torch.zeros(x.size(2), dtype=torch.long, device=x.device)
        
        # Transform: (batch, channels, sequence) -> (batch, sequence, channels)
        x = x.permute(0, 2, 1)
        
        # Add system embeddings
        x = x + self.system_embed(system_ids)
        
        # LSTM forward pass
        x, (h_new, c_new) = self.model(x, (h, c))
        
        # Linear projection to output classes
        x = self.linear(x)
        
        return x, h_new, c_new


class EndpointerSession:
    """Manages a single endpointer streaming session."""
    
    def __init__(self, mimi_model, endpointer_model, device: torch.device):
        self.mimi = mimi_model
        self.endpointer = endpointer_model
        self.device = device
        
        # Initialize LSTM hidden states
        self.h, self.c = self.endpointer.init_hidden(1, device)
        
        # Audio buffering for Mimi (needs 1920 samples = 80ms)
        self.audio_buffer = []
        self.audio_buffer_size = 0
        
        # Mimi expects chunks of 1920 samples (80ms at 24kHz)
        self.mimi_chunk_size = 1920
        
        # Statistics
        self.total_frames_received = 0
        self.total_outputs_sent = 0
        
        logger.info("EndpointerSession initialized")
    
    async def process_audio_frame(self, pcm: np.ndarray) -> Optional[float]:
        """
        Process a single audio frame (960 samples at 24kHz = 40ms).
        Buffers 2 frames to get 1920 samples for Mimi encoding.
        
        Args:
            pcm: Audio frame as float32 numpy array
            
        Returns:
            User-end probability if enough frames buffered, None otherwise
        """
        if len(pcm) > MAX_AUDIO_SAMPLES_PER_MESSAGE:
            raise ValueError(
                f"Audio packet too large: {len(pcm)} samples; max={MAX_AUDIO_SAMPLES_PER_MESSAGE}"
            )
        if len(pcm) != INCOMING_FRAME_SIZE:
            logger.warning(f"Expected {INCOMING_FRAME_SIZE} samples, got {len(pcm)}")
            # Pad or truncate to expected size
            if len(pcm) < INCOMING_FRAME_SIZE:
                pcm = np.pad(pcm, (0, INCOMING_FRAME_SIZE - len(pcm)), mode='constant')
            else:
                pcm = pcm[:INCOMING_FRAME_SIZE]
        
        self.total_frames_received += 1
        
        # Buffer frames until we have enough for Mimi (1920 samples)
        self.audio_buffer.append(pcm)
        self.audio_buffer_size += len(pcm)
        
        # Process when we have 1920 samples (2 frames of 960)
        if self.audio_buffer_size >= self.mimi_chunk_size:
            # Concatenate buffered audio
            audio_chunk = np.concatenate(self.audio_buffer)
            
            # Take exactly 1920 samples
            chunk_to_process = audio_chunk[:self.mimi_chunk_size]
            
            # Keep remainder for next iteration
            remainder = audio_chunk[self.mimi_chunk_size:]
            if len(remainder) > 0:
                self.audio_buffer = [remainder]
                self.audio_buffer_size = len(remainder)
            else:
                self.audio_buffer = []
                self.audio_buffer_size = 0
            
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(chunk_to_process).float().unsqueeze(0).to(self.device)
            
            # Encode with Mimi to get embeddings
            with torch.no_grad():
                codes = self.mimi.encode(audio_tensor.unsqueeze(0))
                embeddings = self.mimi.quantizer.decode(codes)
            
            # Run endpointer inference
            with torch.no_grad():
                output, self.h, self.c = self.endpointer(
                    embeddings, self.h, self.c
                )
                
                # Get probabilities
                logits = output.squeeze(0)
                probs = torch.softmax(logits, dim=-1)
                user_end_prob = probs[-1, USER_END_IDX].item()
            
            self.total_outputs_sent += 1
            
            logger.debug(f"Frame {self.total_frames_received}: Generated prediction #{self.total_outputs_sent}, prob={user_end_prob:.4f}")
            
            return user_end_prob
        
        return None
    
    def get_stats(self) -> dict:
        """Get session statistics."""
        return {
            "frames_received": self.total_frames_received,
            "outputs_sent": self.total_outputs_sent,
            "expected_output_rate_hz": ENDPOINTER_OUTPUT_RATE,
        }


# Global model instances (loaded once on startup)
mimi_model = None
endpointer_model = None
device = None


def load_models():
    """Load Mimi and Endpointer models."""
    global mimi_model, endpointer_model, device
    
    logger.info("Loading models...")
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load Mimi model
    logger.info(f"Loading Mimi model from {HF_MODEL}")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(HF_MODEL)
    mimi_model = checkpoint_info.get_mimi(device=device)
    mimi_model.streaming_forever(batch_size=1)
    logger.info("Mimi model loaded")
    
    # Load Endpointer model
    logger.info(f"Loading Endpointer model from {MODEL_CHECKPOINT}")
    model_kwargs = {
        "input_size": 512,
        "hidden_size": 324,
        "num_layers": 3,
        "output_size": 5,
        "dropout": 0.1,
        "bidirectional": False,
    }
    
    endpointer_model = LSTM_Model(**model_kwargs)
    checkpoint = torch.load(MODEL_CHECKPOINT, map_location=device, weights_only=True)
    endpointer_model.load_state_dict(checkpoint["model_state_dict"])
    endpointer_model.to(device)
    endpointer_model.eval()
    logger.info("Endpointer model loaded")


# FastAPI app
app = FastAPI(title="Endpointer Inference Server")


@app.on_event("startup")
async def startup_event():
    """Load models on server startup."""
    load_models()


@app.get("/api/build_info")
async def build_info():
    """Return build information."""
    return JSONResponse({
        "service": "endpointer",
        "version": "1.0.0",
        "model_checkpoint": MODEL_CHECKPOINT,
        "sample_rate": SAMPLE_RATE,
        "incoming_frame_size": INCOMING_FRAME_SIZE,
        "mimi_chunk_size": MIMI_CHUNK_SIZE,
        "output_rate_hz": ENDPOINTER_OUTPUT_RATE,
        "device": str(device),
    })


@app.websocket("/api/endpointer_stream")
async def endpointer_stream(websocket: WebSocket):
    """
    WebSocket endpoint for streaming endpointer inference.
    
    Receives audio frames and returns user-end probabilities.
    """
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    
    # Create session
    session = EndpointerSession(mimi_model, endpointer_model, device)
    
    # Send ready message
    ready_msg = msgpack.packb({"type": "Ready"}, use_bin_type=True)
    await websocket.send_bytes(ready_msg)
    logger.info("Sent Ready message")
    
    try:
        while True:
            # Receive message
            message_bytes = await websocket.receive_bytes()
            message = msgpack.unpackb(message_bytes)
            
            msg_type = message.get("type")
            
            if msg_type == "Audio":
                # Process audio frame
                pcm_list = message.get("pcm", [])
                pcm = np.array(pcm_list, dtype=np.float32)
                
                # Process frame and get probability if available
                user_end_prob = await session.process_audio_frame(pcm)
                
                if user_end_prob is not None:
                    # Send prediction
                    response = msgpack.packb(
                        {
                            "type": "Prediction",
                            "user_end_probability": user_end_prob,
                            "frame_count": session.total_frames_received,
                        },
                        use_bin_type=True,
                        use_single_float=True,
                    )
                    await websocket.send_bytes(response)
            
            elif msg_type == "GetStats":
                # Send statistics
                stats = session.get_stats()
                response = msgpack.packb(
                    {"type": "Stats", **stats},
                    use_bin_type=True,
                )
                await websocket.send_bytes(response)
            
            else:
                logger.warning(f"Unknown message type: {msg_type}")
    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in WebSocket handler: {e}", exc_info=True)
        error_msg = msgpack.packb(
            {"type": "Error", "message": str(e)},
            use_bin_type=True,
        )
        try:
            await websocket.send_bytes(error_msg)
        except:
            pass
    finally:
        stats = session.get_stats()
        logger.info(f"Session ended. Stats: {stats}")


if __name__ == "__main__":
    import uvicorn
    
    # Parse port from command line
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8092, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    args = parser.parse_args()
    
    logger.info(f"Starting Endpointer Inference Server on {args.host}:{args.port}")
    
    # Run with websockets support and longer timeout
    uvicorn.run(
        app, 
        host=args.host, 
        port=args.port,
        ws="websockets",  # Explicitly use websockets library
        log_level="info",
        timeout_keep_alive=300,  # 5 minutes keepalive timeout
        ws_ping_interval=60,  # Ping every 60 seconds
        ws_ping_timeout=60,  # Wait 60 seconds for pong
    )
