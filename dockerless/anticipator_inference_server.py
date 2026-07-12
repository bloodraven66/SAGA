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
from moshi.modules.transformer import StreamingTransformer


# Set offline mode for HuggingFace
os.environ["HUGGINGFACE_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
# Disable torch.compile / TorchInductor to avoid nvcc permission issues
os.environ["TORCHDYNAMO_DISABLE"] = "1"

logging.basicConfig(level=logging.DEBUG)  # Changed to DEBUG
logger = logging.getLogger(__name__)

# Constants
SAMPLE_RATE = 24000
INCOMING_FRAME_SIZE = 960  # 40ms at 24kHz - actual frame size from frontend
FRAME_RATE = SAMPLE_RATE / INCOMING_FRAME_SIZE  # 25Hz
MIMI_CHUNK_SIZE = 1920  # Mimi requires 80ms chunks (1920 samples at 24kHz)
ENDPOINTER_OUTPUT_RATE = 12.5  # Hz (same as Mimi encoding rate)
FRAMES_PER_OUTPUT = int(FRAME_RATE / ENDPOINTER_OUTPUT_RATE)  # 2 frames of 960 = 1920 samples

# Model configuration
# MODEL_CHECKPOINT = "/mnt/matylda4/udupa/exps/endpointing/NAC-LD-Endpointer/checkpoints/data_mix_2__fc960_transformer_mimi_12.5hz_loss1-01_m3/best_val_acc.pt"
MODEL_CHECKPOINT = "/mnt/matylda4/udupa/exps/endpointing/NAC-LD-Endpointer/checkpoints/data_mix_4__fc960_transformer_mimi_12.5hz_loss1-01_m3/best_val_acc.pt"

HF_MODEL = "kyutai/stt-1b-en_fr"

class FC_Transformer_Model(torch.nn.Module):
    def __init__(
        self, 
        **kwargs
    ):
        super(FC_Transformer_Model, self).__init__()
        self.model1 = self.make_transformer(kwargs)
        self.model2 = self.make_transformer(kwargs)
        self.linear = nn.Linear(kwargs["hidden_size"]*2, 1)
        self.activation = nn.Sigmoid()
    
    def make_transformer(self, kwargs):
        model = StreamingTransformer(
            d_model=kwargs["hidden_size"],
            num_heads=kwargs["num_heads"],
            num_layers=kwargs["num_layers"],
            dim_feedforward=kwargs["dim_feedforward"],
            causal=True,
            context=kwargs["context"],
            positional_embedding=kwargs["positional_embedding"],
            max_period=kwargs["max_period"],
        )
        self.modelname = "transformer"
        return model

    def _forward(self, x):
        
        return x

    def forward(self, x):
        # logging.info(f"{x.shape}")
        # return x
        x = x.reshape(1, -1, x.size(1), x.size(2))  # bs x num_channels x feat_dim x T_frames
        x1 = x[:, 0, :, :].permute(0, 2, 1)
        x2 = x[:, 1, :, :].permute(0, 2, 1)
        x1 = self.model1(x1)
        x2 = self.model2(x2)
        x = torch.cat([x1, x2], dim=2)
        x = self.linear(x)
        return x

class EndpointerSession:
    """Manages a single endpointer streaming session."""
    
    def __init__(self, mimi_model, endpointer_model, device: torch.device):
        self.mimi = mimi_model
        self.endpointer = endpointer_model
        self.device = device
        
        # Audio buffering for Mimi (needs 1920 samples = 80ms)
        self.audio_buffer = []
        self.audio_buffer_size = 0
        
        # Mimi expects chunks of 1920 samples (80ms at 24kHz)
        self.mimi_chunk_size = 1920
        
        # Statistics
        self.total_frames_received = 0
        self.total_outputs_sent = 0
        
        self.zero_stream_cache = None
        self.cached_embeds = None

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
            # audio_tensor_2 = torch.zeros_like(audio_tensor)
            # audio_tensor = torch.cat([audio_tensor, audio_tensor_2], dim=0)
            logger.info(f"input: {audio_tensor.shape}")
            # Encode with Mimi to get embeddings
            with torch.no_grad():
                codes = self.mimi.encode(audio_tensor.unsqueeze(0))
                embeddings = self.mimi.quantizer.decode(codes)
            
            logger.info(f"input embed: {embeddings.shape}")
            current_chunk_size = embeddings.shape[-1]
            if self.cached_embeds is not None:
                embeddings = torch.cat([self.cached_embeds, embeddings], dim=-1)
            self.cached_embeds = embeddings.clone()
            logger.info(f"input embed: {embeddings.shape}")

            if self.zero_stream_cache is None:
                zero_stream_audio = torch.zeros_like(audio_tensor)
                with torch.no_grad():
                    zero_stream_codes = self.mimi.encode(zero_stream_audio.unsqueeze(0))[:, :, :1]
                    self.zero_stream_cache = self.mimi.quantizer.decode(zero_stream_codes)
            

            embeddings = torch.cat([embeddings, self.zero_stream_cache.repeat(1, 1, embeddings.shape[-1])], dim=0)
            logger.info(f"model input: {embeddings.shape}")

            # Run endpointer inference
            with torch.no_grad():
                output = self.endpointer(embeddings)
                logger.info(f"out shape: {output.shape}")
                # Get probabilities
                logits = output.squeeze(0)
                ##sigmoid
                probs = torch.sigmoid(logits)[-current_chunk_size:]
                # exit()
            
            user_end_prob = probs.item()  
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
        "hidden_size": 512,
        "num_heads": 4,
        "num_layers": 6,
        "dim_feedforward": 1024,
        "context": 240,
        "positional_embedding": "rope",
        "max_period": 1000.0
    }
    
    endpointer_model = FC_Transformer_Model(**model_kwargs)
    checkpoint = torch.load(MODEL_CHECKPOINT, map_location=device)
    endpointer_model.load_state_dict(checkpoint["model_state_dict"])
    endpointer_model.to(device)
    endpointer_model.eval()
    logger.info("Endpointer model loaded")

    # print(endpointer_model)


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
    parser.add_argument("--port", type=int, default=8093, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    args = parser.parse_args()

    # session = EndpointerSession(mimi_model, endpointer_model, device)
    # load_models()
    # exit()
    
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
