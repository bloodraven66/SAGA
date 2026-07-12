#!/usr/bin/env python3
"""
Standalone test script for the Endpointer Inference Server.
Streams audio to the server and plots the endpoint probability predictions.
"""

import asyncio
import argparse
import logging
from pathlib import Path

import msgpack
import numpy as np
import matplotlib.pyplot as plt
import websockets
import soundfile as sf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
SAMPLE_RATE = 24000
FRAME_SIZE = 480  # 20ms at 24kHz
ENDPOINTER_OUTPUT_RATE = 12.5  # Hz


async def stream_audio_to_endpointer(
    audio_path: str,
    server_url: str = "ws://localhost:8092/api/endpointer_stream",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stream audio file to the endpointer server and collect predictions.
    
    Args:
        audio_path: Path to the audio file
        server_url: WebSocket URL of the endpointer server
        
    Returns:
        audio: Original audio waveform
        probabilities: Array of user_end probabilities at 12.5Hz
    """
    # Load audio file
    logger.info(f"Loading audio from {audio_path}")
    audio, sr = sf.read(audio_path)
    
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected sample rate {SAMPLE_RATE}, got {sr}")
    
    logger.info(f"Audio loaded: {len(audio)} samples, {len(audio)/sr:.2f} seconds")
    
    # Split audio into frames
    num_frames = len(audio) // FRAME_SIZE
    frames = [
        audio[i * FRAME_SIZE:(i + 1) * FRAME_SIZE]
        for i in range(num_frames)
    ]
    
    logger.info(f"Split into {num_frames} frames of {FRAME_SIZE} samples")
    
    # Connect to server
    logger.info(f"Connecting to {server_url}")
    async with websockets.connect(
        server_url,
        ping_interval=60,  # Send ping every 60 seconds
        ping_timeout=60,   # Wait 60 seconds for pong
        close_timeout=10,  # Wait 10 seconds for close handshake
    ) as websocket:
        logger.info("Connected to server")
        
        # Wait for Ready message
        ready_bytes = await websocket.recv()
        ready_msg = msgpack.unpackb(ready_bytes)
        logger.info(f"Received: {ready_msg}")
        
        if ready_msg.get("type") != "Ready":
            raise RuntimeError(f"Expected Ready message, got {ready_msg}")
        
        # Stream frames and collect predictions
        probabilities = []
        
        # Create a task to receive predictions continuously
        async def receive_predictions():
            preds = []
            pred_count = 0
            try:
                while True:
                    response_bytes = await websocket.recv()
                    response = msgpack.unpackb(response_bytes)
                    
                    if response.get("type") == "Prediction":
                        prob = response.get("user_end_probability")
                        preds.append(prob)
                        pred_count += 1
                        if pred_count % 100 == 0:
                            logger.info(f"Received {pred_count} predictions so far...")
                        logger.debug(f"Received prediction {len(preds)}: {prob:.4f}")
                    elif response.get("type") == "Stats":
                        logger.info(f"Received stats: {response}")
                        break
            except Exception as e:
                logger.info(f"Prediction receiver ended: {e}")
            logger.info(f"Total predictions collected: {len(preds)}")
            return preds
        
        # Start receiving predictions in background
        receive_task = asyncio.create_task(receive_predictions())
        
        # Send all audio frames with pacing to avoid overwhelming the WebSocket
        for idx, frame in enumerate(frames):
            audio_msg = msgpack.packb(
                {"type": "Audio", "pcm": frame.astype(np.float32).tolist()},
                use_bin_type=True,
                use_single_float=True,
            )
            await websocket.send(audio_msg)
            
            # Add small delay every few frames to prevent WebSocket buffer overflow
            # This simulates real-time streaming at 50Hz (20ms per frame)
            if (idx + 1) % 10 == 0:
                await asyncio.sleep(0.001)  # Small pause every 10 frames
            
            # Log progress every 50 frames
            if (idx + 1) % 50 == 0:
                logger.info(f"Sent {idx + 1}/{num_frames} frames")
        
        logger.info(f"All {num_frames} frames sent, waiting for remaining predictions...")
        
        # Wait a bit for all predictions to be generated and sent
        # At 12.5Hz, we expect num_frames/4 predictions
        expected_predictions = num_frames // 4
        logger.info(f"Expecting approximately {expected_predictions} predictions")
        
        # Give server time to process and send all predictions
        # Wait up to 5 seconds for remaining predictions
        await asyncio.sleep(2.0)
        
        # Request stats to signal end
        stats_msg = msgpack.packb({"type": "GetStats"}, use_bin_type=True)
        await websocket.send(stats_msg)
        
        # Wait for all predictions
        probabilities = await receive_task
        
        logger.info(f"Streaming complete. Total predictions: {len(probabilities)}")
    
    # Convert to numpy arrays
    probabilities = np.array(probabilities)
    
    return audio, probabilities


def plot_results(
    audio: np.ndarray,
    probabilities: np.ndarray,
    output_path: str = "endpointer_test_output.png",
):
    """
    Plot audio waveform and endpointer probabilities.
    
    Args:
        audio: Audio waveform
        probabilities: User_end probabilities at 12.5Hz
        output_path: Path to save the plot
    """
    logger.info("Creating plot...")
    
    # Time axes
    t_audio = np.arange(len(audio)) / SAMPLE_RATE  # seconds
    t_probs = np.arange(len(probabilities)) / ENDPOINTER_OUTPUT_RATE  # seconds
    
    # Normalize audio to range [-1, 1] for better visualization
    audio_normalized = audio / (np.max(np.abs(audio)) + 1e-8)
    
    # Renormalize probabilities to [0, 0.4] to overlay on waveform
    probs_renormalized = probabilities * 0.4
    
    # Create figure with larger x-size
    fig, axes = plt.subplots(2, 1, figsize=(24, 8), sharex=True)
    
    # Row 0: Audio waveform with overlaid probabilities
    axes[0].plot(t_audio, audio_normalized, linewidth=0.3, color='blue', alpha=0.6, label='Audio')
    axes[0].fill_between(t_probs, 0, probs_renormalized, color='red', alpha=0.4, label='User End Probability (scaled to 0-0.4)')
    axes[0].set_ylabel("Amplitude / Scaled Probability")
    axes[0].set_title("Audio Waveform with Endpointer Predictions Overlay")
    axes[0].set_ylim(-1, 1)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right')
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Row 1: Endpointer probabilities only (full scale)
    axes[1].plot(t_probs, probabilities, linewidth=1.5, color='red', label='User End Probability')
    axes[1].axhline(y=0.6, color='orange', linestyle='--', label='High Threshold (0.6)', alpha=0.7, linewidth=1.5)
    axes[1].axhline(y=0.4, color='green', linestyle='--', label='Low Threshold (0.4)', alpha=0.7, linewidth=1.5)
    axes[1].fill_between(t_probs, 0, probabilities, where=(probabilities > 0.6), color='red', alpha=0.3, interpolate=True)
    axes[1].fill_between(t_probs, 0, probabilities, where=(probabilities > 0.4) & (probabilities <= 0.6), color='orange', alpha=0.2, interpolate=True)
    axes[1].set_ylabel("Probability")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_title("Endpointer User-End Probability (Full Scale)")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Plot saved to {output_path}")
    
    # Show plot
    plt.show()
    
    # Print statistics
    logger.info("\n" + "="*60)
    logger.info("Statistics:")
    logger.info(f"  Audio duration: {len(audio)/SAMPLE_RATE:.2f} seconds")
    logger.info(f"  Number of predictions: {len(probabilities)}")
    logger.info(f"  Prediction rate: {len(probabilities)/(len(audio)/SAMPLE_RATE):.2f} Hz")
    logger.info(f"  Mean probability: {np.mean(probabilities):.4f}")
    logger.info(f"  Max probability: {np.max(probabilities):.4f}")
    logger.info(f"  Min probability: {np.min(probabilities):.4f}")
    logger.info(f"  Std dev: {np.std(probabilities):.4f}")
    
    # Count threshold crossings
    threshold_06 = np.sum(probabilities > 0.6)
    threshold_04 = np.sum(probabilities > 0.4)
    logger.info(f"  Frames above 0.6: {threshold_06} ({threshold_06/len(probabilities)*100:.1f}%)")
    logger.info(f"  Frames above 0.4: {threshold_04} ({threshold_04/len(probabilities)*100:.1f}%)")
    logger.info("="*60)


async def main():
    parser = argparse.ArgumentParser(description="Test Endpointer Inference Server")
    parser.add_argument(
        "--audio",
        type=str,
        default="/mnt/matylda4/udupa/data/SpokenWOZ/audio_5700_test_resampled_24000/MUL1104.wav",
        help="Path to audio file (24kHz WAV)",
    )
    parser.add_argument(
        "--server",
        type=str,
        default="ws://localhost:8092/api/endpointer_stream",
        help="WebSocket URL of endpointer server",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="endpointer_test_output.png",
        help="Path to save output plot",
    )
    args = parser.parse_args()
    
    # Check if audio file exists
    if not Path(args.audio).exists():
        logger.error(f"Audio file not found: {args.audio}")
        return
    
    try:
        # Stream audio and get predictions
        audio, probabilities = await stream_audio_to_endpointer(
            args.audio, args.server
        )
        
        # Plot results
        plot_results(audio, probabilities, args.output)
        
        logger.info("Test completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
