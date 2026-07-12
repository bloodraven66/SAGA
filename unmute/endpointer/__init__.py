"""Endpointer client for streaming turn-end detection."""
import asyncio
import random
from logging import getLogger
from typing import AsyncIterator, Literal, Union

import msgpack
import numpy as np
import websockets
from fastrtc import audio_to_float32
from pydantic import BaseModel, TypeAdapter

from unmute import metrics as mt
from unmute.exceptions import MissingServiceAtCapacity
from unmute.kyutai_constants import (
    FRAME_TIME_SEC,
    HEADERS,
    SAMPLE_RATE,
    ENDPOINTER_PATH,
    ENDPOINTER_SERVER,
)
from unmute.service_discovery import ServiceWithStartup
from unmute.timer import Stopwatch
from unmute.websocket_utils import WebsocketState

logger = getLogger(__name__)


class EndpointerPredictionMessage(BaseModel):
    type: Literal["Prediction"]
    user_end_probability: float
    frame_count: int


class EndpointerStatsMessage(BaseModel):
    type: Literal["Stats"]
    frames_received: int
    outputs_sent: int
    expected_output_rate_hz: float


class EndpointerErrorMessage(BaseModel):
    type: Literal["Error"]
    message: str


class EndpointerReadyMessage(BaseModel):
    type: Literal["Ready"]


EndpointerMessage = Union[
    EndpointerPredictionMessage,
    EndpointerStatsMessage,
    EndpointerErrorMessage,
    EndpointerReadyMessage,
]
EndpointerMessageAdapter = TypeAdapter(EndpointerMessage)


class Endpointer(ServiceWithStartup):
    """Client for the endpointer inference server."""
    
    def __init__(self, endpointer_instance: str = ENDPOINTER_SERVER):
        self.endpointer_instance = endpointer_instance
        self.websocket: websockets.ClientConnection | None = None
        self.sent_samples = 0
        self.received_predictions = 0
        self.current_probability = 0.0  # Latest user_end probability
        self.time_since_first_audio_sent = Stopwatch(autostart=False)
        
        self.shutdown_complete = asyncio.Event()

    def state(self) -> WebsocketState:
        if not self.websocket:
            return "not_created"
        else:
            d: dict[websockets.protocol.State, WebsocketState] = {
                websockets.protocol.State.CONNECTING: "connecting",
                websockets.protocol.State.OPEN: "connected",
                websockets.protocol.State.CLOSING: "closing",
                websockets.protocol.State.CLOSED: "closed",
            }
            return d[self.websocket.state]

    async def send_audio(self, audio: np.ndarray) -> None:
        """Send audio frame to endpointer server."""
        if audio.ndim != 1:
            raise ValueError(f"Expected 1D array, got {audio.shape=}")

        if audio.dtype != np.float32:
            audio = audio_to_float32(audio)

        logger.debug(f"Endpointer.send_audio: sending {len(audio)} samples")
        
        self.sent_samples += len(audio)
        self.time_since_first_audio_sent.start_if_not_started()

        await self._send({"type": "Audio", "pcm": audio.tolist()})

    async def _send(self, data: dict) -> None:
        """Send an arbitrary message to the endpointer server."""
        to_send = msgpack.packb(data, use_bin_type=True, use_single_float=True)

        if self.websocket:
            await self.websocket.send(to_send)
        else:
            logger.warning("Endpointer websocket not connected")

    async def start_up(self):
        """Connect to endpointer server."""
        logger.info(f"Connecting to Endpointer {self.endpointer_instance}...")
        self.websocket = await websockets.connect(
            self.endpointer_instance + ENDPOINTER_PATH,
            additional_headers=HEADERS,
            ping_interval=60,
            ping_timeout=60,
        )
        logger.info("Connected to Endpointer")

        try:
            message_bytes = await self.websocket.recv()
            message_dict = msgpack.unpackb(message_bytes)  # type: ignore
            message = EndpointerMessageAdapter.validate_python(message_dict)
            if isinstance(message, EndpointerReadyMessage):
                logger.info("Endpointer ready")
                return
            elif isinstance(message, EndpointerErrorMessage):
                raise MissingServiceAtCapacity("endpointer")
            else:
                raise RuntimeError(
                    f"Expected ready or error message, got {message.type}"
                )
        except Exception as e:
            logger.error(f"Error during Endpointer startup: {repr(e)}")
            # Make sure we don't leave a dangling websocket connection
            await self.websocket.close()
            self.websocket = None
            raise

    async def shutdown(self):
        """Shutdown endpointer connection."""
        logger.info("Shutting down Endpointer, receiving last messages")
        if self.shutdown_complete.is_set():
            return

        if self.time_since_first_audio_sent.started:
            logger.info(f"Endpointer session duration: {self.time_since_first_audio_sent.time():.2f}s")
            logger.info(f"Audio duration: {self.sent_samples / SAMPLE_RATE:.2f}s")
            logger.info(f"Predictions received: {self.received_predictions}")

        if not self.websocket:
            raise RuntimeError("Endpointer websocket not connected")
        
        # Request final stats
        await self._send({"type": "GetStats"})
        await self.websocket.close()
        await self.shutdown_complete.wait()

        logger.info("Endpointer shutdown() finished")

    async def __aiter__(self) -> AsyncIterator[EndpointerPredictionMessage]:
        """Iterate over prediction messages from the endpointer."""
        if not self.websocket:
            raise RuntimeError("Endpointer websocket not connected")

        my_id = random.randint(1, int(1e9))

        try:
            async for message_bytes in self.websocket:
                data = msgpack.unpackb(message_bytes)  # type: ignore
                logger.debug(f"{my_id} Endpointer got {data}")
                message: EndpointerMessage = EndpointerMessageAdapter.validate_python(data)

                match message:
                    case EndpointerPredictionMessage():
                        self.current_probability = message.user_end_probability
                        self.received_predictions += 1
                        yield message
                    case EndpointerStatsMessage():
                        logger.info(f"Endpointer stats: {message}")
                        continue
                    case EndpointerReadyMessage():
                        continue
                    case EndpointerErrorMessage():
                        logger.error(f"Endpointer error: {message.message}")
                        raise RuntimeError(f"Endpointer error: {message.message}")
                    case _:
                        raise ValueError(f"Unknown message: {message}")

        except websockets.ConnectionClosedOK:
            pass
        finally:
            self.shutdown_complete.set()
