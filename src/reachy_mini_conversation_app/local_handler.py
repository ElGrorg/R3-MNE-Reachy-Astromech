import asyncio
import logging
import time
import warnings
import numpy as np
from typing import Any, Tuple, List
from numpy.typing import NDArray
from fastrtc import AsyncStreamHandler, AdditionalOutputs, wait_for_item

from faster_whisper import WhisperModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.sentiment_tool import ReportSentimentTool

from reachy_mini_conversation_app.tools.dance import Dance
from reachy_mini_conversation_app.tools.stop_dance import StopDance
from reachy_mini_conversation_app.tools.play_emotion import PlayEmotion

logger = logging.getLogger(__name__)

# Constants
SAMPLE_RATE = 16000  # Whisper expects 16kHz usually
VAD_THRESHOLD = 0.006
SILENCE_DURATION = 1.0
MIN_SPEECH_DURATION = 0.40

class LocalAudioHandler(AsyncStreamHandler):
    """A local handler for STT and Sentiment Analysis."""

    def __init__(self, deps: ToolDependencies):
        """
        Initialize the LocalAudioHandler.

        Args:
            deps: Dependencies required for the tools (robot, movement manager, etc.).
        """
        super().__init__(
            expected_layout="mono",
            output_sample_rate=24000,
            input_sample_rate=SAMPLE_RATE,
        )
        self.deps = deps
        self.output_queue: "asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs]" = asyncio.Queue()
        
        # Audio buffer
        self.audio_buffer: List[NDArray[np.float32]] = []
        self.speaking = False
        self.last_speech_time = 0.0
        self.speech_start_time = 0.0
        
        # AI Models
        self.stt_model = None
        self.sentiment_analyzer = None
        
        # Tools
        self.sentiment_tool = ReportSentimentTool()
        self.dance_tool = Dance()
        self.stop_dance_tool = StopDance()
        self.play_emotion_tool = PlayEmotion()

    def copy(self) -> "LocalAudioHandler":
        """
        Create a copy of the handler.

        Returns:
            A new instance of LocalAudioHandler with the same dependencies.
        """
        logger.info("Creating copy of LocalAudioHandler")
        return LocalAudioHandler(self.deps)

    async def start_up(self) -> None:
        """Initialize models."""
        logger.info("Loading local AI models...")
        try:
            # Run in executor to avoid blocking async loop
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._load_models)
            logger.info("Local AI models loaded.")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")

    def _load_models(self):
        """Load the Whisper model and VADER sentiment analyzer."""
        self.stt_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Process incoming audio frame."""
        _, audio_int16 = frame

        # Convert to float32 first
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        
        # Ensure mono and flatten
        if audio_float32.ndim > 1:
            if audio_float32.shape[0] < audio_float32.shape[1]:
                # (channels, samples) -> average over channels (axis 0)
                audio_float32 = np.mean(audio_float32, axis=0)
            else:
                # (samples, channels) -> average over channels (axis 1)
                audio_float32 = np.mean(audio_float32, axis=1)
        
        # Calculate energy (RMS)
        rms = np.sqrt(np.mean(audio_float32**2))
        
        current_time = time.time()
        
        if rms > VAD_THRESHOLD:
            if not self.speaking:
                self.speaking = True
                self.speech_start_time = current_time
                logger.debug(f"Speech detected! (RMS: {rms:.4f})")
                self.deps.movement_manager.set_listening(True)
                if self.deps.head_wobbler:
                    self.deps.head_wobbler.reset()
            
            self.last_speech_time = current_time
            self.audio_buffer.append(audio_float32)
            
        elif self.speaking:
            # We were speaking, now it's quiet
            self.audio_buffer.append(audio_float32)
            
            if current_time - self.last_speech_time > SILENCE_DURATION:
                # Silence timeout reached, process the buffer
                self.speaking = False
                self.deps.movement_manager.set_listening(False)
                logger.debug("Speech stopped, processing...")
                
                duration = self.last_speech_time - self.speech_start_time
                if duration >= MIN_SPEECH_DURATION:
                    try:
                        await self._process_audio()
                    except Exception as e:
                        logger.error(f"Error in _process_audio: {e}", exc_info=True)
                else:
                    logger.debug(f"Speech too short ({duration:.2f}s), ignoring.")
                
                self.audio_buffer = []

    async def _process_audio(self):
        """Transcribe and analyze sentiment."""
        if not self.audio_buffer:
            return
            
        if not self.stt_model:
            return

        # Concatenate buffer
        full_audio = np.concatenate(self.audio_buffer)
        
        # Sanitize audio to prevent Whisper errors
        full_audio = np.nan_to_num(full_audio, nan=0.0, posinf=0.0, neginf=0.0)
        full_audio = np.clip(full_audio, -1.0, 1.0)
        
        # Check for silence/validity
        if len(full_audio) < 1600: # < 0.1s
             return

        if np.max(np.abs(full_audio)) < 1e-5:
            return
        
        logger.debug(f"Transcribing {len(full_audio)/SAMPLE_RATE:.2f}s of audio...")

        # Transcribe (run in executor)
        loop = asyncio.get_running_loop()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                segments, _ = await loop.run_in_executor(None, self.stt_model.transcribe, full_audio)
                text = " ".join([segment.text for segment in segments]).strip()
            
            if not text:
                logger.debug("No text transcribed.")
                return
                
            logger.debug(f"Transcribed: {text}")
            await self.output_queue.put(AdditionalOutputs({"role": "user", "content": text}))
            
            # 1. Check for keywords FIRST
            keyword_result = await self._check_keywords(text)
            if keyword_result:
                await self.output_queue.put(
                    AdditionalOutputs({
                        "role": "assistant", 
                        "content": f"Action: {keyword_result}",
                        "metadata": {"action": keyword_result}
                    })
                )
                return

            # 2. Sentiment Analysis (Fallback)
            scores = self.sentiment_analyzer.polarity_scores(text)
            compound = scores['compound']
            
            # Map compound score to sentiment/intensity
            sentiment, intensity = self._map_sentiment(compound, scores)
            
            logger.debug(f"Sentiment: {sentiment} ({intensity}) - Scores: {scores}")
            
            # Trigger Tool
            result = await self.sentiment_tool(self.deps, sentiment=sentiment, intensity=intensity)
            
            await self.output_queue.put(
                AdditionalOutputs({
                    "role": "assistant", 
                    "content": f"Detected: {sentiment} ({intensity})",
                    "metadata": {"action": result}
                })
            )
        except Exception as e:
            logger.error(f"Transcription/Processing error: {e}", exc_info=True)

    async def _check_keywords(self, text: str) -> str | None:
        """Check text for keywords and trigger tools."""
        text_lower = text.lower()
        
        # Dance keywords
        if "dance" in text_lower or "dancing" in text_lower:
            if "stop" in text_lower or "don't" in text_lower:
                # Stop dance
                logger.debug("Keyword detected: STOP DANCE")
                await self.stop_dance_tool(self.deps, dummy=True)
                return "stopped dance"
            else:
                # Start dance
                logger.debug("Keyword detected: DANCE")
                await self.dance_tool(self.deps, move="random")
                return "started dance"
        
        # Emotion keywords
        # Emotion keywords
        
        emotions = {
            "happy": "cheerful1", 
            "sad": "sad1",
            "confused": "confused1",
            "mad": "furious1",
            "angry": "furious1",
            "surprised": "surprised1",
            "scared": "scared1",
            "excited": "enthusiastic1",
            "bored": "boredom1",
            "tired": "tired1",
            "thank": "grateful1",
            "yes": "yes1",
            "no": "no1"
        }
        
        for keyword, emotion_move in emotions.items():
            if keyword in text_lower:
                logger.debug(f"Keyword detected: {keyword.upper()} -> {emotion_move}")
                await self.play_emotion_tool(self.deps, emotion=emotion_move)
                return f"played emotion {emotion_move}"
                
        return None

    def _map_sentiment(self, compound: float, scores: dict) -> Tuple[str, str]:
        """Map VADER scores to our sentiment categories."""
        """Map VADER scores to our sentiment categories."""
        
        intensity = "medium"
        if abs(compound) > 0.6:
            intensity = "high"
        elif abs(compound) < 0.3:
            intensity = "low"
            
        if compound >= 0.5:
            return "happy", intensity
        elif compound <= -0.5:
            return "sad", intensity
        elif -0.1 < compound < 0.1:
            return "neutral", "low"
        else:
            # In between
            if compound > 0:
                return "excited", "low"
            else:
                return "confused", "low"

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """
        Emit processed output (audio or additional messages) from the queue.

        Returns:
            The next item from the output queue, or None.
        """
        return await wait_for_item(self.output_queue)

    async def shutdown(self) -> None:
        """Shutdown the handler and release resources."""
        pass
