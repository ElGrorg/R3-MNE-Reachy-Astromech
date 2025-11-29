import logging
import random
import os
from typing import Any, Dict, List

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.dance_emotion_moves import EmotionQueueMove
from reachy_mini_conversation_app.audio.sound_player import SoundPlayer

logger = logging.getLogger(__name__)

# Initialize emotion library
try:
    from reachy_mini.motion.recorded_move import RecordedMoves
    # Note: huggingface_hub automatically reads HF_TOKEN from environment variables
    RECORDED_MOVES = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
    EMOTION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Emotion library not available: {e}")
    RECORDED_MOVES = None
    EMOTION_AVAILABLE = False

class ReportSentimentTool(Tool):
    """Tool to report the sentiment of the user's speech and trigger an emotional response."""

    name = "report_sentiment"
    description = (
        "Report the sentiment of the user's speech. "
        "This tool MUST be called instead of speaking text. "
        "It will trigger an appropriate emotional response (sound + movement) on the robot."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "sentiment": {
                "type": "string",
                "enum": ["happy", "sad", "angry", "surprised", "neutral", "confused", "excited", "scared"],
                "description": "The detected sentiment of the user's speech.",
            },
            "intensity": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": "The intensity of the detected sentiment.",
            },
        },
        "required": ["sentiment", "intensity"],
    }

    async def __call__(self, deps: ToolDependencies, sentiment: str, intensity: str) -> Dict[str, Any]:
        """Execute the sentiment report tool."""
        logger.info(f"Reporting sentiment: {sentiment} (intensity: {intensity})")

        if not EMOTION_AVAILABLE:
            logger.warning("Emotion system not available, skipping movement.")

        # Map sentiment/intensity to emotion moves and sounds
        # Map sentiment/intensity to emotion moves and sounds
        # Available moves: cheerful1, sad1, furious1, surprised1, scared1, confused1, enthusiastic1
        emotion_map = {
            "happy": {"move": "cheerful", "sound_dir": "happy"},
            "excited": {"move": "enthusiastic", "sound_dir": "happy"},
            "sad": {"move": "sad", "sound_dir": "sad"},
            "angry": {"move": "furious", "sound_dir": "mad"},
            "surprised": {"move": "surprised", "sound_dir": "surprised"},
            "scared": {"move": "scared", "sound_dir": "sad"}, # Fallback
            "confused": {"move": "confused", "sound_dir": "surprised"}, # Fallback
            "neutral": {"move": "attentive", "sound_dir": "happy"}, # attentive1/2 exist
        }

        config = emotion_map.get(sentiment.lower(), {"move": "happy", "sound_dir": "happy"})
        
        move_base_name = config["move"]
        sound_dir_base = config["sound_dir"]

        # Determine specific move variant based on intensity
        move_name = None
        if move_base_name and EMOTION_AVAILABLE and RECORDED_MOVES:
            # Library uses numbers like cheerful1, cheerful2, etc.
            # Map intensity to number?
            # low -> 1, medium -> 1, high -> 2?
            suffix = "2" if intensity == "high" else "1"
            move_name = f"{move_base_name}{suffix}"
            
            # Check if move exists in recorded moves
            available_moves = RECORDED_MOVES.list_moves()
            if move_name not in available_moves:
                # Try suffix 1
                move_name = f"{move_base_name}1"
                if move_name not in available_moves:
                     # Try just finding any move starting with base name
                     candidates = [m for m in available_moves if m.startswith(move_base_name)]
                     if candidates:
                         move_name = candidates[0]
                     else:
                         logger.warning(f"Move {move_name} not found, skipping.")
                         move_name = None

        sound_path = None
        try:
            sound_category_dir = None
            if sound_dir_base == "misc":
                sound_category_dir = "misc"
            else:
                suffix = "long" if intensity == "high" else "short"
                sound_category_dir = f"{sound_dir_base}_{suffix}"
            
            current_file = os.path.abspath(__file__)
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
            audio_root = os.path.join(repo_root, "emotion_audio")
            
            target_dir = os.path.join(audio_root, sound_category_dir)
            
            if os.path.exists(target_dir):
                files = [f for f in os.listdir(target_dir) if f.endswith(".wav")]
                if files:
                    sound_file = random.choice(files)
                    sound_path = os.path.join(target_dir, sound_file)
            else:
                logger.warning(f"Audio directory not found: {target_dir}")

        except Exception as e:
            logger.error(f"Error resolving sound path: {e}")

        # Execute actions
        actions_taken = []

        # 1. Play sound
        if sound_path:
            logger.info(f"Playing sound: {sound_path}")
            player = SoundPlayer(sound_path, head_wobbler=deps.head_wobbler)
            player.play()
            actions_taken.append(f"played sound {os.path.basename(sound_path)}")

        # 2. Queue movement
        if move_name and RECORDED_MOVES:
            logger.info(f"Queueing move: {move_name}")
            move = EmotionQueueMove(move_name, RECORDED_MOVES)
            deps.movement_manager.queue_move(move)
            actions_taken.append(f"queued move {move_name}")

        return {
            "status": "success",
            "sentiment_detected": sentiment,
            "intensity": intensity,
            "actions": actions_taken
        }
