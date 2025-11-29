import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# Initialize emotion library
try:
    from reachy_mini.motion.recorded_move import RecordedMoves
    from reachy_mini_conversation_app.dance_emotion_moves import EmotionQueueMove

    # Note: huggingface_hub automatically reads HF_TOKEN from environment variables
    RECORDED_MOVES = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
    EMOTION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Emotion library not available: {e}")
    RECORDED_MOVES = None
    EMOTION_AVAILABLE = False


def get_available_emotions_and_descriptions() -> str:
    """Get formatted list of available emotions with descriptions."""
    if not EMOTION_AVAILABLE:
        return "Emotions not available"

    try:
        emotion_names = RECORDED_MOVES.list_moves()
        output = "Available emotions:\n"
        for name in emotion_names:
            description = RECORDED_MOVES.get(name).description
            output += f" - {name}: {description}\n"
        return output
    except Exception as e:
        return f"Error getting emotions: {e}"


class PlayEmotion(Tool):
    """Play a pre-recorded emotion."""

    name = "play_emotion"
    description = "Play a pre-recorded emotion"
    parameters_schema = {
        "type": "object",
        "properties": {
            "emotion": {
                "type": "string",
                "description": f"""Name of the emotion to play.
                                    Here is a list of the available emotions:
                                    {get_available_emotions_and_descriptions()}
                                    """,
            },
        },
        "required": ["emotion"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Play a pre-recorded emotion."""
        if not EMOTION_AVAILABLE:
            return {"error": "Emotion system not available"}

        emotion_name = kwargs.get("emotion")
        if not emotion_name:
            return {"error": "Emotion name is required"}

        logger.info("Tool call: play_emotion emotion=%s", emotion_name)

        # Check if emotion exists
        try:
            emotion_names = RECORDED_MOVES.list_moves()
            if emotion_name not in emotion_names:
                return {"error": f"Unknown emotion '{emotion_name}'. Available: {emotion_names}"}

            # Add emotion to queue
            movement_manager = deps.movement_manager
            emotion_move = EmotionQueueMove(emotion_name, RECORDED_MOVES)
            movement_manager.queue_move(emotion_move)

            # Play sound
            try:
                import os
                import random
                from reachy_mini_conversation_app.audio.sound_player import SoundPlayer

                # Resolve audio root
                current_file = os.path.abspath(__file__)
                # src/reachy_mini_conversation_app/tools/play_emotion.py -> up 4 levels to root
                repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
                audio_root = os.path.join(repo_root, "emotion_audio")

                # Try to match emotion name to folder
                # emotion_name might be "happy_long", "sad_short", etc.
                # or just "happy" (if that exists as a move)
                
                target_dir = os.path.join(audio_root, emotion_name)
                
                # If exact match doesn't exist, try to find a matching category
                if not os.path.exists(target_dir):
                    # Try to find a folder that starts with the emotion name (e.g. "happy" -> "happy_short")
                    candidates = []
                    if os.path.exists(audio_root):
                        for dirname in os.listdir(audio_root):
                            if dirname.startswith(emotion_name):
                                candidates.append(dirname)
                    
                    if candidates:
                        target_dir = os.path.join(audio_root, random.choice(candidates))
                    else:
                        # Fallback: try to match known categories
                        for category in ["happy", "sad", "mad", "surprised"]:
                            if category in emotion_name:
                                # Pick random intensity
                                suffix = random.choice(["short", "long"])
                                target_dir = os.path.join(audio_root, f"{category}_{suffix}")
                                break
                        else:
                            # Final fallback
                            target_dir = os.path.join(audio_root, "misc")

                if os.path.exists(target_dir):
                    files = [f for f in os.listdir(target_dir) if f.endswith(".wav")]
                    if files:
                        sound_file = random.choice(files)
                        sound_path = os.path.join(target_dir, sound_file)
                        logger.info(f"Playing emotion sound: {sound_path}")
                        player = SoundPlayer(sound_path, head_wobbler=deps.head_wobbler)
                        player.play()
                else:
                    logger.warning(f"Could not find audio directory for emotion: {emotion_name}")

            except Exception as e:
                logger.error(f"Error playing emotion sound: {e}")

            return {"status": "queued", "emotion": emotion_name}

        except Exception as e:
            logger.exception("Failed to play emotion")
            return {"error": f"Failed to play emotion: {e!s}"}
