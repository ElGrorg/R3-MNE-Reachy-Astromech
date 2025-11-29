import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from reachy_mini_conversation_app.openai_realtime import OpenaiRealtimeHandler
from reachy_mini_conversation_app.tools import ToolDependencies, ALL_TOOL_SPECS
from reachy_mini_conversation_app.prompts import SESSION_INSTRUCTIONS


@pytest.mark.asyncio
async def test_startup_parameters():
    """Test that the startup parameters are set correctly."""
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = OpenaiRealtimeHandler(deps)

    # Mock the OpenAI client and its methods
    handler.client = AsyncMock()
    handler.client.beta.realtime.connect = AsyncMock()

    # Call the start_up method
    await handler.start_up()

    # Assert that session.update was called with the correct parameters
    handler.client.beta.realtime.connect.assert_called_once()
    conn = await handler.client.beta.realtime.connect().__aenter__()
    conn.session.update.assert_called_once_with(
        session={
            "turn_detection": {
                "type": "server_vad",
            },
            "input_audio_transcription": {
                "model": "whisper-1",
                "language": "en",
            },
            "instructions": SESSION_INSTRUCTIONS,
            "tools": ALL_TOOL_SPECS,
            "tool_choice": "required",
            "temperature": 0.7,
        },
    )
