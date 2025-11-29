# Reachy Mini Astromech Droid

Turn your Reachy Mini into R3-MNE, a Star Wars-like astromech droid! This application combines local AI processing, expressive movements, and classic droid sounds to bring your robot to life.

![R3-MNE and R2-D2](docs/assets/reachy_and_r2.png)

## Features

- **Astromech Persona**: The robot communicates using pre-recorded "droid speak" audio files, mimicking the emotional beeps and whistles of an R2 unit.
- **Local Speech-to-Text**: Uses `faster-whisper` for fast, private, and offline-capable speech recognition.
- **Sentiment Analysis**: Analyzes your speech using `vaderSentiment` to understand your emotional tone (happy, sad, etc.) and responds with appropriate droid emotions.
- **Expressive Motion**: A layered motion system blends dances, emotional gestures, and subtle "alive" movements like breathing and head wobble.

## Installation

> Windows support is currently experimental and has not been extensively tested. Use with caution.

### Using uv
You can set up the project quickly using [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12.1  # Create a virtual environment with Python 3.12.1
source .venv/bin/activate
uv sync
```

### Using pip

```bash
python -m venv .venv # Create a virtual environment
source .venv/bin/activate
pip install -e .
```

## Running the app

To run this program, you will need to create two virtual environments, one for the Reachy Mini Daemon and one for the R3-MNE app. 
In one terminal, navigate to the project root, activate the virtual environment, and run the Reachy Mini Daemon:

```bash
reachy-mini-daemon
```

In a second terminal, navigate to the project root, activate the virtual environment, and run:

```bash
r3-mne --gradio
```

By default, the app runs in console mode for direct audio interaction. Use the `--gradio` flag to launch a web UI served locally at http://127.0.0.1:7860/ (Currently required when running).
### CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--gradio` | `False` | Launch the Gradio web UI. Without this flag, runs in console mode. Required when running in simulation mode. |
| `--head-tracker` | `None` | Enable head tracking. Options: `yolo`, `none` |
| `--no-camera` | `False` | Disable camera pipeline. |
| `--debug` | `False` | Enable verbose logging for troubleshooting. |

<!-- 
### Examples
- Run on hardware with MediaPipe face tracking:

  ```bash
  reachy-mini-conversation-app --head-tracker mediapipe
  ```

- Disable the camera pipeline (audio-only conversation):

  ```bash
  reachy-mini-conversation-app --no-camera
  ``` -->
<!-- 
## LLM tools exposed to the assistant

| Tool | Action | Dependencies |
|------|--------|--------------|
| `move_head` | Queue a head pose change (left/right/up/down/front). | Core install only. |
| `dance` | Queue a dance from `reachy_mini_dances_library`. | Core install only. |
| `stop_dance` | Clear queued dances. | Core install only. |
| `play_emotion` | Play a recorded emotion clip via Hugging Face assets. | Needs `HF_TOKEN` for the recorded emotions dataset. |
| `stop_emotion` | Clear queued emotions. | Core install only. |
| `do_nothing` | Explicitly remain idle. | Core install only. |

## Using custom profiles
Create custom profiles with dedicated instructions and enabled tools! 

Set `REACHY_MINI_CUSTOM_PROFILE=<name>` to load `src/reachy_mini_conversation_app/profiles/<name>/` (see `.env.example`). If unset, the `default` profile is used.

Each profile requires two files: `instructions.txt` (prompt text) and `tools.txt` (list of allowed tools), and optionally contains custom tools implementations.

### Custom instructions
Write plain-text prompts in `instructions.txt`. To reuse shared prompt pieces, add lines like:
```
[passion_for_lobster_jokes]
[identities/witty_identity]
```
Each placeholder pulls the matching file under `src/reachy_mini_conversation_app/prompts/` (nested paths allowed). See `src/reachy_mini_conversation_app/profiles/example/` for a reference layout.

### Enabling tools
List enabled tools in `tools.txt`, one per line; prefix with `#` to comment out. For example:

```
play_emotion
# move_head

# My custom tool defined locally
sweep_look
```
Tools are resolved first from Python files in the profile folder (custom tools), then from the shared library `src/reachy_mini_conversation_app/tools/` (e.g., `dance`). 

### Custom tools
On top of built-in tools found in the shared library, you can implement custom tools specific to your profile by adding Python files in the profile folder. 
Custom tools must subclass `reachy_mini_conversation_app.tools.core_tools.Tool` (see `profiles/example/sweep_look.py`).




## Development workflow
- Install the dev group extras: `uv sync --group dev` or `pip install -e .[dev]`.
- Run formatting and linting: `ruff check .`.
- Execute the test suite: `pytest`.
- When iterating on robot motions, keep the control loop responsive => offload blocking work using the helpers in `tools.py`. -->

## License
Apache 2.0

## Open Issues

- **Windows Support**: Currently experimental. Users may encounter issues with audio drivers or dependency installation.
- **Latency**: While local processing is fast, the full pipeline (STT -> Sentiment -> Action) can sometimes have a slight delay depending on hardware.
- **Vision & Camera**: Camera integration, face tracking, and vision models (local or cloud) are implemented but currently **untested**. Use at your own risk.

## Next Steps

- **Expanded Droid Vocabulary**: Adding more diverse audio samples for a wider range of emotions.
- **Interactive Games**: Implementing simple games (like "Red Light, Green Light") using the vision system.
- **Improved Face Tracking**: Optimizing the local YOLO/MediaPipe trackers for smoother head movements.
- **Custom Personalities**: Easier configuration to switch between different "droid personalities" (e.g., sassy, helpful, timid).
- **Patch Console Mode**: Currently, R3 cannot receive audio in console mode. It would be ideal if gradio was not necessary
