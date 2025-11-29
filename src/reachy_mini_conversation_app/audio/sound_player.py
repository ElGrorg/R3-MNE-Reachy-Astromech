import wave
import sys
import subprocess
import logging
from threading import Thread

logger = logging.getLogger(__name__)

class SoundPlayer:
    """Plays audio files using available system players (sounddevice, afplay, aplay)."""

    def __init__(self, sound_path, head_wobbler=None):
        """
        Initialize the SoundPlayer.

        Args:
            sound_path: Path to the audio file to play.
            head_wobbler: Optional HeadWobbler instance to animate the robot during playback.
        """
        self.sound_path = sound_path
        self.head_wobbler = head_wobbler
        self.thread = None

    def play(self):
        """Start playback in a separate thread."""
        if self.thread is None or not self.thread.is_alive():
            self.thread = Thread(target=self._play)
            self.thread.start()

    def _play(self):
        """Internal method to handle playback logic."""
        try:
            # Try using sounddevice (Cross-platform, reliable)
            try:
                import sounddevice as sd
                import numpy as np
                
                # Find reSpeaker or USB audio device
                target_device = None
                try:
                    devices = sd.query_devices()
                    for i, dev in enumerate(devices):
                        name = dev.get('name', '').lower()
                        # Check for ReSpeaker or generic USB Audio, and ensure it has output channels
                        if ('respeaker' in name or 'usb' in name) and dev.get('max_output_channels', 0) > 0:
                            target_device = i
                            logger.info(f"Found target audio device: {dev['name']} (index {i})")
                            break
                except Exception as dev_e:
                    logger.warning(f"Error querying devices: {dev_e}")

                with wave.open(self.sound_path, 'rb') as wf:
                    samplerate = wf.getframerate()
                    channels = wf.getnchannels()
                    dtype = np.int16
                    
                    data = wf.readframes(wf.getnframes())
                    audio_array = np.frombuffer(data, dtype=dtype)
                    
                    # Reshape if stereo
                    if channels > 1:
                        audio_array = audio_array.reshape(-1, channels)

                    # Feed head wobbler if available
                    if self.head_wobbler:
                        # HeadWobbler expects mono, so if stereo, take one channel or average
                        # But feed_raw expects (1, N) or (N,)
                        # Let's just pass the array, HeadWobbler might need to handle shape
                        # Actually HeadWobbler.feed_raw takes NDArray[np.int16]
                        # And inside working_loop: pcm = np.asarray(chunk).squeeze(0)
                        # And feed does: buf = ... .reshape(1, -1)
                        # So it expects (1, N).
                        
                        wobble_audio = audio_array
                        if channels > 1:
                            # Take first channel for wobbling
                            wobble_audio = audio_array[:, 0]
                        
                        # Ensure it is (1, N) for consistency with feed()
                        wobble_audio = wobble_audio.reshape(1, -1)
                        
                        # Reset wobbler state for new sound
                        self.head_wobbler.reset()
                        self.head_wobbler.feed_raw(wobble_audio)
                        logger.info("Fed audio to HeadWobbler")
                        
                    logger.info(f"Playing with sounddevice: {self.sound_path} (sr={samplerate}, ch={channels}, device={target_device})")
                    sd.play(audio_array, samplerate, device=target_device)
                    sd.wait()
                    logger.info("sounddevice playback finished")
                return
            except ImportError:
                logger.warning("sounddevice not found, trying fallbacks")
            except Exception as e:
                logger.error(f"sounddevice failed: {e}")

            # Fallback for macOS (afplay)
            if sys.platform == "darwin":
                logger.info(f"Attempting to play sound with afplay: {self.sound_path}")
                # afplay doesn't easily support device selection without extra flags/setup
                result = subprocess.run(["afplay", self.sound_path], capture_output=True, text=True)
                if result.returncode != 0:
                    logger.error(f"afplay failed with code {result.returncode}: {result.stderr}")
                else:
                    logger.info("afplay completed successfully")
                return

            # Fallback for Linux (ALSA)
            if sys.platform == "linux":
                subprocess.run(["aplay", self.sound_path], check=False)
                return

            logger.warning("No suitable audio player found for this platform.")

        except Exception as e:
            logger.error(f"Error playing sound: {e}")
