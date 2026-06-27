import os
import io
import tempfile
import numpy as np
from typing import Optional, Tuple

MODEL_PATH = os.path.join("models", "whisper-small")
_whisper_model = None


def _load_whisper():
    global _whisper_model
    if _whisper_model is None:
        try:
            from faster_whisper import WhisperModel
            path = MODEL_PATH if os.path.exists(MODEL_PATH) else "small"
            _whisper_model = WhisperModel(path, device="cpu", compute_type="int8")
        except Exception as e:
            print(f"[voice_input] Whisper load error: {e}")
            _whisper_model = None
    return _whisper_model


def transcribe_file(audio_path: str) -> Tuple[str, str]:
    """
    Transcribe an audio file.
    Returns (transcript_text, detected_language)
    """
    model = _load_whisper()
    if model is None:
        return "", "unknown"
    try:
        segments, info = model.transcribe(audio_path, beam_size=5, language=None)
        text = " ".join(seg.text for seg in segments).strip()
        lang = info.language
        return text, lang
    except Exception as e:
        print(f"[voice_input] Transcription error: {e}")
        return "", "unknown"


def transcribe_bytes(audio_bytes: bytes, suffix: str = ".wav") -> Tuple[str, str]:
    """Transcribe audio from raw bytes (uploaded file)."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        return transcribe_file(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def record_and_transcribe(duration_seconds: int = 6) -> Tuple[str, str]:
    """Record from microphone and transcribe. Returns (text, lang)."""
    try:
        import sounddevice as sd
        import soundfile as sf

        sample_rate = 16000
        print(f"[voice_input] Recording for {duration_seconds}s...")
        audio = sd.rec(
            int(duration_seconds * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        print("[voice_input] Done recording.")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, sample_rate)
            tmp_path = tmp.name
        try:
            return transcribe_file(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    except Exception as e:
        print(f"[voice_input] Record error: {e}")
        return "", "unknown"


def preload():
    _load_whisper()