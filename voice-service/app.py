"""Local voice service providing STT and TTS endpoints."""

import os
import subprocess
import tempfile
from threading import Lock
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

app = FastAPI(title="Eburon Voice Service", version="1.0.0")

_STT_MODEL_LOCK = Lock()
_STATE: dict[str, Any] = {"stt_model": None}


def get_stt_model() -> Any:
    """Initialize and return the lazy-loaded Whisper model instance."""
    with _STT_MODEL_LOCK:
        if _STATE["stt_model"] is None:
            from faster_whisper import WhisperModel  # Lazy import for local dev linting.

            model_size = os.getenv("STT_MODEL_SIZE", "small")
            device = os.getenv("STT_DEVICE", "cpu")
            compute_type = os.getenv("STT_COMPUTE_TYPE", "int8")
            _STATE["stt_model"] = WhisperModel(
                model_size, device=device, compute_type=compute_type
            )
    return _STATE["stt_model"]


class TTSRequest(BaseModel):
    """Payload for text-to-speech requests."""

    text: str = Field(min_length=1, max_length=5000)
    voice: str | None = None
    speed: int | None = None


@app.get("/health")
def health():
    """Simple readiness probe."""
    return {"ok": True}


@app.post("/stt")
async def stt(audio: UploadFile = File(...)):
    """Transcribe uploaded audio into text using Whisper."""
    suffix = ".webm"
    if audio.filename and "." in audio.filename:
        suffix = f".{audio.filename.rsplit('.', 1)[-1]}"

    input_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            input_path = temp_file.name
            temp_file.write(await audio.read())

        model = get_stt_model()
        segments, _info = model.transcribe(input_path, vad_filter=True)
        text = " ".join(segment.text.strip() for segment in segments).strip()

        return JSONResponse({"text": text})
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"STT failed: {error}") from error
    finally:
        if input_path and os.path.exists(input_path):
            os.remove(input_path)


@app.post("/tts")
async def tts(request: TTSRequest):
    """Generate speech audio from text using espeak-ng."""
    output_path = None
    try:
        output = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        output_path = output.name
        output.close()

        voice = request.voice or os.getenv("TTS_VOICE", "en-us")
        speed = request.speed or int(os.getenv("TTS_SPEED", "170"))

        command = [
            "espeak-ng",
            "-v",
            voice,
            "-s",
            str(speed),
            "-w",
            output_path,
            request.text,
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)

        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="speech.wav",
            background=BackgroundTask(
                lambda: os.path.exists(output_path) and os.remove(output_path)
            ),
        )
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() if error.stderr else str(error)
        raise HTTPException(status_code=500, detail=f"TTS failed: {detail}") from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"TTS failed: {error}") from error
