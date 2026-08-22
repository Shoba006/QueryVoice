import os
import time
import logging
import requests
from typing import Optional, Union, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class TranscriptionResult(BaseModel):
    text: str = Field(..., description="Transcribed query text")
    language: str = Field(default="hi-IN", description="Detected or specified language code")
    confidence: Optional[float] = Field(default=None, description="Confidence score if available")
    raw_latency_ms: float = Field(..., description="STT network & processing latency in milliseconds")
    error: Optional[str] = Field(default=None, description="Error message if transcription failed")
    is_success: bool = Field(default=True, description="Status of transcription step")

class SarvamSTTClient:
    """
    Speech-to-Text client wrapping Sarvam AI's saarika STT model.
    Implements retry logic with exponential backoff and structured error returns.
    """
    def __init__(self, api_key: Optional[str] = None, model: str = "saarika:v2.5", default_language: str = "hi-IN"):
        self.api_key = api_key or os.environ.get("SARVAM_API_KEY")
        self.model = model
        self.default_language = default_language
        self.endpoint = "https://api.sarvam.ai/speech-to-text"

    def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        language_code: Optional[str] = None,
        max_retries: int = 2
    ) -> TranscriptionResult:
        """
        Transcribes audio bytes using Sarvam AI STT API with retries.
        Returns a structured TranscriptionResult even on failure.
        """
        start_time = time.perf_counter()
        lang = language_code or self.default_language

        if not self.api_key or self.api_key == "your_sarvam_api_key_here":
            logger.warning("SARVAM_API_KEY not configured or placeholder detected. Operating in fallback mode.")
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return TranscriptionResult(
                text="यह एक परीक्षण खोज प्रश्न है।",  # Sample Indic test query
                language=lang,
                confidence=0.95,
                raw_latency_ms=round(latency_ms, 2),
                error="SARVAM_API_KEY not provided (Mock Fallback)",
                is_success=True
            )

        headers = {
            "api-subscription-key": self.api_key
        }

        # Multi-part form data
        files = {
            "file": (filename, audio_bytes, "audio/wav")
        }
        data = {
            "model": self.model,
            "language_code": lang,
            "with_timestamps": "false"
        }

        last_error_msg = ""
        backoff = 0.5  # initial delay in seconds

        for attempt in range(max_retries + 1):
            try:
                attempt_start = time.perf_counter()
                response = requests.post(
                    self.endpoint,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    resp_json = response.json()
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    transcript_text = resp_json.get("transcript", "").strip()
                    detected_lang = resp_json.get("language_code", lang)
                    
                    return TranscriptionResult(
                        text=transcript_text,
                        language=detected_lang,
                        confidence=resp_json.get("confidence", 0.90),
                        raw_latency_ms=round(elapsed_ms, 2),
                        is_success=True
                    )
                else:
                    last_error_msg = f"HTTP {response.status_code}: {response.text}"
                    logger.warning(f"Sarvam STT attempt {attempt + 1} failed: {last_error_msg}")

            except Exception as e:
                last_error_msg = str(e)
                logger.warning(f"Sarvam STT attempt {attempt + 1} exception: {last_error_msg}")

            if attempt < max_retries:
                time.sleep(backoff)
                backoff *= 2.0  # exponential backoff

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.error(f"Sarvam STT failed after {max_retries + 1} attempts. Error: {last_error_msg}")
        
        return TranscriptionResult(
            text="",
            language=lang,
            confidence=0.0,
            raw_latency_ms=round(elapsed_ms, 2),
            error=f"STT Error after retries: {last_error_msg}",
            is_success=False
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = SarvamSTTClient()
    # Test fallback with dummy bytes
    # Generate valid 44-byte WAV header for self-test
    wav_header = b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80\x3e\x00\x00\x00\x7d\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00'
    res = client.transcribe(wav_header)
    print("STT Self-Test Result:")
    print(res.model_dump_json(indent=2))
