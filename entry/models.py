"""
Pydantic models for API requests and responses
"""
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime


# Input models for API requests
class PreprocessRequest(BaseModel):
    text: str


class AudioQuality(str, Enum):
    """Audio quality options for TTS"""
    HIGH = "high"      # 24kHz, 16-bit
    MEDIUM = "medium"  # 16kHz, 8-bit
    LOW = "low"        # 8kHz, 8-bit
    AUTO = "auto"      # Automatically determine based on response size


class AudioFormat(str, Enum):
    """Audio format options for TTS"""
    WAV = "wav"        # Standard WAV format
    MP3 = "mp3"        # MP3 compressed format
    AUTO = "auto"      # Automatically determine based on response size


class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    speed: float = 1.0
    pitch: float = 0.0
    language: Optional[str] = None
    quality: AudioQuality = AudioQuality.AUTO
    format: AudioFormat = AudioFormat.AUTO
    breathiness: float = 0.0
    tenseness: float = 0.0
    jitter: float = 0.0
    sultry: float = 0.0
    fiction: bool = False


class TokenizeRequest(BaseModel):
    text: str
    voice: str = "af_sky"


class TTSBatchRequest(BaseModel):
    texts: List[str]
    voice: Optional[str] = "af_sky"
    speed: Optional[float] = 1.0
    use_gpu: Optional[bool] = True
    breathiness: Optional[float] = 0.0
    tenseness: Optional[float] = 0.0
    jitter: Optional[float] = 0.0
    sultry: Optional[float] = 0.0
    fiction: Optional[List[bool]] = None
    quality: AudioQuality = AudioQuality.AUTO
    format: AudioFormat = AudioFormat.AUTO


# Job models
class JobStatusEnum(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TTSJobRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    speed: float = 1.0
    use_gpu: bool = True
    breathiness: float = 0.0
    tenseness: float = 0.0
    jitter: float = 0.0
    sultry: float = 0.0
    fiction: bool = False
    quality: AudioQuality = AudioQuality.AUTO
    format: AudioFormat = AudioFormat.AUTO
    # Metadata
    title: str = "Untitled"
    author: str = "Unknown Author"
    genre: str = "non-fiction"


class TTSJob(BaseModel):
    id: str
    text: str
    voice: Optional[str] = None
    speed: float = 1.0
    use_gpu: bool = True
    breathiness: float = 0.0
    tenseness: float = 0.0
    jitter: float = 0.0
    sultry: float = 0.0
    fiction: bool = False
    # Metadata
    title: str = "Untitled"
    author: str = "Unknown Author"
    genre: str = "non-fiction"


class JobResponse(BaseModel):
    job_id: str
    status: JobStatusEnum
    message: str


class JobStatus(BaseModel):
    job_id: str
    status: JobStatusEnum
    title: str
    author: str
    genre: str
    position_in_queue: Optional[int] = None
    total_in_queue: int
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    audio_base64: Optional[str] = None


class QueueStatus(BaseModel):
    total_jobs: int
    processing_jobs: int
    queued_jobs: int
    jobs: List[JobStatus]


# Response models
class TTSResponse(BaseModel):
    sample_rate: int
    audio_base64: str
    phonemes: str


class BatchTTSResponse(BaseModel):
    audios: List[str]


class TokenizeResponse(BaseModel):
    phonemes: str


class PreprocessResponse(BaseModel):
    original: str
    processed: str


class VoicesResponse(BaseModel):
    voices: List[str]


class VoiceChoicesResponse(BaseModel):
    choices: Dict[str, str]


class VoicePresetsResponse(BaseModel):
    presets: Dict[str, Dict[str, Any]]


class StreamResponse(BaseModel):
    stream_url: str
    stream_id: str