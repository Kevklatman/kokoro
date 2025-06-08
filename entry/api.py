from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Generator
import re
import uuid
import time
from enum import Enum
from datetime import datetime
import queue
import threading
import uvicorn
import base64
import io
import numpy as np
import torch
import asyncio
import os
from starlette.concurrency import run_in_threadpool
from fastapi import Depends, Header
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import login


# Import your existing TTS components
from kokoro import KModel, KPipeline
from entry.audio_effects import apply_emotion_effects



load_dotenv()  # Loads from .env if present

ENV = os.getenv("ENV", "development")
API_KEY = os.getenv("API_KEY", "dev-secret-key")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)

# Create FastAPI app
app = FastAPI(
    title="Kokoro TTS API",
    description="REST API for Kokoro Text-to-Speech Engine",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




# Initialize models and pipelines
CUDA_AVAILABLE = torch.cuda.is_available()
# Define models directory path
MODELS_DIR = os.path.join(os.getcwd(), 'models')
if not os.path.exists(MODELS_DIR) and os.path.exists('/app/models'):
    MODELS_DIR = '/app/models'

print(f"Using models directory: {MODELS_DIR}")

# Initialize models and pipelines
models = {gpu: KModel().to('cuda' if gpu else 'cpu').eval() for gpu in [False] + ([True] if CUDA_AVAILABLE else [])}
pipelines = {lang_code: KPipeline(lang_code=lang_code, model=False, models_dir=MODELS_DIR) for lang_code in 'ab'}
pipelines['a'].g2p.lexicon.golds['kokoro'] = 'kˈOkəɹO'
pipelines['b'].g2p.lexicon.golds['kokoro'] = 'kˈQkəɹQ'

# Voice choices and presets migrated from app.py
CHOICES = {
    '🇺🇸 🚺 Heart ❤️': 'af_heart',
    '🇺🇸 🚺 Bella 🔥': 'af_bella',
    '🇺🇸 🚺 Nicole 🎧': 'af_nicole',
    '🇺🇸 🚺 Aoede': 'af_aoede',
    '🇺🇸 🚺 Kore': 'af_kore',
    '🇺🇸 🚺 Sarah': 'af_sarah',
    '🇺🇸 🚺 Nova': 'af_nova',
    '🇺🇸 🚺 Sky': 'af_sky',
    '🇺🇸 🚺 Alloy': 'af_alloy',
    '🇺🇸 🚺 Jessica': 'af_jessica',
    '🇺🇸 🚺 River': 'af_river',
    '🇺🇸 🚹 Michael': 'am_michael',
    '🇺🇸 🚹 Fenrir': 'am_fenrir',
    '🇬🇧 🚹 Daniel': 'bm_daniel',
}

# === OFFICIAL, CANONICAL PRESETS (do not change without explicit user direction) ===
VOICE_PRESETS = {
    'literature': {
        'voice': 'af_bella',  # Bella
        'speed': 1.1,
        'breathiness': 0.1,
        'tenseness': 0.1,
        'jitter': 0.15,
        'sultry': 0.1
    },
    'articles': {
        'voice': 'af_sky',    # Sky
        'speed': 1.0,
        'breathiness': 0.15,
        'tenseness': 0.5,
        'jitter': 0.3,
        'sultry': 0.1
    }
}
# === END OFFICIAL PRESETS ===

# Use a subset of known working voices (based on the error messages)
VOICES = set(CHOICES.values())

# Safely load voices, skipping any that fail
available_voices = set()
for voice in VOICES:
    try:
        pipelines[voice[0]].load_voice(voice)
        available_voices.add(voice)
        print(f"Successfully loaded voice: {voice}")
    except Exception as e:
        print(f"Failed to load voice {voice}: {str(e)}")

VOICES = available_voices  # Update the VOICES set to only include successfully loaded voices



# --- Environment-based configuration for CORS and API key ---






# --- API Key authentication dependency ---

def get_api_key():
    return API_KEY

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        from fastapi import HTTPException
        raise HTTPException(status_code=401, detail="Invalid API Key")

# Input models for API requests
class PreprocessRequest(BaseModel):
    text: str

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None  # Optional now, will be set based on fiction parameter
    speed: float = 1.0
    use_gpu: bool = CUDA_AVAILABLE
    breathiness: float = 0.0
    tenseness: float = 0.0
    jitter: float = 0.0
    sultry: float = 0.0
    fiction: bool = False  # Whether the text is fiction (True) or non-fiction (False)

class TokenizeRequest(BaseModel):
    text: str
    voice: str = "af_sky"

# --- Example: protect endpoints with API key ---
# from fastapi import Depends
# @app.post("/synthesize")
# async def synthesize(request: TTSRequest, api_key: str = Depends(verify_api_key)):
#     ...

# Job status enum
class JobStatusEnum(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# Job models - ADD THE MISSING TTSJobRequest MODEL
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
from pydantic import BaseModel
from typing import List, Optional

class TTSBatchRequest(BaseModel):
    texts: List[str]
    voice: Optional[str] = "af_sky"
    speed: Optional[float] = 1.0
    use_gpu: Optional[bool] = True
    breathiness: Optional[float] = 0.0
    tenseness: Optional[float] = 0.0
    jitter: Optional[float] = 0.0
    sultry: Optional[float] = 0.0


class TTSBatchRequest(BaseModel):
    texts: List[str]
    voice: Optional[str] = "af_sky"
    speed: Optional[float] = 1.0
    use_gpu: Optional[bool] = True
    breathiness: Optional[float] = 0.0
    tenseness: Optional[float] = 0.0
    jitter: Optional[float] = 0.0
    sultry: Optional[float] = 0.0

# Global job storage and queue
jobs_storage: Dict[str, JobStatus] = {}
job_queue = queue.Queue()
processing_jobs: Dict[str, JobStatus] = {}

# Queue processor (runs in background thread)
def process_queue():
    """Background thread that processes the job queue"""
    while True:
        try:
            job: TTSJob = job_queue.get(timeout=1)
            job_id = job.id
            if job_id in jobs_storage:
                jobs_storage[job_id].status = JobStatusEnum.PROCESSING
                jobs_storage[job_id].started_at = datetime.now()
                processing_jobs[job_id] = jobs_storage[job_id]
                print(f"🔄 Processing job {job_id}: {job.title}")
                try:
                    selected_voice, emotion_preset = select_voice_and_preset(job.voice, None, job.fiction)
                    speed = emotion_preset.get("speed", job.speed) if emotion_preset and job.speed == 1.0 else job.speed
                    breathiness = emotion_preset.get("breathiness", job.breathiness) if emotion_preset and job.breathiness == 0.0 else job.breathiness
                    tenseness = emotion_preset.get("tenseness", job.tenseness) if emotion_preset and job.tenseness == 0.0 else job.tenseness
                    jitter = emotion_preset.get("jitter", job.jitter) if emotion_preset and job.jitter == 0.0 else job.jitter
                    sultry = emotion_preset.get("sultry", job.sultry) if emotion_preset and job.sultry == 0.0 else job.sultry
                    preprocessed_text = preprocess_text(job.text)
                    (sample_rate, audio_data), phonemes = generate_audio(
                        preprocessed_text,
                        selected_voice,
                        speed,
                        job.use_gpu,
                        breathiness,
                        tenseness,
                        jitter,
                        sultry
                    )
                    if audio_data is None:
                        raise Exception("Failed to generate audio")
                    import wave, io, base64, numpy as np
                    audio_buffer = io.BytesIO()
                    channels = 1
                    sampwidth = 2
                    with wave.open(audio_buffer, 'wb') as wav_file:
                        wav_file.setnchannels(channels)
                        wav_file.setsampwidth(sampwidth)
                        wav_file.setframerate(sample_rate)
                        scaled = np.clip(audio_data, -1.0, 1.0)
                        scaled = (scaled * 32767).astype(np.int16)
                        wav_file.writeframes(scaled.tobytes())
                    audio_buffer.seek(0)
                    audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
                    jobs_storage[job_id].status = JobStatusEnum.COMPLETED
                    jobs_storage[job_id].completed_at = datetime.now()
                    jobs_storage[job_id].audio_base64 = audio_base64
                    print(f"✅ Completed job {job_id}: {job.title}")
                except Exception as e:
                    jobs_storage[job_id].status = JobStatusEnum.FAILED
                    jobs_storage[job_id].error_message = str(e)
                    print(f"❌ Failed job {job_id}: {str(e)}")
                finally:
                    if job_id in processing_jobs:
                        del processing_jobs[job_id]
                    job_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"❌ Queue processor error: {str(e)}")

queue_thread = threading.Thread(target=process_queue, daemon=True)
queue_thread.start()

# Text preprocessing function
def preprocess_text(text):
    """Preprocess text to handle paragraph flow properly"""
    text = text.strip()
    text = re.sub(r'([^\n])\n([^\n])', r'\1 \2', text)
    return text

# Helper function to select voice and preset by name

def select_voice_and_preset(requested_voice, preset_name=None, fiction=None):
    """
    Select voice and emotion preset for TTS.
    Priority:
      1. If preset_name is given and valid, use preset (voice and params).
      2. If requested_voice is given, use it with no preset.
      3. If neither is given, use Bella for fiction, Sky for non-fiction (official presets).
    Returns (voice_id, emotion_preset_dict or None)
    """
    # ENFORCE: Always use preset values for literature (fiction) and articles (non-fiction)
    if fiction is not None:
        if fiction:
            # Literature: Bella preset (override everything)
            return 'af_bella', {
                'speed': 1.1,
                'breathiness': 0,
                'tenseness': 0,
                'jitter': 0,
                'sultry': 0
            }
        else:
            # Articles: Sky preset (override everything)
            return 'af_sky', {
                'speed': 1.0,
                'breathiness': 0,
                'tenseness': 0,
                'jitter': 0,
                'sultry': 0
            }
    if preset_name and preset_name in VOICE_PRESETS:
        preset = VOICE_PRESETS[preset_name]
        return preset['voice'], {
            'speed': preset.get('speed', 1.0),
            'breathiness': preset.get('breathiness', 0.0),
            'tenseness': preset.get('tenseness', 0.0),
            'jitter': preset.get('jitter', 0.0),
            'sultry': preset.get('sultry', 0.0)
        }
    if requested_voice:
        return requested_voice, None
    # Fallback: default
    return 'af_sky', None


# Core TTS functionality from original app.py
def forward_gpu(ps, ref_s, speed):
    return models[True](ps, ref_s, speed)

def generate_audio_batch(
    texts,
    voice='af_sky',
    speed=1,
    use_gpu=CUDA_AVAILABLE,
    breathiness=0.0,
    tenseness=0.0,
    jitter=0.0,
    sultry=0.0
):
    """
    Batch version: Generate audio for a list of texts efficiently using model batch support.
    Returns: list of ((sample_rate, audio_data), phonemes)
    """
    pipeline = pipelines[voice[0]]
    pack = pipeline.load_voice(voice)
    use_gpu = use_gpu and CUDA_AVAILABLE

    # Tokenize all texts in batch
    batch_ps = []
    batch_ref_s = []
    for text in texts:
        for _, ps, _ in pipeline(text, voice, speed):
            batch_ps.append(ps)
            batch_ref_s.append(pack[len(ps)-1])
    if not batch_ps:
        return []

    # Pad sequences to the same length
    import torch
    from torch.nn.utils.rnn import pad_sequence
    ps_tensors = [torch.tensor(ps, dtype=torch.long) for ps in batch_ps]
    ref_s_tensors = [torch.tensor(ref, dtype=torch.float32) for ref in batch_ref_s]
    ps_batch = pad_sequence(ps_tensors, batch_first=True)
    ref_s_batch = pad_sequence(ref_s_tensors, batch_first=True)

    # Forward batch through model
    if use_gpu:
        ps_batch = ps_batch.cuda()
        ref_s_batch = ref_s_batch.cuda()
    model = models[use_gpu]
    audio_batch, pred_dur_batch = model.forward_with_tokens(ps_batch, ref_s_batch, speed)
    audio_batch = audio_batch.cpu().numpy() if use_gpu else audio_batch.numpy()

    # Post-process outputs
    results = []
    for idx, audio in enumerate(audio_batch):
        audio = apply_emotion_effects(audio, breathiness, tenseness, jitter, sultry)
        results.append(((24000, audio), batch_ps[idx]))
    return results

def generate_audio(text, voice='af_sky', speed=1, use_gpu=CUDA_AVAILABLE, 
                   breathiness=0.0, tenseness=0.0, jitter=0.0, sultry=0.0):
    """Core function that generates audio from text"""
    pipeline = pipelines[voice[0]]
    pack = pipeline.load_voice(voice)
    use_gpu = use_gpu and CUDA_AVAILABLE
    
    all_audio_chunks = []
    all_phonemes = []
    
    for _, ps, _ in pipeline(text, voice, speed):
        ref_s = pack[len(ps)-1]
        try:
            if use_gpu:
                audio = forward_gpu(ps, ref_s, speed)
            else:
                audio = models[False](ps, ref_s, speed)
                
            audio = apply_emotion_effects(audio, breathiness, tenseness, jitter, sultry)
                
        except Exception as e:
            if use_gpu:
                audio = models[False](ps, ref_s, speed)
                audio = apply_emotion_effects(audio, breathiness, tenseness, jitter, sultry)
            else:
                raise HTTPException(status_code=500, detail=str(e))
        
        all_audio_chunks.append(audio.numpy())
        all_phonemes.append(ps)
    
    if not all_audio_chunks:
        return None, ''
    
    combined_audio = np.concatenate(all_audio_chunks)
    combined_phonemes = '\n'.join(all_phonemes)
    
    return (24000, combined_audio), combined_phonemes

def tokenize_text(text, voice='af_sky'):
    """Tokenize text to phonemes without generating audio"""
    pipeline = pipelines[voice[0]]
    result_phonemes = []
    for _, ps, _ in pipeline(text, voice):
        result_phonemes.append(ps)
    return '\n'.join(result_phonemes)

# API Endpoints
@app.get("/")
async def root():
    return {"message": "Kokoro TTS API is running. Visit /docs for API documentation."}

@app.post("/preprocess")
def preprocess_text_endpoint(request: PreprocessRequest):
    processed = preprocess_text(request.text)
    return {"original": request.text, "processed": processed}

@app.get("/voices")
async def list_voices():
    """List all available voices"""
    return {"voices": list(VOICES)}

@app.get("/voice-choices")
async def list_voice_choices():
    """List user-friendly voice choices (display name and id)"""
    return {"choices": CHOICES}

@app.get("/voice-presets")
async def list_voice_presets():
    """List available voice presets and their parameters"""
    return {"presets": VOICE_PRESETS}

@app.get("/voice-presets/{preset_name}")
async def get_voice_preset(preset_name: str):
    """Get a specific voice preset"""
    if preset_name in VOICE_PRESETS:
        return {"preset": VOICE_PRESETS[preset_name]}
    else:
        raise HTTPException(status_code=404, detail="Preset not found")

@app.get("/choices")
async def list_choices():
    """List user-friendly voice choices (display name and id)"""
    return {"choices": CHOICES}

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """Convert text to speech and return audio as WAV file. Supports presets and auto-mapping fiction/non-fiction."""
    try:
        # Support for presets: if request has 'preset' field, use it
        preset_name = getattr(request, 'preset', None) if hasattr(request, 'preset') else None
        selected_voice, emotion_preset = select_voice_and_preset(request.voice, preset_name, fiction=getattr(request, 'fiction', None))

        if selected_voice not in VOICES:
            raise HTTPException(status_code=400, detail=f"Voice '{selected_voice}' not found. Available voices: {list(VOICES)}")

        # ENFORCE: If emotion_preset is set (literature/articles), override all emotion parameters
        if emotion_preset:
            speed = emotion_preset['speed']
            breathiness = emotion_preset['breathiness']
            tenseness = emotion_preset['tenseness']
            jitter = emotion_preset['jitter']
            sultry = emotion_preset['sultry']
        else:
            speed = request.speed
            breathiness = request.breathiness
            tenseness = request.tenseness
            jitter = request.jitter
            sultry = request.sultry

        preprocessed_text = preprocess_text(request.text)

        (sample_rate, audio_data), phonemes = generate_audio(
            preprocessed_text,
            selected_voice,
            speed,
            request.use_gpu,
            breathiness,
            tenseness,
            jitter,
            sultry
        )
        if audio_data is None:
            raise HTTPException(status_code=500, detail="Audio generation failed")

        import wave, io, base64, numpy as np
        audio_buffer = io.BytesIO()
        channels = 1
        sampwidth = 2
        with wave.open(audio_buffer, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sampwidth)
            wav_file.setframerate(sample_rate)
            scaled = np.clip(audio_data, -1.0, 1.0)
            scaled = (scaled * 32767).astype(np.int16)
            wav_file.writeframes(scaled.tobytes())
        audio_buffer.seek(0)
        return StreamingResponse(audio_buffer, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts/batch")
async def batch_text_to_speech(request: TTSBatchRequest):
    """
    Convert a batch of texts to speech and return a list of base64-encoded audio strings.
    """
    try:
        # ENFORCE: Always use preset values for literature (fiction) and articles (non-fiction)
        # Check if request has 'fiction' attribute (should be a list matching texts)
        fiction_list = getattr(request, 'fiction', None)
        # If not present, treat as all non-fiction (Sky)
        if fiction_list is None:
            fiction_list = [False] * len(request.texts)
        # For each text, select the preset and override all parameters
        results = []
        for idx, text in enumerate(request.texts):
            is_fiction = fiction_list[idx] if idx < len(fiction_list) else False
            if is_fiction:
                # Bella preset
                voice = 'af_bella'
                speed = 1.1
                breathiness = 0.1
                tenseness = 0.1
                jitter = 0.15
                sultry = 0.1
            else:
                # Sky preset
                voice = 'af_sky'
                speed = 1.0
                breathiness = 0.15
                tenseness = 0.5
                jitter = 0.3
                sultry = 0.1
            (sample_rate, audio_data), _ = generate_audio(
                text,
                voice,
                speed,
                request.use_gpu,
                breathiness,
                tenseness,
                jitter,
                sultry
            )
            results.append(((sample_rate, audio_data), None))
        # Encode each audio result as base64 wav
        import io, wave, numpy as np, base64
        audio_base64_list = []
        for (sample_rate, audio_data), _ in results:
            audio_buffer = io.BytesIO()
            channels = 1
            sampwidth = 2
            with wave.open(audio_buffer, 'wb') as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(sampwidth)
                wav_file.setframerate(sample_rate)
                scaled = np.clip(audio_data, -1.0, 1.0)
                scaled = (scaled * 32767).astype(np.int16)
                wav_file.writeframes(scaled.tobytes())
            audio_buffer.seek(0)
            audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
            audio_base64_list.append(audio_base64)
        return {"audios": audio_base64_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts/base64")
async def text_to_speech_base64(request: TTSRequest):
    """Convert text to speech and return audio as base64 encoded string"""
    try:
        selected_voice, emotion_preset = select_voice_and_preset(request.voice, request.fiction)
        
        if selected_voice not in VOICES:
            raise HTTPException(status_code=400, detail=f"Voice '{selected_voice}' not found. Available voices: {list(VOICES)}")
        
        speed = request.speed
        breathiness = request.breathiness
        tenseness = request.tenseness
        jitter = request.jitter
        sultry = request.sultry
        if emotion_preset:
            if request.speed == 1.0 and "speed" in emotion_preset:
                speed = emotion_preset["speed"]
            if request.breathiness == 0.0 and "breathiness" in emotion_preset:
                breathiness = emotion_preset["breathiness"]
            if request.tenseness == 0.0 and "tenseness" in emotion_preset:
                tenseness = emotion_preset["tenseness"]
            if request.jitter == 0.0 and "jitter" in emotion_preset:
                jitter = emotion_preset["jitter"]
            if request.sultry == 0.0 and "sultry" in emotion_preset:
                sultry = emotion_preset["sultry"]

        preprocessed_text = preprocess_text(request.text)
        
        (sample_rate, audio_data), phonemes = generate_audio(
            preprocessed_text, 
            selected_voice,
            speed, 
            request.use_gpu,
            breathiness,
            tenseness,
            jitter,
            sultry
        )
        
        if audio_data is None:
            raise HTTPException(status_code=500, detail="Failed to generate audio")
        
        import wave
        import struct
        
        audio_buffer = io.BytesIO()
        channels = 1
        sampwidth = 2
        
        with wave.open(audio_buffer, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sampwidth)
            wav_file.setframerate(sample_rate)
            
            scaled = np.clip(audio_data, -1.0, 1.0)
            scaled = (scaled * 32767).astype(np.int16)
            wav_file.writeframes(scaled.tobytes())
        
        audio_buffer.seek(0)
        
        audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
        
        return {
            "sample_rate": sample_rate,
            "audio_base64": audio_base64,
            "phonemes": phonemes
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tokenize")
async def tokenize(request: TokenizeRequest):
    """Tokenize text to phonemes without generating audio"""
    try:
        if request.voice not in VOICES:
            raise HTTPException(status_code=400, detail=f"Voice '{request.voice}' not found. Available voices: {list(VOICES)}")
        
        phonemes = tokenize_text(request.text, request.voice)
        
        return {
            "phonemes": phonemes
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# FIXED: Now accepts TTSJobRequest and creates TTSJob internally
@app.post("/tts/submit", response_model=JobResponse)
async def submit_tts_job(job_request: TTSJobRequest):
    """Submit a TTS job to the processing queue"""
    try:
        print(f"📥 Received job submission request:")
        print(f"  Title: {job_request.title}")
        print(f"  Author: {job_request.author}")
        print(f"  Genre: {job_request.genre}")
        print(f"  Fiction: {job_request.fiction}")
        print(f"  Text length: {len(job_request.text)}")
        
        job_id = str(uuid.uuid4())
        
        # Create TTSJob from TTSJobRequest
        job_data = TTSJob(
            id=job_id,
            text=job_request.text,
            voice=job_request.voice,
            speed=job_request.speed,
            use_gpu=job_request.use_gpu,
            breathiness=job_request.breathiness,
            tenseness=job_request.tenseness,
            jitter=job_request.jitter,
            sultry=job_request.sultry,
            fiction=job_request.fiction,
            title=job_request.title,
            author=job_request.author,
            genre=job_request.genre
        )
        
        job_status = JobStatus(
            job_id=job_id,
            status=JobStatusEnum.QUEUED,
            title=job_request.title,
            author=job_request.author,
            genre=job_request.genre,
            total_in_queue=job_queue.qsize() + 1,
            created_at=datetime.now()
        )
        
        jobs_storage[job_id] = job_status
        job_queue.put(job_data)
        
        print(f"📋 Queued job {job_id}: {job_request.title}")
        print(f"📊 Queue size now: {job_queue.qsize()}")
        print(f"📊 Total jobs in storage: {len(jobs_storage)}")
        
        return JobResponse(
            job_id=job_id,
            status=JobStatusEnum.QUEUED,
            message=f"Job queued successfully. Position: {job_queue.qsize()}"
        )
    except Exception as e:
        print(f"❌ Submit job error: {str(e)}")
        import traceback
        print(f"❌ Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tts/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    if job_id not in jobs_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    job_status = jobs_storage[job_id]
    if job_status.status == JobStatusEnum.QUEUED:
        job_status.position_in_queue = max(1, job_queue.qsize())
    job_status.total_in_queue = len(jobs_storage)
    return job_status

@app.get("/tts/queue", response_model=QueueStatus)
async def get_queue_status():
    queued_count = sum(1 for job in jobs_storage.values() if job.status == JobStatusEnum.QUEUED)
    processing_count = len(processing_jobs)
    
    print(f"📊 Queue status requested:")
    print(f"  Total jobs: {len(jobs_storage)}")
    print(f"  Processing: {processing_count}")
    print(f"  Queued: {queued_count}")
    print(f"  Jobs: {[f'{job.job_id[:8]}:{job.status}:{job.title}' for job in jobs_storage.values()]}")
    
    return QueueStatus(
        total_jobs=len(jobs_storage),
        processing_jobs=processing_count,
        queued_jobs=queued_count,
        jobs=list(jobs_storage.values())
    )

@app.delete("/tts/job/{job_id}")
async def cancel_job(job_id: str):
    if job_id not in jobs_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    job_status = jobs_storage[job_id]
    if job_status.status == JobStatusEnum.PROCESSING:
        raise HTTPException(status_code=400, detail="Cannot cancel job that is currently processing")
    if job_status.status == JobStatusEnum.QUEUED:
        del jobs_storage[job_id]
        return {"message": "Job cancelled successfully"}
    return {"message": "Job already completed or failed"}

# HLS Streaming Support
@app.post("/tts/stream")
async def stream_tts(request: TTSRequest, background_tasks: BackgroundTasks):
    """Stream audio as it's being generated using HLS"""
    try:
        # Create a unique ID for this streaming session
        stream_id = str(uuid.uuid4())
        stream_dir = f"streams/{stream_id}"
        os.makedirs(stream_dir, exist_ok=True)
        
        # Create initial playlist file
        playlist_path = f"{stream_dir}/playlist.m3u8"
        with open(playlist_path, "w") as f:
            f.write("#EXTM3U\n")
            f.write("#EXT-X-VERSION:3\n")
            f.write("#EXT-X-TARGETDURATION:2\n")
            f.write("#EXT-X-MEDIA-SEQUENCE:0\n")
        
        # Start audio generation in background
        background_tasks.add_task(
            generate_streaming_audio,
            request.text,
            stream_id,
            request.voice,
            request.speed,
            request.use_gpu,
            request.breathiness,
            request.tenseness,
            request.jitter,
            request.sultry,
            request.fiction
        )
        
        # Return the stream URL
        base_url = str(request.base_url).rstrip('/')
        stream_url = f"{base_url}/streams/{stream_id}/playlist.m3u8"
        return {"stream_url": stream_url, "stream_id": stream_id}
    except Exception as e:
        print(f"❌ Streaming error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Serve stream files
@app.get("/streams/{stream_id}/{file_name}")
async def get_stream_file(stream_id: str, file_name: str):
    """Serve HLS stream files"""
    file_path = f"streams/{stream_id}/{file_name}"
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Stream file not found")
    
    # For m3u8 playlists
    if file_name.endswith(".m3u8"):
        with open(file_path, "r") as f:
            content = f.read()
        return Response(content=content, media_type="application/vnd.apple.mpegurl")
    
    # For TS segments
    elif file_name.endswith(".ts"):
        return StreamingResponse(
            io.open(file_path, "rb"),
            media_type="video/mp2t"
        )
    
    raise HTTPException(status_code=400, detail="Invalid stream file type")

# Function to generate streaming audio
async def generate_streaming_audio(
    text: str, 
    stream_id: str,
    voice: Optional[str] = None,
    speed: float = 1.0,
    use_gpu: bool = CUDA_AVAILABLE,
    breathiness: float = 0.0,
    tenseness: float = 0.0,
    jitter: float = 0.0,
    sultry: float = 0.0,
    fiction: bool = False
):
    """Generate audio in chunks and create HLS stream"""
    try:
        # Process text into sentences or paragraphs
        text_chunks = preprocess_text(text).split('\n\n')
        
        # Select voice based on fiction parameter if not specified
        voice_id, preset = select_voice_and_preset(voice, fiction=fiction)
        
        # Create stream directory
        stream_dir = f"streams/{stream_id}"
        
        # Initialize HLS playlist
        segment_duration = 2  # seconds
        sample_rate = 24000
        
        # Process each chunk
        segment_index = 0
        for i, chunk in enumerate(text_chunks):
            if not chunk.strip():
                continue
                
            # Generate audio for this chunk
            audio_data = await run_in_threadpool(
                lambda: generate_audio(
                    chunk, 
                    voice_id, 
                    speed, 
                    use_gpu, 
                    breathiness, 
                    tenseness, 
                    jitter, 
                    sultry
                )
            )
            
            # Convert to WAV format
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Split into segments of segment_duration
            samples_per_segment = int(segment_duration * sample_rate)
            for j in range(0, len(audio_array), samples_per_segment):
                segment = audio_array[j:j+samples_per_segment]
                
                # Pad last segment if needed
                if len(segment) < samples_per_segment:
                    segment = np.pad(segment, (0, samples_per_segment - len(segment)))
                
                # Save segment as TS file
                segment_file = f"{stream_dir}/segment_{segment_index}.ts"
                
                # Convert to bytes and save
                segment_bytes = segment.tobytes()
                with open(segment_file, "wb") as f:
                    f.write(segment_bytes)
                
                # Update playlist
                with open(f"{stream_dir}/playlist.m3u8", "a") as f:
                    f.write(f"#EXTINF:{segment_duration},\n")
                    f.write(f"segment_{segment_index}.ts\n")
                
                segment_index += 1
                
                # Small delay to simulate real-time generation
                await asyncio.sleep(0.1)
        
        # Mark the end of the playlist
        with open(f"{stream_dir}/playlist.m3u8", "a") as f:
            f.write("#EXT-X-ENDLIST\n")
            
        print(f"✅ Completed streaming audio generation for {stream_id}")
    except Exception as e:
        print(f"❌ Error generating streaming audio: {str(e)}")
        import traceback
        print(f"❌ Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    # Create streams directory if it doesn't exist
    os.makedirs("streams", exist_ok=True)
    uvicorn.run("api:app", host="0.0.0.0", port=8080, reload=False)
