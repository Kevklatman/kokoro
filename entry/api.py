from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
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

# Import your existing TTS components
from kokoro import KModel, KPipeline
from audio_effects import apply_emotion_effects

# Initialize models and pipelines
CUDA_AVAILABLE = torch.cuda.is_available()
models = {gpu: KModel().to('cuda' if gpu else 'cpu').eval() for gpu in [False] + ([True] if CUDA_AVAILABLE else [])}
pipelines = {lang_code: KPipeline(lang_code=lang_code, model=False) for lang_code in 'ab'}
pipelines['a'].g2p.lexicon.golds['kokoro'] = 'kˈOkəɹO'
pipelines['b'].g2p.lexicon.golds['kokoro'] = 'kˈQkəɹQ'

# Use a subset of known working voices (based on the error messages)
VOICES = {
    'af_heart', 'af_bella', 'af_nicole', 'af_aoede', 'af_kore', 'af_sarah', 'af_nova', 
    'af_sky', 'af_alloy', 'af_jessica', 'af_river', 'am_michael', 'am_fenrir',
    'am_kevin', 'am_josh', 'am_adam', 'am_jack', 'bf_ruby', 'bf_selene',
    'bm_michael', 'bm_kevin'
}

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

# Create FastAPI app
app = FastAPI(
    title="Kokoro TTS API",
    description="REST API for Kokoro Text-to-Speech Engine",
    version="1.0.0"
)

# Enable CORS
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    voice: str = "af_heart"

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
                    selected_voice, emotion_preset = select_voice_and_preset(job.voice, job.fiction)
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

# Helper function to select voice based on fiction parameter
def select_voice_and_preset(requested_voice, is_fiction):
    """Select appropriate voice and emotion preset based on fiction parameter"""
    if requested_voice is not None:
        return requested_voice, None  # Use specified voice if provided, no preset emotions
    
    # Default voices with emotion presets
    if is_fiction:
        voice_id = "af_bella"  # 🇺🇸 🚺 Bella 🔥
        emotion_preset = {
            "speed": 1,
            "breathiness": 0.0,
            "tenseness": 0.0,
            "jitter": 0.0,
            "sultry": 0.0
        }
    else:
        voice_id = "af_sky"  # 🇺🇸 🚺 Sky
        emotion_preset = {
            "speed": 1,
            "breathiness": 0.0,
            "tenseness": 0.0,
            "jitter": 0.0,
            "sultry": 0.0
        }
        
    return voice_id, emotion_preset

# Core TTS functionality from original app.py
def forward_gpu(ps, ref_s, speed):
    return models[True](ps, ref_s, speed)

def generate_audio(text, voice='af_heart', speed=1, use_gpu=CUDA_AVAILABLE, 
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

def tokenize_text(text, voice='af_heart'):
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

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """Convert text to speech and return audio as WAV file"""
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
        
        return StreamingResponse(
            audio_buffer, 
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=kokoro_tts.wav",
                "X-Phonemes": base64.b64encode(phonemes.encode()).decode()
            }
        )
    
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

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8080, reload=False)