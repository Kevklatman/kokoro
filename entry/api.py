from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import re
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
    voice: str = "af_heart"
    speed: float = 1.0
    use_gpu: bool = CUDA_AVAILABLE
    breathiness: float = 0.0
    tenseness: float = 0.0
    jitter: float = 0.0
    sultry: float = 0.0

class TokenizeRequest(BaseModel):
    text: str
    voice: str = "af_heart"

# Simple pass-through function (no preprocessing as it's handled on the frontend)
def preprocess_text(text):
    """Simple text pass-through - preprocessing handled on frontend"""
    return text.strip()

# Core TTS functionality from original app.py
def forward_gpu(ps, ref_s, speed):
    return models[True](ps, ref_s, speed)

def generate_audio(text, voice='af_heart', speed=1, use_gpu=CUDA_AVAILABLE, 
                   breathiness=0.0, tenseness=0.0, jitter=0.0, sultry=0.0):
    """Core function that generates audio from text"""
    pipeline = pipelines[voice[0]]
    pack = pipeline.load_voice(voice)
    use_gpu = use_gpu and CUDA_AVAILABLE
    
    # Collect all audio chunks and phoneme strings
    all_audio_chunks = []
    all_phonemes = []
    
    for _, ps, _ in pipeline(text, voice, speed):
        ref_s = pack[len(ps)-1]
        try:
            if use_gpu:
                audio = forward_gpu(ps, ref_s, speed)
            else:
                audio = models[False](ps, ref_s, speed)
                
            # Apply emotion effects
            audio = apply_emotion_effects(audio, breathiness, tenseness, jitter, sultry)
                
        except Exception as e:
            if use_gpu:
                # Fallback to CPU
                audio = models[False](ps, ref_s, speed)
                # Apply emotion effects
                audio = apply_emotion_effects(audio, breathiness, tenseness, jitter, sultry)
            else:
                raise HTTPException(status_code=500, detail=str(e))
        
        # Add this chunk to our collection
        all_audio_chunks.append(audio.numpy())
        all_phonemes.append(ps)
    
    if not all_audio_chunks:
        return None, ''
    
    # Combine all audio chunks into a single numpy array
    combined_audio = np.concatenate(all_audio_chunks)
    combined_phonemes = '\n'.join(all_phonemes)
    
    # Return the combined audio
    return (24000, combined_audio), combined_phonemes

def tokenize_text(text, voice='af_heart'):
    """Tokenize text to phonemes without generating audio"""
    pipeline = pipelines[voice[0]]
    for _, ps, _ in pipeline(text, voice):
        return ps
    return ''

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
    """
    Convert text to speech and return audio as WAV file
    """
    try:
        # Validate voice
        if request.voice not in VOICES:
            raise HTTPException(status_code=400, detail=f"Voice '{request.voice}' not found. Available voices: {list(VOICES)}")
        
        # Preprocess text for better paragraph handling
        preprocessed_text = preprocess_text(request.text)
        
        # Generate audio
        (sample_rate, audio_data), phonemes = generate_audio(
            preprocessed_text, 
            request.voice, 
            request.speed, 
            request.use_gpu,
            request.breathiness,
            request.tenseness,
            request.jitter,
            request.sultry
        )
        
        if audio_data is None:
            raise HTTPException(status_code=500, detail="Failed to generate audio")
        
        # Convert to WAV format without scipy
        # Create a WAV file in memory using wave module from standard library
        import wave
        import struct
        
        # Create an in-memory file-like object
        audio_buffer = io.BytesIO()
        
        # Sample rate, channels, sample width, etc.
        channels = 1  # Mono audio
        sampwidth = 2  # 16-bit audio
        
        # Create the WAV file
        with wave.open(audio_buffer, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sampwidth)
            wav_file.setframerate(sample_rate)
            
            # Convert audio_data to 16-bit integers and write to the WAV file
            # Scale and convert to 16-bit PCM
            scaled = np.clip(audio_data, -1.0, 1.0)
            scaled = (scaled * 32767).astype(np.int16)
            wav_file.writeframes(scaled.tobytes())
        
        audio_buffer.seek(0)
        
        # Return the audio as a streaming response
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
    """
    Convert text to speech and return audio as base64 encoded string
    (easier for mobile apps to handle)
    """
    try:
        # Validate voice
        if request.voice not in VOICES:
            raise HTTPException(status_code=400, detail=f"Voice '{request.voice}' not found. Available voices: {list(VOICES)}")
        
        # Preprocess text for better paragraph handling
        preprocessed_text = preprocess_text(request.text)
        
        # Generate audio
        (sample_rate, audio_data), phonemes = generate_audio(
            preprocessed_text, 
            request.voice, 
            request.speed, 
            request.use_gpu,
            request.breathiness,
            request.tenseness,
            request.jitter,
            request.sultry
        )
        
        if audio_data is None:
            raise HTTPException(status_code=500, detail="Failed to generate audio")
        
        # Convert to WAV format without scipy
        # Create a WAV file in memory using wave module from standard library
        import wave
        import struct
        
        # Create an in-memory file-like object
        audio_buffer = io.BytesIO()
        
        # Sample rate, channels, sample width, etc.
        channels = 1  # Mono audio
        sampwidth = 2  # 16-bit audio
        
        # Create the WAV file
        with wave.open(audio_buffer, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sampwidth)
            wav_file.setframerate(sample_rate)
            
            # Convert audio_data to 16-bit integers and write to the WAV file
            # Scale and convert to 16-bit PCM
            scaled = np.clip(audio_data, -1.0, 1.0)
            scaled = (scaled * 32767).astype(np.int16)
            wav_file.writeframes(scaled.tobytes())
        
        audio_buffer.seek(0)
        
        # Encode as base64
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
    """
    Tokenize text to phonemes without generating audio
    """
    try:
        # Validate voice
        if request.voice not in VOICES:
            raise HTTPException(status_code=400, detail=f"Voice '{request.voice}' not found. Available voices: {list(VOICES)}")
        
        # Tokenize text
        phonemes = tokenize_text(request.text, request.voice)
        
        return {
            "phonemes": phonemes
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8080, reload=False)
