"""
Core TTS functionality
"""
import re
import torch
import numpy as np
from typing import Tuple, Optional, List
from fastapi import HTTPException

from entry.core.models import (
    get_models, get_pipelines, get_voices, get_voice_presets
)
from entry.audio_effects import apply_emotion_effects
from entry.config import get_settings


def preprocess_text(text: str) -> str:
    """Preprocess text to handle paragraph flow properly"""
    text = text.strip()
    text = re.sub(r'([^\n])\n([^\n])', r'\1 \2', text)
    return text


def select_voice_and_preset(
    requested_voice: Optional[str], 
    preset_name: Optional[str] = None, 
    fiction: Optional[bool] = None
) -> Tuple[str, Optional[dict]]:
    """
    Select voice and emotion preset for TTS.
    Priority:
      1. If preset_name is given and valid, use preset (voice and params).
      2. If requested_voice is given, use it with no preset.
      3. If neither is given, use af_sky for fiction and af_heart for non-fiction.
    Returns (voice_id, emotion_preset_dict or None)
    """
    voice_presets = get_voice_presets()
    
    # ENFORCE: Always use preset values for fiction and non-fiction
    if fiction is not None:
        if fiction:
            # Fiction: af_sky preset (override everything)
            return 'af_sky', {
                'speed': 1.1,
                'breathiness': 0,
                'tenseness': 0,
                'jitter': 0,
                'sultry': 0
            }
        else:
            # Non-fiction: af_heart preset (override everything)
            return 'af_heart', {
                'speed': 1.0,
                'breathiness': 0,
                'tenseness': 0,
                'jitter': 0,
                'sultry': 0
            }
    
    if preset_name and preset_name in voice_presets:
        preset = voice_presets[preset_name]
        return preset['voice'], {
            'speed': preset.get('speed', 1.0),
            'breathiness': preset.get('breathiness', 0.0),
            'tenseness': preset.get('tenseness', 0.0),
            'jitter': preset.get('jitter', 0.0),
            'sultry': preset.get('sultry', 0.0)
        }
    
    if requested_voice:
        return requested_voice, None
    
    # Fallback: default to af_sky
    return 'af_heart', None


def forward_gpu(ps, ref_s, speed):
    """Forward pass using GPU model"""
    models = get_models()
    return models[True](ps, ref_s, speed)


def generate_audio(
    text: str, 
    voice: str = 'af_sky', 
    speed: float = 1.0, 
    use_gpu: bool = True, 
    breathiness: float = 0.0, 
    tenseness: float = 0.0, 
    jitter: float = 0.0, 
    sultry: float = 0.0
) -> Tuple[Tuple[int, np.ndarray], str]:
    """Core function that generates audio from text"""
    settings = get_settings()
    pipelines = get_pipelines()
    models = get_models()
    voices = get_voices()
    
    if voice not in voices:
        raise HTTPException(
            status_code=400, 
            detail=f"Voice '{voice}' not found. Available voices: {list(voices)}"
        )
    
    pipeline = pipelines[voice[0]]
    pack = pipeline.load_voice(voice)
    use_gpu = use_gpu and torch.cuda.is_available() and settings.cuda_available
    
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


def generate_audio_batch(
    texts: List[str],
    voice: str = 'af_sky',
    speed: float = 1.0,
    use_gpu: bool = True,
    breathiness: float = 0.0,
    tenseness: float = 0.0,
    jitter: float = 0.0,
    sultry: float = 0.0
) -> List[Tuple[Tuple[int, np.ndarray], str]]:
    """
    Batch version: Generate audio for a list of texts efficiently using model batch support.
    Returns: list of ((sample_rate, audio_data), phonemes)
    """
    settings = get_settings()
    pipelines = get_pipelines()
    models = get_models()
    voices = get_voices()
    
    if voice not in voices:
        raise HTTPException(
            status_code=400, 
            detail=f"Voice '{voice}' not found. Available voices: {list(voices)}"
        )
    
    pipeline = pipelines[voice[0]]
    pack = pipeline.load_voice(voice)
    use_gpu = use_gpu and torch.cuda.is_available() and settings.cuda_available

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


def tokenize_text(text: str, voice: str = 'af_sky') -> str:
    """Tokenize text to phonemes without generating audio"""
    pipelines = get_pipelines()
    voices = get_voices()
    
    if voice not in voices:
        raise HTTPException(
            status_code=400, 
            detail=f"Voice '{voice}' not found. Available voices: {list(voices)}"
        )
    
    pipeline = pipelines[voice[0]]
    result_phonemes = []
    for _, ps, _ in pipeline(text, voice):
        result_phonemes.append(ps)
    return '\n'.join(result_phonemes)