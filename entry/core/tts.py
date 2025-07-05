"""
Core TTS functionality
"""
import re
import torch
import numpy as np
from typing import Tuple, Optional, List
from fastapi import HTTPException
from loguru import logger

from entry.core.models import (
    get_models, get_pipelines, get_voices, get_voice_presets
)
from entry.audio_effects import apply_emotion_effects
from entry.config import get_settings
from entry.utils.dict_utils import extract_dict_values
from entry.utils.list_utils import safe_list_append, safe_list_extend


def is_gpu_available() -> bool:
    """Check if GPU is available and enabled in settings"""
    settings = get_settings()
    return torch.cuda.is_available() and settings.cuda_available


def get_gpu_settings() -> tuple[bool, bool]:
    """Get GPU availability and settings in one call"""
    settings = get_settings()
    cuda_available = torch.cuda.is_available() and settings.cuda_available
    return cuda_available, settings.cuda_available


def validate_model_components(models, pipelines, voices, context: str = "TTS request") -> bool:
    """Validate that all required model components are available"""
    if not models or not pipelines or not voices:
        logger.error(f"Unable to process {context} - missing components: models={bool(models)}, pipelines={bool(pipelines)}, voices={bool(voices)}")
        return False
    return True


def get_model_components():
    """Get and validate all model components"""
    models = get_models()
    pipelines = get_pipelines()
    voices = get_voices()
    
    if not validate_model_components(models, pipelines, voices):
        raise HTTPException(status_code=503, detail="Service not ready - models not initialized")
    
    return models, pipelines, voices


def preprocess_text(text: str) -> str:
    """Preprocess text for TTS processing"""
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
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
    
    # ENFORCE: Only use preset values when fiction is explicitly set (not default False)
    if fiction is True:  # Explicitly fiction
        # Fiction: af_sky preset (override everything)
        return 'af_sky', {
            'speed': 1.0,
            'breathiness': 0,
            'tenseness': 0,
            'jitter': 0,
            'sultry': 0
        }
    elif fiction is False and requested_voice is None:  # Explicitly non-fiction AND no voice specified
        # Non-fiction: af_heart preset (only when no voice specified)
        return 'af_heart', {
            'speed': 1.0,
            'breathiness': 0,
            'tenseness': 0,
            'jitter': 0,
            'sultry': 0
        }
    
    if preset_name and preset_name in voice_presets:
        preset = voice_presets[preset_name]
        param_keys = ['speed', 'breathiness', 'tenseness', 'jitter', 'sultry']
        param_defaults = [1.0, 0.0, 0.0, 0.0, 0.0]
        emotion_params = extract_dict_values(preset, param_keys, param_defaults)
        return preset['voice'], emotion_params
    
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
    voice: Optional[str] = 'af_sky',
    speed: float = 1.0,
    use_gpu: bool = True,
    breathiness: float = 0.0,
    tenseness: float = 0.0,
    jitter: float = 0.0,
    sultry: float = 0.0
) -> Tuple[int, np.ndarray]:
    """
    Generate audio from text using the TTS model.
    
    Args:
        text: Text to convert to speech
        voice: Voice to use (e.g., 'af_sky', 'af_heart')
        speed: Speech speed multiplier
        use_gpu: Whether to use GPU model
        breathiness: Breathiness effect (0.0 to 1.0)
        tenseness: Tenseness effect (0.0 to 1.0)
        jitter: Jitter effect (0.0 to 1.0)
        sultry: Sultry effect (0.0 to 1.0)
        
    Returns:
        Tuple of (sample_rate, audio_data)
    """
    # Get model components
    models, pipelines, voices = get_model_components()
    
    # Handle None voice by selecting a default
    if voice is None:
        voice = 'af_sky'  # Default voice
        logger.info(f"No voice specified, using default: {voice}")
    
    logger.info(f"Generating audio for voice '{voice}' with speed={speed}, use_gpu={use_gpu}")
    logger.info(f"Available pipelines: {list(pipelines.keys())}")
    logger.info(f"Available voices: {list(voices)}")
    
    # Validate voice
    if voice not in voices:
        logger.error(f"Voice '{voice}' not found in available voices: {list(voices)}")
        raise HTTPException(status_code=400, detail=f"Voice '{voice}' not available")
    
    try:
        # Get pipeline for voice
        voice_prefix = voice[0]  # 'a' for 'af_sky', 'b' for 'bf_emma', etc.
        available_keys = list(pipelines.keys())
        
        if voice_prefix not in pipelines:
            logger.error(f"Missing pipeline for voice prefix '{voice_prefix}'. Available: {available_keys}")
            raise HTTPException(status_code=500, detail=f"No pipeline available for voice '{voice}'")
        
        pipeline = pipelines[voice_prefix]
        
        # Try to get the pipeline using the voice prefix as fallback
        if voice not in pipeline.voices:
            logger.warning(f"Using fallback pipeline key '{voice_prefix}' for voice '{voice}'")
            # This might work if the pipeline can handle the voice
            pass
        
        # Preprocess text
        preprocessed_text = preprocess_text(text)
        
        # Load voice reference audio
        try:
            ref_s = pipeline.load_voice(voice)
        except Exception as voice_error:
            logger.error(f"Failed to load voice pack for '{voice}': {str(voice_error)}")
            raise HTTPException(status_code=500, detail=f"Failed to load voice '{voice}'")
        
        # Generate audio using pipeline
        try:
            audio_results = []
            for result in pipeline(preprocessed_text, voice=voice, speed=speed):
                if result.audio is not None:
                    audio_results.append(result.audio.numpy())
            
            if not audio_results:
                raise HTTPException(status_code=500, detail="No audio generated")
            
            # Concatenate audio chunks if multiple
            audio = np.concatenate(audio_results) if len(audio_results) > 1 else audio_results[0]
            
        except Exception as e:
            logger.error(f"Audio generation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error generating audio: {str(e)}")
        
        # Apply emotion effects
        audio = apply_emotion_effects(audio, breathiness, tenseness, jitter, sultry)
        
        return 24000, audio
        
    except Exception as e:
        logger.error(f"Error accessing pipeline for voice '{voice}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating audio: {str(e)}")


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

    # Deduplication optimization: cache identical texts
    unique_texts = {}
    text_to_index = {}
    
    for i, text in enumerate(texts):
        if text not in unique_texts:
            unique_texts[text] = []
            # Process unique text once
            for _, ps, _ in pipeline(text, voice, speed):
                safe_list_append(unique_texts[text], (ps, pack[len(ps)-1]))
        text_to_index[i] = text
    
    # Build batch from deduplicated results
    batch_ps = []
    batch_ref_s = []
    result_mapping = []  # Track which result corresponds to which original text
    
    for i, text in enumerate(texts):
        cached_results = unique_texts[text]
        for ps, ref_s in cached_results:
            safe_list_append(batch_ps, ps)
            safe_list_append(batch_ref_s, ref_s)
            safe_list_append(result_mapping, i)  # Track original text index
    
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
        safe_list_append(results, ((24000, audio), batch_ps[idx]))
    
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
        safe_list_append(result_phonemes, ps)
    return '\n'.join(result_phonemes)