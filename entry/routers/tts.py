"""
TTS API routes
"""
from fastapi import APIRouter, HTTPException, Response, Request
from fastapi.responses import StreamingResponse, JSONResponse
import base64
import numpy as np

from entry.models import (
    TTSRequest, TTSBatchRequest, TokenizeRequest, PreprocessRequest,
    TTSResponse, BatchTTSResponse, TokenizeResponse, PreprocessResponse
)
from entry.core.tts import (
    generate_audio, generate_audio_batch, tokenize_text, 
    preprocess_text, select_voice_and_preset
)
from entry.core.models import get_settings
from entry.utils.audio import audio_to_base64, create_wav_response, create_audio_response, optimize_response_size, audio_to_bytes
from loguru import logger

router = APIRouter()


@router.post("/", response_class=StreamingResponse)
async def text_to_speech(request: TTSRequest):
    """Convert text to speech and return audio"""
    try:
        # Validate that models are loaded and accessible
        from entry.core.models import get_models, get_pipelines, get_voices
        # Pre-log the current state for diagnostics
        models = get_models()
        pipelines = get_pipelines()
        voices = get_voices()
        logger.info(f"TTS endpoint has access to: models={len(models)}, pipelines={len(pipelines)}, voices={len(voices)}")
        
        if not models or not pipelines or not voices:
            logger.error(f"Unable to process TTS request - missing components: models={bool(models)}, pipelines={bool(pipelines)}, voices={bool(voices)}")
            raise HTTPException(status_code=503, detail="TTS system not fully initialized")
        
        # Extract request parameters
        text = request.text
        voice = request.voice
        speed = request.speed
        quality = getattr(request, 'quality', 'auto') if hasattr(request, 'quality') else 'auto'
        format = getattr(request, 'format', 'auto') if hasattr(request, 'format') else 'auto'
        
        logger.info(f"TTS request for voice {voice} with quality {quality} and format {format}")
        
        # Select voice and preset
        preset_name = getattr(request, 'preset', None) if hasattr(request, 'preset') else None
        selected_voice, emotion_preset = select_voice_and_preset(
            request.voice, preset_name, fiction=request.fiction
        )

        # Apply preset values if available (ENFORCE presets)
        if emotion_preset:
            speed = emotion_preset['speed']
            breathiness = emotion_preset['breathiness']
            tenseness = emotion_preset['tenseness']
            jitter = emotion_preset['jitter']
            sultry = emotion_preset['sultry']
        else:
            breathiness = request.breathiness
            tenseness = request.tenseness
            jitter = request.jitter
            sultry = request.sultry

        preprocessed_text = preprocess_text(request.text)

        # Use GPU if available based on settings
        settings = get_settings()
        use_gpu = settings.cuda_available
        
        (sample_rate, audio_data), phonemes = generate_audio(
            preprocessed_text,
            selected_voice,
            speed,
            use_gpu,
            breathiness,
            tenseness,
            jitter,
            sultry
        )
        
        if audio_data is None:
            logger.error("Audio generation failed - returned None")
            raise HTTPException(status_code=500, detail="Audio generation failed - returned None")

        # Handle different quality and format settings
        if quality == 'auto' or format == 'auto':
            # Automatically optimize response size to avoid Cloud Run limits
            audio_buffer, used_quality, used_format = optimize_response_size(
                audio_data, sample_rate, max_size_kb=30000, use_mp3=(format != 'wav')
            )
            logger.info(f"Auto-selected quality: {used_quality}, format: {used_format} for response")
            content_type = "audio/mpeg" if used_format == 'mp3' else "audio/wav"
            file_ext = "mp3" if used_format == 'mp3' else "wav"
        else:
            # Use the specified quality and format
            audio_buffer, content_type = create_audio_response(audio_data, sample_rate, quality=quality, format=format)
            file_ext = "mp3" if format == 'mp3' else "wav"
            
        return StreamingResponse(
            audio_buffer, 
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=tts_{selected_voice}_{quality}.{file_ext}"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch")
async def batch_text_to_speech(request: Request):
    """Generate audio for multiple text inputs using batch processing"""
    try:
        # Parse the request body manually to avoid pydantic validation issues
        request_data = await request.json()
        logger.info(f"Received batch request data: {request_data}")
        
        # Handle both single text (auto-chunked) and multiple texts
        single_text = request_data.get("text")  # Single long text for auto-chunking
        texts = request_data.get("texts", [])  # Array of texts (current behavior)
        
        # Auto-chunk single text using KPipeline if provided
        if single_text and not texts:
            logger.info(f"Auto-chunking single text of length {len(single_text)} characters")
            try:
                # Import KPipeline here to use its chunking logic
                from kokoro import KPipeline
                
                # Determine language based on voice
                lang_code = 'a'  # Default to American English
                if 'uk' in voice.lower() or 'british' in voice.lower():
                    lang_code = 'b'
                
                # Create pipeline for chunking (without model to just get chunks)
                pipeline = KPipeline(lang_code=lang_code, model=False)
                
                # Use pipeline's chunking to split the text
                texts = []
                chunk_results = list(pipeline(single_text, voice=None, speed=1.0))
                texts = [result.graphemes for result in chunk_results if result.graphemes.strip()]
                
                logger.info(f"Auto-chunked into {len(texts)} segments")
                for i, chunk in enumerate(texts[:3]):  # Log first 3 chunks
                    logger.info(f"Chunk {i}: '{chunk[:50]}{'...' if len(chunk) > 50 else ''}'")
                    
            except Exception as e:
                logger.error(f"Error auto-chunking text: {str(e)}")
                # Fallback: simple sentence splitting
                import re
                texts = [s.strip() for s in re.split(r'[.!?]+', single_text) if s.strip()]
                logger.info(f"Fallback chunking into {len(texts)} sentences")
        
        if not texts:
            raise HTTPException(status_code=400, detail="Either 'text' (single) or 'texts' (array) must be provided")
        
        # Extract other parameters with defaults
        voice = request_data.get("voice", "af_sky")
        speed = request_data.get("speed", 1.0)
        breathiness = request_data.get("breathiness", 0.0)
        tenseness = request_data.get("tenseness", 0.0)
        jitter = request_data.get("jitter", 0.0)
        sultry = request_data.get("sultry", 0.0)
        fiction_list = request_data.get("fiction", [False] * len(texts))
        quality = request_data.get("quality", "medium")
        format_type = request_data.get("format", "mp3")
        
        # Convert string enum values if needed
        if isinstance(quality, dict) and "value" in quality:
            quality = quality["value"]
        if isinstance(format_type, dict) and "value" in format_type:
            format_type = format_type["value"]
            
        logger.info(f"Batch TTS request received with {len(texts)} texts")
        logger.info(f"Using format: {format_type}, quality: {quality}")
        
        # Ensure fiction_list is the same length as texts
        if len(fiction_list) < len(texts):
            fiction_list.extend([False] * (len(texts) - len(fiction_list)))
        
        # Group texts by fiction/non-fiction for batch processing
        fiction_texts = []
        fiction_indices = []
        nonfiction_texts = []
        nonfiction_indices = []
        
        for i, (text, is_fiction) in enumerate(zip(texts, fiction_list)):
            if is_fiction:
                fiction_texts.append(text)
                fiction_indices.append(i)
            else:
                nonfiction_texts.append(text)
                nonfiction_indices.append(i)
        
        # Initialize results list with None placeholders
        results = [None] * len(texts)
        use_gpu = get_settings().cuda_available
        
        # Process fiction texts in batch if any
        if fiction_texts:
            logger.info(f"Processing {len(fiction_texts)} fiction texts in batch mode")
            
            try:
                # Use batch processing for fiction texts
                batch_results = generate_audio_batch(
                    fiction_texts, voice, speed, use_gpu,
                    breathiness, tenseness, jitter, sultry
                )
                
                logger.info(f"Fiction batch results received: {len(batch_results)} items")
                
                # Place results in the correct positions
                for idx, res in zip(fiction_indices, batch_results):
                    try:
                        # generate_audio_batch returns ((sample_rate, audio_data), phonemes)
                        if isinstance(res, tuple) and len(res) >= 1:
                            audio_tuple = res[0]  # This should be (sample_rate, audio_data)
                            
                            if isinstance(audio_tuple, tuple) and len(audio_tuple) == 2:
                                sample_rate, audio_data = audio_tuple
                                
                                # Ensure audio_data is a numpy array
                                if not isinstance(audio_data, np.ndarray):
                                    logger.warning(f"Fiction audio data is not a numpy array: {type(audio_data)}, attempting conversion")
                                    try:
                                        # Try to convert to numpy array if it's not already
                                        if isinstance(audio_data, list):
                                            audio_data = np.array(audio_data, dtype=np.float32)
                                        elif isinstance(audio_data, str):
                                            # If it's a string, we can't convert it to audio data
                                            audio_data = np.zeros(1000, dtype=np.float32)
                                    except Exception as conv_err:
                                        logger.error(f"Failed to convert fiction audio data: {str(conv_err)}")
                                        audio_data = np.zeros(1000, dtype=np.float32)
                                
                                results[idx] = (sample_rate, audio_data)
                            else:
                                logger.warning(f"Unexpected fiction audio tuple format: {type(audio_tuple)}")
                                results[idx] = (24000, np.zeros(1000, dtype=np.float32))
                        else:
                            logger.warning(f"Unexpected fiction batch result format: {type(res)}")
                            results[idx] = (24000, np.zeros(1000, dtype=np.float32))
                    except Exception as e:
                        logger.error(f"Error processing fiction batch result for index {idx}: {str(e)}")
                        results[idx] = (24000, np.zeros(1000, dtype=np.float32))  # Fallback to empty audio
            except Exception as e:
                logger.error(f"Error processing fiction batch: {str(e)}")
                # Fall back to processing individually
                for i, text in enumerate(fiction_texts):
                    try:
                        result = generate_audio(
                            text, voice, speed, use_gpu,
                            breathiness, tenseness, jitter, sultry
                        )
                        results[fiction_indices[i]] = result
                    except Exception as inner_e:
                        logger.error(f"Error processing individual fiction text: {str(inner_e)}")
                        # Skip this text
        
        # Process non-fiction texts in batch if any
        if nonfiction_texts:
            logger.info(f"Processing {len(nonfiction_texts)} non-fiction texts in batch mode")
            
            try:
                # Use batch processing for non-fiction texts
                batch_results = generate_audio_batch(
                    nonfiction_texts, voice, speed, use_gpu,
                    breathiness, tenseness, jitter, sultry
                )
                
                logger.info(f"Batch results received: {len(batch_results)} items")
                logger.info(f"Batch results types: {[type(res) for res in batch_results]}")
                
                # Place results in the correct positions
                for idx, res in zip(nonfiction_indices, batch_results):
                    try:
                        # generate_audio_batch returns ((sample_rate, audio_data), phonemes)
                        # We need to extract just (sample_rate, audio_data)
                        logger.info(f"Processing batch result for index {idx}, type: {type(res)}")
                        
                        if isinstance(res, tuple) and len(res) >= 1:
                            # First element should be (sample_rate, audio_data)
                            audio_tuple = res[0]  
                            logger.info(f"Audio tuple type: {type(audio_tuple)}, length: {len(audio_tuple) if isinstance(audio_tuple, tuple) else 'not a tuple'}")
                            
                            # Dump the actual content for debugging
                            if isinstance(audio_tuple, tuple) and len(audio_tuple) == 2:
                                logger.info(f"Sample rate type: {type(audio_tuple[0])}, Audio data type: {type(audio_tuple[1])}")
                            
                            if isinstance(audio_tuple, tuple) and len(audio_tuple) == 2:
                                sample_rate, audio_data = audio_tuple
                                logger.info(f"Successfully extracted sample_rate: {sample_rate}, audio_data shape: {audio_data.shape if hasattr(audio_data, 'shape') else 'unknown'}")
                                
                                # Ensure audio_data is a numpy array
                                if not isinstance(audio_data, np.ndarray):
                                    logger.warning(f"Audio data is not a numpy array: {type(audio_data)}, attempting conversion")
                                    try:
                                        # Try to convert to numpy array if it's not already
                                        if isinstance(audio_data, list):
                                            audio_data = np.array(audio_data, dtype=np.float32)
                                        elif isinstance(audio_data, str):
                                            # If it's a string, we can't convert it to audio data
                                            # Use a small silent audio segment instead
                                            logger.warning(f"Audio data is a string, using silent audio instead")
                                            audio_data = np.zeros(1000, dtype=np.float32)
                                    except Exception as conv_err:
                                        logger.error(f"Failed to convert audio data: {str(conv_err)}")
                                        audio_data = np.zeros(1000, dtype=np.float32)
                                
                                results[idx] = (sample_rate, audio_data)
                            else:
                                logger.warning(f"Unexpected audio tuple format: {type(audio_tuple)}, using default sample rate")
                                # Create a silent audio segment
                                results[idx] = (24000, np.zeros(1000, dtype=np.float32))
                        else:
                            logger.warning(f"Unexpected batch result format: {type(res)}, using default sample rate")
                            results[idx] = (24000, res)
                    except Exception as e:
                        logger.error(f"Error processing batch result for index {idx}: {str(e)}")
                        results[idx] = (24000, np.zeros(1000, dtype=np.float32))  # Fallback to empty audio
            except Exception as e:
                logger.error(f"Error processing non-fiction batch: {str(e)}")
                # Fall back to processing individually
                for i, text in enumerate(nonfiction_texts):
                    try:
                        result = generate_audio(
                            text, voice, speed, use_gpu,
                            breathiness, tenseness, jitter, sultry
                        )
                        results[nonfiction_indices[i]] = result
                    except Exception as inner_e:
                        logger.error(f"Error processing individual non-fiction text: {str(inner_e)}")
                        # Skip this text
        
        # Convert audio data to base64 encoded strings
        audio_base64_list = []
        for i, result in enumerate(results):
            try:
                if result is None:
                    logger.warning(f"Result {i} is None, adding empty string")
                    audio_base64_list.append("")
                    continue
                
                # Ensure we have a proper tuple with sample_rate and audio_data
                if not isinstance(result, tuple) or len(result) != 2:
                    logger.warning(f"Result {i} is not a proper tuple: {type(result)}, content: {result}")
                    audio_base64_list.append("")
                    continue
                
                # Fast path: extract sample_rate and audio_data
                try:
                    sample_rate, audio_data = result
                except ValueError:
                    audio_base64_list.append("")
                    continue
                
                # Quick sample_rate validation and extraction
                if isinstance(sample_rate, tuple) and len(sample_rate) >= 2:
                    actual_sample_rate = sample_rate[0] if isinstance(sample_rate[0], int) else 24000
                    audio_data = sample_rate[1]
                elif isinstance(sample_rate, int) and sample_rate > 0:
                    actual_sample_rate = sample_rate
                else:
                    actual_sample_rate = 24000
                
                # Quick audio_data validation
                if not isinstance(audio_data, np.ndarray):
                    audio_base64_list.append("")
                    continue
                # Convert audio data to specified format
                if format_type == "mp3":
                    encoded_audio = audio_to_base64(audio_data, actual_sample_rate, quality=quality, format='mp3')
                else:  # Default to WAV
                    encoded_audio = audio_to_base64(audio_data, actual_sample_rate, quality=quality, format='wav')
                
                audio_base64_list.append(encoded_audio)
            except Exception:
                audio_base64_list.append("")
        
        # If no texts were provided or all processing failed
        if not audio_base64_list:
            logger.warning("No audio was generated")
            return JSONResponse(content={"audios": []})
        
        # Return a simple dictionary with a list of strings using explicit JSONResponse
        return JSONResponse(content={"audios": audio_base64_list})
        
    except Exception as e:
        logger.error(f"Batch TTS error: {str(e)}")
        # Include traceback for better debugging
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/base64", response_model=TTSResponse)
async def text_to_speech_base64(request: TTSRequest):
    """Convert text to speech and return audio as base64 encoded string"""
    try:
        # Validate that models are loaded and accessible
        from entry.core.models import get_models, get_pipelines, get_voices
        models = get_models()
        pipelines = get_pipelines()
        voices = get_voices()
        logger.info(f"Base64 TTS endpoint has access to: models={len(models)}, pipelines={len(pipelines)}, voices={len(voices)}")
        
        if not models or not pipelines or not voices:
            logger.error(f"Unable to process Base64 TTS request - missing components: models={bool(models)}, pipelines={bool(pipelines)}, voices={bool(voices)}")
            raise HTTPException(status_code=503, detail="TTS system not fully initialized")
        
        selected_voice, emotion_preset = select_voice_and_preset(
            request.voice, fiction=request.fiction
        )
        
        # Apply preset values or use request values
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
        
        # Get GPU availability from settings
        settings = get_settings()
        use_gpu = settings.cuda_available
        
        (sample_rate, audio_data), phonemes = generate_audio(
            preprocessed_text, 
            selected_voice,
            speed, 
            use_gpu,  # Use GPU if available based on settings
            breathiness,
            tenseness,
            jitter,
            sultry
        )
        
        if audio_data is None:
            raise HTTPException(status_code=500, detail="Failed to generate audio")
        
        # Get quality and format parameters
        quality = request.quality.value
        format = request.format.value if hasattr(request, 'format') else 'auto'
        logger.info(f"Base64 TTS request with quality {quality}, format {format}")
        
        if quality == 'auto' or format == 'auto':
            # For base64 responses, optimize size based on quality and format
            audio_buffer, used_quality, used_format = optimize_response_size(
                audio_data, sample_rate, max_size_kb=30000, use_mp3=(format != 'wav')
            )
            # Convert the optimized buffer to base64
            audio_bytes = audio_buffer.getvalue()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            logger.info(f"Auto-selected quality: {used_quality}, format: {used_format} for base64 response")
            # Remember the format for the response
            format = used_format
        else:
            # Use the specified quality and format
            audio_base64 = audio_to_base64(audio_data, sample_rate, quality=quality, format=format)
        
        # Get the actual sample rate after potential downsampling
        actual_sample_rate = sample_rate
        if quality == 'medium' or (quality == 'auto' and used_quality == 'medium'):
            actual_sample_rate = 16000
        elif quality == 'low' or (quality == 'auto' and used_quality == 'low'):
            actual_sample_rate = 8000
        
        return TTSResponse(
            sample_rate=actual_sample_rate,
            audio_base64=audio_base64,
            phonemes=phonemes
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tokenize", response_model=TokenizeResponse)
async def tokenize(request: TokenizeRequest):
    """Tokenize text to phonemes without generating audio"""
    try:
        phonemes = tokenize_text(request.text, request.voice)
        return TokenizeResponse(phonemes=phonemes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/preprocess", response_model=PreprocessResponse)
def preprocess_text_endpoint(request: PreprocessRequest):
    """Preprocess text for TTS"""
    processed = preprocess_text(request.text)
    return PreprocessResponse(original=request.text, processed=processed)