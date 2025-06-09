"""
Job queue management for background TTS processing
"""
import uuid
import queue
import threading
import base64
import wave
import io
import numpy as np
from datetime import datetime
from typing import Dict

from entry.models import TTSJob, JobStatus, JobStatusEnum
from entry.core.tts import select_voice_and_preset, preprocess_text, generate_audio
from entry.utils.audio import audio_to_base64, audio_to_bytes
from loguru import logger


# Global job storage and queue
jobs_storage: Dict[str, JobStatus] = {}
job_queue = queue.Queue()
processing_jobs: Dict[str, JobStatus] = {}


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
                    selected_voice, emotion_preset = select_voice_and_preset(
                        job.voice, None, job.fiction
                    )
                    
                    # Apply preset values if available
                    if emotion_preset:
                        speed = emotion_preset.get("speed", job.speed) if job.speed == 1.0 else job.speed
                        breathiness = emotion_preset.get("breathiness", job.breathiness) if job.breathiness == 0.0 else job.breathiness
                        tenseness = emotion_preset.get("tenseness", job.tenseness) if job.tenseness == 0.0 else job.tenseness
                        jitter = emotion_preset.get("jitter", job.jitter) if job.jitter == 0.0 else job.jitter
                        sultry = emotion_preset.get("sultry", job.sultry) if job.sultry == 0.0 else job.sultry
                    else:
                        speed = job.speed
                        breathiness = job.breathiness
                        tenseness = job.tenseness
                        jitter = job.jitter
                        sultry = job.sultry
                    
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
                    
                    # Get quality and format parameters with defaults for jobs
                    # Jobs can be long, so we use medium quality and MP3 by default
                    quality = getattr(job, 'quality', 'medium') if hasattr(job, 'quality') else 'medium'
                    format = getattr(job, 'format', 'mp3') if hasattr(job, 'format') else 'mp3'
                    
                    if quality == 'auto':
                        # For background jobs, use medium quality by default
                        quality = 'medium'
                    
                    # Default to MP3 for jobs to save space
                    if format == 'auto':
                        format = 'mp3'
                        
                    logger.info(f"Processing job {job_id} with quality {quality}, format {format}")
                    
                    # Use our optimized audio_to_base64 utility
                    audio_base64 = audio_to_base64(audio_data, sample_rate, quality=quality, format=format)
                    
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


def start_queue_processor():
    """Start the background queue processor thread"""
    queue_thread = threading.Thread(target=process_queue, daemon=True)
    queue_thread.start()
    return queue_thread


def submit_job(job_request) -> str:
    """Submit a new job to the queue"""
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
    
    return job_id


def get_job_status(job_id: str) -> JobStatus:
    """Get status of a specific job"""
    if job_id not in jobs_storage:
        raise ValueError("Job not found")
    
    job_status = jobs_storage[job_id]
    if job_status.status == JobStatusEnum.QUEUED:
        job_status.position_in_queue = max(1, job_queue.qsize())
    job_status.total_in_queue = len(jobs_storage)
    
    return job_status


def get_queue_status():
    """Get overall queue status"""
    queued_count = sum(1 for job in jobs_storage.values() if job.status == JobStatusEnum.QUEUED)
    processing_count = len(processing_jobs)
    
    print(f"📊 Queue status requested:")
    print(f"  Total jobs: {len(jobs_storage)}")
    print(f"  Processing: {processing_count}")
    print(f"  Queued: {queued_count}")
    
    return {
        "total_jobs": len(jobs_storage),
        "processing_jobs": processing_count,
        "queued_jobs": queued_count,
        "jobs": list(jobs_storage.values())
    }


def cancel_job(job_id: str) -> str:
    """Cancel a queued job"""
    if job_id not in jobs_storage:
        raise ValueError("Job not found")
    
    job_status = jobs_storage[job_id]
    if job_status.status == JobStatusEnum.PROCESSING:
        raise ValueError("Cannot cancel job that is currently processing")
    
    if job_status.status == JobStatusEnum.QUEUED:
        del jobs_storage[job_id]
        return "Job cancelled successfully"
    
    return "Job already completed or failed"


# Start the queue processor when module is imported
start_queue_processor()