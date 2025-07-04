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
            # Get job from queue with timeout
            job: TTSJob = job_queue.get(timeout=1)
            
            # Update job status to processing
            job.status = JobStatusEnum.PROCESSING
            job.started_at = datetime.now()
            
            logger.info(f"🔄 Processing job {job.id}: {job.title}")
            
            # Extract parameters
            text = job.text
            voice = job.voice
            speed = job.speed
            quality = job.quality
            format = job.format
            breathiness = job.breathiness
            tenseness = job.tenseness
            jitter = job.jitter
            sultry = job.sultry
            
            # Generate audio
            logger.info(f"Processing job {job.id} with quality {quality}, format {format}")
            
            # Use the core TTS function
            result = generate_audio(
                text=text,
                voice=voice,
                speed=speed,
                use_gpu=True,  # Use GPU for background processing
                breathiness=breathiness,
                tenseness=tenseness,
                jitter=jitter,
                sultry=sultry
            )
            
            if result is None:
                raise Exception("Audio generation returned None")
            
            sample_rate, audio_data = result
            
            # Use our optimized audio_to_base64 utility
            audio_base64 = audio_to_base64(audio_data, sample_rate, quality=quality, format=format)
            
            # Update job with results
            job.status = JobStatusEnum.COMPLETED
            job.completed_at = datetime.now()
            job.result = audio_base64
            
            logger.info(f"✅ Completed job {job.id}: {job.title}")
            
            # Mark task as done
            job_queue.task_done()
            
        except queue.Empty:
            # No jobs in queue, continue
            continue
        except Exception as e:
            logger.error(f"❌ Failed job {job.id}: {str(e)}")
            # Update job status to failed
            if 'job' in locals():
                job.status = JobStatusEnum.FAILED
                job.completed_at = datetime.now()
                job.error = str(e)
            continue
        except Exception as e:
            logger.error(f"❌ Queue processor error: {str(e)}")
            continue


def start_queue_processor():
    """Start the background queue processor thread"""
    queue_thread = threading.Thread(target=process_queue, daemon=True)
    queue_thread.start()
    return queue_thread


def submit_job(job_request) -> str:
    """Submit a new job to the queue"""
    job_id = str(uuid.uuid4())
    
    # Create job object
    job_data = TTSJob(
        id=job_id,
        title=job_request.title,
        author=job_request.author,
        genre=job_request.genre,
        text=job_request.text,
        voice=job_request.voice,
        speed=job_request.speed,
        quality=job_request.quality,
        format=job_request.format,
        breathiness=job_request.breathiness,
        tenseness=job_request.tenseness,
        jitter=job_request.jitter,
        sultry=job_request.sultry,
        fiction=job_request.fiction,
        use_gpu=job_request.use_gpu,
        status=JobStatusEnum.QUEUED,
        created_at=datetime.now(),
        position_in_queue=job_queue.qsize() + 1,
        total_in_queue=job_queue.qsize() + 1
    )
    
    # Store job
    jobs_storage[job_id] = job_data
    
    # Add to queue
    job_queue.put(job_data)
    
    logger.info(f"📋 Queued job {job_id}: {job_request.title}")
    logger.info(f"📊 Queue size now: {job_queue.qsize()}")
    
    return job_id


def get_job_status(job_id: str) -> JobStatus:
    """Get status of a specific job"""
    if job_id not in jobs_storage:
        raise ValueError(f"Job {job_id} not found")
    
    job_status = jobs_storage[job_id]
    
    # Update queue position if job is still queued
    if job_status.status == JobStatusEnum.QUEUED:
        job_status.position_in_queue = max(1, job_queue.qsize())
        job_status.total_in_queue = len(jobs_storage)
    
    return job_status


def get_queue_status():
    """Get overall queue status"""
    queued_count = sum(1 for job in jobs_storage.values() if job.status == JobStatusEnum.QUEUED)
    processing_count = sum(1 for job in jobs_storage.values() if job.status == JobStatusEnum.PROCESSING)
    
    logger.info(f"📊 Queue status requested:")
    logger.info(f"  Total jobs: {len(jobs_storage)}")
    logger.info(f"  Processing: {processing_count}")
    logger.info(f"  Queued: {queued_count}")
    
    return {
        "total_jobs": len(jobs_storage),
        "processing_jobs": processing_count,
        "queued_jobs": queued_count,
        "queue_size": job_queue.qsize()
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


# Only start the queue processor if this module is being used directly
# or if explicitly requested, not on every import
if __name__ == "__main__":
    start_queue_processor()