"""
Job management API routes
"""
from fastapi import APIRouter, HTTPException
from loguru import logger
import traceback

from entry.models import TTSJobRequest, JobResponse, JobStatus, QueueStatus
from entry.services.queue import submit_job, get_job_status, get_queue_status, cancel_job
from entry.utils.error_handling import (
    safe_execute, handle_http_error, create_not_found_error, create_validation_error,
    log_operation_start, log_operation_success, log_operation_failure
)


def handle_job_error(e: Exception, context: str = "Job operation") -> None:
    """Consistent error handling for job operations"""
    logger.error(f"❌ {context} error: {str(e)}")
    logger.error(f"❌ Traceback: {traceback.format_exc()}")
    raise HTTPException(status_code=500, detail=str(e))


router = APIRouter()


@router.post("/submit", response_model=JobResponse)
async def submit_tts_job(job_request: TTSJobRequest):
    """Submit a TTS job for background processing"""
    log_operation_start("job submission", title=job_request.title, author=job_request.author)
    
    def submit_job_safe():
        # Validate job request
        if not job_request.text or not job_request.text.strip():
            raise ValueError("Job text cannot be empty")
        
        if len(job_request.text) > 50000:  # Reasonable limit
            raise ValueError("Job text too long (max 50000 characters)")
        
        # Submit job to queue
        job_id = submit_job(job_request)
        
        return JobResponse(
            job_id=job_id,
            status="queued",
            message=f"Job '{job_request.title}' submitted successfully"
        )
    
    try:
        result = safe_execute(submit_job_safe, context="job submission")
        log_operation_success("job submission", job_id=result.job_id)
        return result
    except ValueError as e:
        raise create_validation_error(str(e))
    except Exception as e:
        log_operation_failure("job submission", e, title=job_request.title)
        raise handle_http_error(e, context="job submission")


@router.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status_endpoint(job_id: str):
    """Get the status of a specific job"""
    log_operation_start("job status check", job_id=job_id)
    
    def get_status_safe():
        from entry.services.queue import get_job_status as get_job_status_from_queue
        job_status = get_job_status_from_queue(job_id)
        if job_status is None:
            raise create_not_found_error(job_id, "Job")
        return job_status
    
    try:
        result = safe_execute(get_status_safe, context="job status check")
        log_operation_success("job status check", job_id=job_id, status=result.status)
        return result
    except HTTPException:
        raise
    except Exception as e:
        log_operation_failure("job status check", e, job_id=job_id)
        raise handle_http_error(e, context="job status check")


@router.get("/queue", response_model=QueueStatus)
async def get_queue_status():
    """Get the current queue status"""
    log_operation_start("queue status check")
    
    def get_queue_status_safe():
        from entry.services.queue import get_queue_status as get_queue_status_from_service
        return get_queue_status_from_service()
    
    try:
        result = safe_execute(get_queue_status_safe, context="queue status check")
        log_operation_success("queue status check", total_jobs=result.total_jobs)
        return result
    except Exception as e:
        log_operation_failure("queue status check", e)
        raise handle_http_error(e, context="queue status check")


@router.delete("/job/{job_id}")
async def cancel_job_endpoint(job_id: str):
    """Cancel a queued job"""
    try:
        result = cancel_job(job_id)
        return {"message": result}
    except ValueError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        else:
            raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        handle_job_error(e, "Cancel job")