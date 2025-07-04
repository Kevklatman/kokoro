"""
Job management API routes
"""
from fastapi import APIRouter, HTTPException
from loguru import logger
import traceback

from entry.models import TTSJobRequest, JobResponse, JobStatus, QueueStatus
from entry.services.queue import submit_job, get_job_status, get_queue_status, cancel_job


def handle_job_error(e: Exception, context: str = "Job operation") -> None:
    """Consistent error handling for job operations"""
    logger.error(f"❌ {context} error: {str(e)}")
    logger.error(f"❌ Traceback: {traceback.format_exc()}")
    raise HTTPException(status_code=500, detail=str(e))


router = APIRouter()


@router.post("/submit", response_model=JobResponse)
async def submit_tts_job(job_request: TTSJobRequest):
    """Submit a TTS job to the processing queue"""
    logger.info(f"📥 Received job submission request:")
    logger.info(f"  Title: {job_request.title}")
    logger.info(f"  Author: {job_request.author}")
    logger.info(f"  Genre: {job_request.genre}")
    logger.info(f"  Fiction: {job_request.fiction}")
    logger.info(f"  Text length: {len(job_request.text)}")
    
    try:
        job_id = submit_job(job_request)
        return JobResponse(
            job_id=job_id,
            status="queued",
            message=f"Job queued successfully"
        )
    except Exception as e:
        handle_job_error(e, "Submit job")


@router.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status_endpoint(job_id: str):
    """Get the status of a specific job"""
    try:
        return get_job_status(job_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        handle_job_error(e, "Get job status")


@router.get("/queue", response_model=QueueStatus)
async def get_queue_status_endpoint():
    """Get the current queue status"""
    try:
        queue_data = get_queue_status()
        return QueueStatus(**queue_data)
    except Exception as e:
        handle_job_error(e, "Get queue status")


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