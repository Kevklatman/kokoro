"""
Job management API routes
"""
from fastapi import APIRouter, HTTPException

from entry.models import TTSJobRequest, JobResponse, JobStatus, QueueStatus
from entry.services.queue import submit_job, get_job_status, get_queue_status, cancel_job

router = APIRouter()


@router.post("/submit", response_model=JobResponse)
async def submit_tts_job(job_request: TTSJobRequest):
    """Submit a TTS job to the processing queue"""
    try:
        print(f"📥 Received job submission request:")
        print(f"  Title: {job_request.title}")
        print(f"  Author: {job_request.author}")
        print(f"  Genre: {job_request.genre}")
        print(f"  Fiction: {job_request.fiction}")
        print(f"  Text length: {len(job_request.text)}")
        
        job_id = submit_job(job_request)
        
        return JobResponse(
            job_id=job_id,
            status="queued",
            message=f"Job queued successfully"
        )
        
    except Exception as e:
        print(f"❌ Submit job error: {str(e)}")
        import traceback
        print(f"❌ Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status_endpoint(job_id: str):
    """Get the status of a specific job"""
    try:
        return get_job_status(job_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/queue", response_model=QueueStatus)
async def get_queue_status_endpoint():
    """Get the current queue status"""
    try:
        queue_data = get_queue_status()
        return QueueStatus(**queue_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/job/{job_id}")
async def cancel_job_endpoint(job_id: str):
    """Cancel a queued job"""
    try:
        message = cancel_job(job_id)
        return {"message": message}
    except ValueError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        else:
            raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))