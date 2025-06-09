# Cloud Run Optimization Guide for Kokoro TTS

## Current Performance Issues
- Local Docker: ~45 seconds for inference
- Cloud Run: ~5 minutes for inference (6-7x slower)

## Optimization Recommendations

### 1. Increase Cloud Run Resource Allocation
```bash
gcloud run deploy kokoro-tts \
  --image gcr.io/[YOUR-PROJECT]/kokoro-tts \
  --cpu 2 \
  --memory 4Gi \
  --min-instances 1
```

### 2. Add Model Caching
- The `--min-instances 1` flag keeps at least one instance warm to avoid cold starts
- This ensures the model stays loaded in memory

### 3. Optimize Container Configuration
- Set concurrency to match your CPU allocation:
```bash
gcloud run deploy kokoro-tts \
  --concurrency 10
```

### 4. Environment Variable Tuning
Add these to your Cloud Run service:
```
WORKERS=2  # Match your CPU allocation
WORKER_CONNECTIONS=1000
TIMEOUT=300  # Increase timeout for long-running inferences
```

### 5. Optimize Uvicorn Configuration in Dockerfile
Update your production Dockerfile CMD:
```dockerfile
CMD uvicorn entry.main:app --host 0.0.0.0 --port $PORT --workers ${WORKERS:-1} --timeout-keep-alive ${TIMEOUT:-75}
```

### 6. Consider Memory-CPU Tradeoffs
- Our G2P caching helps reduce repeated computations
- JIT compilation improves inference speed
- Consider increasing cache sizes in production for frequently used phrases

### 7. Monitoring and Profiling
Add these metrics to your FastAPI app:
```python
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    logger.info(f"Request processed in {process_time:.2f} seconds")
    return response
```

### 8. Batch Processing for Multiple Requests
If applicable, implement batch processing to amortize model loading costs across multiple requests.

### 9. Cloud Run Instance Types
Consider using Cloud Run CPU-optimized instances:
```bash
gcloud run deploy kokoro-tts \
  --cpu 2 \
  --memory 4Gi \
  --cpu-boost
```
