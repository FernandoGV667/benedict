#!/usr/bin/env python3
"""
Benedict Web - FastAPI Async YouTube Transcript Service
Optimized for shared hosting
"""

import os
import re
import json
import uuid
import hashlib
import asyncio
import aiofiles
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
import threading
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import uvicorn

# Import your Benedict functions (modularized)
# You'll need to save your core functions in a separate file
from benedict_core import (
    try_fetch_youtube_transcript,
    try_fetch_ytdlp_captions,
    download_audio,
    transcribe_with_whisper,
    select_model,
    _extract_video_id
)

# ============= CONFIGURATION =============
class Settings:
    # Paths
    BASE_DIR = Path(__file__).parent
    JOBS_DIR = BASE_DIR / "jobs"
    TRANSCRIPTS_DIR = BASE_DIR / "transcripts"
    TEMP_DIR = BASE_DIR / "temp_audio"
    
    # Job settings
    MAX_CONCURRENT_JOBS = 3  # Limit for shared hosting
    JOB_TIMEOUT_HOURS = 2
    CLEANUP_INTERVAL_HOURS = 1
    FILE_RETENTION_HOURS = 24
    
    # Create directories
    @classmethod
    def init_dirs(cls):
        for dir_path in [cls.JOBS_DIR, cls.TRANSCRIPTS_DIR, cls.TEMP_DIR]:
            dir_path.mkdir(exist_ok=True)

Settings.init_dirs()

# ============= JOB MANAGER (File-based for shared hosting) =============
class JobManager:
    """Simple file-based job tracking for shared hosting"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=Settings.MAX_CONCURRENT_JOBS)
        self.active_jobs = {}
        self.lock = threading.Lock()
    
    def create_job(self, job_id: str, data: dict) -> None:
        """Create a new job record"""
        job_file = Settings.JOBS_DIR / f"{job_id}.json"
        job_data = {
            "id": job_id,
            "status": "pending",
            "progress": 0,
            "stage": "initializing",
            "created": datetime.utcnow().isoformat(),
            "data": data,
            "result": None,
            "error": None
        }
        with open(job_file, 'w') as f:
            json.dump(job_data, f)
    
    def update_job(self, job_id: str, **kwargs) -> None:
        """Update job status"""
        job_file = Settings.JOBS_DIR / f"{job_id}.json"
        if job_file.exists():
            with open(job_file, 'r') as f:
                job_data = json.load(f)
            job_data.update(kwargs)
            job_data["updated"] = datetime.utcnow().isoformat()
            with open(job_file, 'w') as f:
                json.dump(job_data, f)
    
    def get_job(self, job_id: str) -> Optional[dict]:
        """Get job status"""
        job_file = Settings.JOBS_DIR / f"{job_id}.json"
        if job_file.exists():
            with open(job_file, 'r') as f:
                return json.load(f)
        return None
    
    def cleanup_old_jobs(self) -> None:
        """Remove old job files"""
        cutoff = datetime.utcnow() - timedelta(hours=Settings.FILE_RETENTION_HOURS)
        for job_file in Settings.JOBS_DIR.glob("*.json"):
            try:
                with open(job_file, 'r') as f:
                    job_data = json.load(f)
                created = datetime.fromisoformat(job_data.get("created", datetime.utcnow().isoformat()))
                if created < cutoff:
                    job_file.unlink()
            except:
                pass

# Initialize job manager
job_manager = JobManager()

# ============= ASYNC PROCESSING FUNCTIONS =============
async def process_transcript_async(job_id: str, url: str, language: str, source_priority: str):
    """Async wrapper for transcript processing"""
    try:
        # Run the CPU-intensive work in a thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            job_manager.executor,
            process_transcript_sync,
            job_id, url, language, source_priority
        )
        return result
    except Exception as e:
        job_manager.update_job(job_id, status="failed", error=str(e))
        raise

def process_transcript_sync(job_id: str, url: str, language: str, source_priority: str) -> dict:
    """Synchronous transcript processing (runs in thread pool)"""
    try:
        video_id = _extract_video_id(url)
        
        # Update progress
        job_manager.update_job(job_id, status="processing", progress=5, stage="checking_captions")
        
        # Determine language preferences
        if language == "auto":
            preferred_langs = ["en", "en-US", "en-GB", "es", "es-419", "es-ES"]
        else:
            preferred_langs = [language, f"{language}-US", f"{language}-GB"]
        
        # Try YouTube captions first
        if source_priority in ["auto", "captions"]:
            job_manager.update_job(job_id, progress=10, stage="fetching_youtube_captions")
            
            txt, lang, source = try_fetch_youtube_transcript(url, preferred_langs)
            if txt:
                return save_transcript(job_id, txt, lang, f'YouTube {source} captions', video_id)
            
            job_manager.update_job(job_id, progress=20, stage="fetching_ytdlp_captions")
            
            txt2, lang2, source2 = try_fetch_ytdlp_captions(url, preferred_langs)
            if txt2:
                return save_transcript(job_id, txt2, lang2, f'YouTube {source2} captions', video_id)
        
        # Fall back to Whisper
        if source_priority in ["whisper", "auto"]:
            job_manager.update_job(job_id, progress=30, stage="downloading_audio")
            
            with tempfile.TemporaryDirectory() as tmpdir:
                audio_path, duration = download_audio(url, tmpdir)
                model_size = select_model(duration)
                
                job_manager.update_job(
                    job_id, 
                    progress=40, 
                    stage="transcribing_with_whisper",
                    model=model_size,
                    duration_minutes=round(duration / 60) if duration else None
                )
                
                # Transcribe with progress updates
                def progress_callback(p):
                    job_manager.update_job(job_id, progress=40 + int(p * 50))
                
                transcript, lang = transcribe_with_whisper(audio_path, model_size)
                
                return save_transcript(
                    job_id, transcript, lang, 
                    f'Whisper AI ({model_size} model)', 
                    video_id
                )
        
        raise Exception("No transcript could be extracted")
        
    except Exception as e:
        job_manager.update_job(job_id, status="failed", error=str(e))
        raise

def save_transcript(job_id: str, transcript: str, language: str, source: str, video_id: str) -> dict:
    """Save transcript and return result"""
    file_id = f"{video_id}_{hashlib.md5(f"{language}_{source}".encode()).hexdigest()[:8]}"
    file_path = Settings.TRANSCRIPTS_DIR / f"{file_id}.txt"
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(transcript)
    
    result = {
        "file_id": file_id,
        "language": language,
        "source": source,
        "transcript_preview": transcript[:500],
        "word_count": len(transcript.split()),
        "timestamp": datetime.utcnow().isoformat()
    }
    
    job_manager.update_job(job_id, status="completed", progress=100, result=result)
    return result

# ============= BACKGROUND CLEANUP =============
async def periodic_cleanup():
    """Periodic cleanup of old files"""
    while True:
        await asyncio.sleep(Settings.CLEANUP_INTERVAL_HOURS * 3600)
        try:
            job_manager.cleanup_old_jobs()
            # Clean old transcript files
            cutoff = datetime.utcnow() - timedelta(hours=Settings.FILE_RETENTION_HOURS)
            for file_path in Settings.TRANSCRIPTS_DIR.glob("*.txt"):
                if datetime.fromtimestamp(file_path.stat().st_mtime) < cutoff:
                    file_path.unlink()
        except Exception as e:
            print(f"Cleanup error: {e}")

# ============= FASTAPI APP =============
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app lifecycle"""
    # Startup
    cleanup_task = asyncio.create_task(periodic_cleanup())
    yield
    # Shutdown
    cleanup_task.cancel()

app = FastAPI(
    title="Benedict Transcript Service",
    description="Extract YouTube transcripts using captions or Whisper AI",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= PYDANTIC MODELS =============
class TranscriptRequest(BaseModel):
    url: HttpUrl
    language: str = "auto"
    source: str = "auto"

class JobResponse(BaseModel):
    job_id: str
    status_url: str

# ============= HTML INTERFACE =============
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Benedict - YouTube Transcript Extractor</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 40px;
        }
        
        h1 {
            color: #333;
            margin-bottom: 30px;
            text-align: center;
            font-size: 2.5em;
        }
        
        .input-group {
            margin-bottom: 20px;
        }
        
        input[type="url"] {
            width: 100%;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 16px;
        }
        
        .options {
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
        }
        
        select {
            flex: 1;
            padding: 10px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
        }
        
        button {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
        }
        
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        
        .status-box {
            margin-top: 20px;
            padding: 20px;
            background: #f5f5f5;
            border-radius: 10px;
            display: none;
        }
        
        .status-box.active { display: block; }
        
        .progress-bar {
            width: 100%;
            height: 30px;
            background: #e0e0e0;
            border-radius: 15px;
            overflow: hidden;
            margin: 20px 0;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }
        
        .result-box {
            background: #e8f5e9;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }
        
        .error-box {
            background: #ffebee;
            color: #c62828;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }
        
        .download-btn {
            display: inline-block;
            margin-top: 10px;
            padding: 10px 20px;
            background: #4caf50;
            color: white;
            text-decoration: none;
            border-radius: 5px;
        }
        
        .stats {
            display: flex;
            gap: 20px;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #e0e0e0;
            justify-content: center;
        }
        
        .stat {
            text-align: center;
        }
        
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
        }
        
        .stat-label {
            font-size: 14px;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎬 Benedict</h1>
        <p style="text-align: center; color: #666; margin-bottom: 30px;">
            Fast YouTube transcript extraction powered by AI
        </p>
        
        <form id="transcriptForm">
            <div class="input-group">
                <input type="url" 
                       id="url" 
                       placeholder="https://www.youtube.com/watch?v=..." 
                       required>
            </div>
            
            <div class="options">
                <select id="language">
                    <option value="auto">Auto-detect Language</option>
                    <option value="en">English</option>
                    <option value="es">Spanish</option>
                    <option value="fr">French</option>
                    <option value="de">German</option>
                </select>
                
                <select id="source">
                    <option value="auto">Auto (Fastest)</option>
                    <option value="captions">YouTube Captions Only</option>
                    <option value="whisper">Whisper AI Only</option>
                </select>
            </div>
            
            <button type="submit" id="submitBtn">
                Extract Transcript
            </button>
        </form>
        
        <div id="statusBox" class="status-box">
            <h3>Processing Job</h3>
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill" style="width: 0%">
                    <span id="progressText">0%</span>
                </div>
            </div>
            <p id="statusText">Initializing...</p>
        </div>
        
        <div id="resultBox"></div>
        
        <div class="stats">
            <div class="stat">
                <div class="stat-value" id="activeJobs">0</div>
                <div class="stat-label">Active Jobs</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="totalProcessed">0</div>
                <div class="stat-label">Total Processed</div>
            </div>
        </div>
    </div>
    
    <script>
        let currentJobId = null;
        let pollInterval = null;
        let totalProcessed = parseInt(localStorage.getItem('totalProcessed') || '0');
        
        document.getElementById('totalProcessed').textContent = totalProcessed;
        
        document.getElementById('transcriptForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = {
                url: document.getElementById('url').value,
                language: document.getElementById('language').value,
                source: document.getElementById('source').value
            };
            
            const submitBtn = document.getElementById('submitBtn');
            submitBtn.disabled = true;
            submitBtn.textContent = 'Submitting...';
            
            try {
                const response = await fetch('/api/transcribe', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(formData)
                });
                
                const data = await response.json();
                
                if (data.job_id) {
                    currentJobId = data.job_id;
                    startPolling(currentJobId);
                } else if (data.cached) {
                    showResult(data.result);
                } else {
                    showError(data.error || 'Failed to start job');
                }
            } catch (error) {
                showError('Network error: ' + error.message);
            } finally {
                submitBtn.disabled = false;
                submitBtn.textContent = 'Extract Transcript';
            }
        });
        
        function startPolling(jobId) {
            document.getElementById('statusBox').classList.add('active');
            document.getElementById('resultBox').innerHTML = '';
            
            if (pollInterval) clearInterval(pollInterval);
            
            pollInterval = setInterval(() => checkJobStatus(jobId), 1000);
            checkJobStatus(jobId);
            updateActiveJobs();
        }
        
        async function checkJobStatus(jobId) {
            try {
                const response = await fetch(`/api/job/${jobId}`);
                const data = await response.json();
                
                updateProgress(data);
                
                if (data.status === 'completed') {
                    clearInterval(pollInterval);
                    showResult(data.result);
                    totalProcessed++;
                    localStorage.setItem('totalProcessed', totalProcessed);
                    document.getElementById('totalProcessed').textContent = totalProcessed;
                    updateActiveJobs();
                } else if (data.status === 'failed') {
                    clearInterval(pollInterval);
                    showError(data.error || 'Job failed');
                    updateActiveJobs();
                }
            } catch (error) {
                console.error('Polling error:', error);
            }
        }
        
        function updateProgress(data) {
            const progress = data.progress || 0;
            document.getElementById('progressFill').style.width = progress + '%';
            document.getElementById('progressText').textContent = progress + '%';
            
            let statusText = data.stage || 'Processing';
            if (data.model) statusText += ` (${data.model} model)`;
            if (data.duration_minutes) statusText += ` - ${data.duration_minutes} min audio`;
            
            document.getElementById('statusText').textContent = statusText;
        }
        
        function showResult(result) {
            document.getElementById('statusBox').classList.remove('active');
            document.getElementById('resultBox').innerHTML = `
                <div class="result-box">
                    <h3>✅ Transcript Ready!</h3>
                    <p><strong>Language:</strong> ${result.language}</p>
                    <p><strong>Source:</strong> ${result.source}</p>
                    <p><strong>Word Count:</strong> ${result.word_count}</p>
                    <p><strong>Preview:</strong></p>
                    <p style="margin: 10px 0; padding: 10px; background: white; border-radius: 5px;">
                        ${result.transcript_preview}...
                    </p>
                    <a href="/download/${result.file_id}" class="download-btn">
                        📥 Download Full Transcript
                    </a>
                </div>
            `;
        }
        
        function showError(error) {
            document.getElementById('statusBox').classList.remove('active');
            document.getElementById('resultBox').innerHTML = `
                <div class="error-box">❌ Error: ${error}</div>
            `;
        }
        
        async function updateActiveJobs() {
            try {
                const response = await fetch('/api/stats');
                const data = await response.json();
                document.getElementById('activeJobs').textContent = data.active_jobs;
            } catch (error) {
                console.error('Stats error:', error);
            }
        }
        
        // Update stats periodically
        setInterval(updateActiveJobs, 5000);
        updateActiveJobs();
    </script>
</body>
</html>
'''

# ============= API ENDPOINTS =============
@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main interface"""
    return HTML_TEMPLATE

@app.post("/api/transcribe", response_model=JobResponse)
async def start_transcription(request: TranscriptRequest, background_tasks: BackgroundTasks):
    """Start a new transcription job"""
    try:
        # Validate URL
        video_id = _extract_video_id(str(request.url))
        
        # Check cache (simple file-based)
        cache_file = Settings.TRANSCRIPTS_DIR / f"cache_{video_id}.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                cached = json.load(f)
                if datetime.fromisoformat(cached['timestamp']) > datetime.utcnow() - timedelta(hours=24):
                    return JSONResponse({"cached": True, "result": cached['result']})
        
        # Create new job
        job_id = str(uuid.uuid4())
        job_manager.create_job(job_id, {
            "url": str(request.url),
            "language": request.language,
            "source": request.source
        })
        
        # Start background processing
        background_tasks.add_task(
            process_transcript_async,
            job_id,
            str(request.url),
            request.language,
            request.source
        )
        
        return JobResponse(
            job_id=job_id,
            status_url=f"/api/job/{job_id}"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/job/{job_id}")
async def get_job_status(job_id: str):
    """Get job status"""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.get("/download/{file_id}")
async def download_transcript(file_id: str):
    """Download transcript file"""
    # Sanitize file_id
    file_id = re.sub(r'[^a-zA-Z0-9_-]', '', file_id)
    file_path = Settings.TRANSCRIPTS_DIR / f"{file_id}.txt"
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=f"transcript_{file_id}.txt",
        media_type="text/plain"
    )

@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    active_jobs = sum(1 for f in Settings.JOBS_DIR.glob("*.json") 
                     if json.load(open(f))['status'] == 'processing')
    return {"active_jobs": active_jobs}

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

# ============= RUN THE APP =============
if __name__ == "__main__":
    # For development
    uvicorn.run(app, host="0.0.0.0", port=8000)
