from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import os
import shutil
from typing import Optional
from datetime import datetime
import pandas as pd
import json

from audit_engine import (
    init_log_file, log_upload, load_upload_logs, load_checkpoints,
    extract_pdf_content, extract_docx_content, verify_checkpoints_with_ai,
    generate_report, save_pdf_images, convert_docx_to_pdf_bytes
)

# Directories
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
IMAGES_DIR = "saved_images"

for directory in [UPLOAD_DIR, OUTPUT_DIR, IMAGES_DIR]:
    os.makedirs(directory, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_log_file()
    yield
    # Shutdown logic if needed


app = FastAPI(
    title="BD Audit Automation API",
    description="API for document validation and audit automation",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (if you have frontend separately)
app.mount("/static", StaticFiles(directory="static"), name="static")



@app.get("/")
async def home():
    """Simple welcome endpoint"""
    return {"message": "Welcome to BD Audit Automation API"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...), username: str = Form(...)):
    """Upload and process a single file"""
    try:
        if not file.filename.lower().endswith(('.pdf', '.docx')):
            raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_id = f"{username}_{timestamp}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, file_id)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        with open(file_path, "rb") as f:
            file_bytes = f.read()

        file_ext = file.filename.lower().split(".")[-1]

        if file_ext == "pdf":
            full_text = extract_pdf_content(file_bytes)
        elif file_ext == "docx":
            full_text = extract_docx_content(file_bytes)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        checkpoints = load_checkpoints()
        results = await verify_checkpoints_with_ai(checkpoints, full_text, file_bytes, file_ext)

        pass_count = sum(1 for r in results if r["Status"] == "Pass")
        fail_count = sum(1 for r in results if r["Status"] == "Fail")
        summary = {
            "total_checkpoints": len(checkpoints),
            "passed": pass_count,
            "failed": fail_count
        }

        report_file = generate_report(file.filename, summary, results)
        report_path = os.path.join(OUTPUT_DIR, f"report_{file_id}.docx")

        with open(report_path, "wb") as f:
            f.write(report_file.getvalue())

        if file_ext == "pdf":
            pdf_bytes = file_bytes
        elif file_ext == "docx":
            pdf_bytes = convert_docx_to_pdf_bytes(file_bytes)
        else:
            pdf_bytes = None

        if pdf_bytes:
            save_pdf_images(pdf_bytes, IMAGES_DIR, file_id.replace(".", "_"))

        results_data = {
            "file_id": file_id,
            "filename": file.filename,
            "username": username,
            "summary": summary,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }

        results_path = os.path.join(OUTPUT_DIR, f"results_{file_id}.json")
        with open(results_path, "w") as f:
            json.dump(results_data, f, indent=2)

        log_upload(username, file.filename)

        return {
            "status": "success",
            "message": f"File {file.filename} processed successfully",
            "file_id": file_id,
            "summary": summary
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.get("/results/{file_id}")
async def get_detailed_results(file_id: str):
    """Get detailed results for a specific file"""
    results_path = os.path.join(OUTPUT_DIR, f"results_{file_id}.json")

    if not os.path.exists(results_path):
        raise HTTPException(status_code=404, detail="Results not found")

    with open(results_path, "r") as f:
        results_data = json.load(f)

    return JSONResponse(content=results_data)


@app.get("/download/{file_id}")
async def download_report(file_id: str):
    """Download generated report for a specific file"""
    report_path = os.path.join(OUTPUT_DIR, f"report_{file_id}.docx")

    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Report not found")

    return FileResponse(
        report_path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=f"Validation_Report_{file_id}.docx"
    )


@app.get("/uploads")
async def get_user_uploads(username: Optional[str] = Query(None)):
    """Get upload history for a specific user or all users"""
    df_records = load_upload_logs()

    if username:
        user_records = df_records[df_records["User"] == username]
        data = user_records.to_dict('records') if not user_records.empty else []
    else:
        data = df_records.to_dict('records') if not df_records.empty else []

    return JSONResponse(content={"uploads": data})


@app.get("/dashboard")
async def dashboard():
    """Simple dashboard placeholder"""
    return {"message": "Dashboard API - connect frontend for visualization"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8041)
