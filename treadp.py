# backend/core/ai_reviewer.py
from backend.utils.azure_clients import openai_client, OPENAI_DEPLOYMENT
from io import BytesIO
import base64
import time
import random
import threading

# Concurrency limit for calls to the OpenAI/Azure client.
# Tune this depending on your OpenAI/Azure rate limits and plan.
MAX_CONCURRENT_OPENAI_CALLS = 4
SEMAPHORE = threading.Semaphore(MAX_CONCURRENT_OPENAI_CALLS)


def _retry_on_exception(max_attempts=5, initial_wait=0.5, backoff_factor=2.0, jitter=0.1):
    """
    Decorator factory for retrying functions that raise exceptions (e.g., rate limits).
    Exponential backoff + jitter. Returns the wrapped function's value on success.
    On final failure raises the last exception.
    """
    def decorator(fn):
        def wrapper(*args, **kwargs):
            attempt = 0
            wait = initial_wait
            last_exc = None
            while attempt < max_attempts:
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    # If it's a rate-limit situation, we backoff and retry.
                    attempt += 1
                    if attempt >= max_attempts:
                        break
                    sleep_time = wait + random.uniform(0, jitter)
                    time.sleep(sleep_time)
                    wait *= backoff_factor
            # after exhausting attempts raise last exception
            raise last_exc
        return wrapper
    return decorator


def classify_rule_type(checkpoint: str) -> str:
    """Return 'text_rule' | 'visual_rule' | 'hybrid_rule'"""
    if openai_client is None:
        return "text_rule"
    prompt = f"""
You are an assistant that classifies audit checkpoints.

Checkpoint:
"{checkpoint}"

Decide the rule type:
- text_rule: Can be verified only from text.
- visual_rule: Needs visual/layout check only (logo, page numbers, formatting, signatures).
- hybrid_rule: Needs both text + visual validation.

Answer with only one word: text_rule, visual_rule, or hybrid_rule.
"""
    try:
        @ _retry_on_exception(max_attempts=4, initial_wait=0.6)
        def _call():
            # semaphore to limit concurrent requests
            with SEMAPHORE:
                resp = openai_client.chat.completions.create(
                    model=OPENAI_DEPLOYMENT,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=10
                )
            return resp

        resp = _call()
        return resp.choices[0].message.content.strip().lower()
    except Exception:
        # on any failure, default to text_rule to avoid needless visual analysis
        return "text_rule"


def check_with_text_engine(checkpoint: str, document_text: str) -> str:
    if openai_client is None:
        return "STATUS: FAIL\n(Error: OpenAI client not configured)"

    prompt = f"""
You are auditing a document.

Document Text:
{document_text}

Rule:
{checkpoint}

Instructions:
- Check if this rule is satisfied in the text (including [HEADER]/[FOOTER]/[IMAGE] placeholders).
- End with one line: STATUS: PASS or STATUS: FAIL
- If FAIL, add a short recommendation.
"""
    try:
        @ _retry_on_exception(max_attempts=4, initial_wait=0.8)
        def _call_text():
            with SEMAPHORE:
                resp = openai_client.chat.completions.create(
                    model=OPENAI_DEPLOYMENT,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=300
                )
            return resp

        resp = _call_text()
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error in text engine: {e}\nSTATUS: FAIL"


def encode_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def check_with_vision_engine(checkpoint: str, page_images):
    """Sends first page image + prompt to OpenAI multimodal call; returns textual analysis and aggregated status."""
    results = []
    if openai_client is None or not page_images:
        return ["Error: vision or images not available"], "Fail"

    base_prompt = (
        "You are a document compliance checker.\n\n"
        "Checkpoint Rule:\n{checkpoint}\n\n"
        "Task: Review the provided page image and verify whether the header contains the required elements.\n"
        "For each required element, state Found or Missing. After findings, give STATUS: PASS or STATUS: FAIL.\n"
    ).format(checkpoint=checkpoint)

    try:
        b64 = encode_image_to_base64(page_images[0])
        @ _retry_on_exception(max_attempts=4, initial_wait=1.0)
        def _call_vision():
            with SEMAPHORE:
                response = openai_client.chat.completions.create(
                    model=OPENAI_DEPLOYMENT,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": base_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                        ]
                    }],
                    temperature=0,
                    max_tokens=400
                )
            return response

        response = _call_vision()
        text = response.choices[0].message.content.strip()
        results.append(text)
    except Exception as e:
        results.append(f"Error analyzing image: {e}")

    final_status = "Fail"
    for r in results:
        if "status: pass" in r.lower():
            final_status = "Pass"
            break
    return results, final_status


def check_with_hybrid_engine(checkpoint: str, document_text: str, page_images):
    text_result = check_with_text_engine(checkpoint, document_text)
    text_status = "Pass" if "status: pass" in text_result.lower() else "Fail"
    vision_results, vision_status = check_with_vision_engine(checkpoint, page_images)
    final_status = "Pass" if (text_status == "Pass" and vision_status == "Pass") else "Fail"
    return {
        "text_result": text_result,
        "vision_results": vision_results,
        "status": final_status
    }



...........................................................................................................

# backend/core/audit_logic.py
from backend.core.ai_reviewer import (
    classify_rule_type, check_with_text_engine, check_with_vision_engine, check_with_hybrid_engine
)
from backend.utils.file_ops import pdf_bytes_to_images, convert_docx_to_pdf_bytes
import concurrent.futures
import traceback

def determine_status_from_recommendation(recommendation: str) -> str:
    rec_lower = recommendation.lower() if recommendation else ""
    if "status: pass" in rec_lower:
        return "Pass"
    if "status: fail" in rec_lower:
        return "Fail"
    # heuristics
    if any(p in rec_lower for p in ["checkpoint satisfied", "requirement satisfied", "checkpoint met"]):
        return "Pass"
    return "Fail"


async def verify_checkpoints_with_ai(checkpoints, full_text: str, file_bytes: bytes, file_ext: str, max_workers: int = 6):
    """
    Verifies checkpoints in parallel using ThreadPoolExecutor.

    - max_workers controls degree of parallelism (tune for your environment and API rate limits).
    - Returns list of dicts with keys: S_No, Checkpoint, Status, Recommendation
    """
    results = []
    page_images = []
    # convert to images if available
    try:
        if file_ext == "pdf":
            page_images = pdf_bytes_to_images(file_bytes, dpi=150)
        elif file_ext == "docx":
            pdf_bytes = convert_docx_to_pdf_bytes(file_bytes)
            if pdf_bytes:
                page_images = pdf_bytes_to_images(pdf_bytes, dpi=150)
    except Exception:
        # If conversion fails, just proceed with no images
        page_images = []

    # worker function for one checkpoint
    def _process_checkpoint(idx, checkpoint):
        try:
            rule_type = classify_rule_type(checkpoint)

            if rule_type == "text_rule":
                recommendation = check_with_text_engine(checkpoint, full_text)
                status = determine_status_from_recommendation(recommendation)
            elif rule_type == "visual_rule":
                if page_images:
                    vision_results, status = check_with_vision_engine(checkpoint, page_images)
                    recommendation = "\n".join(vision_results)
                else:
                    status = "Fail"
                    recommendation = "Visual analysis not available"
            elif rule_type == "hybrid_rule":
                if page_images:
                    hybrid = check_with_hybrid_engine(checkpoint, full_text, page_images)
                    recommendation = f"Text: {hybrid['text_result']}\nVision: {hybrid['vision_results']}"
                    status = hybrid["status"]
                else:
                    # Fallback to text-only if vision unavailable
                    recommendation = check_with_text_engine(checkpoint, full_text) + "\n(Visual unavailable)"
                    status = determine_status_from_recommendation(recommendation)
            else:
                status = "Fail"
                recommendation = "Unknown rule type"
        except Exception as e:
            # capture stack trace for debugging, but return a friendly message
            tb = traceback.format_exc()
            recommendation = f"Error processing checkpoint: {e}\n{tb}"
            status = "Fail"

        return {
            "S_No": idx,
            "Checkpoint": checkpoint,
            "Status": status,
            "Recommendation": recommendation
        }

    # Run tasks in parallel while preserving order in final output
    results_by_idx = {}
    # Limit workers to at most number of checkpoints or provided max_workers
    workers = min(max_workers, max(1, len(checkpoints)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {}
        for idx, checkpoint in enumerate(checkpoints, start=1):
            future = executor.submit(_process_checkpoint, idx, checkpoint)
            future_to_idx[future] = idx

        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                res = future.result()
            except Exception as e:
                # Shouldn't normally happen because _process_checkpoint catches exceptions,
                # but keep a safety net here.
                res = {
                    "S_No": idx,
                    "Checkpoint": checkpoints[idx - 1],
                    "Status": "Fail",
                    "Recommendation": f"Unhandled exception: {e}"
                }
            results_by_idx[idx] = res

    # Build ordered list
    ordered_results = [results_by_idx[i] for i in sorted(results_by_idx.keys())]
    return ordered_results
.......................................................................................................

# backend/routers/audit_router.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import os
import json
from datetime import datetime

from backend.core.logger import log_upload, init_log_file
from backend.core.checkpoint_loader import load_checkpoints
from backend.core.document_extractor import extract_pdf_content, extract_docx_content
from backend.core.audit_logic import verify_checkpoints_with_ai
from backend.core.report_generator import generate_report
from backend.utils.file_ops import convert_docx_to_pdf_bytes, save_pdf_images

router = APIRouter(prefix="/audit", tags=["Audit"])

UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
OUTPUT_DIR = os.path.join(os.getcwd(), "output")
IMAGES_DIR = os.path.join(os.getcwd(), "saved_images")

for d in (UPLOAD_DIR, OUTPUT_DIR, IMAGES_DIR):
    os.makedirs(d, exist_ok=True)


@router.post("/run")
async def run_audit(
    username: str = Form(...),
    file: UploadFile = File(...),
    max_workers: int = Form(6)  # optional tuning parameter; default 6
):
    if not file.filename.lower().endswith((".pdf", ".docx")):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX are supported")

    # store upload (use secure filename handling if needed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_id = f"{username}_{timestamp}_{file.filename}"
    upload_path = os.path.join(UPLOAD_DIR, file_id)
    file_content = await file.read()
    with open(upload_path, "wb") as f:
        f.write(file_content)

    # read back bytes (to support further processing)
    with open(upload_path, "rb") as f:
        file_bytes = f.read()

    ext = file.filename.lower().split(".")[-1]
    try:
        if ext == "pdf":
            full_text = extract_pdf_content(file_bytes)
        else:
            full_text = extract_docx_content(file_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract document text: {e}")

    checkpoints = load_checkpoints()

    # Run verification - this is async (uses ThreadPoolExecutor internally)
    results = await verify_checkpoints_with_ai(checkpoints, full_text, file_bytes, ext, max_workers=max_workers)

    pass_count = sum(1 for r in results if r["Status"] == "Pass")
    fail_count = sum(1 for r in results if r["Status"] == "Fail")
    summary = {
        "total_checkpoints": len(checkpoints),
        "passed": pass_count,
        "failed": fail_count
    }

    # Save report
    report_buf = generate_report(file.filename, summary, results)
    report_path = os.path.join(OUTPUT_DIR, f"report_{file_id}.docx")
    with open(report_path, "wb") as f:
        f.write(report_buf.getvalue())

    # Convert docx->pdf then save images (if needed)
    try:
        if ext == "pdf":
            pdf_bytes = file_bytes
        else:
            pdf_bytes = convert_docx_to_pdf_bytes(file_bytes)
        if pdf_bytes:
            save_pdf_images(pdf_bytes, IMAGES_DIR, file_id.replace(".", "_"))
    except Exception:
        # don't fail whole request if image extraction fails
        pass

    # Log upload
    init_log_file()
    log_upload(username, file.filename)

    # persist results JSON for retrieval
    results_data = {
        "file_id": file_id,
        "filename": file.filename,
        "username": username,
        "summary": summary,
        "results": results,
        "timestamp": datetime.now().isoformat()
    }
    with open(os.path.join(OUTPUT_DIR, f"results_{file_id}.json"), "w") as f:
        json.dump(results_data, f, indent=2)

    return JSONResponse(content={"status": "success", "file_id": file_id, "summary": summary})
