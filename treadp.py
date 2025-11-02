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

    ................................................................................................
    #modified backend/utils/azure_clients.py
    import time
import random
from openai import AzureOpenAIError

MAX_RETRIES = 5  # you can adjust (3–5 recommended)
RETRY_BASE_DELAY = 5  # seconds

def safe_chat_completion_create(model, messages, temperature=0, max_tokens=200, client=None):
    """
    Wrapper for openai_client.chat.completions.create with retry handling.
    Automatically retries when Azure rate limit (429) or transient network errors occur.
    """
    if client is None:
        raise RuntimeError("Azure OpenAI client not initialized")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            # Azure throttling errors usually contain code 429 or "RateLimitReached"
            msg = str(e).lower()
            if "ratelimit" in msg or "429" in msg or "quota" in msg:
                wait_time = RETRY_BASE_DELAY * attempt + random.uniform(0, 2)
                print(f"[Retry {attempt}/{MAX_RETRIES}] Rate limit hit. Waiting {wait_time:.1f}s before retry...")
                time.sleep(wait_time)
                continue
            # transient network or timeout
            if "timeout" in msg or "connection" in msg or "temporarily unavailable" in msg:
                wait_time = RETRY_BASE_DELAY * attempt
                print(f"[Retry {attempt}/{MAX_RETRIES}] Transient network error. Waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                continue

            # other errors → don't retry
            raise

    raise RuntimeError(f"Failed after {MAX_RETRIES} retries due to persistent rate limit or network errors.")

    ............................................................................................................
    backend/core/ai_reviewer.py
    import base64
import traceback
from backend.utils.azure_clients import openai_client, OPENAI_DEPLOYMENT, safe_chat_completion_create

# ----------------------------
# 1️⃣ - CLASSIFY RULE TYPE
# ----------------------------
def classify_rule_type(rule_text: str) -> str:
    """
    Classifies whether the audit checkpoint rule should be validated using Text Engine or Vision Engine.
    """
    try:
        prompt = (
            f"Decide if the following checkpoint should be validated using text-based analysis or visual-based analysis.\n"
            f"Return only one word: 'Text' or 'Visual'.\n\nCheckpoint:\n{rule_text}"
        )

        response = safe_chat_completion_create(
            model=OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are a document audit rule classifier."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=10,
            client=openai_client
        )

        result = response.choices[0].message.content.strip()
        return "Visual" if "visual" in result.lower() else "Text"

    except Exception as e:
        print("❌ Error in classify_rule_type:", e)
        traceback.print_exc()
        return "Text"  # fallback to text-based if error


# ----------------------------
# 2️⃣ - TEXT ENGINE CHECK
# ----------------------------
def check_with_text_engine(rule: str, extracted_text: str) -> dict:
    """
    Validates text-based rules using Azure OpenAI GPT model.
    """
    try:
        prompt = (
            f"Checkpoint:\n{rule}\n\n"
            f"Extracted Document Text:\n{extracted_text}\n\n"
            "Evaluate whether the checkpoint criteria are satisfied within the document text. "
            "Reply strictly in JSON with the following keys:\n"
            "{\n"
            "  'Status': 'Pass' or 'Fail',\n"
            "  'Evidence': 'Brief evidence or reason',\n"
            "  'Recommendation': 'Fix suggestion if failed'\n"
            "}"
        )

        response = safe_chat_completion_create(
            model=OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are a compliance document auditor focusing on text analysis."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=400,
            client=openai_client
        )

        return response.choices[0].message.content

    except Exception as e:
        print("❌ Error in check_with_text_engine:", e)
        traceback.print_exc()
        return {
            "Status": "Fail",
            "Evidence": "Error in text engine",
            "Recommendation": str(e)
        }


# ----------------------------
# 3️⃣ - VISION ENGINE CHECK
# ----------------------------
def check_with_vision_engine(rule: str, image_bytes: bytes) -> dict:
    """
    Validates visual-based rules using Azure OpenAI vision-enabled model.
    """
    try:
        b64 = base64.b64encode(image_bytes).decode("utf-8")

        base_prompt = (
            f"Checkpoint:\n{rule}\n\n"
            "Analyze the provided image to determine if the checkpoint is visually satisfied. "
            "Return JSON output with these keys:\n"
            "{\n"
            "  'Status': 'Pass' or 'Fail',\n"
            "  'Evidence': 'Brief evidence',\n"
            "  'Recommendation': 'Fix suggestion if failed'\n"
            "}"
        )

        response = safe_chat_completion_create(
            model=OPENAI_DEPLOYMENT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": base_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                    ],
                }
            ],
            temperature=0,
            max_tokens=400,
            client=openai_client
        )

        return response.choices[0].message.content

    except Exception as e:
        print("❌ Error in check_with_vision_engine:", e)
        traceback.print_exc()
        return {
            "Status": "Fail",
            "Evidence": "Error in vision engine",
            "Recommendation": str(e)
        }

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# backend/core/ai_reviewer.py
from backend.utils.azure_clients import openai_client, OPENAI_DEPLOYMENT
import concurrent.futures
import time

def analyze_checkpoint_with_gpt(checkpoint: str, document_text: str) -> dict:
    """
    Use GPT-4o to verify if the checkpoint is satisfied in the given document text.
    Returns dict with Status + Recommendation.
    """
    if openai_client is None:
        return {"Status": "Fail", "Recommendation": "OpenAI client not configured"}

    prompt = f"""
You are an audit assistant.

Below is the document content and a compliance checkpoint.
Check if the checkpoint is satisfied in the document.

Document:
\"\"\"{document_text[:12000]}\"\"\"  # limit for context safety

Checkpoint:
"{checkpoint}"

Respond in the format:
Recommendation: <short summary of findings or missing information>
Status: PASS or FAIL
"""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200
        )
        content = response.choices[0].message.content.strip()

        status = "Fail"
        if "status: pass" in content.lower():
            status = "Pass"
        elif "status: fail" in content.lower():
            status = "Fail"

        # Extract recommendation part
        recommendation = content.replace("Status:", "").replace("status:", "").strip()
        return {"Status": status, "Recommendation": recommendation}

    except Exception as e:
        return {"Status": "Fail", "Recommendation": f"Error: {e}"}


def parallel_analyze_checkpoints(checkpoints, document_text: str, max_workers: int = 5):
    """
    Run checkpoint verifications in parallel using ThreadPoolExecutor.
    Handles rate limits gracefully by retrying.
    """
    results = []

    def safe_analyze(cp):
        retries = 3
        for attempt in range(retries):
            result = analyze_checkpoint_with_gpt(cp, document_text)
            if "rate limit" in result["Recommendation"].lower():
                time.sleep(2 ** attempt)
                continue
            return result
        return {"Status": "Fail", "Recommendation": "Rate limit exceeded repeatedly"}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_cp = {executor.submit(safe_analyze, cp): cp for cp in checkpoints}
        for future in concurrent.futures.as_completed(future_to_cp):
            cp = future_to_cp[future]
            try:
                res = future.result()
            except Exception as e:
                res = {"Status": "Fail", "Recommendation": f"Error processing: {e}"}
            results.append({"Checkpoint": cp, **res})

    return results
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# backend/core/audit_logic.py
from backend.core.ai_reviewer import parallel_analyze_checkpoints

async def verify_checkpoints_with_ai(checkpoints, full_text: str, file_bytes: bytes, file_ext: str):
    """
    Simplified unified GPT-4o verification — no vision/text/hybrid separation.
    Runs all checkpoints in parallel.
    """
    results = parallel_analyze_checkpoints(checkpoints, full_text, max_workers=5)

    # Add serial number
    for idx, r in enumerate(results, start=1):
        r["S_No"] = idx

    return results
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# backend/routers/audit_router.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import os, json
from datetime import datetime

from backend.core.logger import log_upload, init_log_file
from backend.core.checkpoint_loader import load_checkpoints
from backend.core.document_extractor import extract_pdf_content, extract_docx_content
from backend.core.audit_logic import verify_checkpoints_with_ai
from backend.core.report_generator import generate_report

router = APIRouter(prefix="/audit", tags=["Audit"])

UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
OUTPUT_DIR = os.path.join(os.getcwd(), "output")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@router.post("/run")
async def run_audit(username: str = Form(...), file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".pdf", ".docx")):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX are supported")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_id = f"{username}_{timestamp}_{file.filename}"
    upload_path = os.path.join(UPLOAD_DIR, file_id)
    with open(upload_path, "wb") as f:
        f.write(await file.read())

    with open(upload_path, "rb") as f:
        file_bytes = f.read()

    ext = file.filename.lower().split(".")[-1]
    if ext == "pdf":
        full_text = extract_pdf_content(file_bytes)
    else:
        full_text = extract_docx_content(file_bytes)

    checkpoints = load_checkpoints()
    results = await verify_checkpoints_with_ai(checkpoints, full_text, file_bytes, ext)

    pass_count = sum(1 for r in results if r["Status"] == "Pass")
    fail_count = sum(1 for r in results if r["Status"] == "Fail")

    summary = {
        "total_checkpoints": len(checkpoints),
        "passed": pass_count,
        "failed": fail_count
    }

    report_buf = generate_report(file.filename, summary, results)
    report_path = os.path.join(OUTPUT_DIR, f"report_{file_id}.docx")
    with open(report_path, "wb") as f:
        f.write(report_buf.getvalue())

    init_log_file()
    log_upload(username, file.filename)

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



