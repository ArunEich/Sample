"""
Single Function App with Multiple Functions
- HTTP triggers for API endpoints
- Service Bus trigger for background processing
- NO FastAPI wrapping
"""

import os
import json
import logging
import io  # ✅ Added for BytesIO
from datetime import datetime
import uuid
import azure.functions as func
from azure.storage.blob import BlobServiceClient
from werkzeug.formparser import parse_form_data
# from werkzeug.datastructures import FileStoraget

# Import processing modules
from backend.core.document_extractor import extract_text_with_docintelligence
from backend.core.checkpoint_loader import load_checkpoints_from_blob
from backend.core.audit_logic import verify_checkpoints_with_ai
from backend.core.report_generator import generate_report
from backend.core.logger import log_upload, load_upload_logs

# ✅ Create clean Function App (NO FastAPI)
app = func.FunctionApp()

# Environment variables
BLOB_CONN_STR = os.getenv("AZURE_BLOB_CONN_STR")
BLOB_CONTAINER = os.getenv("BLOB_CONTAINER", "genaipoc")
SERVICE_BUS_CONN_STR = os.getenv("AZURE_SERVICE_BUS_CONN_STR")
SERVICE_BUS_QUEUE = os.getenv("SERVICE_BUS_QUEUE", "audit-queue")

if not BLOB_CONN_STR:
    logging.error("AZURE_BLOB_CONN_STR is not set")

blob_service = BlobServiceClient.from_connection_string(BLOB_CONN_STR)
container_client = blob_service.get_container_client(BLOB_CONTAINER)


# ============================================
# HTTP FUNCTIONS (API Endpoints)
# ============================================

from werkzeug.datastructures import FileStorage
from werkzeug.formparser import parse_form_data

@app.function_name(name="UploadDocument")
@app.route(route="audit/run", methods=["POST"], auth_level=func.AuthLevel.ANONYMOUS)
async def upload_document(req: func.HttpRequest) -> func.HttpResponse:
    try:
        # ✅ Parse multipart form-data manually
        environ = {
            'wsgi.input': io.BytesIO(req.get_body()),
            'CONTENT_LENGTH': str(len(req.get_body())),
            'CONTENT_TYPE': req.headers.get('Content-Type'),
            'REQUEST_METHOD': 'POST'
        }
        stream, form, files = parse_form_data(environ)

        username = form.get('username')
        file: FileStorage = files.get('file')

        if not username or not file:
            return func.HttpResponse(
                json.dumps({"error": "Missing username or file"}),
                status_code=400,
                mimetype="application/json"
            )

        filename = file.filename
        if not filename.lower().endswith(('.pdf', '.docx')):
            return func.HttpResponse(
                json.dumps({"error": "Only PDF and DOCX are supported"}),
                status_code=400,
                mimetype="application/json"
            )

        file_content = file.stream.read()

        # ✅ Continue with blob upload and Service Bus logic
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        file_id = f"{username}_{timestamp}_{uuid.uuid4().hex[:8]}"
        blob_name = f"uploads/{file_id}_{filename}"

        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(file_content, overwrite=True)
        blob_url = blob_client.url

        log_upload(user=username, file_name=filename)

        from azure.servicebus import ServiceBusClient, ServiceBusMessage
        servicebus_client = ServiceBusClient.from_connection_string(SERVICE_BUS_CONN_STR)
        sender = servicebus_client.get_queue_sender(SERVICE_BUS_QUEUE)

        message_data = {
            "file_id": file_id,
            "username": username,
            "filename": filename,
            "blob_container": BLOB_CONTAINER,
            "blob_name": blob_name,
            "blob_url": blob_url,
            "timestamp": timestamp
        }
        sender.send_messages(ServiceBusMessage(json.dumps(message_data)))
        sender.close()
        servicebus_client.close()

        return func.HttpResponse(
            json.dumps({
                "status": "success",
                "message": "File uploaded and processing queued",
                "file_id": file_id,
                "blob_url": blob_url
            }),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        logging.exception(f"Upload failed: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )



@app.function_name(name="ViewReport")
@app.route(route="report/view/{file_id}", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def view_report(req: func.HttpRequest) -> func.HttpResponse:
    """
    GET /api/report/view/{file_id}
    Return detailed results as JSON
    """
    try:
        file_id = req.route_params.get('file_id')
        
        if not file_id:
            return func.HttpResponse(
                json.dumps({"error": "Missing file_id"}),
                status_code=400,
                mimetype="application/json"
            )
        
        blob_name = f"results/results_{file_id}.json"
        blob_client = container_client.get_blob_client(blob_name)
        
        if not blob_client.exists():
            return func.HttpResponse(
                json.dumps({"error": "Results not found"}),
                status_code=404,
                mimetype="application/json"
            )
        
        results_data = json.loads(blob_client.download_blob().readall())
        
        return func.HttpResponse(
            json.dumps(results_data),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.exception(f"View report failed: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )


@app.function_name(name="DownloadReport")
@app.route(route="report/download/{file_id}", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def download_report(req: func.HttpRequest) -> func.HttpResponse:
    """
    GET /api/report/download/{file_id}
    Download report as .docx
    """
    try:
        file_id = req.route_params.get('file_id')
        
        if not file_id:
            return func.HttpResponse(
                json.dumps({"error": "Missing file_id"}),
                status_code=400,
                mimetype="application/json"
            )
        
        blob_name = f"reports/report_{file_id}.docx"
        blob_client = container_client.get_blob_client(blob_name)
        
        if not blob_client.exists():
            return func.HttpResponse(
                json.dumps({"error": "Report not found"}),
                status_code=404,
                mimetype="application/json"
            )
        
        file_bytes = blob_client.download_blob().readall()
        
        return func.HttpResponse(
            body=file_bytes,
            status_code=200,
            mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={
                "Content-Disposition": f"attachment; filename=Validation_Report_{file_id}.docx"
            }
        )
        
    except Exception as e:
        logging.exception(f"Download report failed: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )


@app.function_name(name="GetDashboard")
@app.route(route="report/dashboard", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def get_dashboard(req: func.HttpRequest) -> func.HttpResponse:
    """
    GET /api/report/dashboard?username=xxx
    Get upload history
    """
    try:
        username = req.params.get('username')
        
        logs_data = load_upload_logs()
        
        if username:
            filtered_logs = [
                record for record in logs_data 
                if record.get("User") == username
            ]
            data = filtered_logs
        else:
            data = logs_data
        
        return func.HttpResponse(
            json.dumps({"dashboard": data}),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.exception(f"Dashboard failed: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )


@app.function_name(name="HealthCheck")
@app.route(route="health", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def health_check(req: func.HttpRequest) -> func.HttpResponse:
    """
    GET /api/health
    Health check endpoint
    """
    return func.HttpResponse(
        json.dumps({
            "status": "healthy",
            "service": "BD Audit Automation",
            "timestamp": datetime.utcnow().isoformat()
        }),
        status_code=200,
        mimetype="application/json"
    )


# ============================================
# SERVICE BUS FUNCTION (Background Processing)
# ============================================

@app.function_name(name="ProcessAuditMessage")
@app.service_bus_queue_trigger(
    arg_name="msg",
    queue_name=SERVICE_BUS_QUEUE,
    connection="AZURE_SERVICE_BUS_CONN_STR"
)
async def process_audit_message(msg: func.ServiceBusMessage):
    """
    Service Bus triggered function
    Processes documents asynchronously
    """
    try:
        # Parse message
        raw = msg.get_body()
        if isinstance(raw, bytes):
            raw = raw.decode('utf-8')
        data = json.loads(raw)
        
        logging.info(f"[ProcessAuditMessage] Received: {data}")
        
        file_id = data.get('file_id')
        blob_name = data.get('blob_name')
        filename = data.get('filename')
        username = data.get('username', 'unknown')
        
        if not all([file_id, blob_name, filename]):
            logging.error(f"Invalid message format: {data}")
            return
        
        # Download file from blob
        blob_client = container_client.get_blob_client(blob_name)
        if not blob_client.exists():
            logging.error(f"Blob not found: {blob_name}")
            return
        
        logging.info(f"Downloading blob: {blob_name}")
        file_bytes = blob_client.download_blob().readall()
        ext = filename.lower().split(".")[-1] if filename else ""
        
        # Extract text
        full_text = None
        try:
            logging.info("Extracting text from document")
            full_text = extract_text_with_docintelligence(file_bytes, ext)
            logging.info(f"Extraction successful, length: {len(full_text) if full_text else 0}")
        except Exception as e:
            logging.exception(f"Text extraction failed: {e}")
        
        # Load checkpoints
        checkpoints = []
        try:
            logging.info("Loading checkpoints")
            checkpoints = load_checkpoints_from_blob()
            logging.info(f"Loaded {len(checkpoints)} checkpoints")
        except Exception as e:
            logging.exception(f"Loading checkpoints failed: {e}")
        
        # Verify with AI
        results = []
        try:
            logging.info("Starting AI verification")
            results = await verify_checkpoints_with_ai(checkpoints, full_text, file_bytes, ext)
            logging.info(f"Verification completed, {len(results)} results")
        except Exception as e:
            logging.exception(f"AI verification failed: {e}")
        
        # Generate summary
        summary = {
            "total_checkpoints": len(checkpoints),
            "passed": sum(1 for r in results if r.get("Status") == "Pass"),
            "failed": sum(1 for r in results if r.get("Status") == "Fail"),
            "username": username,
            "filename": filename
        }
        
        # Generate and upload report
        try:
            logging.info("Generating Word report")
            report_buf = generate_report(filename, summary, results)
            report_path = f"reports/report_{file_id}.docx"
            container_client.upload_blob(report_path, report_buf.getvalue(), overwrite=True)
            logging.info(f"Report uploaded: {report_path}")
        except Exception as e:
            logging.exception(f"Report generation failed: {e}")
        
        # Upload results JSON
        try:
            results_json = {
                "file_id": file_id,
                "username": username,
                "filename": filename,
                "summary": summary,
                "results": results,
                "timestamp": datetime.utcnow().isoformat(),
                "blob_name": blob_name
            }
            results_path = f"results/results_{file_id}.json"
            container_client.upload_blob(
                results_path, 
                json.dumps(results_json, indent=2), 
                overwrite=True
            )
            logging.info(f"Results uploaded: {results_path}")
        except Exception as e:
            logging.exception(f"Uploading results failed: {e}")
        
        logging.info(f" Processing completed for file_id={file_id}")
        
    except json.JSONDecodeError as e:
        logging.exception(f"Invalid JSON in message: {e}")
    except Exception as e:
        logging.exception(f"Unexpected error: {e}")