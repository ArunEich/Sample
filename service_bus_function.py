import os
import json
import logging
from datetime import datetime
import azure.functions as func
from azure.storage.blob import BlobServiceClient

# Import core processing functions
from backend.core.document_extractor import extract_text_with_docintelligence
from backend.core.checkpoint_loader import load_checkpoints_from_blob
from backend.core.audit_logic import verify_checkpoints_with_ai
from backend.core.report_generator import generate_report

# ✅ Create Function App (NO FastAPI mixing)
app = func.FunctionApp()

# Environment variables
BLOB_CONN_STR = os.getenv("AZURE_BLOB_CONN_STR")
BLOB_CONTAINER = os.getenv("BLOB_CONTAINER", "genaipoc")
SERVICE_BUS_QUEUE = os.getenv("SERVICE_BUS_QUEUE", "audit-queue")

if not BLOB_CONN_STR:
    logging.error("AZURE_BLOB_CONN_STR is not set; blob operations will fail")

blob_service = BlobServiceClient.from_connection_string(BLOB_CONN_STR)
container_client = blob_service.get_container_client(BLOB_CONTAINER)


@app.function_name(name="ProcessAuditMessage")
@app.service_bus_queue_trigger(
    arg_name="msg",
    queue_name=SERVICE_BUS_QUEUE,
    connection="AZURE_SERVICE_BUS_CONN_STR"
)
async def process_audit_message(msg: func.ServiceBusMessage):
    """
    Service Bus triggered function to process audit documents.
    Triggered when a message arrives in the queue.
    """
    try:
        # Parse message
        raw = msg.get_body()
        if isinstance(raw, bytes):
            raw = raw.decode('utf-8')
        data = json.loads(raw)
        
        logging.info(f"[ProcessAuditMessage] Received message: {data}")

        # Extract message data
        file_id = data.get('file_id')
        blob_name = data.get('blob_name')
        filename = data.get('filename')
        username = data.get('username', 'unknown')

        # Validate required fields
        if not all([file_id, blob_name, filename]):
            logging.error(f"[ProcessAuditMessage] Invalid message format - missing required fields: {data}")
            return

        # Check if blob exists
        blob_client = container_client.get_blob_client(blob_name)
        if not blob_client.exists():
            logging.error(f"[ProcessAuditMessage] Blob not found: {blob_name}")
            return

        # Download document from blob
        logging.info(f"[ProcessAuditMessage] Downloading blob: {blob_name}")
        file_bytes = blob_client.download_blob().readall()
        ext = filename.lower().split(".")[-1] if filename else ""

        # Step 1: Extract text from document
        full_text = None
        try:
            logging.info(f"[ProcessAuditMessage] Extracting text from {filename}")
            full_text = extract_text_with_docintelligence(file_bytes, ext)
            logging.info(f"[ProcessAuditMessage] Text extraction successful. Length: {len(full_text) if full_text else 0}")
        except Exception as e:
            logging.exception(f"[ProcessAuditMessage] Text extraction failed: {e}")

        # Step 2: Load audit checkpoints
        checkpoints = []
        try:
            logging.info("[ProcessAuditMessage] Loading checkpoints from blob")
            checkpoints = load_checkpoints_from_blob()
            logging.info(f"[ProcessAuditMessage] Loaded {len(checkpoints)} checkpoints")
        except Exception as e:
            logging.exception(f"[ProcessAuditMessage] Loading checkpoints failed: {e}")

        # Step 3: Verify checkpoints with AI
        results = []
        try:
            logging.info("[ProcessAuditMessage] Starting AI verification")
            results = await verify_checkpoints_with_ai(checkpoints, full_text, file_bytes, ext)
            logging.info(f"[ProcessAuditMessage] AI verification completed. Results count: {len(results)}")
        except Exception as e:
            logging.exception(f"[ProcessAuditMessage] AI verification failed: {e}")

        # Step 4: Generate summary
        summary = {
            "total_checkpoints": len(checkpoints),
            "passed": sum(1 for r in results if r.get("Status") == "Pass"),
            "failed": sum(1 for r in results if r.get("Status") == "Fail"),
            "username": username,
            "filename": filename
        }

        # Step 5: Generate and upload Word report
        try:
            logging.info("[ProcessAuditMessage] Generating Word report")
            report_buf = generate_report(filename, summary, results)
            report_path = f"reports/report_{file_id}.docx"
            
            container_client.upload_blob(
                report_path, 
                report_buf.getvalue(), 
                overwrite=True
            )
            logging.info(f"[ProcessAuditMessage] Report uploaded to: {report_path}")
        except Exception as e:
            logging.exception(f"[ProcessAuditMessage] Report generation/upload failed: {e}")

        # Step 6: Upload results JSON
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
            logging.info(f"[ProcessAuditMessage] Results JSON uploaded to: {results_path}")
        except Exception as e:
            logging.exception(f"[ProcessAuditMessage] Uploading results JSON failed: {e}")

        logging.info(f"[ProcessAuditMessage] ✅ Processing completed successfully for file_id={file_id}")

    except json.JSONDecodeError as e:
        logging.exception(f"[ProcessAuditMessage] Invalid JSON in message: {e}")
    except Exception as e:
        logging.exception(f"[ProcessAuditMessage] Unexpected error: {e}")


# Optional: Add a simple HTTP endpoint for health checks
@app.function_name(name="HealthCheck")
@app.route(route="health", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def health_check(req: func.HttpRequest) -> func.HttpResponse:
    """Simple health check endpoint"""
    return func.HttpResponse(
        json.dumps({
            "status": "healthy",
            "service": "Service Bus Processor",
            "timestamp": datetime.utcnow().isoformat()
        }),
        mimetype="application/json",
        status_code=200
    )
