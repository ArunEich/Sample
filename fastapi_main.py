# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from backend.routers import audit_router, report_router
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI app"""
    logger.info("🚀 FastAPI application starting up...")
    # Startup logic (if needed)
    yield
    # Shutdown logic (if needed)
    logger.info("🛑 FastAPI application shutting down...")


# Create FastAPI app
app = FastAPI(
    title="BD Audit Automation API",
    version="1.0.0",
    description="API for document audit automation using Azure OpenAI",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ Change this in production to specific origins
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

# Include routers
app.include_router(audit_router.router)
app.include_router(report_router.router)


@app.get("/", tags=["Root"])
def root():
    """Root endpoint"""
    return {
        "message": "BD Audit Automation API running",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "audit": "/audit/run",
            "reports": "/report/dashboard"
        }
    }


@app.get("/health", tags=["Health"])
def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "FastAPI Backend"
    }


# For local development only
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
