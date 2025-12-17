"""
Main FastAPI application for VisaVerse Guardian AI.
This serves as the API Gateway for all microservices.
"""

from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
import logging
import time
import uuid
import os
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, Request, UploadFile, File, Form, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from ..core.models import AuditEntry
from ..core.interfaces import AuditService, OrchestrationService
from ..services.orchestration import TransparentOrchestrationService
from ..services.audit import TransparencyAuditService
from ..services.document import PaddleOCRDocumentService
from ..services.graph import Neo4jGraphReasoningService
from ..services.risk import GeminiRiskAssessmentService
from ..services.memory import MemMachineMemoryService


class Settings(BaseSettings):
    """Application settings."""
    app_name: str = "VisaVerse Guardian AI"
    version: str = "0.1.0"
    debug: bool = False
    allowed_hosts: list = ["*"]
    cors_origins: list = ["*"]
    
    # Google Cloud Platform settings
    gcp_project_id: str = ""
    gcp_region: str = "us-central1"
    
    # Service configuration
    gemini_api_key: str = ""
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    
    # Authentication
    jwt_secret_key: str = "your-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    
    # File upload limits
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    max_files_per_request: int = 10
    allowed_file_types: list = [".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".bmp"]
    
    class Config:
        env_file = ".env"


# Request/Response Models
class VisaApplicationRequest(BaseModel):
    """Request model for visa application processing."""
    visa_type: str = Field(..., description="Type of visa being applied for")
    country: str = Field(..., description="Target country for visa application")
    language: Optional[str] = Field(None, description="Language hint for document processing")


class VisaApplicationResponse(BaseModel):
    """Response model for visa application processing."""
    application_id: str
    trace_id: str
    processing_time_ms: float
    approval_probability: float
    risk_factor_count: int
    recommendation_count: int
    transparency_score: float
    summary: str
    detailed_explanation: str
    actionable_steps: List[str]
    disclaimers: List[str]


class ApplicationStatusResponse(BaseModel):
    """Response model for application status."""
    application_id: str
    status: str
    last_updated: str
    operations_completed: List[str]
    audit_completeness: bool
    processing_steps: int


class HealthCheckResponse(BaseModel):
    """Response model for health checks."""
    status: str
    service: str
    version: str
    timestamp: float
    services: Optional[Dict[str, str]] = None


# Global settings instance
settings = Settings()

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global service instances
orchestration_service: Optional[OrchestrationService] = None
audit_service: Optional[AuditService] = None

# Authentication
security = HTTPBearer(auto_error=False)


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[str]:
    """Get current user from JWT token (simplified for demo)."""
    if not credentials:
        return None
    
    # In production, implement proper JWT validation
    # For now, return a demo user ID
    return "demo_user_123"


def validate_file_upload(file: UploadFile) -> None:
    """Validate uploaded file."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required"
        )
    
    # Check file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in settings.allowed_file_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type {file_ext} not allowed. Allowed types: {settings.allowed_file_types}"
        )
    
    # Check file size (this is approximate, actual size check happens during read)
    if hasattr(file, 'size') and file.size and file.size > settings.max_file_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum allowed size of {settings.max_file_size} bytes"
        )


async def initialize_services():
    """Initialize all services."""
    global orchestration_service, audit_service
    
    try:
        logger.info("Initializing services...")
        
        # Initialize audit service
        audit_service = TransparencyAuditService(storage_config={})
        
        # Initialize individual services
        document_service = PaddleOCRDocumentService()
        graph_service = Neo4jGraphReasoningService(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password
        )
        risk_service = GeminiRiskAssessmentService(
            api_key=settings.gemini_api_key
        )
        memory_service = MemMachineMemoryService(storage_config={})
        
        # Initialize orchestration service
        orchestration_service = TransparentOrchestrationService(
            document_service=document_service,
            graph_service=graph_service,
            risk_service=risk_service,
            memory_service=memory_service,
            audit_service=audit_service
        )
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}")
        raise


async def cleanup_services():
    """Cleanup services on shutdown."""
    global orchestration_service, audit_service
    
    logger.info("Cleaning up services...")
    
    # Cleanup would go here
    orchestration_service = None
    audit_service = None
    
    logger.info("Services cleaned up")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting VisaVerse Guardian AI API Gateway")
    
    # Initialize services
    await initialize_services()
    
    yield
    
    logger.info("Shutting down VisaVerse Guardian AI API Gateway")
    
    # Cleanup services
    await cleanup_services()


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="AI-driven global mobility intelligence platform",
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.allowed_hosts
)


# Request ID and audit middleware
@app.middleware("http")
async def add_request_id_and_audit(request: Request, call_next):
    """Add request ID and audit logging to all requests."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    start_time = time.time()
    
    # Add request ID to response headers
    response = await call_next(request)
    
    execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    response.headers["X-Request-ID"] = request_id
    
    # Log request for audit trail (when audit service is implemented)
    logger.info(
        f"Request {request_id}: {request.method} {request.url.path} "
        f"completed in {execution_time:.2f}ms with status {response.status_code}"
    )
    
    return response


# Health check endpoints
@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Basic health check endpoint."""
    return HealthCheckResponse(
        status="healthy",
        service="api-gateway",
        version=settings.version,
        timestamp=time.time()
    )


@app.get("/health/detailed", response_model=HealthCheckResponse)
async def detailed_health_check():
    """Detailed health check including service dependencies."""
    services_status = {
        "orchestration_service": "healthy" if orchestration_service else "not_initialized",
        "audit_service": "healthy" if audit_service else "not_initialized",
        "document_service": "healthy" if orchestration_service else "not_initialized",
        "graph_service": "healthy" if orchestration_service else "not_initialized",
        "risk_service": "healthy" if orchestration_service else "not_initialized",
        "memory_service": "healthy" if orchestration_service else "not_initialized"
    }
    
    overall_status = "healthy" if all(status == "healthy" for status in services_status.values()) else "degraded"
    
    return HealthCheckResponse(
        status=overall_status,
        service="api-gateway",
        version=settings.version,
        timestamp=time.time(),
        services=services_status
    )


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to VisaVerse Guardian AI",
        "version": settings.version,
        "description": "AI-driven global mobility intelligence platform",
        "docs": "/docs" if settings.debug else "Documentation not available in production",
        "health": "/health",
        "endpoints": {
            "process_application": "/api/v1/applications/process",
            "get_status": "/api/v1/applications/{application_id}/status",
            "health": "/health"
        }
    }


# Main API endpoints
@app.post("/api/v1/applications/process", response_model=VisaApplicationResponse)
async def process_visa_application(
    request: VisaApplicationRequest = Depends(),
    files: List[UploadFile] = File(..., description="Document files to process"),
    user_id: Optional[str] = Depends(get_current_user)
):
    """
    Process a complete visa application with documents.
    
    This endpoint orchestrates the entire visa application processing pipeline:
    1. Document processing and OCR extraction
    2. Compliance validation against visa rules
    3. Risk assessment and probability calculation
    4. Natural language explanation generation
    5. Personalized insights based on user history
    """
    if not orchestration_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Orchestration service not available"
        )
    
    # Validate request
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one document file is required"
        )
    
    if len(files) > settings.max_files_per_request:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many files. Maximum allowed: {settings.max_files_per_request}"
        )
    
    # Validate each file
    for file in files:
        validate_file_upload(file)
    
    try:
        # Read file contents
        file_contents = []
        filenames = []
        
        for file in files:
            content = await file.read()
            
            # Check actual file size
            if len(content) > settings.max_file_size:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File {file.filename} exceeds maximum size of {settings.max_file_size} bytes"
                )
            
            file_contents.append(content)
            filenames.append(file.filename)
        
        # Process application
        result = await orchestration_service.process_visa_application(
            user_id=user_id or "anonymous",
            visa_type=request.visa_type,
            country=request.country,
            documents=file_contents,
            filenames=filenames,
            language=request.language
        )
        
        # Format response
        return VisaApplicationResponse(
            application_id=result["application_id"],
            trace_id=result["trace_id"],
            processing_time_ms=result["processing_time_ms"],
            approval_probability=result["risk_assessment"]["approval_probability"],
            risk_factor_count=result["risk_assessment"]["risk_factor_count"],
            recommendation_count=result["risk_assessment"]["recommendation_count"],
            transparency_score=result["transparency_report"].get("transparency_score", 0.0),
            summary=result["explanation"]["summary"],
            detailed_explanation=result["explanation"]["detailed_explanation"],
            actionable_steps=result["explanation"]["actionable_steps"],
            disclaimers=result["disclaimers"]
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing visa application: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error occurred while processing application"
        )


@app.get("/api/v1/applications/{application_id}/status", response_model=ApplicationStatusResponse)
async def get_application_status(
    application_id: str,
    user_id: Optional[str] = Depends(get_current_user)
):
    """
    Get the current status and results of a visa application.
    
    Returns processing status, completed operations, and results if available.
    """
    if not orchestration_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Orchestration service not available"
        )
    
    try:
        # Convert string to UUID
        from uuid import UUID
        app_uuid = UUID(application_id)
        
        # Get application status
        status_result = await orchestration_service.get_application_status(app_uuid)
        
        if status_result.get("status") == "not_found":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Application not found"
            )
        
        return ApplicationStatusResponse(
            application_id=status_result["application_id"],
            status=status_result["status"],
            last_updated=status_result["last_updated"],
            operations_completed=status_result["operations_completed"],
            audit_completeness=status_result["audit_completeness"],
            processing_steps=status_result["processing_steps"]
        )
    
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid application ID format"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting application status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error occurred while retrieving status"
        )


@app.get("/api/v1/applications/{application_id}/audit")
async def get_application_audit_trail(
    application_id: str,
    user_id: Optional[str] = Depends(get_current_user)
):
    """
    Get the complete audit trail for an application.
    
    Returns all processing steps, data sources, and transparency information.
    """
    if not audit_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Audit service not available"
        )
    
    try:
        # Convert string to UUID
        from uuid import UUID
        app_uuid = UUID(application_id)
        
        # Get audit trail
        audit_entries = await audit_service.get_audit_trail(application_id=app_uuid)
        
        # Get transparency report
        transparency_report = await audit_service.generate_transparency_report(app_uuid)
        
        return {
            "application_id": application_id,
            "audit_entries": [
                {
                    "id": str(entry.id),
                    "timestamp": entry.timestamp.isoformat(),
                    "operation": entry.operation,
                    "service": entry.service,
                    "success": entry.success,
                    "execution_time_ms": entry.execution_time_ms,
                    "trace_id": entry.trace_id
                }
                for entry in audit_entries
            ],
            "transparency_report": transparency_report
        }
    
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid application ID format"
        )
    except Exception as e:
        logger.error(f"Error getting audit trail: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error occurred while retrieving audit trail"
        )


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with proper logging."""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail} - {request.method} {request.url}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "timestamp": datetime.now().isoformat(),
                "request_id": getattr(request.state, "request_id", None)
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions with proper logging."""
    logger.error(f"Unhandled exception: {str(exc)} - {request.method} {request.url}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": 500,
                "message": "Internal server error",
                "timestamp": datetime.now().isoformat(),
                "request_id": getattr(request.state, "request_id", None)
            }
        }
    )


# Additional utility endpoints
@app.get("/api/v1/countries")
async def get_supported_countries():
    """Get list of supported countries for visa applications."""
    # This would typically come from a database or configuration
    return {
        "countries": [
            {"code": "US", "name": "United States"},
            {"code": "CA", "name": "Canada"},
            {"code": "GB", "name": "United Kingdom"},
            {"code": "AU", "name": "Australia"},
            {"code": "DE", "name": "Germany"},
            {"code": "FR", "name": "France"},
            {"code": "JP", "name": "Japan"},
            {"code": "SG", "name": "Singapore"}
        ]
    }


@app.get("/api/v1/visa-types")
async def get_supported_visa_types():
    """Get list of supported visa types."""
    return {
        "visa_types": [
            {"code": "tourist", "name": "Tourist/Visitor Visa"},
            {"code": "business", "name": "Business Visa"},
            {"code": "work", "name": "Work Visa"},
            {"code": "student", "name": "Student Visa"},
            {"code": "family", "name": "Family/Spouse Visa"},
            {"code": "transit", "name": "Transit Visa"}
        ]
    }


@app.get("/api/v1/document-types")
async def get_supported_document_types():
    """Get list of supported document types for upload."""
    return {
        "document_types": [
            {"type": "passport", "name": "Passport", "required": True},
            {"type": "visa_application", "name": "Visa Application Form", "required": True},
            {"type": "employment_letter", "name": "Employment Letter", "required": False},
            {"type": "bank_statement", "name": "Bank Statement", "required": False},
            {"type": "educational_certificate", "name": "Educational Certificate", "required": False},
            {"type": "medical_certificate", "name": "Medical Certificate", "required": False}
        ],
        "supported_formats": settings.allowed_file_types,
        "max_file_size_mb": settings.max_file_size // (1024 * 1024),
        "max_files_per_request": settings.max_files_per_request
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.visaverse.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="info"
    )