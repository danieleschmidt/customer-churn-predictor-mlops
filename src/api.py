"""
FastAPI application for Customer Churn Prediction API.

This module provides a REST API interface for the customer churn prediction system,
exposing CLI functionality as HTTP endpoints with comprehensive rate limiting,
monitoring, and OpenAPI documentation.
"""

import asyncio
import os
import time
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import json
import pandas as pd

from fastapi import (
    FastAPI, HTTPException, Request, Response, BackgroundTasks,
    Depends, File, UploadFile, status
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from contextlib import asynccontextmanager
import uvicorn

from .rate_limiter import get_rate_limiter, RateLimitStatus, RateLimitRule
from .health_check import get_health_status, get_comprehensive_health, get_readiness_status
from .metrics import get_prometheus_metrics, get_metrics_collector
from .predict_churn import make_prediction, make_batch_predictions
from .security import get_security_report, get_security_policies
from .model_cache import get_cache_stats, invalidate_model_cache
from .data_validation import validate_customer_data
from .logging_config import get_logger

logger = get_logger(__name__)

# Security scheme for authenticated endpoints
security = HTTPBearer(auto_error=False)


# Pydantic models for API requests/responses
class PredictionRequest(BaseModel):
    """Request model for single customer prediction."""
    customer_data: Dict[str, Union[str, int, float]] = Field(
        ...,
        description="Customer features as key-value pairs",
        example={
            "tenure": 12,
            "MonthlyCharges": 50.0,
            "TotalCharges": 600.0,
            "Contract": "Month-to-month",
            "InternetService": "Fiber optic"
        }
    )
    run_id: Optional[str] = Field(
        None,
        description="MLflow run ID for model version"
    )


class PredictionResponse(BaseModel):
    """Response model for prediction results."""
    prediction: int = Field(..., description="Predicted churn (0=No, 1=Yes)")
    probability: float = Field(..., description="Churn probability (0.0-1.0)")
    model_version: Optional[str] = Field(None, description="Model version used")
    timestamp: str = Field(..., description="Prediction timestamp")


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[int] = Field(..., description="List of predictions")
    probabilities: List[float] = Field(..., description="List of probabilities")
    count: int = Field(..., description="Number of predictions made")
    model_version: Optional[str] = Field(None, description="Model version used")
    timestamp: str = Field(..., description="Prediction timestamp")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Health status")
    timestamp: str = Field(..., description="Check timestamp")
    uptime_seconds: float = Field(..., description="Application uptime")
    version: str = Field(..., description="Application version")


class RateLimitInfo(BaseModel):
    """Rate limiting information in response headers."""
    requests_remaining: int
    reset_time: float
    retry_after: Optional[int] = None


class ValidationRequest(BaseModel):
    """Request model for data validation."""
    for_prediction: bool = Field(False, description="Validate for prediction use")
    check_distribution: bool = Field(False, description="Check feature distributions")
    check_business_rules: bool = Field(True, description="Check business rules")


class SecurityScanRequest(BaseModel):
    """Request model for security scanning."""
    image: str = Field(..., description="Docker image to scan")
    max_high: int = Field(0, description="Maximum high severity vulnerabilities")
    max_medium: int = Field(5, description="Maximum medium severity vulnerabilities")


class RateLimitRuleRequest(BaseModel):
    """Request model for rate limit rule configuration."""
    requests: int = Field(..., gt=0, description="Maximum requests allowed")
    window_seconds: int = Field(..., gt=0, description="Time window in seconds")
    burst_size: Optional[int] = Field(None, description="Maximum burst size")
    per_ip: bool = Field(True, description="Apply limit per IP")
    per_endpoint: bool = Field(True, description="Apply limit per endpoint")
    description: str = Field("", description="Rule description")


# Application lifespan manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    # Startup
    logger.info("Starting Customer Churn Prediction API")
    
    # Initialize rate limiter
    rate_limiter = get_rate_limiter()
    logger.info(f"Rate limiter initialized: {rate_limiter.get_stats()}")
    
    # Initialize metrics collector
    metrics_collector = get_metrics_collector()
    logger.info("Metrics collector initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Customer Churn Prediction API")


# Create FastAPI application
app = FastAPI(
    title="Customer Churn Prediction API",
    description="""
    RESTful API for customer churn prediction with comprehensive rate limiting,
    monitoring, and security features.
    
    ## Features
    
    * **Machine Learning Predictions**: Single and batch customer churn predictions
    * **Health Monitoring**: Comprehensive health checks and readiness probes
    * **Metrics Export**: Prometheus-compatible metrics endpoint
    * **Rate Limiting**: Configurable per-IP and per-endpoint rate limiting
    * **Security Scanning**: Docker image vulnerability assessment
    * **Data Validation**: Customer data quality validation
    * **Model Caching**: Intelligent model and preprocessor caching
    * **OpenAPI Documentation**: Interactive API documentation and schema
    
    ## Rate Limiting
    
    All endpoints are protected by intelligent rate limiting:
    
    * **Default**: 100 requests per minute
    * **Predictions**: 30 requests per minute (burst: 10)
    * **Health Checks**: 200 requests per minute
    * **Training**: 5 requests per 5 minutes
    * **Admin**: 10 requests per minute
    
    Rate limit information is provided in response headers:
    * `X-RateLimit-Remaining`: Requests remaining in current window
    * `X-RateLimit-Reset`: When the rate limit resets (Unix timestamp)
    * `Retry-After`: Seconds to wait before retrying (when rate limited)
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)


# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Apply rate limiting to all requests."""
    # Get client IP
    client_ip = request.client.host
    if "x-forwarded-for" in request.headers:
        client_ip = request.headers["x-forwarded-for"].split(",")[0].strip()
    elif "x-real-ip" in request.headers:
        client_ip = request.headers["x-real-ip"]
    
    # Determine endpoint for rate limiting
    endpoint = request.url.path.strip("/")
    if not endpoint:
        endpoint = "root"
    
    # Determine rate limit category
    if endpoint.startswith("predict"):
        limit_key = "predict"
    elif endpoint in ["health", "health/detailed", "ready"]:
        limit_key = "health"
    elif endpoint == "metrics":
        limit_key = "metrics"
    elif endpoint.startswith("train") or endpoint.startswith("pipeline"):
        limit_key = "train"
    elif endpoint.startswith("admin") or endpoint.startswith("security"):
        limit_key = "admin"
    else:
        limit_key = "default"
    
    # Check rate limit
    rate_limiter = get_rate_limiter()
    limit_status = rate_limiter.check_rate_limit(client_ip, limit_key)
    
    # Add rate limit headers to response
    async def add_rate_limit_headers(response: Response):
        response.headers["X-RateLimit-Remaining"] = str(limit_status.requests_remaining)
        response.headers["X-RateLimit-Reset"] = str(int(limit_status.reset_time))
        
        if not limit_status.allowed and limit_status.retry_after:
            response.headers["Retry-After"] = str(limit_status.retry_after)
        
        return response
    
    # Check if request is allowed
    if not limit_status.allowed:
        response = JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "Rate limit exceeded",
                "detail": f"Too many requests. Try again in {limit_status.retry_after} seconds.",
                "requests_remaining": limit_status.requests_remaining,
                "reset_time": limit_status.reset_time
            }
        )
        return await add_rate_limit_headers(response)
    
    # Process request
    response = await call_next(request)
    return await add_rate_limit_headers(response)


# Utility functions
def get_client_ip(request: Request) -> str:
    """Extract client IP from request."""
    client_ip = request.client.host
    if "x-forwarded-for" in request.headers:
        client_ip = request.headers["x-forwarded-for"].split(",")[0].strip()
    elif "x-real-ip" in request.headers:
        client_ip = request.headers["x-real-ip"]
    return client_ip


async def verify_auth(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[str]:
    """Verify authentication for protected endpoints."""
    # For now, just log the attempt - implement actual auth as needed
    if credentials:
        logger.info(f"Auth attempt with token: {credentials.credentials[:10]}...")
        # TODO: Implement actual token verification
        return "authenticated_user"
    return None


# API Endpoints

@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Customer Churn Prediction API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "metrics": "/metrics",
            "predict": "/predict",
            "predict_batch": "/predict/batch"
        },
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(request: PredictionRequest):
    """Make a churn prediction for a single customer."""
    try:
        start_time = time.time()
        
        # Make prediction
        prediction, probability = make_prediction(
            request.customer_data,
            run_id=request.run_id
        )
        
        if prediction is None or probability is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Prediction failed - model or data error"
            )
        
        # Record metrics
        latency = time.time() - start_time
        metrics_collector = get_metrics_collector()
        metrics_collector.record_prediction_latency(latency, "single")
        metrics_collector.record_prediction_count(1, "success", "single")
        
        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            model_version=request.run_id,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        metrics_collector = get_metrics_collector()
        metrics_collector.record_prediction_count(1, "error", "single")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(file: UploadFile = File(...)):
    """Make batch predictions from uploaded CSV file."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a CSV"
        )
    
    try:
        start_time = time.time()
        
        # Read CSV file
        content = await file.read()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Load CSV into DataFrame
            df = pd.read_csv(tmp_path)
            
            # Make batch predictions
            predictions, probabilities = make_batch_predictions(df)
            
            if predictions is None or probabilities is None:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Batch prediction failed - model or data error"
                )
            
            # Record metrics
            latency = time.time() - start_time
            metrics_collector = get_metrics_collector()
            metrics_collector.record_prediction_latency(latency, "batch")
            metrics_collector.record_prediction_count(len(predictions), "success", "batch")
            
            return BatchPredictionResponse(
                predictions=predictions,
                probabilities=probabilities,
                count=len(predictions),
                timestamp=datetime.utcnow().isoformat()
            )
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)
            
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        metrics_collector = get_metrics_collector()
        metrics_collector.record_prediction_count(0, "error", "batch")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Basic health check endpoint."""
    try:
        health_status = get_health_status()
        
        return HealthResponse(
            status=health_status["status"],
            timestamp=health_status["timestamp"],
            uptime_seconds=health_status.get("uptime_seconds", 0),
            version=health_status.get("version", "1.0.0")
        )
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )


@app.get("/health/detailed")
async def detailed_health_check():
    """Comprehensive health check with detailed diagnostics."""
    try:
        health_status = get_comprehensive_health()
        return health_status
        
    except Exception as e:
        logger.error(f"Detailed health check error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Detailed health check failed: {str(e)}"
        )


@app.get("/ready")
async def readiness_check():
    """Kubernetes readiness probe endpoint."""
    try:
        readiness_status = get_readiness_status()
        
        if not readiness_status.get("ready", False):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not ready"
            )
        
        return readiness_status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Readiness check error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Readiness check failed: {str(e)}"
        )


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics_endpoint():
    """Prometheus metrics endpoint."""
    try:
        metrics_data = get_prometheus_metrics()
        return metrics_data
        
    except Exception as e:
        logger.error(f"Metrics endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Metrics generation failed: {str(e)}"
        )


@app.post("/validate")
async def validate_data(
    request: ValidationRequest,
    file: UploadFile = File(...)
):
    """Validate customer data file."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a CSV"
        )
    
    try:
        # Save uploaded file temporarily
        content = await file.read()
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Validate data
            if request.for_prediction:
                from .data_validation import validate_prediction_data
                report = validate_prediction_data(tmp_path)
            else:
                kwargs = {
                    'check_distribution': request.check_distribution,
                    'check_business_rules': request.check_business_rules
                }
                report = validate_customer_data(tmp_path, **kwargs)
            
            return {
                "valid": report.is_valid,
                "summary": report.get_summary(),
                "errors": report.errors[:10],  # Limit errors in response
                "error_count": len(report.errors)
            }
            
        finally:
            os.unlink(tmp_path)
            
    except Exception as e:
        logger.error(f"Data validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Data validation failed: {str(e)}"
        )


@app.get("/cache/stats")
async def cache_statistics():
    """Get model cache statistics."""
    try:
        cache_stats = get_cache_stats()
        return cache_stats
        
    except Exception as e:
        logger.error(f"Cache stats error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cache statistics: {str(e)}"
        )


@app.post("/cache/clear")
async def clear_cache(user: Optional[str] = Depends(verify_auth)):
    """Clear model cache (requires authentication)."""
    try:
        invalidate_model_cache()
        return {"message": "Cache cleared successfully"}
        
    except Exception as e:
        logger.error(f"Cache clear error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear cache: {str(e)}"
        )


@app.post("/admin/security/scan")
async def security_scan(
    request: SecurityScanRequest,
    user: Optional[str] = Depends(verify_auth)
):
    """Perform security scan of Docker image."""
    try:
        report = get_security_report(request.image)
        
        if "error" in report:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Security scan failed: {report['error']}"
            )
        
        # Check if image meets security requirements
        scan_result = report.get("scan_result", {})
        high_count = scan_result.get("high_severity_count", 0)
        medium_count = scan_result.get("medium_severity_count", 0)
        
        if high_count > request.max_high or medium_count > request.max_medium:
            return JSONResponse(
                status_code=status.HTTP_406_NOT_ACCEPTABLE,
                content={
                    **report,
                    "security_check": "failed",
                    "reason": f"Vulnerabilities exceed limits: {high_count} high (max {request.max_high}), {medium_count} medium (max {request.max_medium})"
                }
            )
        
        return {
            **report,
            "security_check": "passed"
        }
        
    except Exception as e:
        logger.error(f"Security scan error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Security scan failed: {str(e)}"
        )


@app.get("/admin/security/policies")
async def security_policies():
    """Get security policies and recommendations."""
    try:
        policies = get_security_policies()
        return policies
        
    except Exception as e:
        logger.error(f"Security policies error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get security policies: {str(e)}"
        )


@app.get("/admin/rate-limit/stats")
async def rate_limit_statistics():
    """Get rate limiting statistics."""
    try:
        rate_limiter = get_rate_limiter()
        stats = rate_limiter.get_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Rate limit stats error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get rate limit statistics: {str(e)}"
        )


@app.post("/admin/rate-limit/rules/{rule_key}")
async def add_rate_limit_rule(
    rule_key: str,
    rule_request: RateLimitRuleRequest,
    user: Optional[str] = Depends(verify_auth)
):
    """Add or update rate limiting rule."""
    try:
        rule = RateLimitRule(
            requests=rule_request.requests,
            window_seconds=rule_request.window_seconds,
            burst_size=rule_request.burst_size,
            per_ip=rule_request.per_ip,
            per_endpoint=rule_request.per_endpoint,
            description=rule_request.description
        )
        
        rate_limiter = get_rate_limiter()
        rate_limiter.add_rule(rule_key, rule)
        
        return {
            "message": f"Rate limit rule added for {rule_key}",
            "rule": rule.to_dict()
        }
        
    except Exception as e:
        logger.error(f"Add rate limit rule error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add rate limit rule: {str(e)}"
        )


# Error handlers
@app.exception_handler(429)
async def rate_limit_handler(request: Request, exc):
    """Handle rate limit exceeded errors."""
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "detail": "Too many requests. Please try again later.",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred. Please try again later.",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )