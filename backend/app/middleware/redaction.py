"""
Middleware for automatic data redaction in API responses.

This middleware automatically redacts sensitive information from all API responses
before they are sent to clients, ensuring data protection by design.
"""
import json
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.utils import redact_dict, RedactionConfig, PRESIDIO_AVAILABLE

logger = logging.getLogger(__name__)


class RedactionMiddleware(BaseHTTPMiddleware):
    """
    Middleware that automatically redacts sensitive data from API responses.
    
    This middleware intercepts all responses and applies Presidio-based
    redaction to JSON payloads before sending them to clients.
    
    Features:
    - Automatic PII detection and redaction in response bodies
    - Preserves response structure and types
    - Minimal performance overhead
    - Can be configured per-endpoint using path patterns
    """
    
    def __init__(
        self,
        app: ASGIApp,
        enabled: bool = True,
        config: RedactionConfig = None,
        skip_paths: list = None,
    ):
        """
        Initialize redaction middleware.
        
        Args:
            app: ASGI application
            enabled: Whether redaction is enabled
            config: Redaction configuration (uses default if None)
            skip_paths: List of path patterns to skip redaction (e.g., ['/health', '/docs'])
        """
        super().__init__(app)
        self.enabled = enabled and PRESIDIO_AVAILABLE
        self.config = config or RedactionConfig(
            enabled=True,
            score_threshold=0.35,
            show_entity_type=False,
        )
        self.skip_paths = skip_paths or [
            '/health',
            '/docs',
            '/redoc',
            '/openapi.json',
            '/api/assets',  # Assets are already redacted at source
        ]
        
        if not PRESIDIO_AVAILABLE and enabled:
            logger.warning(
                "RedactionMiddleware enabled but Presidio not available. "
                "Responses will NOT be redacted."
            )
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request and redact the response.
        
        Args:
            request: Incoming request
            call_next: Next middleware/handler in chain
            
        Returns:
            Response with redacted content (if applicable)
        """
        # Get the response from the next handler
        response = await call_next(request)
        
        # Skip redaction if disabled or path is in skip list
        if not self.enabled:
            return response
        
        request_path = request.url.path
        if any(skip in request_path for skip in self.skip_paths):
            return response
        
        # Only process JSON responses
        content_type = response.headers.get('content-type', '')
        if 'application/json' not in content_type:
            return response
        
        # Read the response body
        try:
            body_bytes = b""
            async for chunk in response.body_iterator:
                body_bytes += chunk
            
            # Parse JSON
            body_json = json.loads(body_bytes.decode())
            
            # Apply redaction
            if isinstance(body_json, dict):
                redacted_json = redact_dict(body_json, self.config)
            elif isinstance(body_json, list):
                # Handle list of dicts
                redacted_json = [
                    redact_dict(item, self.config) if isinstance(item, dict) else item
                    for item in body_json
                ]
            else:
                # Not a structure we can redact
                redacted_json = body_json
            
            # Create new response with redacted content
            redacted_body = json.dumps(redacted_json).encode()
            
            # Preserve original headers but update content length
            headers = dict(response.headers)
            headers['content-length'] = str(len(redacted_body))
            
            return Response(
                content=redacted_body,
                status_code=response.status_code,
                headers=headers,
                media_type=response.media_type,
            )
            
        except Exception as e:
            # If redaction fails, log error and return original response
            logger.error(f"Failed to redact response for {request_path}: {e}")
            
            # Reconstruct response with original body
            return Response(
                content=body_bytes,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type,
            )


def configure_redaction_middleware(app, enabled: bool = True):
    """
    Helper function to add redaction middleware to FastAPI app.
    
    Args:
        app: FastAPI application instance
        enabled: Whether to enable redaction
    """
    app.add_middleware(
        RedactionMiddleware,
        enabled=enabled,
        config=RedactionConfig(
            enabled=enabled,
            score_threshold=0.35,
            show_entity_type=False,
        )
    )
    
    if enabled and PRESIDIO_AVAILABLE:
        logger.info("✅ Redaction middleware enabled - All API responses will be sanitized")
    elif enabled and not PRESIDIO_AVAILABLE:
        logger.warning("⚠️  Redaction middleware requested but Presidio not installed")
    else:
        logger.info("ℹ️  Redaction middleware disabled")
