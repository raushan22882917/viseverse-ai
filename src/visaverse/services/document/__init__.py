"""
Document processing services.
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class PaddleOCRDocumentService:
    """
    Lightweight wrapper for PaddleOCR document service.
    Delays heavy initialization until first use.
    """
    
    def __init__(self):
        """Initialize service without loading PaddleOCR yet."""
        self._processor = None
        self._initialized = False
        logger.info("PaddleOCR document service created (not yet initialized)")
    
    def _ensure_initialized(self):
        """Ensure the processor is initialized."""
        if not self._initialized:
            logger.info("Initializing PaddleOCR processor...")
            from .processor import PaddleOCRDocumentProcessor
            self._processor = PaddleOCRDocumentProcessor()
            self._initialized = True
            logger.info("PaddleOCR processor initialized")
    
    async def extract_text_and_data(
        self, 
        document_bytes: bytes, 
        filename: str, 
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract text and structured data from document."""
        self._ensure_initialized()
        
        # For now, return a simplified result until the full processor is ready
        if self._processor:
            return await self._processor.process_document(
                document_bytes=document_bytes,
                filename=filename,
                language=language
            )
        else:
            # Fallback response
            return {
                "text": f"Document processing for {filename} is initializing...",
                "confidence": 0.0,
                "fields": {},
                "processing_time_ms": 0,
                "status": "initializing"
            }
    
    @property
    def is_ready(self) -> bool:
        """Check if the service is ready to process documents."""
        return self._initialized and self._processor is not None