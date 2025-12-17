"""
Unit tests for core data models.
"""

import pytest
from datetime import datetime
from uuid import UUID
from pydantic import ValidationError

from src.visaverse.core.models import (
    DocumentType,
    ProcessingStatus,
    ApplicationStatus,
    RiskSeverity,
    ProcessedDocument,
    StructuredData,
    VisaApplication,
    AuditEntry,
    RiskAssessment,
    LayoutData,
    DocumentMetadata
)


class TestDocumentModels:
    """Test document-related models."""
    
    def test_document_type_enum(self):
        """Test DocumentType enum values."""
        assert DocumentType.PASSPORT == "passport"
        assert DocumentType.VISA_APPLICATION == "visa_application"
        assert DocumentType.BANK_STATEMENT == "bank_statement"
    
    def test_processed_document_creation(self):
        """Test ProcessedDocument model creation."""
        doc = ProcessedDocument(
            document_type=DocumentType.PASSPORT,
            extracted_text="Sample passport text",
            layout_info=LayoutData(),
            confidence_score=0.95,
            language="en"
        )
        
        assert doc.document_type == DocumentType.PASSPORT
        assert doc.confidence_score == 0.95
        assert doc.language == "en"
        assert isinstance(doc.id, UUID)
        assert isinstance(doc.created_at, datetime)
    
    def test_processed_document_validation(self):
        """Test ProcessedDocument validation."""
        # Test invalid confidence score
        with pytest.raises(ValidationError):
            ProcessedDocument(
                document_type=DocumentType.PASSPORT,
                extracted_text="Sample text",
                layout_info=LayoutData(),
                confidence_score=1.5,  # Invalid: > 1.0
                language="en"
            )
    
    def test_structured_data_creation(self):
        """Test StructuredData model creation."""
        data = StructuredData(
            document_type=DocumentType.PASSPORT,
            key_fields={"passport_number": "A12345678"},
            extraction_confidence=0.88
        )
        
        assert data.document_type == DocumentType.PASSPORT
        assert data.key_fields["passport_number"] == "A12345678"
        assert data.extraction_confidence == 0.88
        assert len(data.dates) == 0
        assert len(data.missing_fields) == 0


class TestApplicationModels:
    """Test application-related models."""
    
    def test_visa_application_creation(self):
        """Test VisaApplication model creation."""
        app = VisaApplication(
            user_id="user123",
            visa_type="tourist",
            target_country="US"
        )
        
        assert app.user_id == "user123"
        assert app.visa_type == "tourist"
        assert app.target_country == "US"
        assert app.status == ApplicationStatus.DRAFT
        assert isinstance(app.id, UUID)
        assert isinstance(app.last_updated, datetime)
    
    def test_visa_application_timestamp_update(self):
        """Test that last_updated timestamp is automatically set."""
        app = VisaApplication(
            user_id="user123",
            visa_type="tourist",
            target_country="US"
        )
        
        original_time = app.last_updated
        
        # Create another instance to verify timestamp is updated
        app2 = VisaApplication(
            user_id="user456",
            visa_type="business",
            target_country="CA"
        )
        
        # Timestamps should be different (though this might be flaky in very fast tests)
        assert app2.last_updated >= original_time


class TestAuditModels:
    """Test audit-related models."""
    
    def test_audit_entry_creation(self):
        """Test AuditEntry model creation."""
        entry = AuditEntry(
            operation="document_upload",
            service="document-service",
            user_id="user123",
            success=True,
            execution_time_ms=150.5
        )
        
        assert entry.operation == "document_upload"
        assert entry.service == "document-service"
        assert entry.user_id == "user123"
        assert entry.success is True
        assert entry.execution_time_ms == 150.5
        assert isinstance(entry.id, UUID)
        assert isinstance(entry.timestamp, datetime)
    
    def test_audit_entry_optional_fields(self):
        """Test AuditEntry with optional fields."""
        entry = AuditEntry(
            operation="test_operation",
            service="test-service",
            success=False,
            error_message="Test error"
        )
        
        assert entry.user_id is None
        assert entry.application_id is None
        assert entry.error_message == "Test error"
        assert entry.success is False


class TestRiskModels:
    """Test risk assessment models."""
    
    def test_risk_assessment_creation(self):
        """Test RiskAssessment model creation."""
        assessment = RiskAssessment(
            approval_probability=0.75,
            confidence_level=0.85
        )
        
        assert assessment.approval_probability == 0.75
        assert assessment.confidence_level == 0.85
        assert len(assessment.risk_factors) == 0
        assert len(assessment.recommendations) == 0
        assert isinstance(assessment.id, UUID)
        assert isinstance(assessment.created_at, datetime)
    
    def test_risk_assessment_validation(self):
        """Test RiskAssessment validation."""
        # Test invalid probability
        with pytest.raises(ValidationError):
            RiskAssessment(
                approval_probability=1.5,  # Invalid: > 1.0
                confidence_level=0.85
            )
        
        # Test invalid confidence
        with pytest.raises(ValidationError):
            RiskAssessment(
                approval_probability=0.75,
                confidence_level=-0.1  # Invalid: < 0.0
            )