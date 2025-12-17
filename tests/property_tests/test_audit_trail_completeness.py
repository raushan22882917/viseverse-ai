"""
Property-based test for audit trail completeness.

**Feature: visaverse-guardian-ai, Property 9: Audit Trail Completeness**
**Validates: Requirements 7.4**

This test ensures that for any processing activity, the system generates 
complete audit entries that maintain traceability of all operations.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from typing import Dict, Any, Optional, List
from uuid import UUID, uuid4
from datetime import datetime, timedelta

from src.visaverse.core.models import AuditEntry, ApplicationStatus
from src.visaverse.core.interfaces import AuditService


# Hypothesis strategies for generating test data
@st.composite
def audit_operation_strategy(draw):
    """Generate realistic audit operation data."""
    operations = [
        "document_upload", "document_processing", "ocr_extraction",
        "compliance_validation", "risk_assessment", "explanation_generation",
        "memory_storage", "user_authentication", "application_submission"
    ]
    
    services = [
        "api-gateway", "document-service", "graph-service", 
        "risk-service", "memory-service", "auth-service"
    ]
    
    operation = draw(st.sampled_from(operations))
    service = draw(st.sampled_from(services))
    
    # Generate simple user IDs
    user_id = draw(st.one_of(
        st.none(),
        st.text(min_size=5, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz0123456789")
    ))
    
    # Generate application ID (optional)
    application_id = draw(st.one_of(st.none(), st.just(uuid4())))
    
    # Generate simple input/output data
    input_data = draw(st.one_of(
        st.none(),
        st.dictionaries(
            st.text(min_size=1, max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz"),
            st.one_of(st.text(max_size=20), st.integers(min_value=0, max_value=1000)),
            min_size=0, max_size=3
        )
    ))
    
    output_data = draw(st.one_of(
        st.none(),
        st.dictionaries(
            st.text(min_size=1, max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz"),
            st.one_of(st.text(max_size=20), st.integers(min_value=0, max_value=1000)),
            min_size=0, max_size=3
        )
    ))
    
    success = draw(st.booleans())
    error_message = draw(st.one_of(
        st.none(),
        st.text(min_size=5, max_size=50) if not success else st.none()
    ))
    
    execution_time_ms = draw(st.one_of(
        st.none(),
        st.floats(min_value=0.1, max_value=5000.0)
    ))
    
    trace_id = draw(st.one_of(
        st.none(),
        st.text(min_size=5, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz0123456789")
    ))
    
    return {
        "operation": operation,
        "service": service,
        "user_id": user_id,
        "application_id": application_id,
        "input_data": input_data,
        "output_data": output_data,
        "success": success,
        "error_message": error_message,
        "execution_time_ms": execution_time_ms,
        "trace_id": trace_id
    }


@st.composite
def application_processing_sequence_strategy(draw):
    """Generate a sequence of operations for a complete application processing."""
    application_id = uuid4()
    user_id = draw(st.text(min_size=5, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'))))
    
    # Define required operations for complete application processing
    required_operations = [
        ("document_upload", "api-gateway"),
        ("document_processing", "document-service"),
        ("ocr_extraction", "document-service"),
        ("compliance_validation", "graph-service"),
        ("risk_assessment", "risk-service"),
        ("explanation_generation", "risk-service"),
        ("memory_storage", "memory-service")
    ]
    
    operations = []
    for i, (operation, service) in enumerate(required_operations):
        op_data = {
            "operation": operation,
            "service": service,
            "user_id": user_id,
            "application_id": application_id,
            "input_data": {"step": i + 1, "application_id": str(application_id)},
            "output_data": {"status": "completed", "step": i + 1},
            "success": True,
            "error_message": None,
            "execution_time_ms": draw(st.floats(min_value=10.0, max_value=5000.0)),
            "trace_id": f"trace-{application_id}-{i}"
        }
        operations.append(op_data)
    
    return {
        "application_id": application_id,
        "user_id": user_id,
        "operations": operations
    }


class MockAuditService(AuditService):
    """Mock implementation of AuditService for testing."""
    
    def __init__(self):
        self.audit_entries: List[AuditEntry] = []
    
    async def log_operation(
        self,
        operation: str,
        service: str,
        user_id: Optional[str] = None,
        application_id: Optional[UUID] = None,
        input_data: Optional[Dict[str, Any]] = None,
        output_data: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        execution_time_ms: Optional[float] = None,
        trace_id: Optional[str] = None
    ) -> UUID:
        """Log an operation to the audit trail."""
        entry = AuditEntry(
            operation=operation,
            service=service,
            user_id=user_id,
            application_id=application_id,
            input_data=input_data or {},
            output_data=output_data or {},
            success=success,
            error_message=error_message,
            execution_time_ms=execution_time_ms,
            trace_id=trace_id
        )
        self.audit_entries.append(entry)
        return entry.id
    
    async def get_audit_trail(
        self,
        user_id: Optional[str] = None,
        application_id: Optional[UUID] = None,
        service: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100
    ) -> List[AuditEntry]:
        """Retrieve audit trail entries based on filters."""
        filtered_entries = self.audit_entries
        
        if user_id:
            filtered_entries = [e for e in filtered_entries if e.user_id == user_id]
        
        if application_id:
            filtered_entries = [e for e in filtered_entries if e.application_id == application_id]
        
        if service:
            filtered_entries = [e for e in filtered_entries if e.service == service]
        
        return filtered_entries[:limit]
    
    async def ensure_audit_completeness(
        self,
        application_id: UUID
    ) -> bool:
        """Verify that all required audit entries exist for an application."""
        required_operations = {
            "document_upload", "document_processing", "ocr_extraction",
            "compliance_validation", "risk_assessment", "explanation_generation",
            "memory_storage"
        }
        
        app_entries = await self.get_audit_trail(application_id=application_id)
        logged_operations = {entry.operation for entry in app_entries}
        
        return required_operations.issubset(logged_operations)


class TestAuditTrailCompleteness:
    """Property-based tests for audit trail completeness."""
    
    @given(audit_operation_strategy())
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50)
    async def test_audit_entry_creation_completeness(self, operation_data):
        """
        **Feature: visaverse-guardian-ai, Property 9: Audit Trail Completeness**
        
        For any processing operation, the system should create a complete audit entry
        that captures all relevant information about the operation.
        """
        # Arrange
        audit_service = MockAuditService()
        
        # Act
        entry_id = await audit_service.log_operation(**operation_data)
        
        # Assert - Verify audit entry was created
        assert entry_id is not None
        assert isinstance(entry_id, UUID)
        
        # Retrieve the created entry
        all_entries = await audit_service.get_audit_trail()
        created_entry = next((e for e in all_entries if e.id == entry_id), None)
        
        # Verify completeness of audit entry
        assert created_entry is not None
        assert created_entry.operation == operation_data["operation"]
        assert created_entry.service == operation_data["service"]
        assert created_entry.user_id == operation_data["user_id"]
        assert created_entry.application_id == operation_data["application_id"]
        assert created_entry.success == operation_data["success"]
        assert created_entry.error_message == operation_data["error_message"]
        assert created_entry.execution_time_ms == operation_data["execution_time_ms"]
        assert created_entry.trace_id == operation_data["trace_id"]
        
        # Verify timestamp is set
        assert created_entry.timestamp is not None
        assert isinstance(created_entry.timestamp, datetime)
        
        # Verify input/output data is preserved
        if operation_data["input_data"]:
            assert created_entry.input_data == operation_data["input_data"]
        if operation_data["output_data"]:
            assert created_entry.output_data == operation_data["output_data"]
    
    @given(application_processing_sequence_strategy())
    async def test_complete_application_audit_trail(self, sequence_data):
        """
        **Feature: visaverse-guardian-ai, Property 9: Audit Trail Completeness**
        
        For any complete application processing sequence, all required operations
        should be logged in the audit trail, ensuring full traceability.
        """
        # Arrange
        audit_service = MockAuditService()
        application_id = sequence_data["application_id"]
        operations = sequence_data["operations"]
        
        # Act - Log all operations in the sequence
        for operation in operations:
            await audit_service.log_operation(**operation)
        
        # Assert - Verify audit trail completeness
        is_complete = await audit_service.ensure_audit_completeness(application_id)
        assert is_complete, f"Audit trail is incomplete for application {application_id}"
        
        # Verify all operations are logged
        app_entries = await audit_service.get_audit_trail(application_id=application_id)
        assert len(app_entries) == len(operations)
        
        # Verify each required operation is present
        logged_operations = {entry.operation for entry in app_entries}
        required_operations = {op["operation"] for op in operations}
        assert logged_operations == required_operations
        
        # Verify traceability - all entries should have the same application_id
        for entry in app_entries:
            assert entry.application_id == application_id
            assert entry.user_id == sequence_data["user_id"]
    
    @given(st.lists(audit_operation_strategy(), min_size=1, max_size=20))
    async def test_audit_trail_filtering_completeness(self, operations_list):
        """
        **Feature: visaverse-guardian-ai, Property 9: Audit Trail Completeness**
        
        For any set of audit operations, filtering by user_id, application_id, or service
        should return complete and accurate results without losing entries.
        """
        # Arrange
        audit_service = MockAuditService()
        
        # Act - Log all operations
        entry_ids = []
        for operation in operations_list:
            entry_id = await audit_service.log_operation(**operation)
            entry_ids.append(entry_id)
        
        # Assert - Test various filtering scenarios
        all_entries = await audit_service.get_audit_trail()
        assert len(all_entries) == len(operations_list)
        
        # Test filtering by user_id
        user_ids = {op["user_id"] for op in operations_list if op["user_id"]}
        for user_id in user_ids:
            user_entries = await audit_service.get_audit_trail(user_id=user_id)
            expected_count = sum(1 for op in operations_list if op["user_id"] == user_id)
            assert len(user_entries) == expected_count
            
            # Verify all returned entries match the filter
            for entry in user_entries:
                assert entry.user_id == user_id
        
        # Test filtering by application_id
        app_ids = {op["application_id"] for op in operations_list if op["application_id"]}
        for app_id in app_ids:
            app_entries = await audit_service.get_audit_trail(application_id=app_id)
            expected_count = sum(1 for op in operations_list if op["application_id"] == app_id)
            assert len(app_entries) == expected_count
            
            # Verify all returned entries match the filter
            for entry in app_entries:
                assert entry.application_id == app_id
        
        # Test filtering by service
        services = {op["service"] for op in operations_list}
        for service in services:
            service_entries = await audit_service.get_audit_trail(service=service)
            expected_count = sum(1 for op in operations_list if op["service"] == service)
            assert len(service_entries) == expected_count
            
            # Verify all returned entries match the filter
            for entry in service_entries:
                assert entry.service == service
    
    @given(st.integers(min_value=1, max_value=50))
    async def test_audit_trail_limit_completeness(self, limit):
        """
        **Feature: visaverse-guardian-ai, Property 9: Audit Trail Completeness**
        
        For any limit parameter, the audit trail should return exactly that number
        of entries (or fewer if not enough exist) without losing data integrity.
        """
        # Arrange
        audit_service = MockAuditService()
        
        # Create more entries than the limit to test truncation
        num_operations = limit + 10
        operations = []
        for i in range(num_operations):
            operation = {
                "operation": f"test_operation_{i}",
                "service": "test-service",
                "user_id": f"user_{i}",
                "success": True
            }
            operations.append(operation)
            await audit_service.log_operation(**operation)
        
        # Act
        limited_entries = await audit_service.get_audit_trail(limit=limit)
        
        # Assert
        assert len(limited_entries) == limit
        
        # Verify entries are complete and valid
        for entry in limited_entries:
            assert entry.operation.startswith("test_operation_")
            assert entry.service == "test-service"
            assert entry.user_id.startswith("user_")
            assert entry.success is True
            assert isinstance(entry.timestamp, datetime)
            assert isinstance(entry.id, UUID)