"""
Property-based tests for compliance validation consistency.

**Feature: visaverse-guardian-ai, Property 2: Compliance Validation Consistency**
**Validates: Requirements 1.2, 5.1, 5.2, 5.4**

Property 2: Compliance Validation Consistency
For any extracted document data and visa type/country combination, the system should 
validate against compliance rules and produce consistent results with reasoning paths.
"""

import asyncio
import uuid
from typing import List, Dict, Any
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock

from hypothesis import given, strategies as st, assume, settings
import pytest

from src.visaverse.services.graph.reasoning_engine import Neo4jGraphReasoningService
from src.visaverse.core.models import (
    StructuredData,
    DocumentType,
    ComplianceResult,
    VisaRule,
    RuleType,
    RuleCondition,
    ComparisonOperator,
    Requirement,
    DateField,
    FinancialData
)


# Test data generators
@st.composite
def generate_structured_data(draw):
    """Generate realistic structured data for testing."""
    doc_type = draw(st.sampled_from(list(DocumentType)))
    
    # Generate key fields based on document type
    key_fields = {}
    dates = []
    financial_info = None
    
    if doc_type == DocumentType.PASSPORT:
        key_fields = {
            'passport_number': draw(st.text(alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', min_size=8, max_size=12)),
            'name': draw(st.text(alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ ', min_size=5, max_size=30)),
            'nationality': draw(st.sampled_from(['AMERICAN', 'BRITISH', 'CANADIAN', 'GERMAN', 'FRENCH', 'INDIAN']))
        }
        
        # Add dates
        birth_date = draw(st.datetimes(min_value=datetime(1950, 1, 1), max_value=datetime(2005, 12, 31)))
        issue_date = draw(st.datetimes(min_value=datetime(2015, 1, 1), max_value=datetime(2023, 12, 31)))
        expiry_date = issue_date + timedelta(days=draw(st.integers(1825, 3650)))  # 5-10 years
        
        dates = [
            DateField(field_name='date_of_birth', date_value=birth_date, confidence=0.9, format_detected='DD/MM/YYYY'),
            DateField(field_name='date_of_issue', date_value=issue_date, confidence=0.9, format_detected='DD/MM/YYYY'),
            DateField(field_name='date_of_expiry', date_value=expiry_date, confidence=0.9, format_detected='DD/MM/YYYY')
        ]
    
    elif doc_type == DocumentType.BANK_STATEMENT:
        key_fields = {
            'account_number': draw(st.text(alphabet='0123456789', min_size=10, max_size=15)),
            'account_holder': draw(st.text(alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ ', min_size=5, max_size=30))
        }
        
        # Add financial info
        amounts = []
        for _ in range(draw(st.integers(1, 5))):
            amounts.append({
                'amount': draw(st.floats(min_value=100, max_value=50000)),
                'currency': draw(st.sampled_from(['USD', 'EUR', 'GBP', 'CAD']))
            })
        
        financial_info = FinancialData(
            amounts=amounts,
            currency=amounts[0]['currency'] if amounts else 'USD',
            account_numbers=[key_fields['account_number']],
            transaction_count=len(amounts)
        )
    
    elif doc_type == DocumentType.EMPLOYMENT_LETTER:
        key_fields = {
            'employee_name': draw(st.text(alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ ', min_size=5, max_size=30)),
            'position': draw(st.sampled_from(['Software Engineer', 'Manager', 'Analyst', 'Director', 'Consultant'])),
            'salary': str(draw(st.integers(30000, 200000)))
        }
    
    return StructuredData(
        document_type=doc_type,
        key_fields=key_fields,
        dates=dates,
        financial_info=financial_info,
        missing_fields=draw(st.lists(st.text(min_size=1, max_size=20), max_size=3)),
        extraction_confidence=draw(st.floats(min_value=0.5, max_value=1.0))
    )


@st.composite
def generate_visa_rule(draw):
    """Generate a realistic visa rule for testing."""
    rule_id = uuid.uuid4()
    country = draw(st.sampled_from(['US', 'UK', 'CA', 'DE', 'FR', 'AU']))
    visa_type = draw(st.sampled_from(['Tourist', 'Business', 'Student', 'Work', 'Transit']))
    rule_type = draw(st.sampled_from(list(RuleType)))
    
    # Generate conditions
    conditions = []
    field_names = ['passport_number', 'name', 'nationality', 'salary', 'account_number']
    
    for _ in range(draw(st.integers(1, 3))):
        field = draw(st.sampled_from(field_names))
        operator = draw(st.sampled_from(list(ComparisonOperator)))
        
        if operator in [ComparisonOperator.EXISTS, ComparisonOperator.NOT_EXISTS]:
            value = None
        else:
            value = draw(st.text(min_size=1, max_size=20))
        
        conditions.append(RuleCondition(
            field=field,
            operator=operator,
            value=value,
            description=f"Condition for {field}"
        ))
    
    # Generate requirements
    requirements = []
    for _ in range(draw(st.integers(1, 2))):
        requirements.append(Requirement(
            id=uuid.uuid4(),
            name=f"Requirement {len(requirements) + 1}",
            description=f"Test requirement for {visa_type} visa",
            mandatory=draw(st.booleans())
        ))
    
    return VisaRule(
        id=rule_id,
        country=country,
        visa_type=visa_type,
        rule_type=rule_type,
        conditions=conditions,
        requirements=requirements,
        priority=draw(st.integers(1, 10)),
        description=f"Test rule for {visa_type} visa in {country}"
    )


class MockNeo4jSession:
    """Mock Neo4j session for testing."""
    
    def __init__(self, rules_data: List[Dict[str, Any]] = None):
        self.rules_data = rules_data or []
        self.queries_executed = []
    
    def run(self, query: str, **params):
        """Mock query execution."""
        self.queries_executed.append({'query': query, 'params': params})
        
        # Mock different query responses
        if 'MATCH (c:Country' in query and 'VisaType' in query:
            # Return mock rules
            return MockResult(self.rules_data)
        elif 'CREATE CONSTRAINT' in query or 'CREATE INDEX' in query:
            # Schema operations
            return MockResult([])
        elif 'RETURN 1' in query:
            # Connection test
            return MockResult([{'1': 1}])
        else:
            return MockResult([])
    
    def begin_transaction(self):
        """Mock transaction."""
        return self
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class MockResult:
    """Mock Neo4j result."""
    
    def __init__(self, records: List[Dict[str, Any]]):
        self.records = records
    
    def __iter__(self):
        for record in self.records:
            yield MockRecord(record)
    
    def single(self):
        return MockRecord(self.records[0]) if self.records else None


class MockRecord:
    """Mock Neo4j record."""
    
    def __init__(self, data: Dict[str, Any]):
        self.data = data
    
    def __getitem__(self, key):
        return self.data.get(key)
    
    def get(self, key, default=None):
        return self.data.get(key, default)


class TestComplianceValidationConsistency:
    """Test compliance validation consistency property."""
    
    @pytest.fixture
    def mock_reasoning_service(self):
        """Create a mock reasoning service for testing."""
        service = Neo4jGraphReasoningService("bolt://localhost:7687", "neo4j", "password")
        service._driver = Mock()
        return service
    
    @given(
        document_data=st.lists(generate_structured_data(), min_size=1, max_size=3),
        visa_type=st.sampled_from(['Tourist', 'Business', 'Student', 'Work']),
        country=st.sampled_from(['US', 'UK', 'CA', 'DE', 'FR'])
    )
    @settings(max_examples=50, deadline=10000)
    async def test_compliance_validation_consistency_property(
        self, 
        mock_reasoning_service, 
        document_data, 
        visa_type, 
        country
    ):
        """
        **Feature: visaverse-guardian-ai, Property 2: Compliance Validation Consistency**
        
        Property: For any extracted document data and visa type/country combination, 
        the system should validate against compliance rules and produce consistent 
        results with reasoning paths.
        
        **Validates: Requirements 1.2, 5.1, 5.2, 5.4**
        """
        # Mock the session to return consistent rule data
        mock_rules = [
            {
                'r': {
                    'id': str(uuid.uuid4()),
                    'name': f'{visa_type}_rule_1',
                    'description': f'Test rule for {visa_type} visa',
                    'rule_type': 'mandatory',
                    'priority': 1,
                    'conditions': '[]'
                },
                'requirements': [],
                'document_types': ['passport'],
                'fields': ['passport_number', 'name']
            }
        ]
        
        mock_session = MockNeo4jSession(mock_rules)
        mock_reasoning_service._get_session = Mock(return_value=mock_session)
        
        # Act - Validate compliance multiple times with same input
        results = []
        for _ in range(3):  # Run multiple times to test consistency
            result = await mock_reasoning_service.validate_compliance(
                document_data, visa_type, country
            )
            results.append(result)
        
        # Assert - Results should be consistent
        # Requirement 1.2: Should validate against country-specific compliance rules
        for result in results:
            assert isinstance(result, ComplianceResult)
            assert result.id is not None
            assert isinstance(result.is_compliant, bool)
            assert isinstance(result.violations, list)
            assert isinstance(result.reasoning_path.steps, list)
            assert isinstance(result.required_documents, list)
            assert isinstance(result.satisfied_requirements, list)
            assert 0.0 <= result.confidence <= 1.0
        
        # Requirement 5.1, 5.2: Should produce consistent results for same input
        first_result = results[0]
        for result in results[1:]:
            # Core validation results should be identical
            assert result.is_compliant == first_result.is_compliant
            assert len(result.violations) == len(first_result.violations)
            assert len(result.reasoning_path.steps) == len(first_result.reasoning_path.steps)
            
            # Confidence should be consistent
            assert abs(result.confidence - first_result.confidence) < 0.01
        
        # Requirement 5.4: Should provide reasoning paths
        for result in results:
            assert result.reasoning_path is not None
            assert isinstance(result.reasoning_path.steps, list)
            assert isinstance(result.reasoning_path.conclusion, str)
            assert 0.0 <= result.reasoning_path.confidence <= 1.0
            
            # Each reasoning step should have required fields
            for step in result.reasoning_path.steps:
                assert step.step_number > 0
                assert isinstance(step.rule_applied, str)
                assert isinstance(step.input_data, dict)
                assert isinstance(step.output_result, dict)
                assert 0.0 <= step.confidence <= 1.0
                assert isinstance(step.explanation, str)
    
    @given(
        rules=st.lists(generate_visa_rule(), min_size=1, max_size=3),
        country=st.sampled_from(['US', 'UK', 'CA'])
    )
    @settings(max_examples=30, deadline=8000)
    async def test_rule_storage_and_retrieval_consistency(
        self, 
        mock_reasoning_service, 
        rules, 
        country
    ):
        """
        Test that rule storage and retrieval is consistent.
        
        **Feature: visaverse-guardian-ai, Property 2: Compliance Validation Consistency**
        **Validates: Requirements 5.1, 5.2**
        """
        # Mock successful rule storage
        mock_session = MockNeo4jSession()
        mock_reasoning_service._get_session = Mock(return_value=mock_session)
        
        # Act - Store rules
        await mock_reasoning_service.update_rules(country, rules)
        
        # Assert - Storage operations should be consistent
        assert len(mock_session.queries_executed) > 0
        
        # Should have executed queries for each rule
        rule_creation_queries = [
            q for q in mock_session.queries_executed 
            if 'MERGE (r:Rule' in q['query']
        ]
        assert len(rule_creation_queries) == len(rules)
        
        # Each rule should have consistent parameters
        for i, query in enumerate(rule_creation_queries):
            params = query['params']
            rule = rules[i]
            
            assert params['rule_id'] == str(rule.id)
            assert params['rule_type'] == rule.rule_type.value
            assert params['priority'] == rule.priority
            assert params['description'] == rule.description
    
    async def test_empty_rules_handling_consistency(self, mock_reasoning_service):
        """
        Test consistent handling when no rules are available.
        
        **Feature: visaverse-guardian-ai, Property 2: Compliance Validation Consistency**
        **Validates: Requirements 1.2, 5.4**
        """
        # Mock empty rules response
        mock_session = MockNeo4jSession([])  # No rules
        mock_reasoning_service._get_session = Mock(return_value=mock_session)
        
        # Create test data
        document_data = [StructuredData(
            document_type=DocumentType.PASSPORT,
            key_fields={'passport_number': 'AB123456'},
            dates=[],
            financial_info=None,
            missing_fields=[],
            extraction_confidence=0.9
        )]
        
        # Act - Validate with no rules
        result = await mock_reasoning_service.validate_compliance(
            document_data, 'Tourist', 'XX'  # Non-existent country
        )
        
        # Assert - Should handle gracefully and consistently
        assert isinstance(result, ComplianceResult)
        assert result.is_compliant == False  # Should be non-compliant when no rules
        assert result.confidence == 0.0  # Should have zero confidence
        assert len(result.violations) == 0  # No violations since no rules to violate
        assert result.reasoning_path is not None
        assert len(result.reasoning_path.steps) == 1  # Should have one step explaining no rules
        assert 'no rules' in result.reasoning_path.conclusion.lower()
    
    async def test_error_handling_consistency(self, mock_reasoning_service):
        """
        Test consistent error handling in compliance validation.
        
        **Feature: visaverse-guardian-ai, Property 2: Compliance Validation Consistency**
        **Validates: Requirements 5.4**
        """
        # Mock session that raises an exception
        def failing_session():
            raise Exception("Database connection failed")
        
        mock_reasoning_service._get_session = Mock(side_effect=failing_session)
        
        # Create test data
        document_data = [StructuredData(
            document_type=DocumentType.PASSPORT,
            key_fields={'passport_number': 'AB123456'},
            dates=[],
            financial_info=None,
            missing_fields=[],
            extraction_confidence=0.9
        )]
        
        # Act - Validate with failing database
        result = await mock_reasoning_service.validate_compliance(
            document_data, 'Tourist', 'US'
        )
        
        # Assert - Should handle errors consistently
        assert isinstance(result, ComplianceResult)
        assert result.is_compliant == False  # Should be non-compliant on error
        assert result.confidence == 0.0  # Should have zero confidence on error
        assert result.reasoning_path is not None
        assert len(result.reasoning_path.steps) == 1
        assert 'error' in result.reasoning_path.conclusion.lower()


# Synchronous wrapper tests for pytest
def test_compliance_validation_consistency_property():
    """Synchronous wrapper for the main property test."""
    service = Neo4jGraphReasoningService("bolt://localhost:7687", "neo4j", "password")
    service._driver = Mock()
    test_instance = TestComplianceValidationConsistency()
    
    async def run_test():
        # Create simple test data
        document_data = [StructuredData(
            document_type=DocumentType.PASSPORT,
            key_fields={'passport_number': 'AB123456', 'name': 'JOHN DOE'},
            dates=[],
            financial_info=None,
            missing_fields=[],
            extraction_confidence=0.9
        )]
        
        # Mock rules response
        mock_rules = [{
            'r': {
                'id': str(uuid.uuid4()),
                'name': 'passport_rule',
                'description': 'Passport validation rule',
                'rule_type': 'mandatory',
                'priority': 1,
                'conditions': '[]'
            },
            'requirements': [],
            'document_types': ['passport'],
            'fields': ['passport_number']
        }]
        
        mock_session = MockNeo4jSession(mock_rules)
        service._get_session = Mock(return_value=mock_session)
        
        # Test consistency
        results = []
        for _ in range(2):
            result = await service.validate_compliance(document_data, 'Tourist', 'US')
            results.append(result)
        
        # Verify consistency
        assert len(results) == 2
        assert results[0].is_compliant == results[1].is_compliant
        assert len(results[0].violations) == len(results[1].violations)
        assert abs(results[0].confidence - results[1].confidence) < 0.01
    
    asyncio.run(run_test())


def test_empty_rules_handling():
    """Test handling of empty rules."""
    service = Neo4jGraphReasoningService("bolt://localhost:7687", "neo4j", "password")
    service._driver = Mock()
    test_instance = TestComplianceValidationConsistency()
    
    asyncio.run(test_instance.test_empty_rules_handling_consistency(service))


def test_error_handling():
    """Test error handling consistency."""
    service = Neo4jGraphReasoningService("bolt://localhost:7687", "neo4j", "password")
    service._driver = Mock()
    test_instance = TestComplianceValidationConsistency()
    
    asyncio.run(test_instance.test_error_handling_consistency(service))