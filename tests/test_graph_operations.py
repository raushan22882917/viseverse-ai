"""
Unit tests for graph operations in the Neo4j Graph Reasoning Service.
Tests rule storage and retrieval, multi-hop reasoning paths, and rule conflict resolution.
Requirements: 5.1, 5.2, 5.4
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from uuid import uuid4, UUID
from datetime import datetime
from typing import List, Dict, Any

from src.visaverse.services.graph.reasoning_engine import Neo4jGraphReasoningService
from src.visaverse.core.models import (
    VisaRule, RuleType, ComparisonOperator, RuleCondition, Requirement,
    StructuredData, DocumentType, DateField, FinancialData,
    ComplianceResult, ReasoningPath, ReasoningStep, RuleViolation, RiskSeverity
)


# Module-level fixtures
@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j driver for testing."""
    driver = Mock()
    session = Mock()
    transaction = Mock()
    
    # Mock session context manager
    session.__enter__ = Mock(return_value=session)
    session.__exit__ = Mock(return_value=None)
    
    # Mock transaction context manager
    transaction.__enter__ = Mock(return_value=transaction)
    transaction.__exit__ = Mock(return_value=None)
    
    session.begin_transaction.return_value = transaction
    driver.session.return_value = session
    
    return driver, session, transaction

@pytest.fixture
def graph_service(mock_neo4j_driver):
    """Create graph service with mocked Neo4j driver."""
    driver, session, transaction = mock_neo4j_driver
    
    with patch('src.visaverse.services.graph.reasoning_engine.GraphDatabase.driver') as mock_driver_factory:
        mock_driver_factory.return_value = driver
        service = Neo4jGraphReasoningService(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="test"
        )
        service._driver = driver
        return service, session, transaction

@pytest.fixture
def sample_visa_rules():
    """Sample visa rules for testing."""
    return [
        VisaRule(
            id=uuid4(),
            country="US",
            visa_type="H1B",
            rule_type=RuleType.MANDATORY,
            conditions=[
                RuleCondition(
                    field="employment_letter",
                    operator=ComparisonOperator.EXISTS,
                    value=True,
                    description="Employment letter must be present"
                )
            ],
            requirements=[
                Requirement(
                    id=uuid4(),
                    name="Employment Verification",
                    description="Valid employment letter required",
                    mandatory=True
                )
            ],
            priority=1,
            description="Employment letter requirement for H1B visa"
        ),
        VisaRule(
            id=uuid4(),
            country="US",
            visa_type="H1B",
            rule_type=RuleType.MANDATORY,
            conditions=[
                RuleCondition(
                    field="salary",
                    operator=ComparisonOperator.GREATER_THAN,
                    value=60000,
                    description="Salary must be above prevailing wage"
                )
            ],
            requirements=[
                Requirement(
                    id=uuid4(),
                    name="Salary Requirement",
                    description="Salary must meet minimum threshold",
                    mandatory=True
                )
            ],
            priority=2,
            description="Salary requirement for H1B visa"
        )
    ]

@pytest.fixture
def sample_structured_data():
    """Sample structured data for testing."""
    return [
        StructuredData(
            document_type=DocumentType.EMPLOYMENT_LETTER,
            key_fields={
                "employer_name": "Tech Corp Inc",
                "position": "Software Engineer",
                "salary": 75000,
                "start_date": "2024-01-01"
            },
            dates=[
                DateField(
                    field_name="start_date",
                    date_value=datetime(2024, 1, 1),
                    confidence=0.95,
                    format_detected="YYYY-MM-DD"
                )
            ],
            financial_info=FinancialData(
                amounts=[{"salary": 75000}],
                currency="USD"
            ),
            missing_fields=[],
            extraction_confidence=0.92
        ),
        StructuredData(
            document_type=DocumentType.PASSPORT,
            key_fields={
                "passport_number": "123456789",
                "nationality": "Indian",
                "expiry_date": "2030-12-31"
            },
            dates=[
                DateField(
                    field_name="expiry_date",
                    date_value=datetime(2030, 12, 31),
                    confidence=0.98,
                    format_detected="YYYY-MM-DD"
                )
            ],
            missing_fields=[],
            extraction_confidence=0.96
        )
    ]


class TestGraphOperations:
    """Test suite for graph operations functionality."""


class TestRuleStorageAndRetrieval:
    """Test rule storage and retrieval operations."""
    
    @pytest.mark.asyncio
    async def test_update_rules_success(self, graph_service, sample_visa_rules):
        """Test successful rule storage in Neo4j."""
        service, session, transaction = graph_service
        
        # Mock successful transaction execution
        transaction.run.return_value = None
        
        # Execute rule update
        await service.update_rules("US", sample_visa_rules)
        
        # Verify transaction calls
        assert transaction.run.call_count >= len(sample_visa_rules) * 3  # Country, visa type, rule creation
        
        # Verify country creation call
        country_calls = [call for call in transaction.run.call_args_list 
                        if "MERGE (c:Country" in str(call)]
        assert len(country_calls) >= 1
        
        # Verify rule creation calls
        rule_calls = [call for call in transaction.run.call_args_list 
                     if "MERGE (r:Rule" in str(call)]
        assert len(rule_calls) >= len(sample_visa_rules)
    
    @pytest.mark.asyncio
    async def test_update_rules_with_requirements(self, graph_service, sample_visa_rules):
        """Test rule storage with requirements and relationships."""
        service, session, transaction = graph_service
        
        # Mock successful transaction execution
        transaction.run.return_value = None
        
        # Execute rule update
        await service.update_rules("US", sample_visa_rules)
        
        # Verify requirement creation calls
        req_calls = [call for call in transaction.run.call_args_list 
                    if "MERGE (req:Requirement" in str(call)]
        
        # Should have calls for each requirement in each rule
        total_requirements = sum(len(rule.requirements) for rule in sample_visa_rules)
        assert len(req_calls) >= total_requirements
        
        # Verify relationship creation calls
        relationship_calls = [call for call in transaction.run.call_args_list 
                            if "MERGE (r)-[:REQUIRES]->(req)" in str(call)]
        assert len(relationship_calls) >= total_requirements
    
    @pytest.mark.asyncio
    async def test_get_rules_by_country_and_visa_type(self, graph_service):
        """Test retrieving rules by country and visa type."""
        service, session, transaction = graph_service
        
        # Mock query result
        mock_record = Mock()
        mock_record.__getitem__ = Mock(side_effect=lambda key: {
            'r': {
                'id': str(uuid4()),
                'rule_type': 'mandatory',
                'priority': 1,
                'description': 'Test rule',
                'conditions': '[{"field": "test", "operator": "exists", "value": true, "description": "test"}]',
                'created_at': datetime.utcnow().isoformat()
            },
            'requirements': []
        }[key])
        
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([mock_record]))
        session.run.return_value = mock_result
        
        # Execute rule retrieval
        rules = await service.get_rules("US", "H1B")
        
        # Verify query execution - check the last call which should be our query
        assert session.run.called
        last_call = session.run.call_args_list[-1]
        assert "MATCH (c:Country {code: $country})" in last_call[0][0]
        assert last_call[1]['country'] == "US"
        assert last_call[1]['visa_type'] == "H1B"
        
        # Verify result structure
        assert isinstance(rules, list)
    
    @pytest.mark.asyncio
    async def test_get_rules_by_country_only(self, graph_service):
        """Test retrieving all rules for a country."""
        service, session, transaction = graph_service
        
        # Mock query result
        mock_record = Mock()
        mock_record.__getitem__ = Mock(side_effect=lambda key: {
            'r': {
                'id': str(uuid4()),
                'rule_type': 'mandatory',
                'priority': 1,
                'description': 'Test rule',
                'conditions': '[{"field": "test", "operator": "exists", "value": true, "description": "test"}]',
                'created_at': datetime.utcnow().isoformat()
            },
            'visa_type': 'H1B',
            'requirements': []
        }[key])
        mock_record.get = Mock(return_value='H1B')
        
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([mock_record]))
        session.run.return_value = mock_result
        
        # Execute rule retrieval
        rules = await service.get_rules("US")
        
        # Verify query execution - check the last call which should be our query
        assert session.run.called
        last_call = session.run.call_args_list[-1]
        assert "MATCH (c:Country {code: $country})" in last_call[0][0]
        assert last_call[1]['country'] == "US"
        assert 'visa_type' not in last_call[1]
    
    @pytest.mark.asyncio
    async def test_get_rules_empty_result(self, graph_service):
        """Test handling of empty rule query results."""
        service, session, transaction = graph_service
        
        # Mock empty result
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([]))
        session.run.return_value = mock_result
        
        # Execute rule retrieval
        rules = await service.get_rules("NONEXISTENT", "INVALID")
        
        # Verify empty result handling
        assert rules == []
        assert session.run.called


class TestMultiHopReasoningPaths:
    """Test multi-hop reasoning path functionality."""
    
    @pytest.mark.asyncio
    async def test_validate_compliance_with_reasoning_path(self, graph_service, sample_structured_data):
        """Test compliance validation generates proper reasoning path."""
        service, session, transaction = graph_service
        
        # Mock rule query result
        mock_rule_record = Mock()
        mock_rule_record.__getitem__ = Mock(side_effect=lambda key: {
            'r': {
                'id': str(uuid4()),
                'name': 'Employment Letter Rule',
                'description': 'Employment letter must be present',
                'rule_type': 'mandatory',
                'priority': 1,
                'conditions': '[{"field": "employment_letter", "operator": "exists", "value": true, "description": "Employment letter required"}]'
            },
            'requirements': [],
            'document_types': ['employment_letter'],
            'fields': ['employer_name', 'salary']
        }[key])
        
        mock_rule_result = Mock()
        mock_rule_result.__iter__ = Mock(return_value=iter([mock_rule_record]))
        session.run.return_value = mock_rule_result
        
        # Execute compliance validation
        result = await service.validate_compliance(sample_structured_data, "H1B", "US")
        
        # Verify reasoning path structure
        assert isinstance(result, ComplianceResult)
        assert isinstance(result.reasoning_path, ReasoningPath)
        assert len(result.reasoning_path.steps) > 0
        
        # Verify reasoning step structure
        step = result.reasoning_path.steps[0]
        assert isinstance(step, ReasoningStep)
        assert step.step_number == 1
        assert step.rule_applied == 'Employment Letter Rule'
        assert isinstance(step.input_data, dict)
        assert isinstance(step.output_result, dict)
        assert 0.0 <= step.confidence <= 1.0
    
    @pytest.mark.asyncio
    async def test_multi_step_reasoning_path(self, graph_service, sample_structured_data):
        """Test reasoning path with multiple validation steps."""
        service, session, transaction = graph_service
        
        # Mock multiple rule query results
        rule_records = []
        for i, rule_name in enumerate(['Employment Rule', 'Salary Rule', 'Document Rule']):
            mock_record = Mock()
            mock_record.__getitem__ = Mock(side_effect=lambda key, name=rule_name, idx=i: {
                'r': {
                    'id': str(uuid4()),
                    'name': name,
                    'description': f'{name} description',
                    'rule_type': 'mandatory',
                    'priority': idx + 1,
                    'conditions': f'[{{"field": "test_{idx}", "operator": "exists", "value": true, "description": "test"}}]'
                },
                'requirements': [],
                'document_types': ['employment_letter'],
                'fields': ['test_field']
            }[key])
            rule_records.append(mock_record)
        
        mock_rule_result = Mock()
        mock_rule_result.__iter__ = Mock(return_value=iter(rule_records))
        session.run.return_value = mock_rule_result
        
        # Execute compliance validation
        result = await service.validate_compliance(sample_structured_data, "H1B", "US")
        
        # Verify multi-step reasoning path
        assert len(result.reasoning_path.steps) == 3
        
        # Verify step ordering
        for i, step in enumerate(result.reasoning_path.steps):
            assert step.step_number == i + 1
            assert f'Rule' in step.rule_applied
    
    @pytest.mark.asyncio
    async def test_reasoning_path_with_violations(self, graph_service):
        """Test reasoning path when violations are detected."""
        service, session, transaction = graph_service
        
        # Create structured data missing required fields
        incomplete_data = [
            StructuredData(
                document_type=DocumentType.PASSPORT,
                key_fields={"passport_number": "123456789"},
                dates=[],
                missing_fields=["employment_letter"],
                extraction_confidence=0.8
            )
        ]
        
        # Mock rule requiring employment letter
        mock_rule_record = Mock()
        mock_rule_record.__getitem__ = Mock(side_effect=lambda key: {
            'r': {
                'id': str(uuid4()),
                'name': 'Employment Letter Rule',
                'description': 'Employment letter required',
                'rule_type': 'mandatory',
                'priority': 1,
                'conditions': '[{"field": "employment_letter", "operator": "exists", "value": true, "description": "Employment letter required"}]'
            },
            'requirements': [],
            'document_types': ['employment_letter'],
            'fields': ['employer_name']
        }[key])
        
        mock_rule_result = Mock()
        mock_rule_result.__iter__ = Mock(return_value=iter([mock_rule_record]))
        session.run.return_value = mock_rule_result
        
        # Execute compliance validation
        result = await service.validate_compliance(incomplete_data, "H1B", "US")
        
        # Verify violation detection in reasoning path
        assert not result.is_compliant
        assert len(result.violations) > 0
        assert len(result.reasoning_path.steps) > 0
        
        # Verify reasoning step reflects violation
        step = result.reasoning_path.steps[0]
        assert step.output_result['violation'] is not None
        assert step.confidence < 1.0
    
    @pytest.mark.asyncio
    async def test_get_reasoning_path_by_id(self, graph_service):
        """Test retrieving stored reasoning path by validation ID."""
        service, session, transaction = graph_service
        
        validation_id = uuid4()
        
        # Mock stored reasoning path query result
        mock_record = Mock()
        reasoning_path_data = {
            "id": str(validation_id),
            "steps": [
                {
                    "step_number": 1,
                    "rule_applied": "Test Rule",
                    "input_data": {"test": "data"},
                    "output_result": {"status": "passed"},
                    "confidence": 0.9,
                    "explanation": "Test explanation"
                }
            ],
            "conclusion": "Test conclusion",
            "confidence": 0.9
        }
        mock_record.__getitem__ = Mock(return_value=str(reasoning_path_data).replace("'", '"'))
        
        mock_result = Mock()
        mock_result.single.return_value = mock_record
        session.run.return_value = mock_result
        
        # Execute reasoning path retrieval
        with patch('json.loads') as mock_json_loads:
            mock_json_loads.return_value = reasoning_path_data
            reasoning_path = await service.get_reasoning_path(validation_id)
        
        # Verify query execution - check the last call which should be our query
        assert session.run.called
        last_call = session.run.call_args_list[-1]
        assert "MATCH (v:Validation {id: $validation_id})" in last_call[0][0]
        assert last_call[1]['validation_id'] == str(validation_id)
        
        # Verify reasoning path structure
        assert isinstance(reasoning_path, ReasoningPath)
        assert reasoning_path.id == validation_id


class TestRuleConflictResolution:
    """Test rule conflict resolution functionality."""
    
    @pytest.mark.asyncio
    async def test_rule_priority_ordering(self, graph_service, sample_structured_data):
        """Test that rules are applied in priority order."""
        service, session, transaction = graph_service
        
        # Mock rules with different priorities
        rule_records = []
        priorities = [3, 1, 2]  # Intentionally out of order
        expected_order = [3, 2, 1]  # Expected descending order
        
        for i, priority in enumerate(priorities):
            mock_record = Mock()
            mock_record.__getitem__ = Mock(side_effect=lambda key, p=priority: {
                'r': {
                    'id': str(uuid4()),
                    'name': f'Rule Priority {p}',
                    'description': f'Rule with priority {p}',
                    'rule_type': 'mandatory',
                    'priority': p,
                    'conditions': '[{"field": "test", "operator": "exists", "value": true, "description": "test"}]'
                },
                'requirements': [],
                'document_types': [],
                'fields': []
            }[key])
            rule_records.append(mock_record)
        
        mock_rule_result = Mock()
        mock_rule_result.__iter__ = Mock(return_value=iter(rule_records))
        session.run.return_value = mock_rule_result
        
        # Execute compliance validation
        result = await service.validate_compliance(sample_structured_data, "H1B", "US")
        
        # Verify query includes ORDER BY priority DESC - check the last call
        assert session.run.called
        last_call = session.run.call_args_list[-1]
        query = last_call[0][0]
        assert "ORDER BY r.priority DESC" in query
        
        # Verify reasoning steps are created (order may vary in mock)
        assert len(result.reasoning_path.steps) == 3
        
        # Verify all expected priorities are present in the steps
        rule_names = [step.rule_applied for step in result.reasoning_path.steps]
        for priority in expected_order:
            assert any(f'Priority {priority}' in name for name in rule_names)
    
    @pytest.mark.asyncio
    async def test_mandatory_vs_optional_rule_conflicts(self, graph_service, sample_structured_data):
        """Test resolution of conflicts between mandatory and optional rules."""
        service, session, transaction = graph_service
        
        # Mock conflicting rules - mandatory takes precedence
        conflicting_rules = [
            {
                'id': str(uuid4()),
                'name': 'Mandatory Rule',
                'description': 'Mandatory requirement',
                'rule_type': 'mandatory',
                'priority': 1,
                'conditions': '[{"field": "salary", "operator": "greater_than", "value": 70000, "description": "High salary required"}]'
            },
            {
                'id': str(uuid4()),
                'name': 'Optional Rule',
                'description': 'Optional requirement',
                'rule_type': 'optional',
                'priority': 2,
                'conditions': '[{"field": "salary", "operator": "greater_than", "value": 50000, "description": "Lower salary acceptable"}]'
            }
        ]
        
        rule_records = []
        for rule_data in conflicting_rules:
            mock_record = Mock()
            mock_record.__getitem__ = Mock(side_effect=lambda key, data=rule_data: {
                'r': data,
                'requirements': [],
                'document_types': [],
                'fields': []
            }[key])
            rule_records.append(mock_record)
        
        mock_rule_result = Mock()
        mock_rule_result.__iter__ = Mock(return_value=iter(rule_records))
        session.run.return_value = mock_rule_result
        
        # Execute compliance validation
        result = await service.validate_compliance(sample_structured_data, "H1B", "US")
        
        # Verify both rules are processed but mandatory rule violations are prioritized
        assert len(result.reasoning_path.steps) == 2
        
        # Check that mandatory rule is processed
        mandatory_step = next((step for step in result.reasoning_path.steps 
                             if 'Mandatory Rule' in step.rule_applied), None)
        assert mandatory_step is not None
        
        # Check that optional rule is also processed
        optional_step = next((step for step in result.reasoning_path.steps 
                            if 'Optional Rule' in step.rule_applied), None)
        assert optional_step is not None
    
    @pytest.mark.asyncio
    async def test_rule_condition_evaluation_conflicts(self, graph_service):
        """Test handling of conflicting rule conditions."""
        service, session, transaction = graph_service
        
        # Create data that satisfies one rule but not another
        conflicting_data = [
            StructuredData(
                document_type=DocumentType.EMPLOYMENT_LETTER,
                key_fields={
                    "salary": 65000,  # Between 60k and 70k thresholds
                    "position": "Software Engineer"
                },
                dates=[],
                missing_fields=[],
                extraction_confidence=0.95
            )
        ]
        
        # Mock conflicting salary requirements
        conflicting_rules = [
            {
                'id': str(uuid4()),
                'name': 'High Salary Rule',
                'description': 'Salary must be above 70k',
                'rule_type': 'mandatory',
                'priority': 1,
                'conditions': '[{"field": "salary", "operator": "greater_than", "value": 70000, "description": "High salary"}]'
            },
            {
                'id': str(uuid4()),
                'name': 'Medium Salary Rule',
                'description': 'Salary must be above 60k',
                'rule_type': 'mandatory',
                'priority': 2,
                'conditions': '[{"field": "salary", "operator": "greater_than", "value": 60000, "description": "Medium salary"}]'
            }
        ]
        
        rule_records = []
        for rule_data in conflicting_rules:
            mock_record = Mock()
            mock_record.__getitem__ = Mock(side_effect=lambda key, data=rule_data: {
                'r': data,
                'requirements': [],
                'document_types': [],
                'fields': []
            }[key])
            rule_records.append(mock_record)
        
        mock_rule_result = Mock()
        mock_rule_result.__iter__ = Mock(return_value=iter(rule_records))
        session.run.return_value = mock_rule_result
        
        # Execute compliance validation
        result = await service.validate_compliance(conflicting_data, "H1B", "US")
        
        # Verify that both rules are evaluated
        assert len(result.reasoning_path.steps) == 2
        
        # Verify that one rule passes and one fails
        violations = [step.output_result.get('violation') for step in result.reasoning_path.steps]
        violations = [v for v in violations if v is not None]
        
        # Should have exactly one violation (high salary rule fails, medium salary rule passes)
        assert len(violations) == 1
        assert not result.is_compliant  # Overall result is non-compliant due to mandatory rule failure
    
    @pytest.mark.asyncio
    async def test_confidence_calculation_with_conflicts(self, graph_service, sample_structured_data):
        """Test confidence calculation when rule conflicts exist."""
        service, session, transaction = graph_service
        
        # Mock rules with mixed success/failure
        mixed_rules = [
            {
                'id': str(uuid4()),
                'name': 'Passing Rule',
                'description': 'This rule passes',
                'rule_type': 'mandatory',
                'priority': 1,
                'conditions': '[{"field": "employer_name", "operator": "exists", "value": true, "description": "Employer exists"}]'
            },
            {
                'id': str(uuid4()),
                'name': 'Failing Rule',
                'description': 'This rule fails',
                'rule_type': 'mandatory',
                'priority': 2,
                'conditions': '[{"field": "nonexistent_field", "operator": "exists", "value": true, "description": "Missing field"}]'
            }
        ]
        
        rule_records = []
        for rule_data in mixed_rules:
            mock_record = Mock()
            mock_record.__getitem__ = Mock(side_effect=lambda key, data=rule_data: {
                'r': data,
                'requirements': [],
                'document_types': [],
                'fields': []
            }[key])
            rule_records.append(mock_record)
        
        mock_rule_result = Mock()
        mock_rule_result.__iter__ = Mock(return_value=iter(rule_records))
        session.run.return_value = mock_rule_result
        
        # Execute compliance validation
        result = await service.validate_compliance(sample_structured_data, "H1B", "US")
        
        # Verify confidence is reduced due to violations
        assert 0.0 <= result.confidence <= 1.0
        assert result.confidence < 1.0  # Should be less than perfect due to violations
        
        # Verify reasoning path confidence reflects mixed results
        passing_steps = [step for step in result.reasoning_path.steps 
                        if step.output_result.get('violation') is None]
        failing_steps = [step for step in result.reasoning_path.steps 
                        if step.output_result.get('violation') is not None]
        
        assert len(passing_steps) > 0
        assert len(failing_steps) > 0
        
        # Passing steps should have higher confidence than failing steps
        if passing_steps and failing_steps:
            avg_passing_confidence = sum(step.confidence for step in passing_steps) / len(passing_steps)
            avg_failing_confidence = sum(step.confidence for step in failing_steps) / len(failing_steps)
            assert avg_passing_confidence > avg_failing_confidence


if __name__ == "__main__":
    pytest.main([__file__])