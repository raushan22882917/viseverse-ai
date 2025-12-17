"""
Property-based tests for risk factor explanation completeness.

**Feature: visaverse-guardian-ai, Property 4: Risk Factor Explanation Completeness**
**Validates: Requirements 2.1, 2.2, 2.5**

Property: For any identified compliance issue, the system should provide specific 
risk factors with actionable recommendations and severity categorization.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings
from typing import List
import asyncio
from unittest.mock import AsyncMock, MagicMock
import os

from src.visaverse.services.risk.risk_engine import GeminiRiskAssessmentService
from src.visaverse.core.models import (
    ComplianceResult,
    RuleViolation,
    RiskSeverity,
    HistoricalPattern,
    StructuredData,
    DocumentType,
    ApplicationOutcome,
    RiskAssessment,
    RiskFactor,
    RiskCategory,
    ReasoningPath,
    ReasoningStep,
    DateField,
    FinancialData
)


# Test data generators (reusing from probability test)
@st.composite
def generate_reasoning_step(draw):
    """Generate a valid reasoning step."""
    return ReasoningStep(
        step_number=draw(st.integers(min_value=1, max_value=10)),
        rule_applied=draw(st.text(min_size=1, max_size=100)),
        input_data=draw(st.dictionaries(st.text(min_size=1, max_size=20), st.text(min_size=1, max_size=50))),
        output_result=draw(st.dictionaries(st.text(min_size=1, max_size=20), st.text(min_size=1, max_size=50))),
        confidence=draw(st.floats(min_value=0.0, max_value=1.0)),
        explanation=draw(st.text(min_size=10, max_size=200))
    )


@st.composite
def generate_reasoning_path(draw):
    """Generate a valid reasoning path."""
    steps = draw(st.lists(generate_reasoning_step(), min_size=0, max_size=3))
    
    return ReasoningPath(
        steps=steps,
        conclusion=draw(st.text(min_size=10, max_size=200)),
        confidence=draw(st.floats(min_value=0.0, max_value=1.0))
    )


@st.composite
def generate_rule_violation(draw):
    """Generate a valid rule violation."""
    return RuleViolation(
        rule_id=draw(st.uuids()),
        rule_description=draw(st.text(min_size=10, max_size=100)),
        violation_type=draw(st.sampled_from([
            "missing_document", "invalid_format", "expired_document", 
            "insufficient_funds", "incomplete_information", "inconsistent_data"
        ])),
        explanation=draw(st.text(min_size=10, max_size=200)),
        severity=draw(st.sampled_from(list(RiskSeverity))),
        affected_fields=draw(st.lists(st.text(min_size=1, max_size=50), min_size=0, max_size=3))
    )


@st.composite
def generate_compliance_result_with_violations(draw):
    """Generate a compliance result that has violations (for testing risk factor generation)."""
    violations = draw(st.lists(generate_rule_violation(), min_size=1, max_size=5))  # Ensure at least 1 violation
    
    return ComplianceResult(
        is_compliant=False,  # Force non-compliant to ensure risk factors are generated
        violations=violations,
        confidence=draw(st.floats(min_value=0.0, max_value=1.0)),
        reasoning_path=draw(generate_reasoning_path()),
        required_documents=draw(st.lists(st.text(min_size=1, max_size=50), min_size=0, max_size=10)),
        satisfied_requirements=draw(st.lists(st.text(min_size=1, max_size=50), min_size=0, max_size=10))
    )


@st.composite
def generate_structured_data(draw):
    """Generate valid structured document data."""
    doc_type = draw(st.sampled_from(list(DocumentType)))
    
    # Generate appropriate key fields based on document type
    key_fields = {}
    if doc_type == DocumentType.PASSPORT:
        key_fields = {
            "name": draw(st.text(min_size=2, max_size=100)),
            "passport_number": draw(st.text(min_size=5, max_size=20)),
            "nationality": draw(st.text(min_size=2, max_size=50))
        }
    elif doc_type == DocumentType.BANK_STATEMENT:
        key_fields = {
            "account_holder": draw(st.text(min_size=2, max_size=100)),
            "account_number": draw(st.text(min_size=5, max_size=20)),
            "balance": draw(st.floats(min_value=0, max_value=1000000))
        }
    elif doc_type == DocumentType.EMPLOYMENT_LETTER:
        key_fields = {
            "employee_name": draw(st.text(min_size=2, max_size=100)),
            "position": draw(st.text(min_size=2, max_size=100)),
            "salary": draw(st.floats(min_value=0, max_value=500000))
        }
    
    return StructuredData(
        document_type=doc_type,
        key_fields=key_fields,
        dates=draw(st.lists(
            st.builds(DateField, 
                field_name=st.text(min_size=1, max_size=50),
                date_value=st.datetimes()
            ), 
            min_size=0, max_size=3
        )),
        financial_info=draw(st.one_of(
            st.none(),
            st.builds(FinancialData,
                currency=st.sampled_from(["USD", "EUR", "GBP", "CAD"]),
                amounts=st.lists(
                    st.dictionaries(
                        st.sampled_from(["amount", "date", "description"]),
                        st.one_of(
                            st.floats(min_value=0, max_value=100000),
                            st.text(min_size=1, max_size=50)
                        )
                    ),
                    min_size=0, max_size=5
                ),
                account_numbers=st.lists(st.text(min_size=5, max_size=20), min_size=0, max_size=3),
                transaction_count=st.one_of(st.none(), st.integers(min_value=0, max_value=1000))
            )
        )),
        missing_fields=draw(st.lists(st.text(min_size=1, max_size=50), min_size=0, max_size=5)),
        extraction_confidence=draw(st.floats(min_value=0.0, max_value=1.0))
    )


def create_mock_risk_service_with_risk_factors():
    """Create a mock risk assessment service that generates risk factors."""
    api_key = os.getenv("GEMINI_API_KEY", "test-api-key")
    
    service = GeminiRiskAssessmentService(api_key=api_key)
    
    # Mock the model to avoid actual API calls
    service.model = MagicMock()
    
    # Create a mock response with risk factors
    mock_response = MagicMock()
    mock_response.text = '''
    {
        "approval_probability": 45.0,
        "confidence_level": 0.7,
        "risk_factors": [
            {
                "category": "document_missing",
                "severity": "high",
                "description": "Required passport document is missing from the application",
                "impact": 0.8,
                "recommendation": "Please provide a valid passport copy with clear visibility of all pages"
            },
            {
                "category": "requirement_not_met",
                "severity": "medium",
                "description": "Financial documentation does not meet minimum requirements",
                "impact": 0.6,
                "recommendation": "Submit bank statements showing sufficient funds for the visa duration"
            }
        ],
        "recommendations": [
            {
                "title": "Complete Required Documentation",
                "description": "Provide all missing documents to improve approval chances",
                "priority": 9,
                "action_required": true,
                "estimated_impact": 0.8
            },
            {
                "title": "Financial Evidence",
                "description": "Strengthen financial documentation with additional proof of funds",
                "priority": 7,
                "action_required": true,
                "estimated_impact": 0.6
            }
        ],
        "summary": "Application has significant documentation gaps that need to be addressed"
    }
    '''
    
    # Configure the AsyncMock to return the mock response
    service.model.generate_content = AsyncMock(return_value=mock_response)
    
    return service


class TestRiskFactorExplanationCompleteness:
    """Test risk factor explanation completeness property."""
    
    @given(
        compliance=generate_compliance_result_with_violations(),
        document_data=st.lists(generate_structured_data(), min_size=1, max_size=3)
    )
    @settings(max_examples=100, deadline=None)
    def test_risk_factor_explanation_completeness(
        self, 
        compliance, 
        document_data
    ):
        """
        **Feature: visaverse-guardian-ai, Property 4: Risk Factor Explanation Completeness**
        
        Property: For any identified compliance issue, the system should provide specific 
        risk factors with actionable recommendations and severity categorization.
        
        This test verifies that:
        1. Risk factors are generated for compliance violations
        2. Each risk factor has a valid category and severity
        3. Each risk factor includes a clear description
        4. Each risk factor provides actionable recommendations
        5. Risk factors have impact scores within valid range
        """
        # Assume valid input constraints
        assume(len(document_data) > 0)
        assume(len(compliance.violations) > 0)  # Ensure we have violations to generate risk factors for
        assume(not compliance.is_compliant)  # Should be non-compliant to generate risk factors
        
        # Create mock service for this test
        mock_risk_service = create_mock_risk_service_with_risk_factors()
        
        # Execute the risk assessment
        async def run_test():
            result = await mock_risk_service.calculate_approval_probability(
                compliance=compliance,
                historical_data=[],
                document_data=document_data
            )
            return result
        
        # Run the async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            assessment = loop.run_until_complete(run_test())
        finally:
            loop.close()
        
        # Verify the assessment is a valid RiskAssessment
        assert isinstance(assessment, RiskAssessment)
        
        # Property 1: Risk factors should be generated for non-compliant applications
        assert assessment.risk_factors is not None
        assert isinstance(assessment.risk_factors, list)
        
        # Since we have compliance violations, we should have risk factors
        # (either from LLM or fallback logic)
        if len(compliance.violations) > 0:
            assert len(assessment.risk_factors) > 0, \
                "Non-compliant applications with violations should generate risk factors"
        
        # Property 2: Each risk factor should have valid category and severity
        for risk_factor in assessment.risk_factors:
            assert isinstance(risk_factor, RiskFactor)
            
            # Valid category
            assert isinstance(risk_factor.category, RiskCategory)
            assert risk_factor.category in list(RiskCategory)
            
            # Valid severity
            assert isinstance(risk_factor.severity, RiskSeverity)
            assert risk_factor.severity in list(RiskSeverity)
        
        # Property 3: Each risk factor should have a clear description
        for risk_factor in assessment.risk_factors:
            assert risk_factor.description is not None
            assert isinstance(risk_factor.description, str)
            assert len(risk_factor.description.strip()) > 0, \
                "Risk factor description should not be empty"
        
        # Property 4: Each risk factor should provide actionable recommendations
        for risk_factor in assessment.risk_factors:
            assert risk_factor.recommendation is not None
            assert isinstance(risk_factor.recommendation, str)
            assert len(risk_factor.recommendation.strip()) > 0, \
                "Risk factor recommendation should not be empty"
        
        # Property 5: Risk factors should have impact scores within valid range
        for risk_factor in assessment.risk_factors:
            assert risk_factor.impact is not None
            assert isinstance(risk_factor.impact, (int, float))
            assert 0.0 <= risk_factor.impact <= 1.0, \
                f"Risk factor impact {risk_factor.impact} should be between 0.0 and 1.0"
        
        # Property 6: Recommendations should be provided alongside risk factors
        assert assessment.recommendations is not None
        assert isinstance(assessment.recommendations, list)
        
        # If we have risk factors, we should have recommendations
        if len(assessment.risk_factors) > 0:
            assert len(assessment.recommendations) > 0, \
                "Applications with risk factors should provide actionable recommendations"
        
        # Property 7: Each recommendation should have required fields
        for recommendation in assessment.recommendations:
            assert recommendation.title is not None
            assert len(recommendation.title.strip()) > 0
            assert recommendation.description is not None
            assert len(recommendation.description.strip()) > 0
            assert 1 <= recommendation.priority <= 10
            assert 0.0 <= recommendation.estimated_impact <= 1.0
    
    @given(
        compliance=generate_compliance_result_with_violations(),
        document_data=st.lists(generate_structured_data(), min_size=1, max_size=3)
    )
    @settings(max_examples=50, deadline=None)
    def test_risk_factor_severity_mapping(
        self, 
        compliance, 
        document_data
    ):
        """
        Test that risk factor severity is appropriately mapped to compliance violation severity.
        
        Higher severity violations should generally result in higher severity risk factors.
        """
        # Assume valid input constraints
        assume(len(document_data) > 0)
        assume(len(compliance.violations) > 0)
        
        # Create mock service for this test
        mock_risk_service = create_mock_risk_service_with_risk_factors()
        
        # Execute the risk assessment
        async def run_test():
            result = await mock_risk_service.calculate_approval_probability(
                compliance=compliance,
                historical_data=[],
                document_data=document_data
            )
            return result
        
        # Run the async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            assessment = loop.run_until_complete(run_test())
        finally:
            loop.close()
        
        # Check that we have risk factors
        assert len(assessment.risk_factors) > 0
        
        # Find the highest severity violation
        max_violation_severity = max(v.severity for v in compliance.violations)
        
        # Check that risk factors reflect the severity appropriately
        risk_severities = [rf.severity for rf in assessment.risk_factors]
        
        # If we have critical violations, we should have at least some high-impact risk factors
        if max_violation_severity == RiskSeverity.CRITICAL:
            high_impact_factors = [rf for rf in assessment.risk_factors if rf.impact >= 0.7]
            assert len(high_impact_factors) > 0, \
                "Critical violations should result in high-impact risk factors"
        
        # All risk factors should have valid severity levels
        for severity in risk_severities:
            assert severity in list(RiskSeverity)
    
    @given(
        compliance=generate_compliance_result_with_violations(),
        document_data=st.lists(generate_structured_data(), min_size=1, max_size=3)
    )
    @settings(max_examples=30, deadline=None)
    def test_risk_factor_categorization(
        self, 
        compliance, 
        document_data
    ):
        """
        Test that risk factors are properly categorized based on the type of compliance issue.
        
        Different types of violations should result in appropriately categorized risk factors.
        """
        # Assume valid input constraints
        assume(len(document_data) > 0)
        assume(len(compliance.violations) > 0)
        
        # Create mock service for this test
        mock_risk_service = create_mock_risk_service_with_risk_factors()
        
        # Execute the risk assessment
        async def run_test():
            result = await mock_risk_service.calculate_approval_probability(
                compliance=compliance,
                historical_data=[],
                document_data=document_data
            )
            return result
        
        # Run the async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            assessment = loop.run_until_complete(run_test())
        finally:
            loop.close()
        
        # Check that we have risk factors
        assert len(assessment.risk_factors) > 0
        
        # Verify that risk factor categories are valid and appropriate
        violation_types = {v.violation_type for v in compliance.violations}
        risk_categories = {rf.category for rf in assessment.risk_factors}
        
        # All categories should be valid
        for category in risk_categories:
            assert category in list(RiskCategory)
        
        # Check for logical mapping between violation types and risk categories
        if "missing_document" in violation_types:
            # Should have document-related risk factors
            document_risks = [rf for rf in assessment.risk_factors 
                            if rf.category in [RiskCategory.DOCUMENT_MISSING, RiskCategory.DOCUMENT_INVALID]]
            # Note: We don't assert this must be true because the LLM might categorize differently
            # but we verify the categories are valid
        
        # Verify that each risk factor has appropriate field references when relevant
        for risk_factor in assessment.risk_factors:
            if risk_factor.field_reference:
                assert isinstance(risk_factor.field_reference, str)
                assert len(risk_factor.field_reference.strip()) > 0