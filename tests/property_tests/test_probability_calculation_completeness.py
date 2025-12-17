"""
Property-based tests for probability calculation completeness.

**Feature: visaverse-guardian-ai, Property 3: Probability Calculation Completeness**
**Validates: Requirements 1.3, 1.4**

Property: For any compliance validation result, the system should calculate an approval 
probability between 0-100% with transparent explanations.
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
    DateField,
    FinancialData
)


# Test data generators
@st.composite
def generate_rule_violation(draw):
    """Generate a valid rule violation."""
    return RuleViolation(
        rule_id=draw(st.uuids()),
        rule_description=draw(st.text(min_size=10, max_size=100)),
        violation_type=draw(st.sampled_from([
            "missing_document", "invalid_format", "expired_document", 
            "insufficient_funds", "incomplete_information"
        ])),
        explanation=draw(st.text(min_size=10, max_size=200)),
        severity=draw(st.sampled_from(list(RiskSeverity))),
        affected_fields=draw(st.lists(st.text(min_size=1, max_size=50), min_size=0, max_size=3))
    )


@st.composite
def generate_reasoning_path(draw):
    """Generate a valid reasoning path."""
    from src.visaverse.core.models import ReasoningPath, ReasoningStep
    
    steps = draw(st.lists(
        st.builds(ReasoningStep,
            step_number=st.integers(min_value=1, max_value=10),
            rule_applied=st.text(min_size=1, max_size=100),
            input_data=st.dictionaries(st.text(min_size=1, max_size=20), st.text(min_size=1, max_size=50)),
            output_result=st.dictionaries(st.text(min_size=1, max_size=20), st.text(min_size=1, max_size=50)),
            confidence=st.floats(min_value=0.0, max_value=1.0),
            explanation=st.text(min_size=10, max_size=200)
        ),
        min_size=0, max_size=3
    ))
    
    return ReasoningPath(
        steps=steps,
        conclusion=draw(st.text(min_size=10, max_size=200)),
        confidence=draw(st.floats(min_value=0.0, max_value=1.0))
    )


@st.composite
def generate_compliance_result(draw):
    """Generate a valid compliance result."""
    violations = draw(st.lists(generate_rule_violation(), min_size=0, max_size=5))
    is_compliant = len(violations) == 0 or draw(st.booleans())
    
    return ComplianceResult(
        is_compliant=is_compliant,
        violations=violations,
        confidence=draw(st.floats(min_value=0.0, max_value=1.0)),
        reasoning_path=draw(generate_reasoning_path()),
        required_documents=draw(st.lists(st.text(min_size=1, max_size=50), min_size=0, max_size=10)),
        satisfied_requirements=draw(st.lists(st.text(min_size=1, max_size=50), min_size=0, max_size=10))
    )


@st.composite
def generate_historical_pattern(draw):
    """Generate a valid historical pattern."""
    return HistoricalPattern(
        application_id=draw(st.uuids()).hex,
        visa_type=draw(st.sampled_from(["tourist", "work", "student", "business"])),
        country=draw(st.sampled_from(["US", "CA", "UK", "DE", "AU"])),
        outcome=draw(st.sampled_from(list(ApplicationOutcome))),
        risk_factors=[],
        timestamp=draw(st.datetimes()),
        user_id=draw(st.uuids()).hex
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


def create_mock_risk_service():
    """Create a mock risk assessment service for testing."""
    # Mock the Gemini API key (use a test key)
    api_key = os.getenv("GEMINI_API_KEY", "test-api-key")
    
    service = GeminiRiskAssessmentService(api_key=api_key)
    
    # Mock the model to avoid actual API calls
    service.model = MagicMock()
    
    # Create a proper mock response object
    mock_response = MagicMock()
    mock_response.text = '''
    {
        "approval_probability": 75.0,
        "confidence_level": 0.8,
        "risk_factors": [],
        "recommendations": [],
        "summary": "Test analysis"
    }
    '''
    
    # Configure the AsyncMock to return the mock response
    service.model.generate_content = AsyncMock(return_value=mock_response)
    
    return service


class TestProbabilityCalculationCompleteness:
    """Test probability calculation completeness property."""
    
    @given(
        compliance=generate_compliance_result(),
        historical_data=st.lists(generate_historical_pattern(), min_size=0, max_size=5),
        document_data=st.lists(generate_structured_data(), min_size=1, max_size=5)
    )
    @settings(max_examples=100, deadline=None)
    def test_probability_calculation_completeness(
        self, 
        compliance, 
        historical_data, 
        document_data
    ):
        """
        **Feature: visaverse-guardian-ai, Property 3: Probability Calculation Completeness**
        
        Property: For any compliance validation result, the system should calculate an 
        approval probability between 0-100% with transparent explanations.
        
        This test verifies that:
        1. A probability is always calculated (not None)
        2. The probability is within valid range (0.0 to 1.0)
        3. The result includes explanatory information
        4. The confidence level is provided
        """
        # Assume valid input constraints
        assume(len(document_data) > 0)
        assume(all(doc.extraction_confidence >= 0.0 for doc in document_data))
        
        # Create mock service for this test
        mock_risk_service = create_mock_risk_service()
        
        # Execute the probability calculation
        async def run_test():
            result = await mock_risk_service.calculate_approval_probability(
                compliance=compliance,
                historical_data=historical_data,
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
        
        # Property 1: Probability is always calculated and not None
        assert assessment.approval_probability is not None
        
        # Property 2: Probability is within valid range (0.0 to 1.0)
        assert 0.0 <= assessment.approval_probability <= 1.0, \
            f"Probability {assessment.approval_probability} is outside valid range [0.0, 1.0]"
        
        # Property 3: Confidence level is provided and valid
        assert assessment.confidence_level is not None
        assert 0.0 <= assessment.confidence_level <= 1.0, \
            f"Confidence {assessment.confidence_level} is outside valid range [0.0, 1.0]"
        
        # Property 4: Risk factors list is provided (may be empty)
        assert assessment.risk_factors is not None
        assert isinstance(assessment.risk_factors, list)
        
        # Property 5: Recommendations list is provided (may be empty)
        assert assessment.recommendations is not None
        assert isinstance(assessment.recommendations, list)
        
        # Property 6: Risk breakdown is provided for transparency
        assert assessment.risk_breakdown is not None
        
        # Property 7: If there are compliance violations, they should be reflected
        if not compliance.is_compliant and compliance.violations:
            # Either risk factors should be present or probability should be lower
            assert (len(assessment.risk_factors) > 0 or 
                   assessment.approval_probability < 0.8), \
                "Non-compliant applications should have risk factors or lower probability"
    
    @given(
        compliance=generate_compliance_result(),
        historical_data=st.lists(generate_historical_pattern(), min_size=0, max_size=3),
        document_data=st.lists(generate_structured_data(), min_size=1, max_size=3)
    )
    @settings(max_examples=50, deadline=None)
    def test_probability_calculation_with_api_failure(
        self, 
        compliance, 
        historical_data, 
        document_data
    ):
        """
        Test that probability calculation works even when LLM API fails.
        
        This ensures the system provides fallback probability calculations
        when the primary LLM service is unavailable.
        """
        # Assume valid input constraints
        assume(len(document_data) > 0)
        
        # Create mock service for this test
        mock_risk_service = create_mock_risk_service()
        
        # Mock API failure
        mock_risk_service.model.generate_content.side_effect = Exception("API Error")
        
        # Execute the probability calculation
        async def run_test():
            result = await mock_risk_service.calculate_approval_probability(
                compliance=compliance,
                historical_data=historical_data,
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
        
        # Even with API failure, we should get a valid assessment
        assert isinstance(assessment, RiskAssessment)
        assert assessment.approval_probability is not None
        assert 0.0 <= assessment.approval_probability <= 1.0
        assert assessment.confidence_level is not None
        
        # Fallback should have lower confidence
        assert assessment.confidence_level <= 0.7, \
            "Fallback assessment should have lower confidence"
    
    @given(
        compliance=generate_compliance_result(),
        document_data=st.lists(generate_structured_data(), min_size=1, max_size=3)
    )
    @settings(max_examples=30, deadline=None)
    def test_probability_consistency_with_compliance(
        self, 
        compliance, 
        document_data
    ):
        """
        Test that probability calculations are consistent with compliance results.
        
        Compliant applications should generally have higher probabilities than
        non-compliant ones, all else being equal.
        """
        # Assume valid input constraints
        assume(len(document_data) > 0)
        
        # Create mock service for this test
        mock_risk_service = create_mock_risk_service()
        
        # Mock successful LLM response that respects compliance
        def mock_response_generator(is_compliant):
            base_prob = 80.0 if is_compliant else 30.0
            return MagicMock(text=f'''
            {{
                "approval_probability": {base_prob},
                "confidence_level": 0.8,
                "risk_factors": [],
                "recommendations": [],
                "summary": "Analysis completed"
            }}
            ''')
        
        mock_risk_service.model.generate_content.return_value = mock_response_generator(compliance.is_compliant)
        
        # Execute the probability calculation
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
        
        # Verify basic properties
        assert isinstance(assessment, RiskAssessment)
        assert 0.0 <= assessment.approval_probability <= 1.0
        
        # If compliance has severe violations, probability should reflect this
        if not compliance.is_compliant:
            severe_violations = [v for v in compliance.violations 
                               if v.severity in [RiskSeverity.HIGH, RiskSeverity.CRITICAL]]
            if severe_violations:
                # Should have either low probability or risk factors
                assert (assessment.approval_probability < 0.6 or 
                       len(assessment.risk_factors) > 0), \
                    "Severe compliance violations should result in lower probability or risk factors"