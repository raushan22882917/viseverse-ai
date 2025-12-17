"""
Pytest configuration and shared fixtures for VisaVerse Guardian AI tests.
"""

import pytest
from hypothesis import settings, Verbosity
from typing import Dict, Any
from uuid import uuid4
from datetime import datetime

# Configure Hypothesis for property-based testing
settings.register_profile("default", max_examples=100, verbosity=Verbosity.normal)
settings.register_profile("ci", max_examples=1000, verbosity=Verbosity.verbose)
settings.register_profile("dev", max_examples=10, verbosity=Verbosity.verbose)

# Load the appropriate profile
settings.load_profile("default")


@pytest.fixture
def sample_audit_data() -> Dict[str, Any]:
    """Sample audit data for testing."""
    return {
        "operation": "document_processing",
        "service": "document-service",
        "user_id": "test-user-123",
        "application_id": uuid4(),
        "input_data": {"filename": "passport.pdf", "language": "en"},
        "output_data": {"document_type": "passport", "confidence": 0.95},
        "success": True,
        "execution_time_ms": 1250.5,
        "timestamp": datetime.utcnow()
    }


@pytest.fixture
def sample_application_id():
    """Sample application ID for testing."""
    return uuid4()


@pytest.fixture
def sample_user_id():
    """Sample user ID for testing."""
    return "test-user-123"