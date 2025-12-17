"""
Core data models for VisaVerse Guardian AI system.
All models use Pydantic for validation and serialization.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class DocumentType(str, Enum):
    """Supported document types for visa applications."""
    PASSPORT = "passport"
    VISA_APPLICATION = "visa_application"
    EMPLOYMENT_LETTER = "employment_letter"
    BANK_STATEMENT = "bank_statement"
    EDUCATIONAL_CERTIFICATE = "educational_certificate"
    MEDICAL_CERTIFICATE = "medical_certificate"


class ProcessingStatus(str, Enum):
    """Document processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ApplicationStatus(str, Enum):
    """Visa application status."""
    DRAFT = "draft"
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"


class RiskCategory(str, Enum):
    """Categories of risk factors."""
    DOCUMENT_MISSING = "document_missing"
    DOCUMENT_INVALID = "document_invalid"
    REQUIREMENT_NOT_MET = "requirement_not_met"
    INCONSISTENCY = "inconsistency"
    HISTORICAL_PATTERN = "historical_pattern"


class RiskSeverity(str, Enum):
    """Severity levels for risk factors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RuleType(str, Enum):
    """Types of visa rules."""
    MANDATORY = "mandatory"
    CONDITIONAL = "conditional"
    OPTIONAL = "optional"


class ComparisonOperator(str, Enum):
    """Operators for rule conditions."""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"


class ApplicationOutcome(str, Enum):
    """Possible outcomes for visa applications."""
    APPROVED = "approved"
    REJECTED = "rejected"
    PENDING = "pending"
    WITHDRAWN = "withdrawn"


# Core Data Models

class DocumentMetadata(BaseModel):
    """Metadata for uploaded documents."""
    filename: str
    file_size: int
    mime_type: str
    upload_timestamp: datetime = Field(default_factory=datetime.utcnow)
    language: Optional[str] = None
    page_count: Optional[int] = None


class LayoutData(BaseModel):
    """Layout information extracted from documents."""
    bounding_boxes: List[Dict[str, Any]] = Field(default_factory=list)
    text_regions: List[Dict[str, Any]] = Field(default_factory=list)
    image_regions: List[Dict[str, Any]] = Field(default_factory=list)


class TableData(BaseModel):
    """Table data extracted from documents."""
    headers: List[str] = Field(default_factory=list)
    rows: List[List[str]] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class DateField(BaseModel):
    """Date field extracted from documents."""
    field_name: str
    date_value: datetime
    confidence: float = Field(ge=0.0, le=1.0)
    format_detected: str


class FinancialData(BaseModel):
    """Financial information extracted from documents."""
    amounts: List[Dict[str, Any]] = Field(default_factory=list)
    currency: Optional[str] = None
    account_numbers: List[str] = Field(default_factory=list)
    transaction_count: Optional[int] = None


class QualityScore(BaseModel):
    """Document quality assessment."""
    overall_score: float = Field(ge=0.0, le=1.0)
    text_clarity: float = Field(ge=0.0, le=1.0)
    image_quality: float = Field(ge=0.0, le=1.0)
    completeness: float = Field(ge=0.0, le=1.0)
    issues: List[str] = Field(default_factory=list)


class StructuredData(BaseModel):
    """Structured data extracted from documents."""
    document_type: DocumentType
    key_fields: Dict[str, Any] = Field(default_factory=dict)
    dates: List[DateField] = Field(default_factory=list)
    financial_info: Optional[FinancialData] = None
    missing_fields: List[str] = Field(default_factory=list)
    extraction_confidence: float = Field(ge=0.0, le=1.0)


class ProcessedDocument(BaseModel):
    """Document after OCR processing."""
    id: UUID = Field(default_factory=uuid4)
    document_type: DocumentType
    extracted_text: str
    layout_info: LayoutData
    table_data: List[TableData] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0)
    language: str
    structured_data: Optional[StructuredData] = None
    quality_score: Optional[QualityScore] = None
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Document(BaseModel):
    """Complete document model."""
    id: UUID = Field(default_factory=uuid4)
    type: DocumentType
    content: str
    metadata: DocumentMetadata
    extracted_data: Optional[StructuredData] = None
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class RuleCondition(BaseModel):
    """Condition for visa rules."""
    field: str
    operator: ComparisonOperator
    value: Any
    description: str


class Requirement(BaseModel):
    """Individual requirement for visa applications."""
    id: UUID = Field(default_factory=uuid4)
    name: str
    description: str
    mandatory: bool = True
    conditions: List[RuleCondition] = Field(default_factory=list)


class VisaRule(BaseModel):
    """Visa rule definition."""
    id: UUID = Field(default_factory=uuid4)
    country: str
    visa_type: str
    rule_type: RuleType
    conditions: List[RuleCondition] = Field(default_factory=list)
    requirements: List[Requirement] = Field(default_factory=list)
    priority: int = Field(ge=1, le=10)
    description: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class RiskFactor(BaseModel):
    """Risk factor identified in application."""
    id: UUID = Field(default_factory=uuid4)
    category: RiskCategory
    severity: RiskSeverity
    description: str
    impact: float = Field(ge=0.0, le=1.0)
    recommendation: str
    field_reference: Optional[str] = None


class Recommendation(BaseModel):
    """Actionable recommendation for applicant."""
    id: UUID = Field(default_factory=uuid4)
    title: str
    description: str
    priority: int = Field(ge=1, le=10)
    action_required: bool = True
    estimated_impact: float = Field(ge=0.0, le=1.0)


class ReasoningStep(BaseModel):
    """Individual step in reasoning path."""
    step_number: int
    rule_applied: str
    input_data: Dict[str, Any]
    output_result: Dict[str, Any]
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str


class ReasoningPath(BaseModel):
    """Complete reasoning path for decisions."""
    id: UUID = Field(default_factory=uuid4)
    steps: List[ReasoningStep] = Field(default_factory=list)
    conclusion: str
    confidence: float = Field(ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class RuleViolation(BaseModel):
    """Violation of visa rules."""
    rule_id: UUID
    rule_description: str
    violation_type: str
    severity: RiskSeverity
    affected_fields: List[str] = Field(default_factory=list)
    explanation: str


class ComplianceResult(BaseModel):
    """Result of compliance validation."""
    id: UUID = Field(default_factory=uuid4)
    is_compliant: bool
    violations: List[RuleViolation] = Field(default_factory=list)
    reasoning_path: ReasoningPath
    required_documents: List[str] = Field(default_factory=list)
    satisfied_requirements: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class RiskBreakdown(BaseModel):
    """Detailed breakdown of risk assessment."""
    document_risks: List[RiskFactor] = Field(default_factory=list)
    compliance_risks: List[RiskFactor] = Field(default_factory=list)
    historical_risks: List[RiskFactor] = Field(default_factory=list)
    total_risk_score: float = Field(ge=0.0, le=1.0)


class RiskAssessment(BaseModel):
    """Complete risk assessment for application."""
    id: UUID = Field(default_factory=uuid4)
    approval_probability: float = Field(ge=0.0, le=1.0)
    risk_factors: List[RiskFactor] = Field(default_factory=list)
    recommendations: List[Recommendation] = Field(default_factory=list)
    confidence_level: float = Field(ge=0.0, le=1.0)
    risk_breakdown: Optional[RiskBreakdown] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ExplanationResult(BaseModel):
    """Natural language explanation of assessment."""
    id: UUID = Field(default_factory=uuid4)
    summary: str
    detailed_explanation: str
    actionable_steps: List[str] = Field(default_factory=list)
    risk_breakdown: RiskBreakdown
    language: str = "en"
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PersonalizedInsight(BaseModel):
    """Personalized insight based on user history."""
    id: UUID = Field(default_factory=uuid4)
    insight_type: str
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    based_on_applications: List[UUID] = Field(default_factory=list)
    recommendation: Optional[str] = None


class HistoricalPattern(BaseModel):
    """Historical pattern from user's past applications."""
    application_id: UUID
    visa_type: str
    country: str
    outcome: ApplicationOutcome
    risk_factors: List[RiskFactor] = Field(default_factory=list)
    timestamp: datetime
    lessons_learned: List[str] = Field(default_factory=list)


class ApplicationData(BaseModel):
    """Complete application data for storage."""
    id: UUID = Field(default_factory=uuid4)
    user_id: str
    visa_type: str
    target_country: str
    documents: List[Document] = Field(default_factory=list)
    structured_data: List[StructuredData] = Field(default_factory=list)
    compliance_result: Optional[ComplianceResult] = None
    risk_assessment: Optional[RiskAssessment] = None
    explanation: Optional[ExplanationResult] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class VisaApplication(BaseModel):
    """Complete visa application model."""
    id: UUID = Field(default_factory=uuid4)
    user_id: str
    visa_type: str
    target_country: str
    documents: List[Document] = Field(default_factory=list)
    status: ApplicationStatus = ApplicationStatus.DRAFT
    risk_assessment: Optional[RiskAssessment] = None
    submission_date: Optional[datetime] = None
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    
    @field_validator('last_updated', mode='before')
    @classmethod
    def set_updated_timestamp(cls, v):
        return datetime.utcnow()


class AuditEntry(BaseModel):
    """Audit trail entry for system operations."""
    id: UUID = Field(default_factory=uuid4)
    operation: str
    service: str
    user_id: Optional[str] = None
    application_id: Optional[UUID] = None
    input_data: Dict[str, Any] = Field(default_factory=dict)
    output_data: Dict[str, Any] = Field(default_factory=dict)
    success: bool
    error_message: Optional[str] = None
    execution_time_ms: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    trace_id: Optional[str] = None