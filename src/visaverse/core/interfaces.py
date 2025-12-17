"""
Abstract base classes and interfaces for VisaVerse Guardian AI services.
These define the contracts that all service implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from uuid import UUID

from .models import (
    ProcessedDocument,
    StructuredData,
    QualityScore,
    ComplianceResult,
    ReasoningPath,
    VisaRule,
    RiskAssessment,
    ExplanationResult,
    HistoricalPattern,
    PersonalizedInsight,
    ApplicationData,
    ApplicationOutcome,
    AuditEntry
)


class DocumentProcessingService(ABC):
    """Abstract interface for document processing service."""
    
    @abstractmethod
    async def process_documents(
        self, 
        files: List[bytes], 
        filenames: List[str],
        language: Optional[str] = None
    ) -> List[ProcessedDocument]:
        """
        Process uploaded documents using OCR and extract structured data.
        
        Args:
            files: List of document file contents as bytes
            filenames: List of original filenames
            language: Optional language hint for OCR processing
            
        Returns:
            List of processed documents with extracted data
        """
        pass
    
    @abstractmethod
    async def extract_structured_data(
        self, 
        document: ProcessedDocument
    ) -> StructuredData:
        """
        Extract structured data from processed document.
        
        Args:
            document: Processed document from OCR
            
        Returns:
            Structured data extracted from document
        """
        pass
    
    @abstractmethod
    async def validate_document_quality(
        self, 
        document: ProcessedDocument
    ) -> QualityScore:
        """
        Assess the quality of document processing results.
        
        Args:
            document: Processed document to assess
            
        Returns:
            Quality score and assessment details
        """
        pass


class GraphReasoningService(ABC):
    """Abstract interface for graph-based reasoning service."""
    
    @abstractmethod
    async def validate_compliance(
        self,
        data: List[StructuredData],
        visa_type: str,
        country: str
    ) -> ComplianceResult:
        """
        Validate application data against visa compliance rules.
        
        Args:
            data: List of structured data from documents
            visa_type: Type of visa being applied for
            country: Target country for visa application
            
        Returns:
            Compliance validation results with reasoning
        """
        pass
    
    @abstractmethod
    async def get_reasoning_path(
        self, 
        validation_id: UUID
    ) -> ReasoningPath:
        """
        Retrieve detailed reasoning path for a validation.
        
        Args:
            validation_id: ID of the compliance validation
            
        Returns:
            Detailed reasoning path and steps
        """
        pass
    
    @abstractmethod
    async def update_rules(
        self, 
        country: str, 
        rules: List[VisaRule]
    ) -> None:
        """
        Update visa rules for a specific country.
        
        Args:
            country: Country code for rules
            rules: List of visa rules to update
        """
        pass
    
    @abstractmethod
    async def get_rules(
        self, 
        country: str, 
        visa_type: Optional[str] = None
    ) -> List[VisaRule]:
        """
        Retrieve visa rules for country and visa type.
        
        Args:
            country: Country code
            visa_type: Optional visa type filter
            
        Returns:
            List of applicable visa rules
        """
        pass


class RiskAssessmentService(ABC):
    """Abstract interface for risk assessment service."""
    
    @abstractmethod
    async def calculate_approval_probability(
        self,
        compliance: ComplianceResult,
        historical_data: List[HistoricalPattern],
        document_data: List[StructuredData]
    ) -> RiskAssessment:
        """
        Calculate visa approval probability and identify risk factors.
        
        Args:
            compliance: Compliance validation results
            historical_data: User's historical application patterns
            document_data: Structured data from current documents
            
        Returns:
            Risk assessment with probability and recommendations
        """
        pass
    
    @abstractmethod
    async def generate_explanation(
        self,
        assessment: RiskAssessment,
        language: str = "en"
    ) -> ExplanationResult:
        """
        Generate natural language explanation of risk assessment.
        
        Args:
            assessment: Risk assessment to explain
            language: Language for explanation (ISO code)
            
        Returns:
            Natural language explanation and recommendations
        """
        pass
    
    @abstractmethod
    async def identify_cross_document_issues(
        self,
        documents: List[StructuredData]
    ) -> List[Dict[str, Any]]:
        """
        Identify inconsistencies across multiple documents.
        
        Args:
            documents: List of structured document data
            
        Returns:
            List of identified cross-document issues
        """
        pass


class MemoryService(ABC):
    """Abstract interface for persistent memory service."""
    
    @abstractmethod
    async def store_application(
        self, 
        user_id: str, 
        application: ApplicationData
    ) -> UUID:
        """
        Store application data for future reference and learning.
        
        Args:
            user_id: Unique user identifier
            application: Complete application data
            
        Returns:
            Stored application ID
        """
        pass
    
    @abstractmethod
    async def get_user_history(
        self, 
        user_id: str
    ) -> List[HistoricalPattern]:
        """
        Retrieve user's historical application patterns.
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            List of historical patterns and outcomes
        """
        pass
    
    @abstractmethod
    async def update_outcome(
        self, 
        application_id: UUID, 
        outcome: ApplicationOutcome
    ) -> None:
        """
        Update the outcome of a stored application.
        
        Args:
            application_id: ID of stored application
            outcome: Final outcome of the application
        """
        pass
    
    @abstractmethod
    async def get_personalized_insights(
        self,
        user_id: str,
        current_application: ApplicationData
    ) -> List[PersonalizedInsight]:
        """
        Generate personalized insights based on user history.
        
        Args:
            user_id: Unique user identifier
            current_application: Current application being processed
            
        Returns:
            List of personalized insights and recommendations
        """
        pass
    
    @abstractmethod
    async def learn_from_patterns(
        self, 
        applications: List[ApplicationData]
    ) -> Dict[str, Any]:
        """
        Learn patterns from multiple applications for system improvement.
        
        Args:
            applications: List of applications to learn from
            
        Returns:
            Learned patterns and insights
        """
        pass


class AuditService(ABC):
    """Abstract interface for audit trail service."""
    
    @abstractmethod
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
        """
        Log an operation to the audit trail.
        
        Args:
            operation: Name of the operation performed
            service: Service that performed the operation
            user_id: Optional user identifier
            application_id: Optional application identifier
            input_data: Optional input data for the operation
            output_data: Optional output data from the operation
            success: Whether the operation succeeded
            error_message: Optional error message if operation failed
            execution_time_ms: Optional execution time in milliseconds
            trace_id: Optional trace ID for request correlation
            
        Returns:
            Audit entry ID
        """
        pass
    
    @abstractmethod
    async def get_audit_trail(
        self,
        user_id: Optional[str] = None,
        application_id: Optional[UUID] = None,
        service: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100
    ) -> List[AuditEntry]:
        """
        Retrieve audit trail entries based on filters.
        
        Args:
            user_id: Optional user filter
            application_id: Optional application filter
            service: Optional service filter
            start_time: Optional start time filter (ISO format)
            end_time: Optional end time filter (ISO format)
            limit: Maximum number of entries to return
            
        Returns:
            List of audit entries matching filters
        """
        pass
    
    @abstractmethod
    async def ensure_audit_completeness(
        self,
        application_id: UUID
    ) -> bool:
        """
        Verify that all required audit entries exist for an application.
        
        Args:
            application_id: Application to check audit completeness for
            
        Returns:
            True if audit trail is complete, False otherwise
        """
        pass


class OrchestrationService(ABC):
    """Abstract interface for service orchestration."""
    
    @abstractmethod
    async def process_visa_application(
        self,
        user_id: str,
        visa_type: str,
        country: str,
        documents: List[bytes],
        filenames: List[str],
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Orchestrate complete visa application processing pipeline.
        
        Args:
            user_id: Unique user identifier
            visa_type: Type of visa being applied for
            country: Target country
            documents: List of document file contents
            filenames: List of original filenames
            language: Optional language hint
            
        Returns:
            Complete processing results including risk assessment and explanations
        """
        pass
    
    @abstractmethod
    async def get_application_status(
        self,
        application_id: UUID
    ) -> Dict[str, Any]:
        """
        Get current status and results of an application.
        
        Args:
            application_id: Application identifier
            
        Returns:
            Current application status and available results
        """
        pass