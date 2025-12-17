"""
Orchestration Service implementation with full traceability and transparency.
Coordinates all services while maintaining complete audit trails.
"""

import logging
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime
import asyncio
import time

from ...core.interfaces import (
    OrchestrationService,
    DocumentProcessingService,
    GraphReasoningService,
    RiskAssessmentService,
    MemoryService,
    AuditService
)
from ...core.models import (
    ApplicationData,
    ProcessedDocument,
    StructuredData,
    ComplianceResult,
    RiskAssessment,
    ExplanationResult,
    HistoricalPattern,
    PersonalizedInsight,
    DocumentType
)


logger = logging.getLogger(__name__)


class TransparentOrchestrationService(OrchestrationService):
    """
    Orchestration service with complete traceability and transparency.
    """
    
    def __init__(
        self,
        document_service: DocumentProcessingService,
        graph_service: GraphReasoningService,
        risk_service: RiskAssessmentService,
        memory_service: MemoryService,
        audit_service: AuditService
    ):
        """
        Initialize the orchestration service with all dependencies.
        
        Args:
            document_service: Document processing service
            graph_service: Graph reasoning service
            risk_service: Risk assessment service
            memory_service: Memory service
            audit_service: Audit service for traceability
        """
        self.document_service = document_service
        self.graph_service = graph_service
        self.risk_service = risk_service
        self.memory_service = memory_service
        self.audit_service = audit_service
        
        logger.info("Initialized transparent orchestration service")
    
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
        Orchestrate complete visa application processing pipeline with full traceability.
        
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
        application_id = uuid4()
        trace_id = str(uuid4())
        start_time = time.time()
        
        try:
            logger.info(f"Starting visa application processing for user {user_id}, application {application_id}")
            
            # Step 1: Document Processing with audit trail
            processed_documents = await self._process_documents_with_audit(
                documents, filenames, language, user_id, application_id, trace_id
            )
            
            # Step 2: Extract structured data with audit trail
            structured_data = await self._extract_structured_data_with_audit(
                processed_documents, user_id, application_id, trace_id
            )
            
            # Step 3: Compliance validation with audit trail
            compliance_result = await self._validate_compliance_with_audit(
                structured_data, visa_type, country, user_id, application_id, trace_id
            )
            
            # Step 4: Get user history with audit trail
            user_history = await self._get_user_history_with_audit(
                user_id, application_id, trace_id
            )
            
            # Step 5: Risk assessment with audit trail
            risk_assessment = await self._calculate_risk_with_audit(
                compliance_result, user_history, structured_data, user_id, application_id, trace_id
            )
            
            # Step 6: Generate explanation with audit trail
            explanation = await self._generate_explanation_with_audit(
                risk_assessment, language or "en", user_id, application_id, trace_id
            )
            
            # Step 7: Store application data with audit trail
            await self._store_application_with_audit(
                user_id, application_id, visa_type, country, structured_data,
                compliance_result, risk_assessment, trace_id
            )
            
            # Step 8: Generate personalized insights with audit trail
            insights = await self._generate_insights_with_audit(
                user_id, application_id, structured_data, visa_type, country, trace_id
            )
            
            # Step 9: Generate transparency report
            transparency_report = await self.audit_service.generate_transparency_report(application_id)
            
            # Calculate total processing time
            total_time = (time.time() - start_time) * 1000
            
            # Log final orchestration result
            await self.audit_service.log_operation(
                operation="visa_application_processing",
                service="orchestration",
                user_id=user_id,
                application_id=application_id,
                input_data={
                    "visa_type": visa_type,
                    "country": country,
                    "document_count": len(documents),
                    "language": language
                },
                output_data={
                    "approval_probability": risk_assessment.approval_probability,
                    "risk_factor_count": len(risk_assessment.risk_factors),
                    "recommendation_count": len(risk_assessment.recommendations),
                    "transparency_score": transparency_report.get('transparency_score', 0.0)
                },
                success=True,
                execution_time_ms=total_time,
                trace_id=trace_id
            )
            
            # Build comprehensive response
            response = {
                "application_id": str(application_id),
                "trace_id": trace_id,
                "processing_time_ms": total_time,
                "documents": [self._serialize_document(doc) for doc in processed_documents],
                "structured_data": [self._serialize_structured_data(data) for data in structured_data],
                "compliance": self._serialize_compliance_result(compliance_result),
                "risk_assessment": self._serialize_risk_assessment(risk_assessment),
                "explanation": self._serialize_explanation(explanation),
                "personalized_insights": [self._serialize_insight(insight) for insight in insights],
                "transparency_report": transparency_report,
                "disclaimers": [
                    "This analysis is for informational purposes only and does not constitute legal advice.",
                    "Visa application outcomes depend on many factors beyond this analysis.",
                    "Always consult with qualified immigration professionals for official guidance."
                ]
            }
            
            logger.info(f"Completed visa application processing for application {application_id}")
            return response
        
        except Exception as e:
            # Log error with audit trail
            error_time = (time.time() - start_time) * 1000
            await self.audit_service.log_operation(
                operation="visa_application_processing",
                service="orchestration",
                user_id=user_id,
                application_id=application_id,
                input_data={
                    "visa_type": visa_type,
                    "country": country,
                    "document_count": len(documents)
                },
                success=False,
                error_message=str(e),
                execution_time_ms=error_time,
                trace_id=trace_id
            )
            
            logger.error(f"Error processing visa application: {str(e)}")
            raise
    
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
        try:
            # Get audit trail for application
            audit_entries = await self.audit_service.get_audit_trail(
                application_id=application_id
            )
            
            if not audit_entries:
                return {
                    "application_id": str(application_id),
                    "status": "not_found",
                    "message": "No processing records found for this application"
                }
            
            # Determine processing status
            operations = {entry.operation for entry in audit_entries}
            
            status = "in_progress"
            if "visa_application_processing" in operations:
                # Check if final operation was successful
                final_entry = max(audit_entries, key=lambda x: x.timestamp)
                if final_entry.operation == "visa_application_processing" and final_entry.success:
                    status = "completed"
                elif not final_entry.success:
                    status = "failed"
            
            # Build status response
            response = {
                "application_id": str(application_id),
                "status": status,
                "last_updated": max(entry.timestamp for entry in audit_entries).isoformat(),
                "operations_completed": list(operations),
                "audit_completeness": await self.audit_service.ensure_audit_completeness(application_id),
                "processing_steps": len(audit_entries)
            }
            
            # Add results if processing is complete
            if status == "completed":
                final_result = next(
                    (entry for entry in audit_entries 
                     if entry.operation == "visa_application_processing" and entry.success),
                    None
                )
                if final_result and final_result.output_data:
                    response["results"] = final_result.output_data
            
            return response
        
        except Exception as e:
            logger.error(f"Error getting application status: {str(e)}")
            return {
                "application_id": str(application_id),
                "status": "error",
                "error": str(e)
            }
    
    async def _process_documents_with_audit(
        self, documents: List[bytes], filenames: List[str], language: Optional[str],
        user_id: str, application_id: UUID, trace_id: str
    ) -> List[ProcessedDocument]:
        """Process documents with audit trail."""
        start_time = time.time()
        
        try:
            processed_docs = await self.document_service.process_documents(
                documents, filenames, language
            )
            
            execution_time = (time.time() - start_time) * 1000
            
            await self.audit_service.log_operation(
                operation="document_processing",
                service="document_service",
                user_id=user_id,
                application_id=application_id,
                input_data={
                    "document_count": len(documents),
                    "filenames": filenames,
                    "language": language
                },
                output_data={
                    "processed_count": len(processed_docs),
                    "document_types": [doc.document_type.value for doc in processed_docs]
                },
                success=True,
                execution_time_ms=execution_time,
                trace_id=trace_id
            )
            
            return processed_docs
        
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            await self.audit_service.log_operation(
                operation="document_processing",
                service="document_service",
                user_id=user_id,
                application_id=application_id,
                input_data={"document_count": len(documents)},
                success=False,
                error_message=str(e),
                execution_time_ms=execution_time,
                trace_id=trace_id
            )
            raise
    
    async def _extract_structured_data_with_audit(
        self, processed_docs: List[ProcessedDocument], user_id: str,
        application_id: UUID, trace_id: str
    ) -> List[StructuredData]:
        """Extract structured data with audit trail."""
        start_time = time.time()
        
        try:
            structured_data = []
            for doc in processed_docs:
                data = await self.document_service.extract_structured_data(doc)
                structured_data.append(data)
            
            execution_time = (time.time() - start_time) * 1000
            
            await self.audit_service.log_operation(
                operation="data_extraction",
                service="document_service",
                user_id=user_id,
                application_id=application_id,
                input_data={
                    "document_count": len(processed_docs)
                },
                output_data={
                    "extracted_count": len(structured_data),
                    "average_confidence": sum(d.extraction_confidence for d in structured_data) / len(structured_data)
                },
                success=True,
                execution_time_ms=execution_time,
                trace_id=trace_id
            )
            
            return structured_data
        
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            await self.audit_service.log_operation(
                operation="data_extraction",
                service="document_service",
                user_id=user_id,
                application_id=application_id,
                success=False,
                error_message=str(e),
                execution_time_ms=execution_time,
                trace_id=trace_id
            )
            raise
    
    async def _validate_compliance_with_audit(
        self, structured_data: List[StructuredData], visa_type: str, country: str,
        user_id: str, application_id: UUID, trace_id: str
    ) -> ComplianceResult:
        """Validate compliance with audit trail."""
        start_time = time.time()
        
        try:
            compliance = await self.graph_service.validate_compliance(
                structured_data, visa_type, country
            )
            
            execution_time = (time.time() - start_time) * 1000
            
            await self.audit_service.log_operation(
                operation="compliance_validation",
                service="graph_service",
                user_id=user_id,
                application_id=application_id,
                input_data={
                    "visa_type": visa_type,
                    "country": country,
                    "document_count": len(structured_data)
                },
                output_data={
                    "is_compliant": compliance.is_compliant,
                    "violation_count": len(compliance.violations),
                    "confidence": compliance.confidence
                },
                success=True,
                execution_time_ms=execution_time,
                trace_id=trace_id
            )
            
            return compliance
        
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            await self.audit_service.log_operation(
                operation="compliance_validation",
                service="graph_service",
                user_id=user_id,
                application_id=application_id,
                success=False,
                error_message=str(e),
                execution_time_ms=execution_time,
                trace_id=trace_id
            )
            raise
    
    async def _get_user_history_with_audit(
        self, user_id: str, application_id: UUID, trace_id: str
    ) -> List[HistoricalPattern]:
        """Get user history with audit trail."""
        start_time = time.time()
        
        try:
            history = await self.memory_service.get_user_history(user_id)
            
            execution_time = (time.time() - start_time) * 1000
            
            await self.audit_service.log_operation(
                operation="history_retrieval",
                service="memory_service",
                user_id=user_id,
                application_id=application_id,
                input_data={"user_id": user_id},
                output_data={"history_count": len(history)},
                success=True,
                execution_time_ms=execution_time,
                trace_id=trace_id
            )
            
            return history
        
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            await self.audit_service.log_operation(
                operation="history_retrieval",
                service="memory_service",
                user_id=user_id,
                application_id=application_id,
                success=False,
                error_message=str(e),
                execution_time_ms=execution_time,
                trace_id=trace_id
            )
            raise
    
    async def _calculate_risk_with_audit(
        self, compliance: ComplianceResult, history: List[HistoricalPattern],
        structured_data: List[StructuredData], user_id: str, application_id: UUID, trace_id: str
    ) -> RiskAssessment:
        """Calculate risk assessment with audit trail."""
        start_time = time.time()
        
        try:
            risk_assessment = await self.risk_service.calculate_approval_probability(
                compliance, history, structured_data
            )
            
            execution_time = (time.time() - start_time) * 1000
            
            await self.audit_service.log_operation(
                operation="risk_assessment",
                service="risk_service",
                user_id=user_id,
                application_id=application_id,
                input_data={
                    "compliance_status": compliance.is_compliant,
                    "history_count": len(history)
                },
                output_data={
                    "approval_probability": risk_assessment.approval_probability,
                    "risk_factor_count": len(risk_assessment.risk_factors),
                    "confidence_level": risk_assessment.confidence_level
                },
                success=True,
                execution_time_ms=execution_time,
                trace_id=trace_id
            )
            
            return risk_assessment
        
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            await self.audit_service.log_operation(
                operation="risk_assessment",
                service="risk_service",
                user_id=user_id,
                application_id=application_id,
                success=False,
                error_message=str(e),
                execution_time_ms=execution_time,
                trace_id=trace_id
            )
            raise
    
    async def _generate_explanation_with_audit(
        self, risk_assessment: RiskAssessment, language: str,
        user_id: str, application_id: UUID, trace_id: str
    ) -> ExplanationResult:
        """Generate explanation with audit trail."""
        start_time = time.time()
        
        try:
            explanation = await self.risk_service.generate_explanation(
                risk_assessment, language
            )
            
            execution_time = (time.time() - start_time) * 1000
            
            await self.audit_service.log_operation(
                operation="explanation_generation",
                service="risk_service",
                user_id=user_id,
                application_id=application_id,
                input_data={
                    "language": language,
                    "approval_probability": risk_assessment.approval_probability
                },
                output_data={
                    "explanation_length": len(explanation.detailed_explanation),
                    "actionable_steps": len(explanation.actionable_steps)
                },
                success=True,
                execution_time_ms=execution_time,
                trace_id=trace_id
            )
            
            return explanation
        
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            await self.audit_service.log_operation(
                operation="explanation_generation",
                service="risk_service",
                user_id=user_id,
                application_id=application_id,
                success=False,
                error_message=str(e),
                execution_time_ms=execution_time,
                trace_id=trace_id
            )
            raise
    
    async def _store_application_with_audit(
        self, user_id: str, application_id: UUID, visa_type: str, country: str,
        structured_data: List[StructuredData], compliance: ComplianceResult,
        risk_assessment: RiskAssessment, trace_id: str
    ) -> None:
        """Store application data with audit trail."""
        start_time = time.time()
        
        try:
            # Create application data object
            application_data = ApplicationData(
                id=application_id,
                user_id=user_id,
                visa_type=visa_type,
                country=country,
                documents=structured_data,
                submission_date=datetime.now(),
                compliance_score=compliance.confidence,
                approval_probability=risk_assessment.approval_probability
            )
            
            stored_id = await self.memory_service.store_application(user_id, application_data)
            
            execution_time = (time.time() - start_time) * 1000
            
            await self.audit_service.log_operation(
                operation="application_storage",
                service="memory_service",
                user_id=user_id,
                application_id=application_id,
                input_data={
                    "visa_type": visa_type,
                    "country": country
                },
                output_data={
                    "stored_id": str(stored_id)
                },
                success=True,
                execution_time_ms=execution_time,
                trace_id=trace_id
            )
        
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            await self.audit_service.log_operation(
                operation="application_storage",
                service="memory_service",
                user_id=user_id,
                application_id=application_id,
                success=False,
                error_message=str(e),
                execution_time_ms=execution_time,
                trace_id=trace_id
            )
            raise
    
    async def _generate_insights_with_audit(
        self, user_id: str, application_id: UUID, structured_data: List[StructuredData],
        visa_type: str, country: str, trace_id: str
    ) -> List[PersonalizedInsight]:
        """Generate personalized insights with audit trail."""
        start_time = time.time()
        
        try:
            # Create current application data for insights
            current_app = ApplicationData(
                id=application_id,
                user_id=user_id,
                visa_type=visa_type,
                country=country,
                documents=structured_data,
                submission_date=datetime.now()
            )
            
            insights = await self.memory_service.get_personalized_insights(
                user_id, current_app
            )
            
            execution_time = (time.time() - start_time) * 1000
            
            await self.audit_service.log_operation(
                operation="insight_generation",
                service="memory_service",
                user_id=user_id,
                application_id=application_id,
                input_data={
                    "visa_type": visa_type,
                    "country": country
                },
                output_data={
                    "insight_count": len(insights)
                },
                success=True,
                execution_time_ms=execution_time,
                trace_id=trace_id
            )
            
            return insights
        
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            await self.audit_service.log_operation(
                operation="insight_generation",
                service="memory_service",
                user_id=user_id,
                application_id=application_id,
                success=False,
                error_message=str(e),
                execution_time_ms=execution_time,
                trace_id=trace_id
            )
            return []  # Return empty list on error for insights
    
    # Serialization methods for response formatting
    def _serialize_document(self, doc: ProcessedDocument) -> Dict[str, Any]:
        """Serialize processed document for response."""
        return {
            "document_type": doc.document_type.value,
            "confidence_score": doc.confidence_score,
            "language": doc.language,
            "page_count": getattr(doc, 'page_count', 1)
        }
    
    def _serialize_structured_data(self, data: StructuredData) -> Dict[str, Any]:
        """Serialize structured data for response."""
        return {
            "document_type": data.document_type.value,
            "extraction_confidence": data.extraction_confidence,
            "key_fields_count": len(data.key_fields),
            "dates_count": len(data.dates),
            "missing_fields_count": len(data.missing_fields)
        }
    
    def _serialize_compliance_result(self, compliance: ComplianceResult) -> Dict[str, Any]:
        """Serialize compliance result for response."""
        return {
            "is_compliant": compliance.is_compliant,
            "confidence": compliance.confidence,
            "violation_count": len(compliance.violations),
            "violations": [
                {
                    "type": v.violation_type,
                    "severity": v.severity.value,
                    "explanation": v.explanation
                }
                for v in compliance.violations
            ]
        }
    
    def _serialize_risk_assessment(self, assessment: RiskAssessment) -> Dict[str, Any]:
        """Serialize risk assessment for response."""
        return {
            "approval_probability": assessment.approval_probability,
            "confidence_level": assessment.confidence_level,
            "risk_factor_count": len(assessment.risk_factors),
            "recommendation_count": len(assessment.recommendations),
            "risk_factors": [
                {
                    "category": rf.category.value,
                    "severity": rf.severity.value,
                    "description": rf.description,
                    "impact": rf.impact,
                    "recommendation": rf.recommendation
                }
                for rf in assessment.risk_factors
            ],
            "recommendations": [
                {
                    "title": rec.title,
                    "description": rec.description,
                    "priority": rec.priority,
                    "action_required": rec.action_required,
                    "estimated_impact": rec.estimated_impact
                }
                for rec in assessment.recommendations
            ]
        }
    
    def _serialize_explanation(self, explanation: ExplanationResult) -> Dict[str, Any]:
        """Serialize explanation result for response."""
        return {
            "summary": explanation.summary,
            "detailed_explanation": explanation.detailed_explanation,
            "actionable_steps": explanation.actionable_steps,
            "language": explanation.language
        }
    
    def _serialize_insight(self, insight: PersonalizedInsight) -> Dict[str, Any]:
        """Serialize personalized insight for response."""
        return {
            "type": insight.insight_type,
            "title": insight.title,
            "description": insight.description,
            "relevance_score": insight.relevance_score,
            "confidence_level": insight.confidence_level,
            "recommendation": insight.recommendation,
            "historical_context": insight.historical_context
        }