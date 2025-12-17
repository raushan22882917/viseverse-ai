"""
Audit Service implementation for reasoning traceability and transparency.
Handles data source tracking, reasoning path validation, and transparency reporting.
"""

import logging
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime
import json
import hashlib
from dataclasses import asdict

from ...core.interfaces import AuditService
from ...core.models import (
    AuditEntry,
    ReasoningPath,
    StructuredData,
    ComplianceResult,
    RiskAssessment,
    ExplanationResult
)


logger = logging.getLogger(__name__)


class TransparencyAuditService(AuditService):
    """
    Audit service for maintaining complete traceability and transparency.
    """
    
    def __init__(self, storage_config: Dict[str, Any]):
        """
        Initialize the audit service.
        
        Args:
            storage_config: Configuration for audit storage backend
        """
        self.storage_config = storage_config
        
        # In-memory storage for development (replace with persistent storage)
        self._audit_entries: Dict[UUID, AuditEntry] = {}
        self._data_lineage: Dict[str, List[Dict[str, Any]]] = {}
        self._reasoning_traces: Dict[UUID, Dict[str, Any]] = {}
        
        logger.info("Initialized transparency audit service")
    
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
        try:
            entry_id = uuid4()
            
            # Create data hash for integrity verification
            data_hash = self._create_data_hash(input_data, output_data)
            
            # Create audit entry
            audit_entry = AuditEntry(
                id=entry_id,
                timestamp=datetime.now(),
                operation=operation,
                service=service,
                user_id=user_id,
                application_id=application_id,
                input_data=input_data or {},
                output_data=output_data or {},
                success=success,
                error_message=error_message,
                execution_time_ms=execution_time_ms,
                trace_id=trace_id or str(uuid4()),
                data_hash=data_hash
            )
            
            # Store audit entry
            self._audit_entries[entry_id] = audit_entry
            
            # Track data lineage
            await self._track_data_lineage(audit_entry)
            
            logger.info(f"Logged operation {operation} from {service} with ID {entry_id}")
            return entry_id
        
        except Exception as e:
            logger.error(f"Error logging operation: {str(e)}")
            raise
    
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
        try:
            entries = list(self._audit_entries.values())
            
            # Apply filters
            if user_id:
                entries = [e for e in entries if e.user_id == user_id]
            
            if application_id:
                entries = [e for e in entries if e.application_id == application_id]
            
            if service:
                entries = [e for e in entries if e.service == service]
            
            if start_time:
                start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                entries = [e for e in entries if e.timestamp >= start_dt]
            
            if end_time:
                end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                entries = [e for e in entries if e.timestamp <= end_dt]
            
            # Sort by timestamp (most recent first)
            entries.sort(key=lambda x: x.timestamp, reverse=True)
            
            # Apply limit
            entries = entries[:limit]
            
            logger.info(f"Retrieved {len(entries)} audit entries")
            return entries
        
        except Exception as e:
            logger.error(f"Error retrieving audit trail: {str(e)}")
            return []
    
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
        try:
            # Get all audit entries for this application
            entries = await self.get_audit_trail(application_id=application_id)
            
            # Define required operations for complete audit trail
            required_operations = {
                'document_processing',
                'compliance_validation',
                'risk_assessment',
                'explanation_generation'
            }
            
            # Check which operations have been logged
            logged_operations = {entry.operation for entry in entries}
            
            # Check if all required operations are present
            missing_operations = required_operations - logged_operations
            
            if missing_operations:
                logger.warning(f"Missing audit entries for application {application_id}: {missing_operations}")
                return False
            
            # Verify data integrity
            for entry in entries:
                if not self._verify_data_integrity(entry):
                    logger.warning(f"Data integrity check failed for audit entry {entry.id}")
                    return False
            
            logger.info(f"Audit trail is complete for application {application_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error checking audit completeness: {str(e)}")
            return False
    
    async def track_data_source(
        self,
        data_id: str,
        source_type: str,
        source_details: Dict[str, Any],
        processing_steps: List[Dict[str, Any]]
    ) -> None:
        """
        Track the source and processing history of data.
        
        Args:
            data_id: Unique identifier for the data
            source_type: Type of data source (e.g., 'document', 'user_input', 'api')
            source_details: Details about the data source
            processing_steps: List of processing steps applied to the data
        """
        try:
            lineage_entry = {
                'data_id': data_id,
                'source_type': source_type,
                'source_details': source_details,
                'processing_steps': processing_steps,
                'timestamp': datetime.now().isoformat(),
                'integrity_hash': self._create_data_hash(source_details, processing_steps)
            }
            
            if data_id not in self._data_lineage:
                self._data_lineage[data_id] = []
            
            self._data_lineage[data_id].append(lineage_entry)
            
            logger.info(f"Tracked data lineage for {data_id}")
        
        except Exception as e:
            logger.error(f"Error tracking data source: {str(e)}")
            raise
    
    async def validate_reasoning_path(
        self,
        reasoning_path: ReasoningPath,
        source_data: List[StructuredData]
    ) -> Dict[str, Any]:
        """
        Validate that reasoning path is grounded in source data.
        
        Args:
            reasoning_path: Reasoning path to validate
            source_data: Source data that reasoning should be based on
            
        Returns:
            Validation results with grounding verification
        """
        try:
            validation_result = {
                'is_valid': True,
                'grounding_score': 0.0,
                'ungrounded_steps': [],
                'data_references': [],
                'validation_timestamp': datetime.now().isoformat()
            }
            
            # Extract all data references from source data
            data_references = self._extract_data_references(source_data)
            validation_result['data_references'] = data_references
            
            # Validate each reasoning step
            grounded_steps = 0
            total_steps = len(reasoning_path.steps)
            
            for i, step in enumerate(reasoning_path.steps):
                is_grounded = self._validate_step_grounding(step, data_references)
                
                if is_grounded:
                    grounded_steps += 1
                else:
                    validation_result['ungrounded_steps'].append({
                        'step_index': i,
                        'step_description': step.description,
                        'reason': 'No supporting data found'
                    })
            
            # Calculate grounding score
            if total_steps > 0:
                validation_result['grounding_score'] = grounded_steps / total_steps
            
            # Mark as invalid if grounding score is too low
            if validation_result['grounding_score'] < 0.8:
                validation_result['is_valid'] = False
            
            logger.info(f"Validated reasoning path: {validation_result['grounding_score']:.2f} grounding score")
            return validation_result
        
        except Exception as e:
            logger.error(f"Error validating reasoning path: {str(e)}")
            return {'is_valid': False, 'error': str(e)}
    
    async def generate_transparency_report(
        self,
        application_id: UUID
    ) -> Dict[str, Any]:
        """
        Generate comprehensive transparency report for an application.
        
        Args:
            application_id: Application to generate report for
            
        Returns:
            Comprehensive transparency report
        """
        try:
            # Get audit trail for application
            audit_entries = await self.get_audit_trail(application_id=application_id)
            
            # Build transparency report
            report = {
                'application_id': str(application_id),
                'report_timestamp': datetime.now().isoformat(),
                'audit_completeness': await self.ensure_audit_completeness(application_id),
                'processing_pipeline': self._build_processing_pipeline(audit_entries),
                'data_sources': self._identify_data_sources(audit_entries),
                'decision_factors': self._extract_decision_factors(audit_entries),
                'transparency_score': 0.0,
                'disclaimers': self._generate_disclaimers()
            }
            
            # Calculate transparency score
            report['transparency_score'] = self._calculate_transparency_score(report)
            
            logger.info(f"Generated transparency report for application {application_id}")
            return report
        
        except Exception as e:
            logger.error(f"Error generating transparency report: {str(e)}")
            return {'error': str(e)}
    
    def _create_data_hash(self, input_data: Optional[Dict], output_data: Optional[Dict]) -> str:
        """Create hash for data integrity verification."""
        combined_data = {
            'input': input_data or {},
            'output': output_data or {}
        }
        
        data_str = json.dumps(combined_data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def _verify_data_integrity(self, entry: AuditEntry) -> bool:
        """Verify data integrity of audit entry."""
        expected_hash = self._create_data_hash(entry.input_data, entry.output_data)
        return entry.data_hash == expected_hash
    
    async def _track_data_lineage(self, audit_entry: AuditEntry) -> None:
        """Track data lineage from audit entry."""
        if audit_entry.input_data:
            for key, value in audit_entry.input_data.items():
                if isinstance(value, dict) and 'data_id' in value:
                    data_id = value['data_id']
                    await self.track_data_source(
                        data_id=data_id,
                        source_type=audit_entry.service,
                        source_details={'operation': audit_entry.operation},
                        processing_steps=[{
                            'step': audit_entry.operation,
                            'timestamp': audit_entry.timestamp.isoformat(),
                            'service': audit_entry.service
                        }]
                    )
    
    def _extract_data_references(self, source_data: List[StructuredData]) -> List[Dict[str, Any]]:
        """Extract all data references from source data."""
        references = []
        
        for data in source_data:
            # Extract key fields as references
            for field_name, field_value in data.key_fields.items():
                references.append({
                    'source': 'document',
                    'document_type': data.document_type.value,
                    'field_name': field_name,
                    'field_value': str(field_value),
                    'confidence': data.extraction_confidence
                })
            
            # Extract dates as references
            for date_field in data.dates:
                references.append({
                    'source': 'document',
                    'document_type': data.document_type.value,
                    'field_name': date_field.field_name,
                    'field_value': date_field.date_value.isoformat(),
                    'confidence': data.extraction_confidence
                })
        
        return references
    
    def _validate_step_grounding(self, step, data_references: List[Dict[str, Any]]) -> bool:
        """Validate that a reasoning step is grounded in data."""
        # Simple validation - check if step mentions any data fields
        step_text = step.description.lower()
        
        for ref in data_references:
            field_name = ref['field_name'].lower()
            field_value = str(ref['field_value']).lower()
            
            if field_name in step_text or field_value in step_text:
                return True
        
        return False
    
    def _build_processing_pipeline(self, audit_entries: List[AuditEntry]) -> List[Dict[str, Any]]:
        """Build processing pipeline from audit entries."""
        pipeline = []
        
        # Sort entries by timestamp
        sorted_entries = sorted(audit_entries, key=lambda x: x.timestamp)
        
        for entry in sorted_entries:
            pipeline.append({
                'operation': entry.operation,
                'service': entry.service,
                'timestamp': entry.timestamp.isoformat(),
                'success': entry.success,
                'execution_time_ms': entry.execution_time_ms
            })
        
        return pipeline
    
    def _identify_data_sources(self, audit_entries: List[AuditEntry]) -> List[Dict[str, Any]]:
        """Identify all data sources from audit entries."""
        sources = []
        
        for entry in audit_entries:
            if entry.operation == 'document_processing' and entry.input_data:
                sources.append({
                    'type': 'document',
                    'details': entry.input_data,
                    'timestamp': entry.timestamp.isoformat()
                })
        
        return sources
    
    def _extract_decision_factors(self, audit_entries: List[AuditEntry]) -> List[Dict[str, Any]]:
        """Extract decision factors from audit entries."""
        factors = []
        
        for entry in audit_entries:
            if entry.operation == 'risk_assessment' and entry.output_data:
                output = entry.output_data
                if 'risk_factors' in output:
                    for risk_factor in output['risk_factors']:
                        factors.append({
                            'type': 'risk_factor',
                            'category': risk_factor.get('category', 'unknown'),
                            'description': risk_factor.get('description', ''),
                            'impact': risk_factor.get('impact', 0.0)
                        })
        
        return factors
    
    def _calculate_transparency_score(self, report: Dict[str, Any]) -> float:
        """Calculate overall transparency score."""
        score = 0.0
        
        # Audit completeness (40% weight)
        if report['audit_completeness']:
            score += 0.4
        
        # Processing pipeline completeness (30% weight)
        pipeline_steps = len(report['processing_pipeline'])
        if pipeline_steps >= 4:  # Expected minimum steps
            score += 0.3
        elif pipeline_steps >= 2:
            score += 0.15
        
        # Data source tracking (20% weight)
        if report['data_sources']:
            score += 0.2
        
        # Decision factor extraction (10% weight)
        if report['decision_factors']:
            score += 0.1
        
        return min(1.0, score)
    
    def _generate_disclaimers(self) -> List[str]:
        """Generate standard disclaimers for transparency."""
        return [
            "This analysis is for informational purposes only and does not constitute legal advice.",
            "Visa application outcomes depend on many factors beyond this analysis.",
            "Always consult with qualified immigration professionals for official guidance.",
            "The AI system's predictions are based on available data and may not reflect all relevant factors.",
            "Final visa decisions are made by immigration authorities, not by this system."
        ]