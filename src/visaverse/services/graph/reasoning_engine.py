"""
Neo4j-based Graph Reasoning Service implementation.
Handles visa rule modeling, compliance validation, and multi-hop reasoning.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID, uuid4
from datetime import datetime
import json

from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import ServiceUnavailable, AuthError

from ...core.interfaces import GraphReasoningService
from ...core.models import (
    ComplianceResult,
    ReasoningPath,
    VisaRule,
    StructuredData,
    RuleViolation,
    ReasoningStep,
    RiskSeverity,
    RuleType,
    ComparisonOperator,
    DocumentType
)


logger = logging.getLogger(__name__)


class Neo4jGraphReasoningService(GraphReasoningService):
    """
    Neo4j-based graph reasoning service for visa compliance validation.
    """
    
    def __init__(self, uri: str, user: str, password: str):
        """
        Initialize the Neo4j graph reasoning service.
        
        Args:
            uri: Neo4j database URI
            user: Database username
            password: Database password
        """
        self.uri = uri
        self.user = user
        self.password = password
        self._driver: Optional[Driver] = None
        
        # Initialize database connection
        self._connect()
        
        # Initialize schema if needed
        self._initialize_schema()
    
    def _connect(self) -> None:
        """Establish connection to Neo4j database."""
        try:
            self._driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.user, self.password)
            )
            # Test connection
            with self._driver.session() as session:
                session.run("RETURN 1")
            logger.info("Successfully connected to Neo4j database")
        except (ServiceUnavailable, AuthError) as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            self._driver = None
    
    def _get_session(self) -> Session:
        """Get Neo4j session."""
        if not self._driver:
            self._connect()
        
        if not self._driver:
            raise RuntimeError("No Neo4j connection available")
        
        return self._driver.session()
    
    def _initialize_schema(self) -> None:
        """Initialize Neo4j schema with constraints and indexes."""
        try:
            with self._get_session() as session:
                # Create constraints
                constraints = [
                    "CREATE CONSTRAINT country_code IF NOT EXISTS FOR (c:Country) REQUIRE c.code IS UNIQUE",
                    "CREATE CONSTRAINT visa_type_id IF NOT EXISTS FOR (v:VisaType) REQUIRE v.id IS UNIQUE",
                    "CREATE CONSTRAINT document_type_name IF NOT EXISTS FOR (d:DocumentType) REQUIRE d.name IS UNIQUE",
                    "CREATE CONSTRAINT requirement_id IF NOT EXISTS FOR (r:Requirement) REQUIRE r.id IS UNIQUE",
                    "CREATE CONSTRAINT field_name IF NOT EXISTS FOR (f:Field) REQUIRE f.name IS UNIQUE",
                    "CREATE CONSTRAINT rule_id IF NOT EXISTS FOR (r:Rule) REQUIRE r.id IS UNIQUE"
                ]
                
                for constraint in constraints:
                    try:
                        session.run(constraint)
                    except Exception as e:
                        # Constraint might already exist
                        logger.debug(f"Constraint creation result: {str(e)}")
                
                # Create indexes for performance
                indexes = [
                    "CREATE INDEX country_name IF NOT EXISTS FOR (c:Country) ON (c.name)",
                    "CREATE INDEX visa_type_name IF NOT EXISTS FOR (v:VisaType) ON (v.name)",
                    "CREATE INDEX rule_priority IF NOT EXISTS FOR (r:Rule) ON (r.priority)"
                ]
                
                for index in indexes:
                    try:
                        session.run(index)
                    except Exception as e:
                        logger.debug(f"Index creation result: {str(e)}")
                
                logger.info("Neo4j schema initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j schema: {str(e)}")
    
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
            target_country: Target country for visa application
            
        Returns:
            Compliance validation results with reasoning
        """
        validation_id = uuid4()
        violations = []
        reasoning_steps = []
        required_documents = []
        satisfied_requirements = []
        
        try:
            with self._get_session() as session:
                # Get applicable rules for the visa type and country
                rules = await self._get_applicable_rules(session, country, visa_type)
                
                if not rules:
                    logger.warning(f"No rules found for {visa_type} visa in {country}")
                    # Create default reasoning path
                    reasoning_path = ReasoningPath(
                        id=validation_id,
                        steps=[ReasoningStep(
                            step_number=1,
                            rule_applied="No rules available",
                            input_data={"country": country, "visa_type": visa_type},
                            output_result={"status": "no_rules"},
                            confidence=0.0,
                            explanation="No visa rules found for the specified country and visa type"
                        )],
                        conclusion="Cannot validate compliance - no rules available",
                        confidence=0.0
                    )
                    
                    return ComplianceResult(
                        id=validation_id,
                        is_compliant=False,
                        violations=[],
                        reasoning_path=reasoning_path,
                        required_documents=[],
                        satisfied_requirements=[],
                        confidence=0.0
                    )
                
                # Validate each rule
                step_number = 1
                for rule in rules:
                    step_result = await self._validate_rule(
                        session, rule, data, step_number
                    )
                    
                    reasoning_steps.append(step_result['step'])
                    
                    if step_result['violation']:
                        violations.append(step_result['violation'])
                    
                    if step_result['required_docs']:
                        required_documents.extend(step_result['required_docs'])
                    
                    if step_result['satisfied_reqs']:
                        satisfied_requirements.extend(step_result['satisfied_reqs'])
                    
                    step_number += 1
                
                # Determine overall compliance
                is_compliant = len(violations) == 0
                
                # Calculate confidence based on rule coverage and data quality
                confidence = self._calculate_confidence(data, rules, violations)
                
                # Create reasoning path
                conclusion = "Compliant" if is_compliant else f"Non-compliant ({len(violations)} violations)"
                reasoning_path = ReasoningPath(
                    id=validation_id,
                    steps=reasoning_steps,
                    conclusion=conclusion,
                    confidence=confidence
                )
                
                return ComplianceResult(
                    id=validation_id,
                    is_compliant=is_compliant,
                    violations=violations,
                    reasoning_path=reasoning_path,
                    required_documents=list(set(required_documents)),
                    satisfied_requirements=list(set(satisfied_requirements)),
                    confidence=confidence
                )
        
        except Exception as e:
            logger.error(f"Error validating compliance: {str(e)}")
            # Return failed validation result
            reasoning_path = ReasoningPath(
                id=validation_id,
                steps=[ReasoningStep(
                    step_number=1,
                    rule_applied="Error",
                    input_data={"error": str(e)},
                    output_result={"status": "error"},
                    confidence=0.0,
                    explanation=f"Validation failed due to error: {str(e)}"
                )],
                conclusion="Validation failed due to system error",
                confidence=0.0
            )
            
            return ComplianceResult(
                id=validation_id,
                is_compliant=False,
                violations=[],
                reasoning_path=reasoning_path,
                required_documents=[],
                satisfied_requirements=[],
                confidence=0.0
            )
    
    async def _get_applicable_rules(
        self, 
        session: Session, 
        country: str, 
        visa_type: str
    ) -> List[Dict[str, Any]]:
        """Get applicable rules for country and visa type."""
        query = """
        MATCH (c:Country {code: $country})-[:OFFERS]->(v:VisaType {name: $visa_type})
        MATCH (v)-[:GOVERNED_BY]->(r:Rule)
        OPTIONAL MATCH (r)-[:REQUIRES]->(req:Requirement)
        OPTIONAL MATCH (req)-[:APPLIES_TO]->(dt:DocumentType)
        OPTIONAL MATCH (req)-[:VALIDATES]->(f:Field)
        RETURN r, collect(DISTINCT req) as requirements, 
               collect(DISTINCT dt.name) as document_types,
               collect(DISTINCT f.name) as fields
        ORDER BY r.priority DESC
        """
        
        result = session.run(query, country=country, visa_type=visa_type)
        rules = []
        
        for record in result:
            rule_node = record['r']
            requirements = record['requirements']
            document_types = record['document_types']
            fields = record['fields']
            
            rule_dict = {
                'id': rule_node['id'],
                'name': rule_node.get('name', ''),
                'description': rule_node.get('description', ''),
                'rule_type': rule_node.get('rule_type', 'mandatory'),
                'priority': rule_node.get('priority', 1),
                'conditions': json.loads(rule_node.get('conditions', '[]')),
                'requirements': [dict(req) for req in requirements if req],
                'document_types': [dt for dt in document_types if dt],
                'fields': [f for f in fields if f]
            }
            rules.append(rule_dict)
        
        return rules
    
    async def _validate_rule(
        self, 
        session: Session, 
        rule: Dict[str, Any], 
        data: List[StructuredData], 
        step_number: int
    ) -> Dict[str, Any]:
        """Validate a single rule against the provided data."""
        rule_id = rule['id']
        rule_name = rule['name']
        rule_type = rule['rule_type']
        conditions = rule['conditions']
        requirements = rule['requirements']
        
        violation = None
        required_docs = []
        satisfied_reqs = []
        
        # Check if required document types are present
        required_doc_types = set(rule['document_types'])
        available_doc_types = set(doc.document_type.value for doc in data)
        
        missing_doc_types = required_doc_types - available_doc_types
        if missing_doc_types and rule_type == 'mandatory':
            violation = RuleViolation(
                rule_id=rule_id,
                rule_description=rule['description'],
                violation_type="missing_document",
                severity=RiskSeverity.HIGH,
                affected_fields=list(missing_doc_types),
                explanation=f"Required document types missing: {', '.join(missing_doc_types)}"
            )
            required_docs.extend(missing_doc_types)
        
        # Check field-level conditions
        for condition in conditions:
            field_name = condition.get('field')
            operator = condition.get('operator')
            expected_value = condition.get('value')
            
            field_found = False
            field_value = None
            
            # Search for field in all documents
            for doc in data:
                if field_name in doc.key_fields:
                    field_found = True
                    field_value = doc.key_fields[field_name]
                    break
                
                # Check dates
                for date_field in doc.dates:
                    if date_field.field_name == field_name:
                        field_found = True
                        field_value = date_field.date_value
                        break
            
            # Validate condition
            condition_met = self._evaluate_condition(
                field_value, operator, expected_value, field_found
            )
            
            if not condition_met and rule_type == 'mandatory':
                if not violation:  # Don't overwrite existing violation
                    violation = RuleViolation(
                        rule_id=rule_id,
                        rule_description=rule['description'],
                        violation_type="condition_not_met",
                        severity=RiskSeverity.MEDIUM,
                        affected_fields=[field_name],
                        explanation=f"Field '{field_name}' does not meet condition: {operator} {expected_value}"
                    )
            elif condition_met:
                satisfied_reqs.append(f"{rule_name}:{field_name}")
        
        # Create reasoning step
        step = ReasoningStep(
            step_number=step_number,
            rule_applied=rule_name,
            input_data={
                "rule_id": rule_id,
                "rule_type": rule_type,
                "conditions": conditions,
                "available_documents": [doc.document_type.value for doc in data]
            },
            output_result={
                "violation": violation.dict() if violation else None,
                "satisfied_requirements": satisfied_reqs,
                "required_documents": required_docs
            },
            confidence=0.9 if not violation else 0.7,
            explanation=f"Validated rule '{rule_name}' - {'PASSED' if not violation else 'FAILED'}"
        )
        
        return {
            'step': step,
            'violation': violation,
            'required_docs': required_docs,
            'satisfied_reqs': satisfied_reqs
        }
    
    def _evaluate_condition(
        self, 
        field_value: Any, 
        operator: str, 
        expected_value: Any, 
        field_found: bool
    ) -> bool:
        """Evaluate a single condition."""
        if operator == ComparisonOperator.EXISTS:
            return field_found
        
        if operator == ComparisonOperator.NOT_EXISTS:
            return not field_found
        
        if not field_found:
            return False
        
        try:
            if operator == ComparisonOperator.EQUALS:
                return str(field_value).lower() == str(expected_value).lower()
            
            elif operator == ComparisonOperator.NOT_EQUALS:
                return str(field_value).lower() != str(expected_value).lower()
            
            elif operator == ComparisonOperator.CONTAINS:
                return str(expected_value).lower() in str(field_value).lower()
            
            elif operator == ComparisonOperator.NOT_CONTAINS:
                return str(expected_value).lower() not in str(field_value).lower()
            
            elif operator == ComparisonOperator.GREATER_THAN:
                return float(field_value) > float(expected_value)
            
            elif operator == ComparisonOperator.LESS_THAN:
                return float(field_value) < float(expected_value)
            
            else:
                logger.warning(f"Unknown operator: {operator}")
                return False
        
        except (ValueError, TypeError) as e:
            logger.warning(f"Error evaluating condition: {str(e)}")
            return False
    
    def _calculate_confidence(
        self, 
        data: List[StructuredData], 
        rules: List[Dict[str, Any]], 
        violations: List[RuleViolation]
    ) -> float:
        """Calculate confidence score for the validation."""
        if not rules:
            return 0.0
        
        # Base confidence on data quality
        data_confidence = sum(doc.extraction_confidence for doc in data) / len(data) if data else 0.0
        
        # Reduce confidence based on violations
        violation_penalty = len(violations) * 0.1
        
        # Rule coverage factor
        rule_coverage = min(1.0, len(rules) / 10)  # Assume 10 rules is good coverage
        
        confidence = (data_confidence * 0.5 + rule_coverage * 0.3 + (1.0 - violation_penalty) * 0.2)
        
        return max(0.0, min(1.0, confidence))
    
    async def get_reasoning_path(self, validation_id: UUID) -> ReasoningPath:
        """
        Retrieve detailed reasoning path for a validation.
        
        Args:
            validation_id: ID of the compliance validation
            
        Returns:
            Detailed reasoning path and steps
        """
        try:
            with self._get_session() as session:
                # Query for stored reasoning path
                query = """
                MATCH (v:Validation {id: $validation_id})
                RETURN v.reasoning_path as reasoning_path
                """
                
                result = session.run(query, validation_id=str(validation_id))
                record = result.single()
                
                if record and record['reasoning_path']:
                    reasoning_data = json.loads(record['reasoning_path'])
                    return ReasoningPath(**reasoning_data)
                else:
                    # Return empty reasoning path if not found
                    return ReasoningPath(
                        id=validation_id,
                        steps=[],
                        conclusion="Reasoning path not found",
                        confidence=0.0
                    )
        
        except Exception as e:
            logger.error(f"Error retrieving reasoning path: {str(e)}")
            return ReasoningPath(
                id=validation_id,
                steps=[],
                conclusion=f"Error retrieving reasoning path: {str(e)}",
                confidence=0.0
            )
    
    async def update_rules(self, country: str, rules: List[VisaRule]) -> None:
        """
        Update visa rules for a specific country.
        
        Args:
            country: Country code for rules
            rules: List of visa rules to update
        """
        try:
            with self._get_session() as session:
                # Start transaction
                with session.begin_transaction() as tx:
                    # Create or update country
                    tx.run("""
                        MERGE (c:Country {code: $country})
                        ON CREATE SET c.name = $country, c.created_at = datetime()
                        ON MATCH SET c.updated_at = datetime()
                    """, country=country)
                    
                    # Process each rule
                    for rule in rules:
                        # Create or update visa type
                        tx.run("""
                            MERGE (v:VisaType {name: $visa_type})
                            ON CREATE SET v.id = $visa_type_id, v.created_at = datetime()
                            ON MATCH SET v.updated_at = datetime()
                        """, visa_type=rule.visa_type, visa_type_id=str(uuid4()))
                        
                        # Create rule
                        tx.run("""
                            MERGE (r:Rule {id: $rule_id})
                            SET r.name = $name,
                                r.description = $description,
                                r.rule_type = $rule_type,
                                r.priority = $priority,
                                r.conditions = $conditions,
                                r.created_at = CASE WHEN r.created_at IS NULL THEN datetime() ELSE r.created_at END,
                                r.updated_at = datetime()
                        """, 
                            rule_id=str(rule.id),
                            name=f"{rule.visa_type}_{rule.rule_type}_{rule.priority}",
                            description=rule.description,
                            rule_type=rule.rule_type.value,
                            priority=rule.priority,
                            conditions=json.dumps([{
                                'field': cond.field,
                                'operator': cond.operator.value,
                                'value': cond.value,
                                'description': cond.description
                            } for cond in rule.conditions])
                        )
                        
                        # Link country -> visa type -> rule
                        tx.run("""
                            MATCH (c:Country {code: $country})
                            MATCH (v:VisaType {name: $visa_type})
                            MATCH (r:Rule {id: $rule_id})
                            MERGE (c)-[:OFFERS]->(v)
                            MERGE (v)-[:GOVERNED_BY]->(r)
                        """, country=country, visa_type=rule.visa_type, rule_id=str(rule.id))
                        
                        # Process requirements
                        for requirement in rule.requirements:
                            # Create requirement
                            tx.run("""
                                MERGE (req:Requirement {id: $req_id})
                                SET req.name = $name,
                                    req.description = $description,
                                    req.mandatory = $mandatory,
                                    req.created_at = CASE WHEN req.created_at IS NULL THEN datetime() ELSE req.created_at END,
                                    req.updated_at = datetime()
                            """, 
                                req_id=str(requirement.id),
                                name=requirement.name,
                                description=requirement.description,
                                mandatory=requirement.mandatory
                            )
                            
                            # Link rule -> requirement
                            tx.run("""
                                MATCH (r:Rule {id: $rule_id})
                                MATCH (req:Requirement {id: $req_id})
                                MERGE (r)-[:REQUIRES]->(req)
                            """, rule_id=str(rule.id), req_id=str(requirement.id))
                    
                    logger.info(f"Successfully updated {len(rules)} rules for country {country}")
        
        except Exception as e:
            logger.error(f"Error updating rules: {str(e)}")
            raise
    
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
        try:
            with self._get_session() as session:
                if visa_type:
                    query = """
                    MATCH (c:Country {code: $country})-[:OFFERS]->(v:VisaType {name: $visa_type})
                    MATCH (v)-[:GOVERNED_BY]->(r:Rule)
                    OPTIONAL MATCH (r)-[:REQUIRES]->(req:Requirement)
                    RETURN r, collect(req) as requirements
                    ORDER BY r.priority DESC
                    """
                    result = session.run(query, country=country, visa_type=visa_type)
                else:
                    query = """
                    MATCH (c:Country {code: $country})-[:OFFERS]->(v:VisaType)
                    MATCH (v)-[:GOVERNED_BY]->(r:Rule)
                    OPTIONAL MATCH (r)-[:REQUIRES]->(req:Requirement)
                    RETURN r, v.name as visa_type, collect(req) as requirements
                    ORDER BY r.priority DESC
                    """
                    result = session.run(query, country=country)
                
                rules = []
                for record in result:
                    rule_node = record['r']
                    requirements = record['requirements']
                    rule_visa_type = visa_type or record.get('visa_type', '')
                    
                    # Parse conditions
                    conditions_data = json.loads(rule_node.get('conditions', '[]'))
                    conditions = []
                    for cond_data in conditions_data:
                        from ...core.models import RuleCondition
                        conditions.append(RuleCondition(
                            field=cond_data['field'],
                            operator=ComparisonOperator(cond_data['operator']),
                            value=cond_data['value'],
                            description=cond_data['description']
                        ))
                    
                    # Parse requirements
                    req_objects = []
                    for req in requirements:
                        if req:
                            from ...core.models import Requirement
                            req_objects.append(Requirement(
                                id=UUID(req['id']),
                                name=req['name'],
                                description=req['description'],
                                mandatory=req.get('mandatory', True)
                            ))
                    
                    rule = VisaRule(
                        id=UUID(rule_node['id']),
                        country=country,
                        visa_type=rule_visa_type,
                        rule_type=RuleType(rule_node.get('rule_type', 'mandatory')),
                        conditions=conditions,
                        requirements=req_objects,
                        priority=rule_node.get('priority', 1),
                        description=rule_node.get('description', ''),
                        created_at=datetime.fromisoformat(rule_node.get('created_at', datetime.utcnow().isoformat()))
                    )
                    rules.append(rule)
                
                return rules
        
        except Exception as e:
            logger.error(f"Error retrieving rules: {str(e)}")
            return []
    
    def close(self) -> None:
        """Close the Neo4j driver connection."""
        if self._driver:
            self._driver.close()
            logger.info("Neo4j connection closed")