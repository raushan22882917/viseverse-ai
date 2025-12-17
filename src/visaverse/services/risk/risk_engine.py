"""
Gemini LLM-based Risk Assessment Service implementation.
Handles probability calculation, risk factor identification, and explanation generation.
"""

import logging
from typing import List, Optional, Dict, Any
from uuid import uuid4
import json
import re
from datetime import datetime

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from ...core.interfaces import RiskAssessmentService
from ...core.models import (
    ComplianceResult,
    HistoricalPattern,
    StructuredData,
    RiskAssessment,
    ExplanationResult,
    RiskFactor,
    Recommendation,
    RiskBreakdown,
    RiskCategory,
    RiskSeverity,
    DocumentType
)


logger = logging.getLogger(__name__)


class GeminiRiskAssessmentService(RiskAssessmentService):
    """
    Gemini LLM-based risk assessment service for visa applications.
    """
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        """
        Initialize the Gemini risk assessment service.
        
        Args:
            api_key: Google Gemini API key
            model_name: Gemini model to use
        """
        self.api_key = api_key
        self.model_name = model_name
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Initialize model with safety settings
        self.model = genai.GenerativeModel(
            model_name=model_name,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
        )
        
        logger.info(f"Initialized Gemini risk assessment service with model: {model_name}")
    
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
        try:
            # Prepare input data for LLM analysis
            analysis_prompt = self._build_analysis_prompt(
                compliance, historical_data, document_data
            )
            
            # Generate risk assessment using Gemini
            response = await self._generate_risk_analysis(analysis_prompt)
            
            # Parse LLM response into structured risk assessment
            risk_assessment = self._parse_risk_response(response, compliance, document_data)
            
            return risk_assessment
        
        except Exception as e:
            logger.error(f"Error calculating approval probability: {str(e)}")
            # Return fallback risk assessment
            return self._create_fallback_assessment(compliance, document_data)
    
    def _build_analysis_prompt(
        self,
        compliance: ComplianceResult,
        historical_data: List[HistoricalPattern],
        document_data: List[StructuredData]
    ) -> str:
        """Build comprehensive analysis prompt for Gemini."""
        
        prompt = """You are a visa application risk assessment expert. Analyze the following visa application data and provide a comprehensive risk assessment.

COMPLIANCE ANALYSIS:
"""
        
        # Add compliance information
        prompt += f"Overall Compliance: {'COMPLIANT' if compliance.is_compliant else 'NON-COMPLIANT'}\n"
        prompt += f"Confidence Level: {compliance.confidence:.2f}\n"
        prompt += f"Violations Found: {len(compliance.violations)}\n\n"
        
        if compliance.violations:
            prompt += "COMPLIANCE VIOLATIONS:\n"
            for i, violation in enumerate(compliance.violations, 1):
                prompt += f"{i}. {violation.violation_type}: {violation.explanation} (Severity: {violation.severity.value})\n"
            prompt += "\n"
        
        # Add document information
        prompt += "DOCUMENT ANALYSIS:\n"
        for doc in document_data:
            prompt += f"- {doc.document_type.value.title()}: "
            prompt += f"Confidence {doc.extraction_confidence:.2f}, "
            prompt += f"Key fields: {len(doc.key_fields)}, "
            prompt += f"Missing fields: {len(doc.missing_fields)}\n"
        
        # Add historical context
        if historical_data:
            prompt += f"\nHISTORICAL CONTEXT:\n"
            prompt += f"Previous applications: {len(historical_data)}\n"
            
            outcomes = {}
            for pattern in historical_data:
                outcome = pattern.outcome.value
                outcomes[outcome] = outcomes.get(outcome, 0) + 1
            
            for outcome, count in outcomes.items():
                prompt += f"- {outcome.title()}: {count}\n"
        
        prompt += """
ASSESSMENT REQUIREMENTS:
1. Calculate approval probability (0-100%)
2. Identify specific risk factors with categories and severity
3. Provide actionable recommendations
4. Base analysis ONLY on the provided data - do not hallucinate

RESPONSE FORMAT (JSON):
{
    "approval_probability": <number 0-100>,
    "confidence_level": <number 0.0-1.0>,
    "risk_factors": [
        {
            "category": "<category>",
            "severity": "<low|medium|high|critical>",
            "description": "<description>",
            "impact": <number 0.0-1.0>,
            "recommendation": "<actionable recommendation>"
        }
    ],
    "recommendations": [
        {
            "title": "<title>",
            "description": "<description>",
            "priority": <number 1-10>,
            "action_required": <boolean>,
            "estimated_impact": <number 0.0-1.0>
        }
    ],
    "summary": "<brief summary of assessment>"
}

Provide your assessment:"""
        
        return prompt
    
    async def _generate_risk_analysis(self, prompt: str) -> str:
        """Generate risk analysis using Gemini LLM."""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating risk analysis: {str(e)}")
            raise
    
    def _parse_risk_response(
        self, 
        response: str, 
        compliance: ComplianceResult,
        document_data: List[StructuredData]
    ) -> RiskAssessment:
        """Parse LLM response into structured risk assessment."""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in LLM response")
            
            json_str = json_match.group(0)
            analysis = json.loads(json_str)
            
            # Parse risk factors
            risk_factors = []
            for rf_data in analysis.get('risk_factors', []):
                risk_factors.append(RiskFactor(
                    category=self._parse_risk_category(rf_data.get('category', 'unknown')),
                    severity=self._parse_risk_severity(rf_data.get('severity', 'medium')),
                    description=rf_data.get('description', ''),
                    impact=float(rf_data.get('impact', 0.5)),
                    recommendation=rf_data.get('recommendation', ''),
                    field_reference=rf_data.get('field_reference')
                ))
            
            # Parse recommendations
            recommendations = []
            for rec_data in analysis.get('recommendations', []):
                recommendations.append(Recommendation(
                    title=rec_data.get('title', ''),
                    description=rec_data.get('description', ''),
                    priority=int(rec_data.get('priority', 5)),
                    action_required=bool(rec_data.get('action_required', True)),
                    estimated_impact=float(rec_data.get('estimated_impact', 0.5))
                ))
            
            # Create risk breakdown
            risk_breakdown = self._create_risk_breakdown(risk_factors, compliance, document_data)
            
            return RiskAssessment(
                approval_probability=float(analysis.get('approval_probability', 50.0)) / 100.0,
                risk_factors=risk_factors,
                recommendations=recommendations,
                confidence_level=float(analysis.get('confidence_level', 0.5)),
                risk_breakdown=risk_breakdown
            )
        
        except Exception as e:
            logger.error(f"Error parsing risk response: {str(e)}")
            return self._create_fallback_assessment(compliance, document_data)
    
    def _parse_risk_category(self, category_str: str) -> RiskCategory:
        """Parse risk category from string."""
        category_map = {
            'document_missing': RiskCategory.DOCUMENT_MISSING,
            'document_invalid': RiskCategory.DOCUMENT_INVALID,
            'requirement_not_met': RiskCategory.REQUIREMENT_NOT_MET,
            'inconsistency': RiskCategory.INCONSISTENCY,
            'historical_pattern': RiskCategory.HISTORICAL_PATTERN
        }
        
        category_lower = category_str.lower().replace(' ', '_')
        return category_map.get(category_lower, RiskCategory.REQUIREMENT_NOT_MET)
    
    def _parse_risk_severity(self, severity_str: str) -> RiskSeverity:
        """Parse risk severity from string."""
        severity_map = {
            'low': RiskSeverity.LOW,
            'medium': RiskSeverity.MEDIUM,
            'high': RiskSeverity.HIGH,
            'critical': RiskSeverity.CRITICAL
        }
        
        return severity_map.get(severity_str.lower(), RiskSeverity.MEDIUM)
    
    def _create_risk_breakdown(
        self,
        risk_factors: List[RiskFactor],
        compliance: ComplianceResult,
        document_data: List[StructuredData]
    ) -> RiskBreakdown:
        """Create detailed risk breakdown."""
        
        # Categorize risks
        document_risks = [rf for rf in risk_factors if rf.category in [
            RiskCategory.DOCUMENT_MISSING, RiskCategory.DOCUMENT_INVALID
        ]]
        
        compliance_risks = [rf for rf in risk_factors if rf.category == RiskCategory.REQUIREMENT_NOT_MET]
        
        historical_risks = [rf for rf in risk_factors if rf.category == RiskCategory.HISTORICAL_PATTERN]
        
        # Calculate total risk score
        if risk_factors:
            total_risk_score = sum(rf.impact for rf in risk_factors) / len(risk_factors)
        else:
            total_risk_score = 0.0 if compliance.is_compliant else 0.5
        
        return RiskBreakdown(
            document_risks=document_risks,
            compliance_risks=compliance_risks,
            historical_risks=historical_risks,
            total_risk_score=min(1.0, total_risk_score)
        )
    
    def _create_fallback_assessment(
        self,
        compliance: ComplianceResult,
        document_data: List[StructuredData]
    ) -> RiskAssessment:
        """Create fallback risk assessment when LLM fails."""
        
        # Basic probability calculation based on compliance
        if compliance.is_compliant:
            base_probability = 0.8
        else:
            base_probability = 0.3
        
        # Adjust based on document quality
        doc_quality = sum(doc.extraction_confidence for doc in document_data) / len(document_data)
        probability = base_probability * doc_quality
        
        # Create basic risk factors from violations
        risk_factors = []
        for violation in compliance.violations:
            risk_factors.append(RiskFactor(
                category=RiskCategory.REQUIREMENT_NOT_MET,
                severity=violation.severity,
                description=violation.explanation,
                impact=0.7 if violation.severity == RiskSeverity.HIGH else 0.4,
                recommendation=f"Address {violation.violation_type} issue"
            ))
        
        # Create basic recommendations
        recommendations = []
        if not compliance.is_compliant:
            recommendations.append(Recommendation(
                title="Address Compliance Issues",
                description="Resolve all compliance violations before resubmitting",
                priority=9,
                action_required=True,
                estimated_impact=0.8
            ))
        
        risk_breakdown = self._create_risk_breakdown(risk_factors, compliance, document_data)
        
        return RiskAssessment(
            approval_probability=probability,
            risk_factors=risk_factors,
            recommendations=recommendations,
            confidence_level=0.6,  # Lower confidence for fallback
            risk_breakdown=risk_breakdown
        )
    
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
        try:
            # Build explanation prompt
            explanation_prompt = self._build_explanation_prompt(assessment, language)
            
            # Generate explanation using Gemini
            response = await self._generate_explanation_text(explanation_prompt)
            
            # Parse response into structured explanation
            explanation = self._parse_explanation_response(response, assessment, language)
            
            return explanation
        
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return self._create_fallback_explanation(assessment, language)
    
    def _build_explanation_prompt(self, assessment: RiskAssessment, language: str) -> str:
        """Build explanation generation prompt."""
        
        language_names = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ko': 'Korean',
            'ar': 'Arabic',
            'hi': 'Hindi'
        }
        
        target_language = language_names.get(language, 'English')
        
        prompt = f"""You are a visa application advisor. Explain the following risk assessment in clear, user-friendly {target_language}.

RISK ASSESSMENT DATA:
Approval Probability: {assessment.approval_probability:.1%}
Confidence Level: {assessment.confidence_level:.1%}
Total Risk Factors: {len(assessment.risk_factors)}

RISK FACTORS:
"""
        
        for i, risk in enumerate(assessment.risk_factors, 1):
            prompt += f"{i}. {risk.category.value.replace('_', ' ').title()}: {risk.description} (Impact: {risk.impact:.1%})\n"
        
        prompt += f"\nRECOMMENDATIONS:\n"
        for i, rec in enumerate(assessment.recommendations, 1):
            prompt += f"{i}. {rec.title}: {rec.description}\n"
        
        prompt += f"""
EXPLANATION REQUIREMENTS:
1. Write in clear, non-technical {target_language}
2. Provide a brief summary (2-3 sentences)
3. Explain the main factors affecting approval chances
4. List actionable steps the applicant can take
5. Be encouraging but realistic
6. Do not provide legal advice - only analysis

RESPONSE FORMAT (JSON):
{{
    "summary": "<brief 2-3 sentence summary>",
    "detailed_explanation": "<detailed explanation of factors>",
    "actionable_steps": ["<step 1>", "<step 2>", "<step 3>"]
}}

Provide your explanation in {target_language}:"""
        
        return prompt
    
    async def _generate_explanation_text(self, prompt: str) -> str:
        """Generate explanation text using Gemini."""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating explanation text: {str(e)}")
            raise
    
    def _parse_explanation_response(
        self, 
        response: str, 
        assessment: RiskAssessment,
        language: str
    ) -> ExplanationResult:
        """Parse explanation response into structured result."""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in explanation response")
            
            json_str = json_match.group(0)
            explanation_data = json.loads(json_str)
            
            return ExplanationResult(
                summary=explanation_data.get('summary', 'Risk assessment completed'),
                detailed_explanation=explanation_data.get('detailed_explanation', 'Analysis based on provided documents'),
                actionable_steps=explanation_data.get('actionable_steps', []),
                risk_breakdown=assessment.risk_breakdown or RiskBreakdown(
                    document_risks=[],
                    compliance_risks=[],
                    historical_risks=[],
                    total_risk_score=0.5
                ),
                language=language
            )
        
        except Exception as e:
            logger.error(f"Error parsing explanation response: {str(e)}")
            return self._create_fallback_explanation(assessment, language)
    
    def _create_fallback_explanation(
        self, 
        assessment: RiskAssessment, 
        language: str
    ) -> ExplanationResult:
        """Create fallback explanation when LLM fails."""
        
        # Basic explanations in English (could be extended for other languages)
        if assessment.approval_probability > 0.7:
            summary = "Your visa application has a good chance of approval based on the provided documents."
        elif assessment.approval_probability > 0.4:
            summary = "Your visa application has moderate approval chances with some areas for improvement."
        else:
            summary = "Your visa application faces significant challenges that should be addressed."
        
        detailed_explanation = f"Based on our analysis, your application has a {assessment.approval_probability:.1%} probability of approval. "
        
        if assessment.risk_factors:
            detailed_explanation += f"We identified {len(assessment.risk_factors)} risk factors that may impact your application. "
        
        detailed_explanation += "Please review the recommendations to improve your application."
        
        # Generate basic actionable steps
        actionable_steps = []
        for rec in assessment.recommendations[:3]:  # Top 3 recommendations
            actionable_steps.append(rec.description)
        
        if not actionable_steps:
            actionable_steps = ["Review all document requirements", "Ensure all information is accurate", "Consider consulting with an immigration advisor"]
        
        return ExplanationResult(
            summary=summary,
            detailed_explanation=detailed_explanation,
            actionable_steps=actionable_steps,
            risk_breakdown=assessment.risk_breakdown or RiskBreakdown(
                document_risks=[],
                compliance_risks=[],
                historical_risks=[],
                total_risk_score=0.5
            ),
            language=language
        )
    
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
        if len(documents) < 2:
            return []  # No cross-document analysis possible
        
        issues = []
        
        try:
            # Build cross-document analysis prompt
            analysis_prompt = self._build_cross_document_prompt(documents)
            
            # Generate analysis using Gemini
            response = await self._generate_cross_document_analysis(analysis_prompt)
            
            # Parse response into issues
            issues = self._parse_cross_document_response(response)
            
        except Exception as e:
            logger.error(f"Error identifying cross-document issues: {str(e)}")
            # Fallback to basic consistency checks
            issues = self._basic_cross_document_checks(documents)
        
        return issues
    
    def _build_cross_document_prompt(self, documents: List[StructuredData]) -> str:
        """Build cross-document analysis prompt."""
        
        prompt = """Analyze the following documents for consistency and identify any discrepancies or missing information.

DOCUMENTS TO ANALYZE:
"""
        
        for i, doc in enumerate(documents, 1):
            prompt += f"\nDocument {i} - {doc.document_type.value.title()}:\n"
            prompt += f"Key Fields: {json.dumps(doc.key_fields, indent=2)}\n"
            
            if doc.dates:
                prompt += "Dates:\n"
                for date_field in doc.dates:
                    prompt += f"  - {date_field.field_name}: {date_field.date_value.strftime('%Y-%m-%d')}\n"
            
            if doc.financial_info:
                prompt += f"Financial Info: {doc.financial_info.currency}, {len(doc.financial_info.amounts)} transactions\n"
        
        prompt += """
ANALYSIS REQUIREMENTS:
1. Check name consistency across all documents
2. Verify date consistency (birth dates, etc.)
3. Identify missing document relationships
4. Flag any contradictory information
5. Note incomplete document sets

RESPONSE FORMAT (JSON):
{
    "issues": [
        {
            "type": "<issue_type>",
            "description": "<description>",
            "affected_documents": ["<doc1>", "<doc2>"],
            "severity": "<low|medium|high|critical>",
            "recommendation": "<how to fix>"
        }
    ]
}

Provide your analysis:"""
        
        return prompt
    
    async def _generate_cross_document_analysis(self, prompt: str) -> str:
        """Generate cross-document analysis using Gemini."""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating cross-document analysis: {str(e)}")
            raise
    
    def _parse_cross_document_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse cross-document analysis response."""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                return []
            
            json_str = json_match.group(0)
            analysis = json.loads(json_str)
            
            return analysis.get('issues', [])
        
        except Exception as e:
            logger.error(f"Error parsing cross-document response: {str(e)}")
            return []
    
    def _basic_cross_document_checks(self, documents: List[StructuredData]) -> List[Dict[str, Any]]:
        """Perform basic cross-document consistency checks."""
        issues = []
        
        # Check name consistency
        names = {}
        for doc in documents:
            doc_type = doc.document_type.value
            
            # Extract names from different document types
            if doc_type == 'passport' and 'name' in doc.key_fields:
                names['passport'] = doc.key_fields['name']
            elif doc_type == 'employment_letter' and 'employee_name' in doc.key_fields:
                names['employment'] = doc.key_fields['employee_name']
            elif doc_type == 'bank_statement' and 'account_holder' in doc.key_fields:
                names['bank'] = doc.key_fields['account_holder']
            elif doc_type == 'educational_certificate' and 'student_name' in doc.key_fields:
                names['education'] = doc.key_fields['student_name']
        
        # Check for name inconsistencies
        if len(set(names.values())) > 1:
            issues.append({
                'type': 'name_inconsistency',
                'description': 'Names do not match consistently across documents',
                'affected_documents': list(names.keys()),
                'severity': 'high',
                'recommendation': 'Ensure all documents use the exact same name format'
            })
        
        # Check for missing document relationships
        doc_types = {doc.document_type for doc in documents}
        
        if DocumentType.EMPLOYMENT_LETTER in doc_types and DocumentType.BANK_STATEMENT not in doc_types:
            issues.append({
                'type': 'missing_financial_proof',
                'description': 'Employment letter provided but no financial documentation',
                'affected_documents': ['employment_letter'],
                'severity': 'medium',
                'recommendation': 'Provide bank statements or other financial proof'
            })
        
        return issues
    
    async def synthesize_recommendations(
        self,
        risk_assessment: RiskAssessment,
        compliance: ComplianceResult,
        user_history: List[HistoricalPattern]
    ) -> List[Recommendation]:
        """
        Synthesize comprehensive recommendations based on risk assessment and user history.
        
        Args:
            risk_assessment: Current risk assessment
            compliance: Compliance validation results
            user_history: User's historical patterns
            
        Returns:
            Synthesized and prioritized recommendations
        """
        try:
            # Build recommendation synthesis prompt
            synthesis_prompt = self._build_recommendation_prompt(
                risk_assessment, compliance, user_history
            )
            
            # Generate recommendations using Gemini
            response = await self._generate_recommendation_synthesis(synthesis_prompt)
            
            # Parse response into recommendations
            recommendations = self._parse_recommendation_response(response)
            
            return recommendations
        
        except Exception as e:
            logger.error(f"Error synthesizing recommendations: {str(e)}")
            return self._create_fallback_recommendations(risk_assessment, compliance)
    
    def _build_recommendation_prompt(
        self,
        risk_assessment: RiskAssessment,
        compliance: ComplianceResult,
        user_history: List[HistoricalPattern]
    ) -> str:
        """Build recommendation synthesis prompt."""
        
        prompt = """You are a visa application advisor. Synthesize comprehensive, actionable recommendations based on the following analysis.

CURRENT RISK ASSESSMENT:
Approval Probability: {:.1%}
Risk Factors: {}
""".format(risk_assessment.approval_probability, len(risk_assessment.risk_factors))
        
        if risk_assessment.risk_factors:
            prompt += "\nIDENTIFIED RISKS:\n"
            for i, risk in enumerate(risk_assessment.risk_factors, 1):
                prompt += f"{i}. {risk.category.value.replace('_', ' ').title()}: {risk.description}\n"
        
        prompt += f"\nCOMPLIANCE STATUS: {'COMPLIANT' if compliance.is_compliant else 'NON-COMPLIANT'}\n"
        
        if compliance.violations:
            prompt += "VIOLATIONS:\n"
            for violation in compliance.violations:
                prompt += f"- {violation.violation_type}: {violation.explanation}\n"
        
        if user_history:
            prompt += f"\nUSER HISTORY: {len(user_history)} previous applications\n"
            
            # Analyze historical patterns
            outcomes = {}
            common_issues = {}
            for pattern in user_history:
                outcome = pattern.outcome.value
                outcomes[outcome] = outcomes.get(outcome, 0) + 1
                
                for risk in pattern.risk_factors:
                    issue = risk.category.value
                    common_issues[issue] = common_issues.get(issue, 0) + 1
            
            prompt += "Historical outcomes: " + ", ".join([f"{k}: {v}" for k, v in outcomes.items()]) + "\n"
            
            if common_issues:
                prompt += "Recurring issues: " + ", ".join([f"{k}: {v}" for k, v in common_issues.items()]) + "\n"
        
        prompt += """
RECOMMENDATION REQUIREMENTS:
1. Provide specific, actionable recommendations
2. Prioritize by impact and urgency
3. Consider user's historical patterns
4. Include both immediate fixes and long-term improvements
5. Be encouraging but realistic
6. Focus on what the user can control

RESPONSE FORMAT (JSON):
{
    "recommendations": [
        {
            "title": "<clear title>",
            "description": "<detailed description>",
            "priority": <1-10, 10 being highest>,
            "action_required": <true/false>,
            "estimated_impact": <0.0-1.0>,
            "timeline": "<immediate|short_term|long_term>",
            "category": "<document|compliance|process|preparation>"
        }
    ]
}

Provide your recommendations:"""
        
        return prompt
    
    async def _generate_recommendation_synthesis(self, prompt: str) -> str:
        """Generate recommendation synthesis using Gemini."""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating recommendation synthesis: {str(e)}")
            raise
    
    def _parse_recommendation_response(self, response: str) -> List[Recommendation]:
        """Parse recommendation synthesis response."""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in recommendation response")
            
            json_str = json_match.group(0)
            data = json.loads(json_str)
            
            recommendations = []
            for rec_data in data.get('recommendations', []):
                recommendations.append(Recommendation(
                    title=rec_data.get('title', ''),
                    description=rec_data.get('description', ''),
                    priority=int(rec_data.get('priority', 5)),
                    action_required=bool(rec_data.get('action_required', True)),
                    estimated_impact=float(rec_data.get('estimated_impact', 0.5)),
                    timeline=rec_data.get('timeline', 'short_term'),
                    category=rec_data.get('category', 'general')
                ))
            
            # Sort by priority (highest first)
            recommendations.sort(key=lambda x: x.priority, reverse=True)
            
            return recommendations
        
        except Exception as e:
            logger.error(f"Error parsing recommendation response: {str(e)}")
            return []
    
    def _create_fallback_recommendations(
        self,
        risk_assessment: RiskAssessment,
        compliance: ComplianceResult
    ) -> List[Recommendation]:
        """Create fallback recommendations when LLM fails."""
        
        recommendations = []
        
        # Address compliance violations
        if not compliance.is_compliant:
            recommendations.append(Recommendation(
                title="Address Compliance Violations",
                description="Review and resolve all identified compliance issues before resubmitting your application",
                priority=10,
                action_required=True,
                estimated_impact=0.9,
                timeline="immediate",
                category="compliance"
            ))
        
        # Address high-impact risk factors
        high_risk_factors = [rf for rf in risk_assessment.risk_factors if rf.impact > 0.6]
        if high_risk_factors:
            recommendations.append(Recommendation(
                title="Address High-Impact Risk Factors",
                description=f"Focus on resolving {len(high_risk_factors)} high-impact issues that significantly affect your approval chances",
                priority=9,
                action_required=True,
                estimated_impact=0.8,
                timeline="immediate",
                category="document"
            ))
        
        # Document quality improvement
        recommendations.append(Recommendation(
            title="Improve Document Quality",
            description="Ensure all documents are clear, complete, and properly formatted",
            priority=7,
            action_required=False,
            estimated_impact=0.6,
            timeline="short_term",
            category="document"
        ))
        
        # General preparation advice
        recommendations.append(Recommendation(
            title="Review Application Thoroughly",
            description="Double-check all information for accuracy and completeness before final submission",
            priority=6,
            action_required=True,
            estimated_impact=0.5,
            timeline="immediate",
            category="process"
        ))
        
        return recommendations