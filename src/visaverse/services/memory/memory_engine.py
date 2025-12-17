"""
MemMachine-based Memory Service implementation.
Handles persistent storage, pattern recognition, and personalized insights.
"""

import logging
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime, timedelta
import json
import asyncio
from collections import defaultdict, Counter

from ...core.interfaces import MemoryService
from ...core.models import (
    ApplicationData,
    HistoricalPattern,
    PersonalizedInsight,
    ApplicationOutcome,
    RiskFactor,
    RiskCategory,
    RiskSeverity,
    DocumentType,
    StructuredData
)


logger = logging.getLogger(__name__)


class MemMachineMemoryService(MemoryService):
    """
    MemMachine-based memory service for persistent learning and pattern recognition.
    """
    
    def __init__(self, storage_config: Dict[str, Any]):
        """
        Initialize the MemMachine memory service.
        
        Args:
            storage_config: Configuration for MemMachine storage backend
        """
        self.storage_config = storage_config
        
        # In-memory storage for development (replace with actual MemMachine integration)
        self._applications: Dict[UUID, ApplicationData] = {}
        self._user_applications: Dict[str, List[UUID]] = defaultdict(list)
        self._outcomes: Dict[UUID, ApplicationOutcome] = {}
        self._patterns_cache: Dict[str, List[HistoricalPattern]] = {}
        self._cache_expiry: Dict[str, datetime] = {}
        
        logger.info("Initialized MemMachine memory service")
    
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
        try:
            # Generate unique application ID
            application_id = uuid4()
            
            # Store application data
            self._applications[application_id] = application
            self._user_applications[user_id].append(application_id)
            
            # Invalidate user's pattern cache
            if user_id in self._patterns_cache:
                del self._patterns_cache[user_id]
                del self._cache_expiry[user_id]
            
            logger.info(f"Stored application {application_id} for user {user_id}")
            return application_id
        
        except Exception as e:
            logger.error(f"Error storing application: {str(e)}")
            raise
    
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
        try:
            # Check cache first
            if self._is_cache_valid(user_id):
                return self._patterns_cache[user_id]
            
            # Build historical patterns from stored applications
            patterns = []
            application_ids = self._user_applications.get(user_id, [])
            
            for app_id in application_ids:
                application = self._applications.get(app_id)
                if not application:
                    continue
                
                # Get outcome if available
                outcome = self._outcomes.get(app_id, ApplicationOutcome.PENDING)
                
                # Extract risk factors from application
                risk_factors = self._extract_risk_factors_from_application(application)
                
                pattern = HistoricalPattern(
                    application_id=app_id,
                    visa_type=application.visa_type,
                    country=application.country,
                    outcome=outcome,
                    risk_factors=risk_factors,
                    timestamp=application.submission_date,
                    document_types=[doc.document_type for doc in application.documents],
                    compliance_score=getattr(application, 'compliance_score', 0.5),
                    approval_probability=getattr(application, 'approval_probability', 0.5)
                )
                patterns.append(pattern)
            
            # Sort by timestamp (most recent first)
            patterns.sort(key=lambda x: x.timestamp, reverse=True)
            
            # Cache the results
            self._patterns_cache[user_id] = patterns
            self._cache_expiry[user_id] = datetime.now() + timedelta(hours=1)
            
            logger.info(f"Retrieved {len(patterns)} historical patterns for user {user_id}")
            return patterns
        
        except Exception as e:
            logger.error(f"Error retrieving user history: {str(e)}")
            return []
    
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
        try:
            if application_id not in self._applications:
                raise ValueError(f"Application {application_id} not found")
            
            # Store the outcome
            self._outcomes[application_id] = outcome
            
            # Find and invalidate user's cache
            for user_id, app_ids in self._user_applications.items():
                if application_id in app_ids:
                    if user_id in self._patterns_cache:
                        del self._patterns_cache[user_id]
                        del self._cache_expiry[user_id]
                    break
            
            logger.info(f"Updated outcome for application {application_id}: {outcome.value}")
        
        except Exception as e:
            logger.error(f"Error updating outcome: {str(e)}")
            raise
    
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
        try:
            # Get user's historical patterns
            history = await self.get_user_history(user_id)
            
            if not history:
                return self._generate_first_time_insights(current_application)
            
            insights = []
            
            # Analyze historical success patterns
            success_insights = self._analyze_success_patterns(history, current_application)
            insights.extend(success_insights)
            
            # Identify recurring issues
            recurring_insights = self._identify_recurring_issues(history, current_application)
            insights.extend(recurring_insights)
            
            # Compare with similar applications
            comparison_insights = self._compare_with_similar_applications(history, current_application)
            insights.extend(comparison_insights)
            
            # Improvement recommendations
            improvement_insights = self._generate_improvement_insights(history, current_application)
            insights.extend(improvement_insights)
            
            # Sort by relevance score
            insights.sort(key=lambda x: x.relevance_score, reverse=True)
            
            logger.info(f"Generated {len(insights)} personalized insights for user {user_id}")
            return insights[:10]  # Return top 10 insights
        
        except Exception as e:
            logger.error(f"Error generating personalized insights: {str(e)}")
            return []
    
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
        try:
            if not applications:
                return {}
            
            # Analyze document type patterns
            doc_patterns = self._analyze_document_patterns(applications)
            
            # Analyze risk factor patterns
            risk_patterns = self._analyze_risk_patterns(applications)
            
            # Analyze country-specific patterns
            country_patterns = self._analyze_country_patterns(applications)
            
            # Analyze success factors
            success_factors = self._analyze_success_factors(applications)
            
            learned_patterns = {
                'document_patterns': doc_patterns,
                'risk_patterns': risk_patterns,
                'country_patterns': country_patterns,
                'success_factors': success_factors,
                'total_applications': len(applications),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Learned patterns from {len(applications)} applications")
            return learned_patterns
        
        except Exception as e:
            logger.error(f"Error learning from patterns: {str(e)}")
            return {}
    
    def _is_cache_valid(self, user_id: str) -> bool:
        """Check if user's pattern cache is still valid."""
        if user_id not in self._cache_expiry:
            return False
        return datetime.now() < self._cache_expiry[user_id]
    
    def _extract_risk_factors_from_application(self, application: ApplicationData) -> List[RiskFactor]:
        """Extract risk factors from application data."""
        risk_factors = []
        
        # Check for missing documents
        required_docs = {DocumentType.PASSPORT, DocumentType.VISA_APPLICATION}
        provided_docs = {doc.document_type for doc in application.documents}
        
        for missing_doc in required_docs - provided_docs:
            risk_factors.append(RiskFactor(
                category=RiskCategory.DOCUMENT_MISSING,
                severity=RiskSeverity.HIGH,
                description=f"Missing {missing_doc.value.replace('_', ' ').title()}",
                impact=0.8,
                recommendation=f"Provide {missing_doc.value.replace('_', ' ').title()}"
            ))
        
        # Check document quality
        for doc in application.documents:
            if hasattr(doc, 'extraction_confidence') and doc.extraction_confidence < 0.7:
                risk_factors.append(RiskFactor(
                    category=RiskCategory.DOCUMENT_INVALID,
                    severity=RiskSeverity.MEDIUM,
                    description=f"Low quality {doc.document_type.value.replace('_', ' ')}",
                    impact=0.5,
                    recommendation="Provide higher quality document scan"
                ))
        
        return risk_factors
    
    def _generate_first_time_insights(self, application: ApplicationData) -> List[PersonalizedInsight]:
        """Generate insights for first-time users."""
        insights = []
        
        insights.append(PersonalizedInsight(
            insight_type="welcome",
            title="Welcome to VisaVerse Guardian AI",
            description="This is your first application analysis. We'll learn from your patterns to provide better insights over time.",
            relevance_score=0.9,
            confidence_level=1.0,
            recommendation="Complete your application thoroughly for the best analysis",
            historical_context=None
        ))
        
        # Basic document completeness insight
        doc_types = {doc.document_type for doc in application.documents}
        if len(doc_types) < 3:
            insights.append(PersonalizedInsight(
                insight_type="document_completeness",
                title="Consider Additional Documents",
                description="Applications with more supporting documents typically have higher approval rates.",
                relevance_score=0.7,
                confidence_level=0.8,
                recommendation="Add employment letters, bank statements, or educational certificates if applicable",
                historical_context=None
            ))
        
        return insights
    
    def _analyze_success_patterns(
        self, 
        history: List[HistoricalPattern], 
        current_application: ApplicationData
    ) -> List[PersonalizedInsight]:
        """Analyze user's historical success patterns."""
        insights = []
        
        # Find successful applications
        successful_apps = [h for h in history if h.outcome == ApplicationOutcome.APPROVED]
        
        if successful_apps:
            # Analyze common success factors
            success_doc_types = Counter()
            for app in successful_apps:
                for doc_type in app.document_types:
                    success_doc_types[doc_type] += 1
            
            # Check if current application follows success patterns
            current_doc_types = {doc.document_type for doc in current_application.documents}
            
            # Find documents that were present in successful applications but missing now
            for doc_type, count in success_doc_types.most_common(3):
                if doc_type not in current_doc_types and count > len(successful_apps) * 0.5:
                    insights.append(PersonalizedInsight(
                        insight_type="success_pattern",
                        title=f"Consider Adding {doc_type.value.replace('_', ' ').title()}",
                        description=f"You included this document in {count}/{len(successful_apps)} successful applications.",
                        relevance_score=0.8,
                        confidence_level=count / len(successful_apps),
                        recommendation=f"Add {doc_type.value.replace('_', ' ')} to match your successful pattern",
                        historical_context=f"Present in {count} successful applications"
                    ))
        
        return insights
    
    def _identify_recurring_issues(
        self, 
        history: List[HistoricalPattern], 
        current_application: ApplicationData
    ) -> List[PersonalizedInsight]:
        """Identify recurring issues from user's history."""
        insights = []
        
        # Count recurring risk categories
        risk_counter = Counter()
        for pattern in history:
            for risk in pattern.risk_factors:
                risk_counter[risk.category] += 1
        
        # Identify patterns that appear in multiple applications
        for risk_category, count in risk_counter.items():
            if count >= 2:  # Appears in 2+ applications
                insights.append(PersonalizedInsight(
                    insight_type="recurring_issue",
                    title=f"Recurring Issue: {risk_category.value.replace('_', ' ').title()}",
                    description=f"This issue appeared in {count} of your previous applications.",
                    relevance_score=0.9,
                    confidence_level=min(1.0, count / len(history)),
                    recommendation="Pay special attention to avoiding this recurring issue",
                    historical_context=f"Occurred in {count}/{len(history)} applications"
                ))
        
        return insights
    
    def _compare_with_similar_applications(
        self, 
        history: List[HistoricalPattern], 
        current_application: ApplicationData
    ) -> List[PersonalizedInsight]:
        """Compare current application with similar historical ones."""
        insights = []
        
        # Find similar applications (same visa type and country)
        similar_apps = [
            h for h in history 
            if h.visa_type == current_application.visa_type 
            and h.country == current_application.country
        ]
        
        if similar_apps:
            # Calculate average approval probability for similar apps
            approved_count = sum(1 for app in similar_apps if app.outcome == ApplicationOutcome.APPROVED)
            success_rate = approved_count / len(similar_apps)
            
            insights.append(PersonalizedInsight(
                insight_type="historical_comparison",
                title=f"Your {current_application.visa_type} Success Rate",
                description=f"You've applied for {current_application.visa_type} visas to {current_application.country} {len(similar_apps)} times with {success_rate:.1%} success rate.",
                relevance_score=0.8,
                confidence_level=min(1.0, len(similar_apps) / 3),
                recommendation="Learn from previous applications to improve your chances" if success_rate < 0.7 else "Your track record looks good for this visa type",
                historical_context=f"{approved_count}/{len(similar_apps)} previous applications approved"
            ))
        
        return insights
    
    def _generate_improvement_insights(
        self, 
        history: List[HistoricalPattern], 
        current_application: ApplicationData
    ) -> List[PersonalizedInsight]:
        """Generate improvement recommendations based on history."""
        insights = []
        
        # Analyze improvement trends
        if len(history) >= 2:
            recent_apps = history[:3]  # Last 3 applications
            older_apps = history[3:6] if len(history) > 3 else []
            
            if older_apps:
                recent_avg_score = sum(app.approval_probability for app in recent_apps) / len(recent_apps)
                older_avg_score = sum(app.approval_probability for app in older_apps) / len(older_apps)
                
                if recent_avg_score > older_avg_score + 0.1:
                    insights.append(PersonalizedInsight(
                        insight_type="improvement_trend",
                        title="Improving Application Quality",
                        description=f"Your recent applications show {(recent_avg_score - older_avg_score):.1%} improvement in quality.",
                        relevance_score=0.7,
                        confidence_level=0.8,
                        recommendation="Continue following your recent application patterns",
                        historical_context=f"Recent avg: {recent_avg_score:.1%}, Previous avg: {older_avg_score:.1%}"
                    ))
                elif older_avg_score > recent_avg_score + 0.1:
                    insights.append(PersonalizedInsight(
                        insight_type="quality_decline",
                        title="Application Quality Needs Attention",
                        description="Your recent applications show lower quality compared to earlier ones.",
                        relevance_score=0.9,
                        confidence_level=0.8,
                        recommendation="Review your earlier successful applications for best practices",
                        historical_context=f"Recent avg: {recent_avg_score:.1%}, Previous avg: {older_avg_score:.1%}"
                    ))
        
        return insights
    
    def _analyze_document_patterns(self, applications: List[ApplicationData]) -> Dict[str, Any]:
        """Analyze document type patterns across applications."""
        doc_counter = Counter()
        doc_success_rates = defaultdict(list)
        
        for app in applications:
            doc_types = {doc.document_type for doc in app.documents}
            for doc_type in doc_types:
                doc_counter[doc_type.value] += 1
                # Assume success if approval_probability > 0.7
                success = getattr(app, 'approval_probability', 0.5) > 0.7
                doc_success_rates[doc_type.value].append(success)
        
        # Calculate success rates for each document type
        doc_analysis = {}
        for doc_type, successes in doc_success_rates.items():
            success_rate = sum(successes) / len(successes)
            doc_analysis[doc_type] = {
                'frequency': doc_counter[doc_type],
                'success_rate': success_rate,
                'total_applications': len(successes)
            }
        
        return doc_analysis
    
    def _analyze_risk_patterns(self, applications: List[ApplicationData]) -> Dict[str, Any]:
        """Analyze risk factor patterns across applications."""
        risk_counter = Counter()
        
        for app in applications:
            # Extract risk factors from application (simplified)
            risk_factors = self._extract_risk_factors_from_application(app)
            for risk in risk_factors:
                risk_counter[risk.category.value] += 1
        
        return dict(risk_counter)
    
    def _analyze_country_patterns(self, applications: List[ApplicationData]) -> Dict[str, Any]:
        """Analyze country-specific patterns."""
        country_stats = defaultdict(lambda: {'count': 0, 'success_rate': 0, 'successes': []})
        
        for app in applications:
            country = app.country
            country_stats[country]['count'] += 1
            
            # Assume success if approval_probability > 0.7
            success = getattr(app, 'approval_probability', 0.5) > 0.7
            country_stats[country]['successes'].append(success)
        
        # Calculate success rates
        for country, stats in country_stats.items():
            if stats['successes']:
                stats['success_rate'] = sum(stats['successes']) / len(stats['successes'])
            del stats['successes']  # Remove raw data
        
        return dict(country_stats)
    
    def _analyze_success_factors(self, applications: List[ApplicationData]) -> Dict[str, Any]:
        """Analyze factors that contribute to success."""
        successful_apps = [
            app for app in applications 
            if getattr(app, 'approval_probability', 0.5) > 0.7
        ]
        
        if not successful_apps:
            return {}
        
        # Analyze common characteristics of successful applications
        success_factors = {
            'average_document_count': sum(len(app.documents) for app in successful_apps) / len(successful_apps),
            'common_visa_types': Counter(app.visa_type for app in successful_apps),
            'common_countries': Counter(app.country for app in successful_apps),
            'total_successful': len(successful_apps),
            'success_rate': len(successful_apps) / len(applications)
        }
        
        return success_factors