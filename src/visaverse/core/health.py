"""
Health check utilities for VisaVerse Guardian AI services.
Provides comprehensive health monitoring for all system components.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import time

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check definition."""
    name: str
    check_function: Callable
    timeout_seconds: float = 10.0
    critical: bool = True
    description: str = ""


@dataclass
class HealthResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    duration_ms: float
    timestamp: datetime
    critical: bool
    details: Optional[Dict[str, Any]] = None


class HealthChecker:
    """
    Comprehensive health checking system.
    """
    
    def __init__(self):
        self.checks: Dict[str, HealthCheck] = {}
        self.last_results: Dict[str, HealthResult] = {}
        self.check_history: Dict[str, List[HealthResult]] = {}
        self.max_history_per_check = 100
    
    def register_check(
        self,
        name: str,
        check_function: Callable,
        timeout_seconds: float = 10.0,
        critical: bool = True,
        description: str = ""
    ):
        """
        Register a health check.
        
        Args:
            name: Unique name for the health check
            check_function: Function that returns True if healthy, False otherwise
            timeout_seconds: Timeout for the check
            critical: Whether this check is critical for overall health
            description: Human-readable description of the check
        """
        self.checks[name] = HealthCheck(
            name=name,
            check_function=check_function,
            timeout_seconds=timeout_seconds,
            critical=critical,
            description=description
        )
        
        if name not in self.check_history:
            self.check_history[name] = []
        
        logger.info(f"Registered health check '{name}' (critical: {critical})")
    
    async def run_check(self, name: str) -> HealthResult:
        """
        Run a specific health check.
        
        Args:
            name: Name of the health check to run
            
        Returns:
            Health check result
        """
        if name not in self.checks:
            return HealthResult(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Health check '{name}' not found",
                duration_ms=0,
                timestamp=datetime.now(),
                critical=False
            )
        
        check = self.checks[name]
        start_time = time.time()
        
        try:
            # Run the check with timeout
            if asyncio.iscoroutinefunction(check.check_function):
                is_healthy = await asyncio.wait_for(
                    check.check_function(),
                    timeout=check.timeout_seconds
                )
            else:
                is_healthy = await asyncio.wait_for(
                    asyncio.to_thread(check.check_function),
                    timeout=check.timeout_seconds
                )
            
            duration_ms = (time.time() - start_time) * 1000
            
            if is_healthy:
                result = HealthResult(
                    name=name,
                    status=HealthStatus.HEALTHY,
                    message="Check passed",
                    duration_ms=duration_ms,
                    timestamp=datetime.now(),
                    critical=check.critical
                )
            else:
                result = HealthResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message="Check failed",
                    duration_ms=duration_ms,
                    timestamp=datetime.now(),
                    critical=check.critical
                )
        
        except asyncio.TimeoutError:
            duration_ms = (time.time() - start_time) * 1000
            result = HealthResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check timed out after {check.timeout_seconds}s",
                duration_ms=duration_ms,
                timestamp=datetime.now(),
                critical=check.critical
            )
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = HealthResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed with error: {str(e)}",
                duration_ms=duration_ms,
                timestamp=datetime.now(),
                critical=check.critical
            )
        
        # Store result
        self.last_results[name] = result
        self.check_history[name].append(result)
        
        # Limit history size
        if len(self.check_history[name]) > self.max_history_per_check:
            self.check_history[name] = self.check_history[name][-self.max_history_per_check:]
        
        return result
    
    async def run_all_checks(self) -> Dict[str, HealthResult]:
        """
        Run all registered health checks.
        
        Returns:
            Dictionary of check names to results
        """
        if not self.checks:
            return {}
        
        # Run all checks concurrently
        tasks = [
            self.run_check(name) for name in self.checks.keys()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        check_results = {}
        for i, (name, result) in enumerate(zip(self.checks.keys(), results)):
            if isinstance(result, Exception):
                check_results[name] = HealthResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check execution failed: {str(result)}",
                    duration_ms=0,
                    timestamp=datetime.now(),
                    critical=self.checks[name].critical
                )
            else:
                check_results[name] = result
        
        return check_results
    
    async def get_overall_health(self) -> Dict[str, Any]:
        """
        Get overall system health status.
        
        Returns:
            Overall health summary
        """
        check_results = await self.run_all_checks()
        
        if not check_results:
            return {
                "status": HealthStatus.UNKNOWN.value,
                "message": "No health checks configured",
                "timestamp": datetime.now().isoformat(),
                "checks": {}
            }
        
        # Determine overall status
        critical_checks = [r for r in check_results.values() if r.critical]
        non_critical_checks = [r for r in check_results.values() if not r.critical]
        
        # Check critical services
        critical_unhealthy = [r for r in critical_checks if r.status == HealthStatus.UNHEALTHY]
        critical_healthy = [r for r in critical_checks if r.status == HealthStatus.HEALTHY]
        
        # Check non-critical services
        non_critical_unhealthy = [r for r in non_critical_checks if r.status == HealthStatus.UNHEALTHY]
        
        # Determine overall status
        if critical_unhealthy:
            overall_status = HealthStatus.UNHEALTHY
            message = f"{len(critical_unhealthy)} critical service(s) unhealthy"
        elif non_critical_unhealthy:
            overall_status = HealthStatus.DEGRADED
            message = f"{len(non_critical_unhealthy)} non-critical service(s) unhealthy"
        else:
            overall_status = HealthStatus.HEALTHY
            message = "All services healthy"
        
        # Calculate statistics
        total_checks = len(check_results)
        healthy_checks = len([r for r in check_results.values() if r.status == HealthStatus.HEALTHY])
        unhealthy_checks = len([r for r in check_results.values() if r.status == HealthStatus.UNHEALTHY])
        
        return {
            "status": overall_status.value,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "statistics": {
                "total_checks": total_checks,
                "healthy_checks": healthy_checks,
                "unhealthy_checks": unhealthy_checks,
                "success_rate": healthy_checks / total_checks if total_checks > 0 else 0
            },
            "checks": {
                name: {
                    "status": result.status.value,
                    "message": result.message,
                    "duration_ms": result.duration_ms,
                    "critical": result.critical,
                    "timestamp": result.timestamp.isoformat()
                }
                for name, result in check_results.items()
            }
        }
    
    def get_check_history(self, name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get history for a specific health check.
        
        Args:
            name: Name of the health check
            limit: Maximum number of historical results to return
            
        Returns:
            List of historical results
        """
        if name not in self.check_history:
            return []
        
        history = self.check_history[name][-limit:]
        
        return [
            {
                "status": result.status.value,
                "message": result.message,
                "duration_ms": result.duration_ms,
                "timestamp": result.timestamp.isoformat(),
                "critical": result.critical
            }
            for result in history
        ]
    
    def get_check_statistics(self, name: str, time_window_minutes: int = 60) -> Dict[str, Any]:
        """
        Get statistics for a specific health check.
        
        Args:
            name: Name of the health check
            time_window_minutes: Time window for statistics
            
        Returns:
            Check statistics
        """
        if name not in self.check_history:
            return {"error": f"No history found for check '{name}'"}
        
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        recent_results = [
            r for r in self.check_history[name]
            if r.timestamp >= cutoff_time
        ]
        
        if not recent_results:
            return {
                "check_name": name,
                "time_window_minutes": time_window_minutes,
                "message": "No recent results"
            }
        
        total_checks = len(recent_results)
        healthy_checks = len([r for r in recent_results if r.status == HealthStatus.HEALTHY])
        unhealthy_checks = len([r for r in recent_results if r.status == HealthStatus.UNHEALTHY])
        
        durations = [r.duration_ms for r in recent_results]
        avg_duration = sum(durations) / len(durations)
        min_duration = min(durations)
        max_duration = max(durations)
        
        return {
            "check_name": name,
            "time_window_minutes": time_window_minutes,
            "total_checks": total_checks,
            "healthy_checks": healthy_checks,
            "unhealthy_checks": unhealthy_checks,
            "success_rate": healthy_checks / total_checks,
            "average_duration_ms": avg_duration,
            "min_duration_ms": min_duration,
            "max_duration_ms": max_duration,
            "last_check": recent_results[-1].timestamp.isoformat() if recent_results else None
        }


# Default health checks for common services
async def database_health_check() -> bool:
    """Basic database connectivity check."""
    try:
        # This would be implemented based on actual database
        # For now, return True as placeholder
        return True
    except Exception:
        return False


async def external_api_health_check() -> bool:
    """Check external API connectivity."""
    try:
        # This would check actual external APIs (Gemini, etc.)
        # For now, return True as placeholder
        return True
    except Exception:
        return False


async def memory_health_check() -> bool:
    """Check system memory usage."""
    try:
        import psutil
        memory = psutil.virtual_memory()
        # Consider unhealthy if memory usage > 90%
        return memory.percent < 90
    except Exception:
        return False


async def disk_health_check() -> bool:
    """Check disk space."""
    try:
        import psutil
        disk = psutil.disk_usage('/')
        # Consider unhealthy if disk usage > 90%
        return disk.percent < 90
    except Exception:
        return False


# Global health checker instance
health_checker = HealthChecker()


def setup_default_health_checks():
    """Set up default health checks."""
    health_checker.register_check(
        "memory",
        memory_health_check,
        timeout_seconds=5.0,
        critical=False,
        description="System memory usage check"
    )
    
    health_checker.register_check(
        "disk",
        disk_health_check,
        timeout_seconds=5.0,
        critical=False,
        description="System disk space check"
    )
    
    health_checker.register_check(
        "database",
        database_health_check,
        timeout_seconds=10.0,
        critical=True,
        description="Database connectivity check"
    )
    
    health_checker.register_check(
        "external_apis",
        external_api_health_check,
        timeout_seconds=15.0,
        critical=True,
        description="External API connectivity check"
    )
    
    logger.info("Default health checks configured")


async def get_health_status() -> Dict[str, Any]:
    """Get comprehensive health status."""
    return await health_checker.get_overall_health()