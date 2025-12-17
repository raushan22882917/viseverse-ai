"""
Resilience and error handling utilities for VisaVerse Guardian AI.
Implements circuit breakers, retry logic, and graceful degradation patterns.
"""

import asyncio
import logging
import time
from typing import Any, Callable, Dict, Optional, TypeVar, Union
from functools import wraps
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
import random
import json

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Number of failures before opening
    recovery_timeout: int = 60  # Seconds before trying half-open
    success_threshold: int = 3  # Successes needed to close from half-open
    timeout: float = 30.0  # Request timeout in seconds


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_attempts: int = 3
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 60.0  # Maximum delay in seconds
    exponential_base: float = 2.0  # Exponential backoff base
    jitter: bool = True  # Add random jitter to delays


class CircuitBreaker:
    """
    Circuit breaker implementation for external service calls.
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.next_attempt_time: Optional[datetime] = None
        
        logger.info(f"Initialized circuit breaker '{name}' with config: {config}")
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: When circuit is open
            Exception: Original function exceptions when circuit is closed
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info(f"Circuit breaker '{self.name}' moved to HALF_OPEN state")
            else:
                raise CircuitBreakerOpenError(f"Circuit breaker '{self.name}' is OPEN")
        
        try:
            # Execute the function with timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs),
                timeout=self.config.timeout
            )
            
            # Handle success
            await self._on_success()
            return result
            
        except Exception as e:
            # Handle failure
            await self._on_failure(e)
            raise
    
    async def _on_success(self):
        """Handle successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info(f"Circuit breaker '{self.name}' moved to CLOSED state")
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0  # Reset failure count on success
    
    async def _on_failure(self, exception: Exception):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        logger.warning(f"Circuit breaker '{self.name}' recorded failure {self.failure_count}: {str(exception)}")
        
        if self.state == CircuitState.CLOSED and self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            self.next_attempt_time = datetime.now() + timedelta(seconds=self.config.recovery_timeout)
            logger.error(f"Circuit breaker '{self.name}' moved to OPEN state")
        elif self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.next_attempt_time = datetime.now() + timedelta(seconds=self.config.recovery_timeout)
            logger.error(f"Circuit breaker '{self.name}' moved back to OPEN state")
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt to reset."""
        return (
            self.next_attempt_time is not None and
            datetime.now() >= self.next_attempt_time
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "next_attempt_time": self.next_attempt_time.isoformat() if self.next_attempt_time else None
        }


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class RetryableError(Exception):
    """Base class for errors that should trigger retries."""
    pass


class NonRetryableError(Exception):
    """Base class for errors that should not trigger retries."""
    pass


def with_retry(config: RetryConfig):
    """
    Decorator for adding retry logic to functions.
    
    Args:
        config: Retry configuration
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                        
                except NonRetryableError:
                    # Don't retry non-retryable errors
                    raise
                    
                except Exception as e:
                    last_exception = e
                    
                    if attempt == config.max_attempts - 1:
                        # Last attempt, don't wait
                        break
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        config.base_delay * (config.exponential_base ** attempt),
                        config.max_delay
                    )
                    
                    # Add jitter if enabled
                    if config.jitter:
                        delay *= (0.5 + random.random() * 0.5)
                    
                    logger.warning(
                        f"Attempt {attempt + 1}/{config.max_attempts} failed for {func.__name__}: {str(e)}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    
                    await asyncio.sleep(delay)
            
            # All attempts failed
            logger.error(f"All {config.max_attempts} attempts failed for {func.__name__}")
            raise last_exception
        
        return wrapper
    return decorator


class ServiceHealthChecker:
    """
    Health checker for external services.
    """
    
    def __init__(self):
        self.service_status: Dict[str, Dict[str, Any]] = {}
        self.check_interval = 30  # seconds
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    def register_service(self, name: str, health_check_func: Callable, **kwargs):
        """
        Register a service for health checking.
        
        Args:
            name: Service name
            health_check_func: Function to check service health
            **kwargs: Additional service metadata
        """
        self.service_status[name] = {
            "health_check_func": health_check_func,
            "status": "unknown",
            "last_check": None,
            "last_success": None,
            "consecutive_failures": 0,
            "metadata": kwargs
        }
        
        logger.info(f"Registered service '{name}' for health checking")
    
    async def start_monitoring(self):
        """Start continuous health monitoring."""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("Started service health monitoring")
    
    async def stop_monitoring(self):
        """Stop health monitoring."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped service health monitoring")
    
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                await self._check_all_services()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {str(e)}")
                await asyncio.sleep(self.check_interval)
    
    async def _check_all_services(self):
        """Check health of all registered services."""
        for service_name in self.service_status:
            try:
                await self._check_service_health(service_name)
            except Exception as e:
                logger.error(f"Error checking health of service '{service_name}': {str(e)}")
    
    async def _check_service_health(self, service_name: str):
        """Check health of a specific service."""
        service_info = self.service_status[service_name]
        health_check_func = service_info["health_check_func"]
        
        try:
            # Execute health check with timeout
            is_healthy = await asyncio.wait_for(
                health_check_func() if asyncio.iscoroutinefunction(health_check_func) else health_check_func(),
                timeout=10.0
            )
            
            # Update status
            service_info["status"] = "healthy" if is_healthy else "unhealthy"
            service_info["last_check"] = datetime.now()
            
            if is_healthy:
                service_info["last_success"] = datetime.now()
                service_info["consecutive_failures"] = 0
            else:
                service_info["consecutive_failures"] += 1
                
        except Exception as e:
            service_info["status"] = "error"
            service_info["last_check"] = datetime.now()
            service_info["consecutive_failures"] += 1
            service_info["last_error"] = str(e)
            
            logger.warning(f"Health check failed for service '{service_name}': {str(e)}")
    
    def get_service_status(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific service."""
        service_info = self.service_status.get(service_name)
        if not service_info:
            return None
        
        return {
            "name": service_name,
            "status": service_info["status"],
            "last_check": service_info["last_check"].isoformat() if service_info["last_check"] else None,
            "last_success": service_info["last_success"].isoformat() if service_info["last_success"] else None,
            "consecutive_failures": service_info["consecutive_failures"],
            "metadata": service_info["metadata"]
        }
    
    def get_all_service_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all services."""
        return {
            name: self.get_service_status(name)
            for name in self.service_status
        }


class GracefulDegradation:
    """
    Graceful degradation manager for handling service failures.
    """
    
    def __init__(self):
        self.fallback_handlers: Dict[str, Callable] = {}
        self.degradation_levels: Dict[str, str] = {}
    
    def register_fallback(self, service_name: str, fallback_func: Callable):
        """
        Register a fallback function for a service.
        
        Args:
            service_name: Name of the service
            fallback_func: Fallback function to use when service fails
        """
        self.fallback_handlers[service_name] = fallback_func
        logger.info(f"Registered fallback handler for service '{service_name}'")
    
    async def execute_with_fallback(self, service_name: str, primary_func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with fallback on failure.
        
        Args:
            service_name: Name of the service
            primary_func: Primary function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Result from primary function or fallback
        """
        try:
            # Try primary function
            if asyncio.iscoroutinefunction(primary_func):
                return await primary_func(*args, **kwargs)
            else:
                return primary_func(*args, **kwargs)
                
        except Exception as e:
            logger.warning(f"Primary function failed for service '{service_name}': {str(e)}")
            
            # Try fallback if available
            fallback_func = self.fallback_handlers.get(service_name)
            if fallback_func:
                logger.info(f"Using fallback handler for service '{service_name}'")
                self.degradation_levels[service_name] = "degraded"
                
                try:
                    if asyncio.iscoroutinefunction(fallback_func):
                        return await fallback_func(*args, **kwargs)
                    else:
                        return fallback_func(*args, **kwargs)
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed for service '{service_name}': {str(fallback_error)}")
                    self.degradation_levels[service_name] = "failed"
                    raise
            else:
                logger.error(f"No fallback available for service '{service_name}'")
                self.degradation_levels[service_name] = "failed"
                raise
    
    def get_degradation_status(self) -> Dict[str, str]:
        """Get current degradation status for all services."""
        return self.degradation_levels.copy()


# Global instances
circuit_breakers: Dict[str, CircuitBreaker] = {}
health_checker = ServiceHealthChecker()
graceful_degradation = GracefulDegradation()


def get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """
    Get or create a circuit breaker.
    
    Args:
        name: Circuit breaker name
        config: Configuration (uses default if not provided)
        
    Returns:
        Circuit breaker instance
    """
    if name not in circuit_breakers:
        if config is None:
            config = CircuitBreakerConfig()
        circuit_breakers[name] = CircuitBreaker(name, config)
    
    return circuit_breakers[name]


async def resilient_call(
    service_name: str,
    func: Callable,
    *args,
    circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
    retry_config: Optional[RetryConfig] = None,
    fallback_func: Optional[Callable] = None,
    **kwargs
) -> Any:
    """
    Execute a function with full resilience patterns.
    
    Args:
        service_name: Name of the service
        func: Function to execute
        *args: Function arguments
        circuit_breaker_config: Circuit breaker configuration
        retry_config: Retry configuration
        fallback_func: Fallback function
        **kwargs: Function keyword arguments
        
    Returns:
        Function result
    """
    # Get circuit breaker
    circuit_breaker = get_circuit_breaker(service_name, circuit_breaker_config)
    
    # Wrap function with retry if configured
    if retry_config:
        func = with_retry(retry_config)(func)
    
    # Execute with circuit breaker and fallback
    if fallback_func:
        graceful_degradation.register_fallback(service_name, fallback_func)
        return await graceful_degradation.execute_with_fallback(
            service_name,
            lambda: circuit_breaker.call(func, *args, **kwargs)
        )
    else:
        return await circuit_breaker.call(func, *args, **kwargs)


def get_resilience_status() -> Dict[str, Any]:
    """Get overall resilience system status."""
    return {
        "circuit_breakers": {name: cb.get_status() for name, cb in circuit_breakers.items()},
        "service_health": health_checker.get_all_service_status(),
        "degradation_status": graceful_degradation.get_degradation_status(),
        "timestamp": datetime.now().isoformat()
    }