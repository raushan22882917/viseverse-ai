"""
Comprehensive logging and monitoring utilities for VisaVerse Guardian AI.
Provides structured logging, metrics collection, and performance monitoring.
"""

import logging
import time
import json
import asyncio
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from functools import wraps
from collections import defaultdict, deque
import threading
import psutil
import os

# Configure structured logging
class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'application_id'):
            log_entry['application_id'] = record.application_id
        if hasattr(record, 'service'):
            log_entry['service'] = record.service
        if hasattr(record, 'operation'):
            log_entry['operation'] = record.operation
        if hasattr(record, 'execution_time_ms'):
            log_entry['execution_time_ms'] = record.execution_time_ms
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


def setup_logging(log_level: str = "INFO", structured: bool = True):
    """
    Set up application logging.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        structured: Whether to use structured JSON logging
    """
    level = getattr(logging, log_level.upper())
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    if structured:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Suppress noisy third-party loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('google').setLevel(logging.WARNING)


@dataclass
class MetricPoint:
    """A single metric data point."""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str]
    metric_type: str  # counter, gauge, histogram, timer


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations."""
    operation: str
    service: str
    duration_ms: float
    success: bool
    timestamp: datetime
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    error_type: Optional[str] = None


class MetricsCollector:
    """
    Metrics collection and aggregation system.
    """
    
    def __init__(self, max_points: int = 10000):
        self.max_points = max_points
        self.metrics: deque = deque(maxlen=max_points)
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def increment_counter(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        with self._lock:
            key = self._make_key(name, tags or {})
            self.counters[key] += value
            
            self.metrics.append(MetricPoint(
                name=name,
                value=value,
                timestamp=datetime.now(),
                tags=tags or {},
                metric_type="counter"
            ))
    
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric value."""
        with self._lock:
            key = self._make_key(name, tags or {})
            self.gauges[key] = value
            
            self.metrics.append(MetricPoint(
                name=name,
                value=value,
                timestamp=datetime.now(),
                tags=tags or {},
                metric_type="gauge"
            ))
    
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a histogram value."""
        with self._lock:
            key = self._make_key(name, tags or {})
            self.histograms[key].append(value)
            
            # Keep only recent values (last 1000)
            if len(self.histograms[key]) > 1000:
                self.histograms[key] = self.histograms[key][-1000:]
            
            self.metrics.append(MetricPoint(
                name=name,
                value=value,
                timestamp=datetime.now(),
                tags=tags or {},
                metric_type="histogram"
            ))
    
    def record_timer(self, name: str, duration_ms: float, tags: Optional[Dict[str, str]] = None):
        """Record a timer value."""
        with self._lock:
            key = self._make_key(name, tags or {})
            self.timers[key].append(duration_ms)
            
            # Keep only recent values (last 1000)
            if len(self.timers[key]) > 1000:
                self.timers[key] = self.timers[key][-1000:]
            
            self.metrics.append(MetricPoint(
                name=name,
                value=duration_ms,
                timestamp=datetime.now(),
                tags=tags or {},
                metric_type="timer"
            ))
    
    def get_counter(self, name: str, tags: Optional[Dict[str, str]] = None) -> float:
        """Get counter value."""
        key = self._make_key(name, tags or {})
        return self.counters.get(key, 0.0)
    
    def get_gauge(self, name: str, tags: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get gauge value."""
        key = self._make_key(name, tags or {})
        return self.gauges.get(key)
    
    def get_histogram_stats(self, name: str, tags: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Get histogram statistics."""
        key = self._make_key(name, tags or {})
        values = self.histograms.get(key, [])
        
        if not values:
            return {}
        
        sorted_values = sorted(values)
        count = len(sorted_values)
        
        return {
            "count": count,
            "min": sorted_values[0],
            "max": sorted_values[-1],
            "mean": sum(sorted_values) / count,
            "p50": sorted_values[int(count * 0.5)],
            "p90": sorted_values[int(count * 0.9)],
            "p95": sorted_values[int(count * 0.95)],
            "p99": sorted_values[int(count * 0.99)]
        }
    
    def get_timer_stats(self, name: str, tags: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Get timer statistics."""
        return self.get_histogram_stats(name, tags)
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metrics."""
        with self._lock:
            return {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms": {k: self.get_histogram_stats("", {}) for k in self.histograms.keys()},
                "timers": {k: self.get_timer_stats("", {}) for k in self.timers.keys()},
                "timestamp": datetime.now().isoformat()
            }
    
    def _make_key(self, name: str, tags: Dict[str, str]) -> str:
        """Create a unique key for metric with tags."""
        if not tags:
            return name
        
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}[{tag_str}]"


class PerformanceMonitor:
    """
    Performance monitoring and profiling system.
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.performance_data: deque = deque(maxlen=1000)
        self._lock = threading.Lock()
    
    def record_performance(self, metrics: PerformanceMetrics):
        """Record performance metrics."""
        with self._lock:
            self.performance_data.append(metrics)
        
        # Update metrics collector
        tags = {
            "service": metrics.service,
            "operation": metrics.operation,
            "success": str(metrics.success)
        }
        
        if metrics.error_type:
            tags["error_type"] = metrics.error_type
        
        self.metrics_collector.record_timer(
            "operation_duration",
            metrics.duration_ms,
            tags
        )
        
        self.metrics_collector.increment_counter(
            "operation_count",
            1.0,
            tags
        )
        
        if not metrics.success:
            self.metrics_collector.increment_counter(
                "operation_errors",
                1.0,
                tags
            )
    
    def get_performance_summary(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get performance summary for the specified time window."""
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        
        with self._lock:
            recent_data = [
                m for m in self.performance_data
                if m.timestamp >= cutoff_time
            ]
        
        if not recent_data:
            return {"message": "No performance data available"}
        
        # Aggregate by service and operation
        service_stats = defaultdict(lambda: {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_duration_ms": 0,
            "min_duration_ms": float('inf'),
            "max_duration_ms": 0,
            "operations": defaultdict(lambda: {
                "count": 0,
                "success_count": 0,
                "total_duration_ms": 0,
                "min_duration_ms": float('inf'),
                "max_duration_ms": 0
            })
        })
        
        for metrics in recent_data:
            service = metrics.service
            operation = metrics.operation
            
            # Service-level stats
            service_stats[service]["total_requests"] += 1
            service_stats[service]["total_duration_ms"] += metrics.duration_ms
            service_stats[service]["min_duration_ms"] = min(
                service_stats[service]["min_duration_ms"],
                metrics.duration_ms
            )
            service_stats[service]["max_duration_ms"] = max(
                service_stats[service]["max_duration_ms"],
                metrics.duration_ms
            )
            
            if metrics.success:
                service_stats[service]["successful_requests"] += 1
            else:
                service_stats[service]["failed_requests"] += 1
            
            # Operation-level stats
            op_stats = service_stats[service]["operations"][operation]
            op_stats["count"] += 1
            op_stats["total_duration_ms"] += metrics.duration_ms
            op_stats["min_duration_ms"] = min(op_stats["min_duration_ms"], metrics.duration_ms)
            op_stats["max_duration_ms"] = max(op_stats["max_duration_ms"], metrics.duration_ms)
            
            if metrics.success:
                op_stats["success_count"] += 1
        
        # Calculate averages and success rates
        summary = {}
        for service, stats in service_stats.items():
            total_requests = stats["total_requests"]
            
            service_summary = {
                "total_requests": total_requests,
                "success_rate": stats["successful_requests"] / total_requests if total_requests > 0 else 0,
                "average_duration_ms": stats["total_duration_ms"] / total_requests if total_requests > 0 else 0,
                "min_duration_ms": stats["min_duration_ms"] if stats["min_duration_ms"] != float('inf') else 0,
                "max_duration_ms": stats["max_duration_ms"],
                "operations": {}
            }
            
            for operation, op_stats in stats["operations"].items():
                op_count = op_stats["count"]
                service_summary["operations"][operation] = {
                    "count": op_count,
                    "success_rate": op_stats["success_count"] / op_count if op_count > 0 else 0,
                    "average_duration_ms": op_stats["total_duration_ms"] / op_count if op_count > 0 else 0,
                    "min_duration_ms": op_stats["min_duration_ms"] if op_stats["min_duration_ms"] != float('inf') else 0,
                    "max_duration_ms": op_stats["max_duration_ms"]
                }
            
            summary[service] = service_summary
        
        return {
            "time_window_minutes": time_window_minutes,
            "total_operations": len(recent_data),
            "services": summary,
            "timestamp": datetime.now().isoformat()
        }


class SystemMonitor:
    """
    System resource monitoring.
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None
    
    async def start_monitoring(self, interval_seconds: int = 30):
        """Start system monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_task = asyncio.create_task(self._monitor_loop(interval_seconds))
    
    async def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
    
    async def _monitor_loop(self, interval_seconds: int):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in system monitoring: {str(e)}")
                await asyncio.sleep(interval_seconds)
    
    async def _collect_system_metrics(self):
        """Collect system metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.metrics_collector.set_gauge("system_cpu_percent", cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.metrics_collector.set_gauge("system_memory_percent", memory.percent)
        self.metrics_collector.set_gauge("system_memory_available_mb", memory.available / 1024 / 1024)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        self.metrics_collector.set_gauge("system_disk_percent", disk.percent)
        self.metrics_collector.set_gauge("system_disk_free_gb", disk.free / 1024 / 1024 / 1024)
        
        # Process-specific metrics
        process = psutil.Process(os.getpid())
        process_memory = process.memory_info()
        self.metrics_collector.set_gauge("process_memory_rss_mb", process_memory.rss / 1024 / 1024)
        self.metrics_collector.set_gauge("process_memory_vms_mb", process_memory.vms / 1024 / 1024)
        self.metrics_collector.set_gauge("process_cpu_percent", process.cpu_percent())
        
        # Network connections
        connections = len(process.connections())
        self.metrics_collector.set_gauge("process_connections", connections)


def monitor_performance(
    service: str,
    operation: str,
    metrics_collector: Optional[MetricsCollector] = None,
    performance_monitor: Optional[PerformanceMonitor] = None
):
    """
    Decorator for monitoring function performance.
    
    Args:
        service: Service name
        operation: Operation name
        metrics_collector: Metrics collector instance
        performance_monitor: Performance monitor instance
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            request_id = getattr(kwargs.get('request', None), 'state', {}).get('request_id', None)
            user_id = kwargs.get('user_id', None)
            success = True
            error_type = None
            
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_type = type(e).__name__
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000
                
                # Record performance metrics
                if performance_monitor:
                    performance_metrics = PerformanceMetrics(
                        operation=operation,
                        service=service,
                        duration_ms=duration_ms,
                        success=success,
                        timestamp=datetime.now(),
                        request_id=request_id,
                        user_id=user_id,
                        error_type=error_type
                    )
                    performance_monitor.record_performance(performance_metrics)
                
                # Log performance
                logger = logging.getLogger(f"{service}.{operation}")
                extra = {
                    'service': service,
                    'operation': operation,
                    'execution_time_ms': duration_ms,
                    'success': success
                }
                
                if request_id:
                    extra['request_id'] = request_id
                if user_id:
                    extra['user_id'] = user_id
                if error_type:
                    extra['error_type'] = error_type
                
                if success:
                    logger.info(f"Operation completed successfully", extra=extra)
                else:
                    logger.error(f"Operation failed", extra=extra)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For synchronous functions, create async wrapper and run it
            return asyncio.run(async_wrapper(*args, **kwargs))
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# Global instances
metrics_collector = MetricsCollector()
performance_monitor = PerformanceMonitor(metrics_collector)
system_monitor = SystemMonitor(metrics_collector)


def get_monitoring_status() -> Dict[str, Any]:
    """Get overall monitoring system status."""
    return {
        "metrics": metrics_collector.get_all_metrics(),
        "performance": performance_monitor.get_performance_summary(),
        "system_monitoring_active": system_monitor.monitoring,
        "timestamp": datetime.now().isoformat()
    }