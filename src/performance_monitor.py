import time
import psutil
import threading
import logging
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """
    Comprehensive performance monitoring system for production environments.
    Tracks system resources, API performance, and model metrics.
    """
    
    def __init__(self, monitoring_interval=60, max_history=1000):
        self.monitoring_interval = monitoring_interval
        self.max_history = max_history
        
        # Thread-safe metrics storage
        self.lock = threading.RLock()
        
        # Metrics collections
        self.system_metrics = deque(maxlen=max_history)
        self.api_metrics = deque(maxlen=max_history)
        self.model_metrics = defaultdict(lambda: deque(maxlen=max_history))
        
        # Performance counters
        self.request_counter = 0
        self.error_counter = 0
        self.total_response_time = 0
        
        # Monitoring thread
        self.monitoring_thread = None
        self.is_monitoring = False
        
        # Alert thresholds
        self.thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'response_time_ms': 1000.0
        }
        
        logger.info("Performance monitor initialized")
    
    def start_monitoring(self):
        """Start the performance monitoring thread"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop the performance monitoring thread"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                self._collect_system_metrics()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            system_metric = {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / (1024**3),
                'disk_percent': disk.percent,
                'disk_used_gb': disk.used / (1024**3)
            }
            
            with self.lock:
                self.system_metrics.append(system_metric)
            
            # Check alerts
            self._check_alerts(system_metric)
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
    
    def record_api_request(self, endpoint, method, response_time_ms, status_code):
        """Record API request metrics"""
        api_metric = {
            'timestamp': datetime.now().isoformat(),
            'endpoint': endpoint,
            'method': method,
            'response_time_ms': response_time_ms,
            'status_code': status_code
        }
        
        with self.lock:
            self.api_metrics.append(api_metric)
            self.request_counter += 1
            self.total_response_time += response_time_ms
            
            if status_code >= 400:
                self.error_counter += 1
    
    def record_model_prediction(self, model_name, processing_time_ms, confidence):
        """Record model prediction metrics"""
        model_metric = {
            'timestamp': datetime.now().isoformat(),
            'processing_time_ms': processing_time_ms,
            'confidence': confidence
        }
        
        with self.lock:
            self.model_metrics[model_name].append(model_metric)
    
    def get_current_status(self):
        """Get current system status"""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            status = {
                'timestamp': datetime.now().isoformat(),
                'status': 'healthy',
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': disk.percent
            }
            
            # Check health
            if (cpu_percent > self.thresholds['cpu_percent'] or 
                memory.percent > self.thresholds['memory_percent']):
                status['status'] = 'warning'
            
            return status
            
        except Exception as e:
            return {
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error': str(e)
            }
    
    def get_api_summary(self, hours=24):
        """Get API performance summary"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.lock:
            recent_metrics = [
                m for m in self.api_metrics 
                if datetime.fromisoformat(m['timestamp']) > cutoff_time
            ]
        
        if not recent_metrics:
            return {
                'total_requests': 0,
                'average_response_time': 0,
                'error_rate': 0
            }
        
        total_requests = len(recent_metrics)
        avg_response_time = sum(m['response_time_ms'] for m in recent_metrics) / total_requests
        error_count = sum(1 for m in recent_metrics if m['status_code'] >= 400)
        error_rate = (error_count / total_requests) * 100
        
        return {
            'total_requests': total_requests,
            'average_response_time': avg_response_time,
            'error_rate': error_rate,
            'error_count': error_count
        }
    
    def get_model_summary(self, model_name=None):
        """Get model performance summary"""
        with self.lock:
            if model_name:
                metrics = self.model_metrics.get(model_name, [])
                if not metrics:
                    return {'total_predictions': 0}
                
                total = len(metrics)
                avg_time = sum(m['processing_time_ms'] for m in metrics) / total
                avg_confidence = sum(m['confidence'] for m in metrics) / total
                
                return {
                    'total_predictions': total,
                    'average_processing_time': avg_time,
                    'average_confidence': avg_confidence
                }
            else:
                return {name: self.get_model_summary(name) 
                       for name in self.model_metrics.keys()}
    
    def _check_alerts(self, system_metric):
        """Check for alert conditions"""
        alerts = []
        
        if system_metric['cpu_percent'] > self.thresholds['cpu_percent']:
            alerts.append(f"High CPU: {system_metric['cpu_percent']:.1f}%")
        
        if system_metric['memory_percent'] > self.thresholds['memory_percent']:
            alerts.append(f"High Memory: {system_metric['memory_percent']:.1f}%")
        
        if alerts:
            for alert in alerts:
                logger.warning(f"ALERT: {alert}")
    
    def export_metrics(self, output_path):
        """Export metrics to JSON file"""
        with self.lock:
            data = {
                'timestamp': datetime.now().isoformat(),
                'system_metrics': list(self.system_metrics),
                'api_metrics': list(self.api_metrics),
                'model_metrics': {k: list(v) for k, v in self.model_metrics.items()}
            }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Metrics exported to {output_path}")

def demonstrate_monitor():
    """Demonstrate monitoring capabilities"""
    print("📊 Performance Monitor Demonstration")
    print("=" * 50)
    
    monitor = PerformanceMonitor(monitoring_interval=5)
    monitor.start_monitoring()
    
    # Simulate requests
    monitor.record_api_request('/api/predict', 'POST', 45.2, 200)
    monitor.record_api_request('/api/models', 'GET', 12.5, 200)
    monitor.record_model_prediction('custom_cnn', 45.2, 0.95)
    
    status = monitor.get_current_status()
    print(f"✅ System status: {status['status']}")
    
    api_summary = monitor.get_api_summary()
    print(f"✅ API requests: {api_summary['total_requests']}")
    
    monitor.stop_monitoring()
    
    print("\n📈 Features:")
    print("• System resource monitoring")
    print("• API performance tracking")
    print("• Model metrics collection")
    print("• Automated alerting")
    print("• Metrics export")

if __name__ == "__main__":
    demonstrate_monitor() 