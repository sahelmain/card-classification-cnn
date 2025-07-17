import sqlite3
import json
import uuid
from datetime import datetime
import logging
import os
import threading
from contextlib import contextmanager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Comprehensive database manager for prediction logging and analytics.
    Provides thread-safe operations and automatic schema management.
    """
    
    def __init__(self, db_path='data/predictions.db'):
        self.db_path = db_path
        self.lock = threading.RLock()  # Thread-safe operations
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self.init_database()
        
        logger.info(f"Database manager initialized with path: {db_path}")
    
    @contextmanager
    def get_connection(self):
        """Thread-safe database connection context manager"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            try:
                yield conn
            finally:
                conn.close()
    
    def init_database(self):
        """Initialize database schema with all required tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Predictions table for logging all predictions
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id TEXT PRIMARY KEY,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    model_name TEXT NOT NULL,
                    predicted_class TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    processing_time_ms REAL,
                    image_size TEXT,
                    user_ip TEXT,
                    session_id TEXT,
                    top_predictions TEXT,  -- JSON string
                    is_correct BOOLEAN,    -- For feedback
                    actual_class TEXT,     -- For feedback
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Model performance tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    date DATE NOT NULL,
                    total_predictions INTEGER DEFAULT 0,
                    average_confidence REAL DEFAULT 0.0,
                    average_processing_time REAL DEFAULT 0.0,
                    accuracy_rate REAL DEFAULT 0.0,
                    error_count INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(model_name, date)
                )
            ''')
            
            # User sessions for analytics
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_ip TEXT,
                    user_agent TEXT,
                    start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_activity DATETIME DEFAULT CURRENT_TIMESTAMP,
                    prediction_count INTEGER DEFAULT 0
                )
            ''')
            
            # System logs
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    level TEXT NOT NULL,  -- INFO, WARNING, ERROR
                    component TEXT NOT NULL,  -- API, MODEL, DATABASE
                    message TEXT NOT NULL,
                    details TEXT,  -- JSON string for additional data
                    user_ip TEXT,
                    session_id TEXT
                )
            ''')
            
            # Model usage statistics
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    usage_count INTEGER DEFAULT 0,
                    last_used DATETIME,
                    total_confidence REAL DEFAULT 0.0,
                    total_processing_time REAL DEFAULT 0.0,
                    success_count INTEGER DEFAULT 0,
                    error_count INTEGER DEFAULT 0,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(model_name)
                )
            ''')
            
            conn.commit()
            logger.info("Database schema initialized successfully")
    
    def log_prediction(self, prediction_data):
        """Log a prediction to the database"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            prediction_id = str(uuid.uuid4())
            top_predictions_json = json.dumps(prediction_data.get('top_predictions', []))
            
            cursor.execute('''
                INSERT INTO predictions (
                    id, model_name, predicted_class, confidence, 
                    processing_time_ms, image_size, user_ip, session_id, 
                    top_predictions
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                prediction_id,
                prediction_data.get('model_name'),
                prediction_data.get('predicted_class'),
                prediction_data.get('confidence'),
                prediction_data.get('processing_time_ms'),
                prediction_data.get('image_size'),
                prediction_data.get('user_ip'),
                prediction_data.get('session_id'),
                top_predictions_json
            ))
            
            conn.commit()
            logger.info(f"Prediction logged with ID: {prediction_id}")
            return prediction_id
    
    def update_model_stats(self, model_name, processing_time, confidence, success=True):
        """Update model usage statistics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO model_stats (
                    model_name, usage_count, last_used, total_confidence,
                    total_processing_time, success_count, error_count
                ) VALUES (
                    ?, 
                    COALESCE((SELECT usage_count FROM model_stats WHERE model_name = ?), 0) + 1,
                    CURRENT_TIMESTAMP,
                    COALESCE((SELECT total_confidence FROM model_stats WHERE model_name = ?), 0) + ?,
                    COALESCE((SELECT total_processing_time FROM model_stats WHERE model_name = ?), 0) + ?,
                    COALESCE((SELECT success_count FROM model_stats WHERE model_name = ?), 0) + ?,
                    COALESCE((SELECT error_count FROM model_stats WHERE model_name = ?), 0) + ?
                )
            ''', (
                model_name, model_name, model_name, confidence, 
                model_name, processing_time, model_name, 1 if success else 0,
                model_name, 0 if success else 1
            ))
            
            conn.commit()
    
    def log_system_event(self, level, component, message, details=None, user_ip=None, session_id=None):
        """Log system events for monitoring and debugging"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            details_json = json.dumps(details) if details else None
            
            cursor.execute('''
                INSERT INTO system_logs (
                    level, component, message, details, user_ip, session_id
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (level, component, message, details_json, user_ip, session_id))
            
            conn.commit()
    
    def get_analytics_summary(self, days=30):
        """Get comprehensive analytics summary"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Basic stats
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_predictions,
                    COUNT(DISTINCT session_id) as unique_sessions,
                    AVG(confidence) as avg_confidence,
                    AVG(processing_time_ms) as avg_processing_time
                FROM predictions 
                WHERE timestamp >= datetime('now', '-{} days')
            '''.format(days))
            
            basic_stats = dict(cursor.fetchone())
            
            # Model performance
            cursor.execute('''
                SELECT 
                    model_name,
                    COUNT(*) as usage_count,
                    AVG(confidence) as avg_confidence,
                    AVG(processing_time_ms) as avg_processing_time
                FROM predictions 
                WHERE timestamp >= datetime('now', '-{} days')
                GROUP BY model_name
                ORDER BY usage_count DESC
            '''.format(days))
            
            model_performance = [dict(row) for row in cursor.fetchall()]
            
            return {
                'basic_stats': basic_stats,
                'model_performance': model_performance,
                'period_days': days
            }
    
    def cleanup_old_data(self, days_to_keep=90):
        """Clean up old data to manage database size"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Clean old predictions
            cursor.execute('''
                DELETE FROM predictions 
                WHERE timestamp < datetime('now', '-{} days')
            '''.format(days_to_keep))
            
            predictions_deleted = cursor.rowcount
            
            # Clean old system logs
            cursor.execute('''
                DELETE FROM system_logs 
                WHERE timestamp < datetime('now', '-{} days')
            '''.format(days_to_keep))
            
            logs_deleted = cursor.rowcount
            
            conn.commit()
            
            logger.info(f"Cleanup completed: {predictions_deleted} predictions, {logs_deleted} logs deleted")
            
            return {
                'predictions_deleted': predictions_deleted,
                'logs_deleted': logs_deleted
            }

def demonstrate_database():
    """Demonstrate database functionality"""
    print("🗄️ Database Manager Demonstration")
    print("=" * 50)
    
    # Initialize database
    db = DatabaseManager('demo_predictions.db')
    
    # Log sample prediction
    sample_prediction = {
        'model_name': 'custom_cnn',
        'predicted_class': 'Ace Of Spades',
        'confidence': 0.95,
        'processing_time_ms': 45.2,
        'image_size': '200x200',
        'user_ip': '192.168.1.100',
        'session_id': 'demo_session_123',
        'top_predictions': [
            {'class': 'Ace Of Spades', 'confidence': 0.95},
            {'class': 'King Of Spades', 'confidence': 0.03}
        ]
    }
    
    prediction_id = db.log_prediction(sample_prediction)
    print(f"✅ Prediction logged with ID: {prediction_id}")
    
    # Update model stats
    db.update_model_stats('custom_cnn', 45.2, 0.95, success=True)
    print("✅ Model statistics updated")
    
    # Log system event
    db.log_system_event('INFO', 'API', 'Prediction completed successfully', 
                       details={'model': 'custom_cnn', 'confidence': 0.95})
    print("✅ System event logged")
    
    print("\n📊 Key Features:")
    print("• Comprehensive prediction logging")
    print("• Model performance tracking")
    print("• User session management")
    print("• System event logging")
    print("• Analytics and reporting")

if __name__ == "__main__":
    demonstrate_database() 