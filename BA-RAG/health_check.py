#!/usr/bin/env python3
"""
Simple health check endpoint for the GraphQL to ChromaDB pipeline.
This can be enabled via the HEALTH_CHECK_ENABLED environment variable.
"""

import os
import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime
import threading
import time

logger = logging.getLogger(__name__)


class HealthCheckHandler(BaseHTTPRequestHandler):
    """HTTP request handler for health checks."""
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/health':
            self.send_health_response()
        elif self.path == '/status':
            self.send_status_response()
        else:
            self.send_response(404)
            self.end_headers()
    
    def send_health_response(self):
        """Send basic health check response."""
        try:
            health_status = {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'service': 'graphql-chromadb-pipeline'
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(health_status).encode())
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self.send_response(500)
            self.end_headers()
    
    def send_status_response(self):
        """Send detailed status response."""
        try:
            # Check if log file exists and get last sync time
            log_file = '/app/app.log'
            last_sync = None
            
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    for line in reversed(lines[-100:]):  # Check last 100 lines
                        if 'Successfully synced' in line:
                            # Extract timestamp from log line
                            try:
                                timestamp_str = line.split(' - ')[0]
                                last_sync = timestamp_str
                                break
                            except:
                                pass
            
            status = {
                'status': 'running',
                'timestamp': datetime.now().isoformat(),
                'service': 'graphql-chromadb-pipeline',
                'last_sync': last_sync,
                'config': {
                    'graphql_endpoint': os.getenv('GRAPHQL_ENDPOINT', 'not_configured'),
                    'chromadb_host': os.getenv('CHROMADB_HOST', 'not_configured'),
                    'chromadb_port': os.getenv('CHROMADB_PORT', 'not_configured'),
                    'collection': os.getenv('CHROMADB_COLLECTION', 'graphql_data'),
                    'run_mode': os.getenv('RUN_MODE', 'continuous'),
                    'sync_interval': os.getenv('SYNC_INTERVAL_SECONDS', '300')
                }
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(status, indent=2).encode())
            
        except Exception as e:
            logger.error(f"Status check failed: {e}")
            error_response = {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(error_response).encode())
    
    def log_message(self, format, *args):
        """Override log_message to reduce noise."""
        pass


class HealthCheckServer:
    """Health check HTTP server."""
    
    def __init__(self, port=8080):
        self.port = port
        self.server = None
        self.thread = None
    
    def start(self):
        """Start the health check server in a separate thread."""
        try:
            self.server = HTTPServer(('0.0.0.0', self.port), HealthCheckHandler)
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.thread.start()
            logger.info(f"Health check server started on port {self.port}")
        except Exception as e:
            logger.error(f"Failed to start health check server: {e}")
    
    def stop(self):
        """Stop the health check server."""
        if self.server:
            self.server.shutdown()
            logger.info("Health check server stopped")


def start_health_check_if_enabled():
    """Start health check server if enabled via environment variable."""
    if os.getenv('HEALTH_CHECK_ENABLED', 'false').lower() == 'true':
        health_server = HealthCheckServer()
        health_server.start()
        return health_server
    return None


if __name__ == '__main__':
    # For standalone testing
    logging.basicConfig(level=logging.INFO)
    server = HealthCheckServer()
    server.start()
    
    try:
        print("Health check server running on http://localhost:8080")
        print("Endpoints:")
        print("  GET /health  - Basic health check")
        print("  GET /status  - Detailed status")
        print("Press Ctrl+C to stop...")
        
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()
        print("\nHealth check server stopped") 