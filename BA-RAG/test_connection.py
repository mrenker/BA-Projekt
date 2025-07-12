#!/usr/bin/env python3
"""
Test connection script for GraphQL to ChromaDB pipeline
This script helps diagnose connection issues and test the setup
"""

import os
import sys
import requests
import json
from typing import Dict, Any

def test_graphql_endpoint(endpoint: str, headers: Dict[str, str] = None) -> bool:
    """Test if GraphQL endpoint is accessible."""
    print(f"Testing GraphQL endpoint: {endpoint}")
    
    try:
        # Simple introspection query
        query = """
        query IntrospectionQuery {
            __schema {
                queryType {
                    name
                }
            }
        }
        """
        
        payload = {
            'query': query,
            'variables': {}
        }
        
        response = requests.post(
            endpoint,
            json=payload,
            headers=headers or {},
            timeout=10
        )
        
        if response.status_code == 200:
            print("✓ GraphQL endpoint is accessible")
            return True
        else:
            print(f"✗ GraphQL endpoint returned status {response.status_code}")
            print(f"Response: {response.text[:200]}")
            return False
            
    except requests.exceptions.ConnectionError as e:
        print(f"✗ Connection failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_chromadb_connection(host: str, port: int) -> bool:
    """Test if ChromaDB is accessible."""
    print(f"Testing ChromaDB connection: {host}:{port}")
    
    try:
        # Try v2 API first
        url = f"http://{host}:{port}/api/v2/tenants/default_tenant/databases/default_database"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            print("✓ ChromaDB v2 API is accessible")
            return True
        else:
            # Try v1 API as fallback
            url = f"http://{host}:{port}/api/v1/heartbeat"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                print("✓ ChromaDB v1 API is accessible")
                return True
            else:
                print(f"✗ ChromaDB returned status {response.status_code}")
                return False
            
    except requests.exceptions.ConnectionError as e:
        print(f"✗ ChromaDB connection failed: {e}")
        return False
    except Exception as e:
        print(f"✗ ChromaDB error: {e}")
        return False

def main():
    """Main test function."""
    print("GraphQL to ChromaDB Connection Test")
    print("=" * 40)
    
    # Get environment variables
    graphql_endpoint = os.getenv('GRAPHQL_ENDPOINT', 'http://localhost:4000/graphql')
    graphql_headers_str = os.getenv('GRAPHQL_HEADERS', '{}')
    chromadb_host = os.getenv('CHROMADB_HOST', 'localhost')
    chromadb_port = int(os.getenv('CHROMADB_PORT', '8000'))
    
    # Parse headers
    try:
        graphql_headers = json.loads(graphql_headers_str) if graphql_headers_str else {}
    except json.JSONDecodeError:
        print(f"Warning: Invalid GRAPHQL_HEADERS format: {graphql_headers_str}")
        graphql_headers = {}
    
    print(f"Configuration:")
    print(f"  GraphQL Endpoint: {graphql_endpoint}")
    print(f"  GraphQL Headers: {graphql_headers}")
    print(f"  ChromaDB Host: {chromadb_host}")
    print(f"  ChromaDB Port: {chromadb_port}")
    print()
    
    # Test connections
    graphql_ok = test_graphql_endpoint(graphql_endpoint, graphql_headers)
    print()
    chromadb_ok = test_chromadb_connection(chromadb_host, chromadb_port)
    print()
    
    # Summary
    print("Test Results:")
    print("=" * 40)
    print(f"GraphQL: {'✓ OK' if graphql_ok else '✗ FAILED'}")
    print(f"ChromaDB: {'✓ OK' if chromadb_ok else '✗ FAILED'}")
    
    if not graphql_ok:
        print()
        print("To fix GraphQL connection:")
        print("1. Make sure your GraphQL service is running")
        print("2. Check the GRAPHQL_ENDPOINT URL")
        print("3. Verify authentication headers if required")
        print("4. If using Docker, ensure the service is accessible")
    
    if not chromadb_ok:
        print()
        print("To fix ChromaDB connection:")
        print("1. Make sure ChromaDB is running")
        print("2. Check CHROMADB_HOST and CHROMADB_PORT")
        print("3. If using Docker, ensure ChromaDB container is accessible")
    
    if graphql_ok and chromadb_ok:
        print()
        print("✓ All connections successful! You can now run the main application.")
        return 0
    else:
        print()
        print("✗ Some connections failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 