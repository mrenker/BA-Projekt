#!/usr/bin/env python3
"""
GraphQL to ChromaDB Data Pipeline

This application pulls data from a GraphQL backend and stores it in ChromaDB.
"""

import os
import sys
import json
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

import requests
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Import health check functionality
from health_check import start_health_check_if_enabled

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)


class GraphQLClient:
    """GraphQL client for fetching data from the backend."""
    
    def __init__(self, endpoint: str, headers: Optional[Dict[str, str]] = None):
        self.endpoint = endpoint
        self.headers = headers or {}
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def execute_query(self, query: str, variables: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute a GraphQL query."""
        payload = {
            'query': query,
            'variables': variables or {}
        }
        
        try:
            logger.info(f"Executing GraphQL query to: {self.endpoint}")
            logger.debug(f"Query: {query}")
            logger.debug(f"Variables: {variables}")
            
            response = self.session.post(
                self.endpoint,
                json=payload,
                timeout=30
            )
            
            logger.info(f"Response status: {response.status_code}")
            logger.debug(f"Response headers: {dict(response.headers)}")
            
            # Log response content for debugging
            response_text = response.text
            logger.debug(f"Response content (first 500 chars): {response_text[:500]}")
            
            response.raise_for_status()
            
            # Check if response is empty
            if not response_text.strip():
                logger.error("Received empty response from GraphQL endpoint")
                raise Exception("Empty response from GraphQL endpoint")
            
            result = response.json()
            if 'errors' in result:
                logger.error(f"GraphQL errors: {result['errors']}")
                raise Exception(f"GraphQL query failed: {result['errors']}")
            
            return result.get('data', {})
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse response: {e}")
            logger.error(f"Response content: {response.text if 'response' in locals() else 'No response'}")
            raise


class ChromaDBManager:
    """ChromaDB manager for storing and retrieving documents."""
    
    def __init__(self, host: str, port: int, collection_name: str, embedding_model: str = "all-MiniLM-L6-v2"):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        self.client = chromadb.HttpClient(
            host=host,
            port=port,
            settings=Settings(
                allow_reset=True,
                anonymized_telemetry=False
            )
        )
        
        logger.info(f"Initialized ChromaDB client with host: {host}, port: {port}, collection name: {collection_name}")

        # Initialize embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        
        logger.info(f"Initialized ChromaDB client with embedding model: {embedding_model}")

        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Connected to existing collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Created new collection: {collection_name}")
    
    def add_documents(self, documents: List[str], metadatas: List[Dict], ids: List[str]):
        """Add documents to ChromaDB."""
        if not documents:
            return
        
        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(documents)} documents to ChromaDB")
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {e}")
            raise
    
    def upsert_documents(self, documents: List[str], metadatas: List[Dict], ids: List[str]):
        """Upsert documents to ChromaDB (add or update)."""
        if not documents:
            return
        
        try:
            self.collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Upserted {len(documents)} documents to ChromaDB")
        except Exception as e:
            logger.error(f"Failed to upsert documents to ChromaDB: {e}")
            raise


class DataPipeline:
    """Main data pipeline for syncing GraphQL data to ChromaDB."""
    
    def __init__(self):
        # Load environment variables
        self.graphql_endpoint = os.getenv('GRAPHQL_ENDPOINT', 'http://localhost:4000/graphql')
        self.graphql_headers = self._parse_headers(os.getenv('GRAPHQL_HEADERS', ''))
        self.chromadb_host = os.getenv('CHROMADB_HOST', 'localhost')
        self.chromadb_port = int(os.getenv('CHROMADB_PORT', '8000'))
        self.collection_name = os.getenv('CHROMADB_COLLECTION', 'graphql_data')
        self.sync_interval = int(os.getenv('SYNC_INTERVAL_SECONDS', '300'))  # 5 minutes default
        self.embedding_model = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        
        logger.info(f"EMBEDDING_MODEL: {self.embedding_model}")

        # Initialize components
        self.graphql_client = GraphQLClient(self.graphql_endpoint, self.graphql_headers)
        self.chromadb_manager = ChromaDBManager(
            host=self.chromadb_host,
            port=self.chromadb_port,
            collection_name=self.collection_name,
            embedding_model=self.embedding_model
        )
        
        logger.info(f"Initialized pipeline with GraphQL endpoint: {self.graphql_endpoint}")
        logger.info(f"ChromaDB: {self.chromadb_host}:{self.chromadb_port}/{self.collection_name}")
        
        # Test GraphQL endpoint connectivity
        self._test_graphql_connectivity()
    
    def _test_graphql_connectivity(self):
        """Test if the GraphQL endpoint is accessible."""
        try:
            logger.info(f"Testing GraphQL endpoint connectivity: {self.graphql_endpoint}")
            
            # Simple introspection query to test connectivity
            test_query = """
            query IntrospectionQuery {
                __schema {
                    queryType {
                        name
                    }
                }
            }
            """
            
            response = self.graphql_client.execute_query(test_query)
            logger.info("GraphQL endpoint is accessible and responding")
            
        except Exception as e:
            logger.error(f"GraphQL endpoint is not accessible: {e}")
            logger.error("Please check:")
            logger.error("1. The GraphQL endpoint URL is correct")
            logger.error("2. The GraphQL service is running")
            logger.error("3. Network connectivity to the endpoint")
            logger.error("4. Authentication headers if required")
            raise Exception(f"GraphQL endpoint connectivity test failed: {e}")
    
    @staticmethod
    def _parse_headers(headers_str: str) -> Dict[str, str]:
        """Parse headers string into dictionary."""
        if not headers_str:
            return {}
        
        headers = {}
        for header in headers_str.split(','):
            if ':' in header:
                key, value = header.split(':', 1)
                headers[key.strip()] = value.strip()
        return headers
    
    def fetch_all_data(self) -> List[Dict[str, Any]]:
        """Fetch all data from GraphQL backend using the specific query from graphql_dump.py."""
        try:
            logger.info("Fetching proposals and comments from GraphQL...")
            
            # Use the fetch_proposals_and_comments function from graphql_dump.py
            proposals = self._fetch_proposals_and_comments()
            
            logger.info(f"Successfully fetched {len(proposals)} proposals with comments")
            return proposals
            
        except Exception as e:
            logger.error(f"Failed to fetch data: {e}")
            raise
    
    def _fetch_proposals_and_comments(self) -> List[Dict[str, Any]]:
        """Fetch proposals and comments using the specific GraphQL query."""
        graphql_query = """
        query Comments {
            comments {
                nodes {
                    ancestry
                    body
                    cachedVotesDown
                    cachedVotesTotal
                    cachedVotesUp
                    cached_votes_down
                    cached_votes_total
                    cached_votes_up
                    commentableId
                    commentableType
                    commentable_id
                    commentable_type
                    confidenceScore
                    confidence_score
                    id
                    publicCreatedAt
                    public_created_at
                }
                pageInfo {
                    endCursor
                    hasNextPage
                    hasPreviousPage
                    startCursor
                }
            }
            proposals {
                nodes {
                    cachedVotesUp
                    cached_votes_up
                    commentsCount
                    comments_count
                    confidenceScore
                    confidence_score
                    description
                    geozoneId
                    geozone_id
                    hotScore
                    hot_score
                    id
                    publicCreatedAt
                    public_created_at
                    retiredAt
                    retiredExplanation
                    retiredReason
                    retired_at
                    retired_explanation
                    retired_reason
                    summary
                    title
                    videoUrl
                    video_url
                }
            }
        }
        """
        
        response = self.graphql_client.execute_query(graphql_query)
        data = response
        
        proposals = data['proposals']['nodes']
        comments = data['comments']['nodes']
        
        # Convert proposal id to int
        for p in proposals:
            try:
                p['id'] = int(p['id'])
            except Exception:
                pass
        
        # Attach comments to proposals
        proposal_map = {p['id']: p for p in proposals}
        for p in proposals:
            p['comments'] = []
        
        for c in comments:
            pid = c.get('commentableId') or c.get('commentable_id')
            try:
                pid = int(pid)
            except Exception:
                continue
            if pid and pid in proposal_map:
                proposal_map[pid]['comments'].append(c)
        
        return proposals
    
    def transform_data_for_chromadb(self, data: List[Dict[str, Any]]) -> tuple:
        """Transform GraphQL data for ChromaDB storage."""
        documents = []
        metadatas = []
        ids = []
        
        for item in data:
            # Create document text with improved field detection
            doc_parts = self._extract_text_content(item)
            document = "\n".join(doc_parts) if doc_parts else str(item)
            
            # Create metadata (exclude large text fields and internal fields)
            metadata = self._create_metadata(item)
            
            # Generate ID with type information
            item_id = self._generate_item_id(item, len(ids))
            
            documents.append(document)
            metadatas.append(metadata)
            ids.append(item_id)
        
        return documents, metadatas, ids
    
    def _extract_text_content(self, item: Dict[str, Any]) -> List[str]:
        """Extract text content from an item for document creation."""
        doc_parts = []
        
        # Common text fields to look for
        text_fields = [
            'title', 'name', 'label', 'heading', 'subject',
            'content', 'body', 'description', 'summary', 'text', 'message',
            'excerpt', 'abstract', 'bio', 'about', 'details'
        ]
        
        # First, try specific text fields
        for field in text_fields:
            if field in item and item[field]:
                value = str(item[field]).strip()
                if value:
                    doc_parts.append(f"{field.title()}: {value}")
        
        # If no specific text fields found, include all string fields
        if not doc_parts:
            for key, value in item.items():
                if (isinstance(value, str) and 
                    len(value) > 10 and  # Only include substantial text
                    not key.startswith('_') and  # Skip internal fields
                    key.lower() not in ['id', 'slug', 'url', 'uri', 'link']):  # Skip ID-like fields
                    doc_parts.append(f"{key.title()}: {value}")
        
        return doc_parts
    
    def _create_metadata(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata for an item."""
        metadata = {}
        
        # Include all fields except large text content
        large_text_fields = {'content', 'body', 'description', 'text', 'message', 'bio', 'about', 'details'}
        
        for key, value in item.items():
            # Skip large text fields from metadata
            if key.lower() in large_text_fields:
                continue
            
            # Only include simple types in metadata
            if isinstance(value, (str, int, float, bool, type(None))):
                # Truncate long strings in metadata
                if isinstance(value, str) and len(value) > 200:
                    metadata[key] = value[:200] + "..."
                else:
                    metadata[key] = value
            elif isinstance(value, (list, dict)):
                # Convert complex types to string representation
                metadata[f"{key}_summary"] = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
        
        # Add sync timestamp
        metadata['synced_at'] = datetime.now().isoformat()
        
        return metadata
    
    def _generate_item_id(self, item: Dict[str, Any], fallback_index: int) -> str:
        """Generate a unique ID for an item."""
        # Try common ID fields
        id_fields = ['id', 'uid', 'uuid', '_id', 'objectId', 'pk']
        
        for field in id_fields:
            if field in item and item[field]:
                # Include source type in ID if available
                source_type = item.get('_source_type', 'proposal')
                if source_type:
                    return f"{source_type}_{item[field]}"
                return str(item[field])
        
        # Generate ID based on source type and index
        source_type = item.get('_source_type', 'proposal')
        source_field = item.get('_source_field', '')
        
        if source_field:
            return f"{source_type}_{source_field}_{fallback_index}"
        else:
            return f"{source_type}_{fallback_index}"
    
    def sync_data(self):
        """Sync data from GraphQL to ChromaDB."""
        logger.info("Starting data sync...")
        
        try:
            # Fetch data from GraphQL
            data = self.fetch_all_data()
            
            if not data:
                logger.warning("No data fetched from GraphQL")
                return
            
            logger.info("Data successfully fetched from GraphQL")

            # Transform data for ChromaDB
            documents, metadatas, ids = self.transform_data_for_chromadb(data)
            
            # Store in ChromaDB (using upsert to handle updates)
            self.chromadb_manager.upsert_documents(documents, metadatas, ids)
            
            logger.info(f"Successfully synced {len(data)} items to ChromaDB")

            # Read string from file
            grounding_file = os.getenv('GROUNDING_FILE', 'grounding.txt')
            if os.path.exists(grounding_file):
                with open(grounding_file, 'r') as f:
                    grounding = [f.read()]
                    grounding_ids = ["grounding_1"]

                    self.chromadb_manager.upsert_documents(grounding, [{'source': 'grounding'}], grounding_ids)
                    logger.info(f"Loaded grounding from {grounding_file}")
            
        except Exception as e:
            logger.error(f"Data sync failed: {e}")
            raise
    
    def run_continuous_sync(self):
        """Run continuous data synchronization."""
        logger.info(f"Starting continuous sync with interval: {self.sync_interval} seconds")
        
        while True:
            try:
                self.sync_data()
                logger.info(f"Waiting {self.sync_interval} seconds before next sync...")
                time.sleep(self.sync_interval)
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, shutting down...")
                break
            except Exception as e:
                logger.error(f"Sync iteration failed: {e}")
                logger.info(f"Waiting {self.sync_interval} seconds before retry...")
                time.sleep(self.sync_interval)
    
    def run_once(self):
        """Run data sync once and exit."""
        logger.info("Running one-time data sync...")
        self.sync_data()
        logger.info("One-time sync completed")


def main():
    """Main application entry point."""
    logger.info("Starting GraphQL to ChromaDB pipeline...")
    
    # Start health check server if enabled
    health_server = start_health_check_if_enabled()
    
    try:
        pipeline = DataPipeline()
        
        # Check if we should run once or continuously
        run_mode = os.getenv('RUN_MODE', 'continuous').lower()
        
        # Add test mode for debugging
        if run_mode == 'test':
            logger.info("Running in test mode - checking connectivity only")
            # The connectivity test is already done in __init__
            logger.info("All connectivity tests passed")
            return
        
        if run_mode == 'once':
            pipeline.run_once()
        else:
            pipeline.run_continuous_sync()
            
    except Exception as e:
        logger.error(f"Application failed: {e}")
        
        # If it's a GraphQL connectivity issue, provide helpful guidance
        if "GraphQL endpoint connectivity test failed" in str(e):
            logger.error("")
            logger.error("Troubleshooting Steps:")
            logger.error("1. Create a .env file from env.example:")
            logger.error("   cp env.example .env")
            logger.error("2. Edit .env file and set GRAPHQL_ENDPOINT to your actual GraphQL API URL")
            logger.error("3. If using Docker, make sure the GraphQL service is accessible from the container")
            logger.error("4. Check if authentication headers are required and set GRAPHQL_HEADERS")
            logger.error("")
            logger.error("Example .env configuration:")
            logger.error("GRAPHQL_ENDPOINT=https://your-api.com/graphql")
            logger.error("GRAPHQL_HEADERS={\"Authorization\": \"Bearer your-token\"}")
            logger.error("CHROMADB_HOST=chroma")
            logger.error("CHROMADB_PORT=8000")
        
        sys.exit(1)
    finally:
        # Stop health check server if it was started
        if health_server:
            health_server.stop()


if __name__ == "__main__":
    main() 