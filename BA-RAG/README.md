# GraphQL to ChromaDB Pipeline

A dockerized Python application that pulls data from a GraphQL backend and stores it in ChromaDB using vector embeddings for semantic search.

## Features

- 🚀 **GraphQL Integration**: Fetches data from any GraphQL API with authentication support
- 🗃️ **ChromaDB Storage**: Stores data in ChromaDB with automatic embeddings
- 🔄 **Continuous Sync**: Supports both one-time and continuous data synchronization
- 📦 **Dockerized**: Fully containerized for easy deployment
- ⚙️ **Configurable**: Extensive configuration through environment variables
- 🔍 **Pagination**: Handles large datasets with automatic pagination
- 📊 **Logging**: Comprehensive logging for monitoring and debugging
- 🔒 **Security**: Runs as non-root user in container
- 🔎 **Auto-Discovery**: Automatically discovers and queries all available GraphQL types
- 🎯 **Smart Data Extraction**: Intelligently extracts text content for optimal embeddings

## Quick Start

### 1. Clone and Configure

```bash
# Copy the environment configuration
cp env.example .env

# Edit .env with your configuration
nano .env
```

### 2. Required Environment Variables

```bash
# GraphQL Configuration
GRAPHQL_ENDPOINT=https://your-graphql-api.com/graphql

# ChromaDB Configuration
CHROMADB_HOST=your-chromadb-host.com
CHROMADB_PORT=8000
```

### 3. Run with Docker Compose

```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up -d --build
```

### 4. Run with Docker

```bash
# Build the image
docker build -t graphql-chromadb-pipeline .

# Run the container
docker run --env-file .env graphql-chromadb-pipeline
```

## Configuration

### Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `GRAPHQL_ENDPOINT` | GraphQL API endpoint URL | `https://api.example.com/graphql` |
| `CHROMADB_HOST` | ChromaDB server hostname | `your-chromadb-host.com` |

### Optional Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GRAPHQL_HEADERS` | `{}` | HTTP headers as JSON string |
| `GRAPHQL_QUERY` | Generic query | Custom GraphQL query |
| `CHROMADB_PORT` | `8000` | ChromaDB server port |
| `CHROMADB_COLLECTION` | `graphql_data` | Collection name in ChromaDB |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `SYNC_INTERVAL_SECONDS` | `300` | Sync interval for continuous mode |
| `BATCH_SIZE` | `100` | Pagination batch size |
| `RUN_MODE` | `continuous` | Run mode: `continuous` or `once` |

### Schema Auto-Discovery Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTO_DISCOVER_SCHEMA` | `true` | Enable automatic GraphQL schema discovery |
| `MAX_QUERY_DEPTH` | `3` | Maximum depth for auto-generated queries |
| `EXCLUDED_TYPES` | `` | Comma-separated list of types/fields to exclude |
| `INCLUDED_TYPES` | `` | Comma-separated list of types/fields to include (if set, only these will be used) |

## Schema Auto-Discovery

The application can automatically discover your GraphQL schema and ingest data from all available types without manual configuration.

### How It Works

1. **Schema Introspection**: Automatically discovers all available GraphQL types and fields
2. **Query Generation**: Generates optimized queries for each discoverable type
3. **Smart Pagination**: Detects and handles different pagination patterns (Relay, offset-based, etc.)
4. **Data Extraction**: Intelligently extracts text content for optimal ChromaDB storage

### Exploring Your Schema

Use the included discovery utility to explore your GraphQL schema:

```bash
# Using environment variables
python discover_schema.py

# Or set variables inline
GRAPHQL_ENDPOINT=https://your-api.com/graphql python discover_schema.py

# In Docker
docker run --env-file .env your-image python discover_schema.py
```

The discovery utility will show you:
- All available queryable types
- Generated queries for each type
- Sample data structures
- Interactive testing capabilities

### Configuration Examples

#### Include Only Specific Types
```bash
AUTO_DISCOVER_SCHEMA=true
INCLUDED_TYPES=User,Post,Comment
```

#### Exclude Sensitive Types
```bash
AUTO_DISCOVER_SCHEMA=true
EXCLUDED_TYPES=PrivateData,InternalConfig,AdminSettings
```

#### Disable Auto-Discovery (Use Manual Query)
```bash
AUTO_DISCOVER_SCHEMA=false
GRAPHQL_QUERY=query { posts { id title content author { name } } }
```

## Usage Examples

### Example 1: Public API (Countries GraphQL)

```bash
# .env configuration
GRAPHQL_ENDPOINT=https://countries.trevorblades.com/
GRAPHQL_HEADERS={}
GRAPHQL_QUERY=query { countries { code name emoji capital } }
CHROMADB_HOST=localhost
CHROMADB_PORT=8000
```

### Example 2: GitHub API

```bash
# .env configuration
GRAPHQL_ENDPOINT=https://api.github.com/graphql
GRAPHQL_HEADERS={"Authorization": "Bearer ghp_your_github_token"}
GRAPHQL_QUERY=query($limit: Int, $offset: Int) { viewer { repositories(first: $limit) { nodes { name description createdAt url } } } }
```

### Example 3: Custom API with Authentication

```bash
# .env configuration
GRAPHQL_ENDPOINT=https://your-api.com/graphql
GRAPHQL_HEADERS={"Authorization": "Bearer your-token", "Content-Type": "application/json"}
CHROMADB_HOST=your-chromadb-instance.com
CHROMADB_PORT=443
CHROMADB_COLLECTION=my_custom_collection
```

## GraphQL Query Customization

The application expects a GraphQL query that supports pagination with `$limit` and `$offset` variables. If you don't provide a custom query, it uses this default:

```graphql
query GetAllData($limit: Int, $offset: Int) {
  items(limit: $limit, offset: $offset) {
    id
    title
    content
    createdAt
    updatedAt
    tags
  }
}
```

### Custom Query Examples

#### Simple Query (No Pagination)
```graphql
query {
  users {
    id
    name
    email
  }
}
```

#### Query with Pagination
```graphql
query GetPosts($limit: Int, $offset: Int) {
  posts(limit: $limit, offset: $offset) {
    id
    title
    content
    author {
      name
    }
    publishedAt
  }
}
```

## Data Transformation

The application automatically transforms GraphQL data for ChromaDB storage:

- **Documents**: Combines `title` and `content` fields (customizable)
- **Metadata**: Stores all other fields except large text content
- **IDs**: Uses GraphQL `id` field or generates sequential IDs
- **Embeddings**: Automatically generated using sentence transformers

### Customizing Data Transformation

Edit the `transform_data_for_chromadb` method in `main.py`:

```python
def transform_data_for_chromadb(self, data: List[Dict[str, Any]]) -> tuple:
    documents = []
    metadatas = []
    ids = []
    
    for item in data:
        # Customize document creation
        document = f"{item.get('title', '')} {item.get('description', '')}"
        
        # Customize metadata
        metadata = {
            'source': 'graphql',
            'type': item.get('type'),
            'created_at': item.get('createdAt'),
        }
        
        documents.append(document)
        metadatas.append(metadata)
        ids.append(str(item['id']))
    
    return documents, metadatas, ids
```

## Deployment

### Docker Compose (Recommended)

```yaml
version: '3.8'
services:
  pipeline:
    build: .
    environment:
      - GRAPHQL_ENDPOINT=https://your-api.com/graphql
      - CHROMADB_HOST=your-chromadb-host.com
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: graphql-chromadb-pipeline
spec:
  replicas: 1
  selector:
    matchLabels:
      app: graphql-chromadb-pipeline
  template:
    metadata:
      labels:
        app: graphql-chromadb-pipeline
    spec:
      containers:
      - name: pipeline
        image: graphql-chromadb-pipeline:latest
        env:
        - name: GRAPHQL_ENDPOINT
          value: "https://your-api.com/graphql"
        - name: CHROMADB_HOST
          value: "chromadb-service"
```

### Environment-specific Configurations

#### Production
```bash
RUN_MODE=continuous
SYNC_INTERVAL_SECONDS=1800  # 30 minutes
BATCH_SIZE=1000
```

#### Development
```bash
RUN_MODE=once
BATCH_SIZE=10
```

## Monitoring and Logging

### Log Files
- Application logs: `./logs/app.log`
- Container logs: `docker-compose logs -f`

### Log Levels
The application logs at INFO level by default. Key log messages include:
- Connection status to GraphQL and ChromaDB
- Data fetch progress and batch information
- Sync completion status
- Error messages and troubleshooting info

### Health Monitoring
```bash
# Check container status
docker-compose ps

# View recent logs
docker-compose logs --tail=50 graphql-chromadb-pipeline

# Follow logs in real-time
docker-compose logs -f graphql-chromadb-pipeline
```

## Troubleshooting

### Common Issues

#### 1. ChromaDB Connection Failed
```
Error: Failed to connect to ChromaDB
```
**Solution**: Verify `CHROMADB_HOST` and `CHROMADB_PORT` are correct.

#### 2. GraphQL Authentication Error
```
Error: GraphQL query failed: [{'message': 'Unauthorized'}]
```
**Solution**: Check your `GRAPHQL_HEADERS` authentication token.

#### 3. Out of Memory
```
Error: Container killed (OOMKilled)
```
**Solution**: Reduce `BATCH_SIZE` or increase container memory limits.

#### 4. Permission Denied (Model Cache)
```
Error: [Errno 13] Permission denied: '/home/appuser'
```
**Solution**: 
- Rebuild the Docker image: `docker-compose build --no-cache`
- Or manually create cache directory: `mkdir -p cache && chmod 755 cache`
- The application will fallback to default embeddings if model download fails

#### 5. ChromaDB Tenant Issues
```
Error: Could not connect to tenant default_tenant
```
**Solution**: This is now automatically handled, but you can also set:
```bash
CHROMADB_TENANT=your_tenant_name
CHROMADB_DATABASE=your_database_name
```

### Debug Mode
```bash
# Run with debug logging
docker run --env-file .env -e PYTHONPATH=/app your-image python -c "
import logging
logging.getLogger().setLevel(logging.DEBUG)
exec(open('main.py').read())
"
```

## Development

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python main.py
```

### Testing
```bash
# Test GraphQL connection
python -c "
from main import GraphQLClient
client = GraphQLClient('https://countries.trevorblades.com/')
result = client.execute_query('query { countries(filter: {code: {eq: \"US\"}}) { name } }')
print(result)
"

# Test ChromaDB connection
python -c "
import chromadb
client = chromadb.HttpClient(host='localhost', port=8000)
print(client.heartbeat())
"
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs for error details
3. Open an issue with relevant logs and configuration 