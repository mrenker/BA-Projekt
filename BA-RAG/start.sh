#!/bin/bash

# GraphQL to ChromaDB Pipeline Startup Script
# This script helps you get started quickly with the application

set -e

echo "🚀 GraphQL to ChromaDB Pipeline Setup"
echo "======================================"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found. Creating from template..."
    if [ -f "env.example" ]; then
        cp env.example .env
        echo "✅ Created .env file from env.example"
        echo ""
        echo "📝 Please edit .env file with your configuration:"
        echo "   - GRAPHQL_ENDPOINT: Your GraphQL API endpoint"
        echo "   - CHROMADB_HOST: Your ChromaDB server host"
        echo "   - CHROMADB_PORT: Your ChromaDB server port"
        echo ""
        echo "Run this script again after configuring .env"
        exit 1
    else
        echo "❌ env.example file not found!"
        exit 1
    fi
fi

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose >/dev/null 2>&1; then
    echo "❌ docker-compose is not installed. Please install docker-compose and try again."
    exit 1
fi

# Load environment variables to check configuration
export $(cat .env | grep -v ^# | xargs)

# Validate required environment variables
if [ -z "$GRAPHQL_ENDPOINT" ]; then
    echo "❌ GRAPHQL_ENDPOINT is not set in .env file"
    exit 1
fi

if [ -z "$CHROMADB_HOST" ]; then
    echo "❌ CHROMADB_HOST is not set in .env file"
    exit 1
fi

echo "✅ Configuration validated"
echo ""
echo "📊 Current Configuration:"
echo "   GraphQL Endpoint: $GRAPHQL_ENDPOINT"
echo "   ChromaDB Host: $CHROMADB_HOST"
echo "   ChromaDB Port: ${CHROMADB_PORT:-8000}"
echo "   Collection: ${CHROMADB_COLLECTION:-graphql_data}"
echo "   Run Mode: ${RUN_MODE:-continuous}"
echo "   Sync Interval: ${SYNC_INTERVAL_SECONDS:-300} seconds"
echo ""

# Create logs directory
mkdir -p logs

# Create cache directory for model persistence
mkdir -p cache

# Parse command line arguments
BACKGROUND=false
BUILD=false
LOGS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--daemon)
            BACKGROUND=true
            shift
            ;;
        -b|--build)
            BUILD=true
            shift
            ;;
        -l|--logs)
            LOGS=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -d, --daemon    Run in background (daemon mode)"
            echo "  -b, --build     Force rebuild of Docker image"
            echo "  -l, --logs      Show logs after starting"
            echo "  -h, --help      Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                # Start in foreground"
            echo "  $0 -d             # Start in background"
            echo "  $0 -b -d          # Rebuild and start in background"
            echo "  $0 -d -l          # Start in background and show logs"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Build command
BUILD_CMD="docker-compose up"
if [ "$BUILD" = true ]; then
    BUILD_CMD="$BUILD_CMD --build"
fi

if [ "$BACKGROUND" = true ]; then
    BUILD_CMD="$BUILD_CMD -d"
fi

echo "🔧 Starting pipeline..."
echo "Command: $BUILD_CMD"
echo ""

# Run the application
eval $BUILD_CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Pipeline started successfully!"
    
    if [ "$BACKGROUND" = true ]; then
        echo ""
        echo "📊 Service is running in the background"
        echo "Use the following commands to manage the service:"
        echo "   docker-compose ps                    # Check status"
        echo "   docker-compose logs -f               # View logs"
        echo "   docker-compose stop                  # Stop service"
        echo "   docker-compose down                  # Stop and remove"
        echo ""
        
        # Show logs if requested
        if [ "$LOGS" = true ]; then
            echo "📋 Showing recent logs (Ctrl+C to exit):"
            echo ""
            docker-compose logs -f
        fi
    else
        echo ""
        echo "Press Ctrl+C to stop the pipeline"
    fi
else
    echo ""
    echo "❌ Failed to start pipeline"
    exit 1
fi 