#!/bin/bash

# PostgreSQL + pgAdmin Docker Setup Script
# Usage: ./setup.sh [command]

set -e

COMPOSE_FILE="docker-compose.yml"

if [ -f "docker-compose.override.yml" ]; then
    COMPOSE_FILE="$COMPOSE_FILE:docker-compose.override.yml"
fi

case "${1:-help}" in
    "start")
        echo "Starting PostgreSQL and pgAdmin..."
        docker-compose -f $COMPOSE_FILE up -d
        echo "Services started!"
        echo "PostgreSQL: localhost:${POSTGRES_PORT:-5432}"
        echo "pgAdmin: http://localhost:${PGADMIN_PORT:-5050}"
        ;;
    "stop")
        echo "Stopping services..."
        docker-compose -f $COMPOSE_FILE down
        ;;
    "restart")
        echo "Restarting services..."
        docker-compose -f $COMPOSE_FILE restart
        ;;
    "logs")
        docker-compose -f $COMPOSE_FILE logs -f
        ;;
    "build")
        echo "Building custom PostgreSQL image..."
        docker-compose -f $COMPOSE_FILE build --no-cache
        ;;
    "reset")
        echo "Resetting database (this will delete all data)..."
        docker-compose -f $COMPOSE_FILE down -v
        docker-compose -f $COMPOSE_FILE up -d
        ;;
    "status")
        docker-compose -f $COMPOSE_FILE ps
        ;;
    "shell")
        echo "Connecting to PostgreSQL shell..."
        docker-compose -f $COMPOSE_FILE exec postgres psql -U ${POSTGRES_USER:-rag_user} -d ${POSTGRES_DB:-rag_db}
        ;;
    "help"|*)
        echo "PostgreSQL + pgAdmin Docker Setup"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  start   - Start all services"
        echo "  stop    - Stop all services"
        echo "  restart - Restart all services"
        echo "  logs    - Show logs from all services"
        echo "  build   - Build custom PostgreSQL image"
        echo "  reset   - Reset database (deletes all data)"
        echo "  status  - Show service status"
        echo "  shell   - Connect to PostgreSQL shell"
        echo "  help    - Show this help message"
        echo ""
        echo "Environment variables can be set in .env file"
        ;;
esac
