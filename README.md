

# Postgres RAG Web App

This application lets you upload, embed, and search text documents using PostgreSQL with pgvector. It provides a modern web UI for chat-based retrieval, document upload, and vector search.

## Capabilities

- Upload and manage text documents via web interface
- Ingest and embed documents into PostgreSQL with pgvector
- Chat with your documents using semantic search (RAG)
- View and search document chunks and embeddings
- Monitor ingestion jobs and view job history
- Configure chunking, embedding models, and database settings
- REST API for integration with other tools

## Quick Start

1. **Clone this repository**
2. **Install Docker and Docker Compose**
3. **Start the database and pgAdmin:**
	```bash
	docker-compose up -d
	```
4. **Install Python dependencies:**
	```bash
	pip install -r requirements.txt
	pip install -r web_requirements.txt
	```
5. **Start the web interface:**
	```bash
	python web_interface.py
	```
6. **Visit the app:**
	- Open your browser to http://localhost:5000

## Requirements

- Python 3.8+
- Docker (for PostgreSQL + pgvector)
- See `requirements.txt` and `web_requirements.txt` for Python packages

## Usage

1. **Upload**: Go to the Upload page and add your text files.
2. **Configure**: Set chunk size, embedding model, and database options.
3. **Ingest**: Start the job and monitor progress.
4. **Chat**: Use the Chat page to ask questions and see relevant document results with scores and metadata.
5. **Explore**: Browse embeddings, job history, and manage your data.

## Database

The app uses PostgreSQL with the pgvector extension for vector search. All configuration is handled via Docker Compose. Default credentials are set in the compose file and can be changed as needed.

## API

REST endpoints are available for job status, cancellation, and chunk search. See the code or web interface for details.

---
For more details, see `WEB_INTERFACE_README.md` or the in-app help.
