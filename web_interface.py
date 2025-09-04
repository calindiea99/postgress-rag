#!/usr/bin/env python3
"""
Flask Web Interface for Text Ingestion

A web-based interface for the text ingestion pipeline that allows users to:
- Upload text files
- Configure ingestion parameters
- Monitor ingestion progress
- View results and statistics
- Query ingested documents

Usage:
    python web_interface.py
    # Then visit http://localhost:5000
"""

import os
import sys
import json
import uuid
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import logging

# Flask imports
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

# Import our ingestion pipeline
from ingestion import TextIngestionPipeline

# Database imports
import psycopg2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IngestionManager:
    """Manages ingestion jobs and their status."""

    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.jobs = {}  # job_id -> job_info (cache)
        self.active_job = None
        self._setup_database()
        self._load_jobs_from_database()

    def _setup_database(self):
        """Setup database connection and ensure jobs table exists."""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            # Create jobs table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_jobs (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    job_id VARCHAR(36) UNIQUE NOT NULL,
                    status VARCHAR(20) NOT NULL DEFAULT 'pending',
                    config JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    progress INTEGER DEFAULT 0,
                    message TEXT,
                    results JSONB,
                    error TEXT
                )
            """)

            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS ingestion_jobs_job_id_idx ON ingestion_jobs(job_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS ingestion_jobs_status_idx ON ingestion_jobs(status)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS ingestion_jobs_created_at_idx ON ingestion_jobs(created_at DESC)
            """)

            conn.commit()
            cursor.close()
            conn.close()
            logger.info("Jobs database table setup completed")

        except Exception as e:
            logger.error(f"Failed to setup jobs database: {e}")
            raise

    def _load_jobs_from_database(self):
        """Load all jobs from database into memory cache."""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT job_id, status, config, created_at, updated_at, progress, message, results, error
                FROM ingestion_jobs
                ORDER BY created_at DESC
            """)

            rows = cursor.fetchall()
            for row in rows:
                job_id, status, config, created_at, updated_at, progress, message, results, error = row

                self.jobs[job_id] = {
                    'id': job_id,
                    'status': status,
                    'config': config,
                    'created_at': created_at.isoformat() if created_at else datetime.now().isoformat(),
                    'updated_at': updated_at.isoformat() if updated_at else datetime.now().isoformat(),
                    'progress': progress or 0,
                    'message': message or '',
                    'results': results,
                    'error': error
                }

                # Set active job if there's a running job
                if status == 'running':
                    self.active_job = job_id

            cursor.close()
            conn.close()
            logger.info(f"Loaded {len(self.jobs)} jobs from database")

        except Exception as e:
            logger.error(f"Failed to load jobs from database: {e}")
            # Continue with empty jobs dict if database load fails

    def _save_job_to_database(self, job_id: str):
        """Save a job to the database."""
        if job_id not in self.jobs:
            return

        job = self.jobs[job_id]
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            # Upsert job data
            cursor.execute("""
                INSERT INTO ingestion_jobs (job_id, status, config, created_at, updated_at, progress, message, results, error)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (job_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    config = EXCLUDED.config,
                    updated_at = EXCLUDED.updated_at,
                    progress = EXCLUDED.progress,
                    message = EXCLUDED.message,
                    results = EXCLUDED.results,
                    error = EXCLUDED.error
            """, (
                job_id,
                job['status'],
                json.dumps(job['config']) if job['config'] else None,
                job['created_at'],
                datetime.now().isoformat(),
                job['progress'],
                job['message'],
                json.dumps(job['results']) if job['results'] else None,
                job['error']
            ))

            conn.commit()
            cursor.close()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to save job {job_id} to database: {e}")

    def create_job(self, config: Dict[str, Any]) -> str:
        """Create a new ingestion job."""
        job_id = str(uuid.uuid4())
        job = {
            'id': job_id,
            'status': 'pending',
            'config': config,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'progress': 0,
            'message': 'Job created',
            'results': None,
            'error': None
        }

        self.jobs[job_id] = job
        self._save_job_to_database(job_id)
        return job_id

    def start_job(self, job_id: str):
        """Start an ingestion job in a background thread."""
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")

        job = self.jobs[job_id]
        if job['status'] != 'pending':
            raise ValueError(f"Job {job_id} is not in pending status")

        # Mark as running
        job['status'] = 'running'
        job['message'] = 'Starting ingestion...'
        self.active_job = job_id
        self._save_job_to_database(job_id)

        # Start in background thread
        thread = threading.Thread(target=self._run_job, args=(job_id,))
        thread.daemon = True
        thread.start()

    def _run_job(self, job_id: str):
        """Run the ingestion job."""
        job = self.jobs[job_id]

        try:
            # Create pipeline
            pipeline = TextIngestionPipeline(job['config'])

            # Override logging to capture progress
            class ProgressHandler(logging.Handler):
                def emit(self, record):
                    job['message'] = record.getMessage()
                    if 'Progress:' in record.getMessage():
                        # Extract progress from message like "Progress: 50/100 documents ingested"
                        try:
                            parts = record.getMessage().split()
                            current = int(parts[1].split('/')[0])
                            total = int(parts[1].split('/')[1])
                            job['progress'] = int((current / total) * 100)
                            self._save_job_to_database(job_id)
                        except:
                            pass

            progress_handler = ProgressHandler()
            pipeline.logger.addHandler(progress_handler)

            # Run pipeline
            pipeline.run()

            # Success
            job['status'] = 'completed'
            job['progress'] = 100
            job['message'] = 'Ingestion completed successfully!'
            job['results'] = pipeline.get_statistics()
            self._save_job_to_database(job_id)

        except Exception as e:
            # Error
            job['status'] = 'failed'
            job['error'] = str(e)
            job['message'] = f'Ingestion failed: {str(e)}'
            self._save_job_to_database(job_id)
            logger.error(f"Job {job_id} failed: {e}")

        finally:
            if self.active_job == job_id:
                self.active_job = None

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a job."""
        return self.jobs.get(job_id, {'error': 'Job not found'})

    def get_all_jobs(self) -> List[Dict[str, Any]]:
        """Get all jobs."""
        return list(self.jobs.values())

    def cancel_job(self, job_id: str):
        """Cancel a running job."""
        if job_id in self.jobs:
            job = self.jobs[job_id]
            if job['status'] == 'running':
                job['status'] = 'cancelled'
                job['message'] = 'Job cancelled by user'
                self._save_job_to_database(job_id)
                if self.active_job == job_id:
                    self.active_job = None

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = Path('./uploads')
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)

# Default configuration
DEFAULT_CONFIG = {
    'input_dir': Path('./uploads'),
    'collection_name': 'web_ingestion_docs',
    'file_pattern': '**/*.txt',
    'chunk_size': 1000,
    'chunk_overlap': 200,
    'batch_size': 50,
    'embedding_model': 'sentence-transformer',
    'model_name': 'all-MiniLM-L6-v2',
    'database': {
        'host': 'localhost',
        'port': 5432,
        'database': 'rag_db',
        'user': 'rag_user',
        'password': 'rag_password'
    },
    'overwrite': False,
    'verbose': True,
    'log_file': False
}

# Initialize ingestion manager
ingestion_manager = IngestionManager(DEFAULT_CONFIG['database'])

@app.route('/')
def index():
    """Main dashboard page."""
    jobs = ingestion_manager.get_all_jobs()
    recent_jobs = sorted(jobs, key=lambda x: x['created_at'], reverse=True)[:10]

    return render_template('index.html',
                         jobs=recent_jobs,
                         active_job=ingestion_manager.active_job)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """File upload page."""
    if request.method == 'POST':
        # Get uploaded files
        uploaded_files = request.files.getlist('files')

        if not uploaded_files or all(f.filename == '' for f in uploaded_files):
            flash('No files selected', 'error')
            return redirect(request.url)

        # Save files
        saved_files = []
        for file in uploaded_files:
            if file.filename == '':
                continue

            filename = secure_filename(file.filename)
            filepath = app.config['UPLOAD_FOLDER'] / filename
            file.save(filepath)
            saved_files.append(filename)

        flash(f'Successfully uploaded {len(saved_files)} files', 'success')
        return redirect(url_for('configure', files=','.join(saved_files)))

    return render_template('upload.html')

@app.route('/configure', methods=['GET', 'POST'])
def configure():
    """Configuration page."""
    files = request.args.get('files', '').split(',') if request.args.get('files') else []

    if request.method == 'POST':
        # Build configuration from form
        config = DEFAULT_CONFIG.copy()

        # Update from form data
        config.update({
            'collection_name': request.form.get('collection_name', config['collection_name']),
            'chunk_size': int(request.form.get('chunk_size', config['chunk_size'])),
            'chunk_overlap': int(request.form.get('chunk_overlap', config['chunk_overlap'])),
            'batch_size': int(request.form.get('batch_size', config['batch_size'])),
            'embedding_model': request.form.get('embedding_model', config['embedding_model']),
            'model_name': request.form.get('model_name', config['model_name']),
            'overwrite': 'overwrite' in request.form,
            'verbose': 'verbose' in request.form
        })

        # Database settings
        config['database'].update({
            'host': request.form.get('db_host', config['database']['host']),
            'port': int(request.form.get('db_port', config['database']['port'])),
            'database': request.form.get('db_name', config['database']['database']),
            'user': request.form.get('db_user', config['database']['user']),
            'password': request.form.get('db_password', config['database']['password'])
        })

        # Create job
        job_id = ingestion_manager.create_job(config)

        # Start job
        try:
            ingestion_manager.start_job(job_id)
            flash('Ingestion job started successfully!', 'success')
            return redirect(url_for('job_status', job_id=job_id))
        except Exception as e:
            flash(f'Failed to start job: {str(e)}', 'error')
            return redirect(url_for('configure'))

    return render_template('configure.html',
                         config=DEFAULT_CONFIG,
                         files=files)

@app.route('/job/<job_id>')
def job_status(job_id):
    """Job status page."""
    job = ingestion_manager.get_job_status(job_id)
    
    # Check if job was not found
    if job.get('error') == 'Job not found':
        flash(f"Job not found: {job_id}", 'error')
        return redirect(url_for('index'))
    
    return render_template('job_status.html', job=job)

@app.route('/api/job/<job_id>/status')
def api_job_status(job_id):
    """API endpoint for job status."""
    job = ingestion_manager.get_job_status(job_id)
    
    # Create a JSON-serializable copy of the job
    serializable_job = {}
    for key, value in job.items():
        if isinstance(value, Path):
            serializable_job[key] = str(value)
        elif isinstance(value, dict):
            # Handle nested dictionaries (like config)
            serializable_job[key] = {}
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, Path):
                    serializable_job[key][sub_key] = str(sub_value)
                else:
                    serializable_job[key][sub_key] = sub_value
        else:
            serializable_job[key] = value
    
    return jsonify(serializable_job)

@app.route('/api/job/<job_id>/cancel', methods=['POST'])
def api_cancel_job(job_id):
    """API endpoint to cancel a job."""
    try:
        ingestion_manager.cancel_job(job_id)
        return jsonify({'success': True, 'message': 'Job cancelled'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/jobs')
def jobs():
    """Jobs history page."""
    jobs = ingestion_manager.get_all_jobs()
    jobs_sorted = sorted(jobs, key=lambda x: x['created_at'], reverse=True)
    return render_template('jobs.html', jobs=jobs_sorted)

@app.route('/settings')
def settings():
    """Settings page."""
    return render_template('settings.html', config=DEFAULT_CONFIG)

@app.route('/chat')
def chat():
    """Chat interface page."""
    return render_template('chat.html')

@app.route('/embeddings')
def embeddings():
    """Embeddings/chunks visualization page."""
    return render_template('embeddings.html')

@app.route('/catalog')
def catalog():
    """Document catalog page."""
    return render_template('catalog.html')

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """API endpoint for chat queries."""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'success': False, 'error': 'Query is required'}), 400

        query = data['query'].strip()
        if not query:
            return jsonify({'success': False, 'error': 'Query cannot be empty'}), 400

        # Import here to avoid circular imports
        from ingestion import TextIngestionPipeline

        # Create a temporary pipeline to query the vector store
        # Use the same config as the web interface
        config = DEFAULT_CONFIG.copy()
        config['input_dir'] = Path('./uploads')  # Ensure it's a Path object

        # Initialize pipeline and vector store for querying
        pipeline = TextIngestionPipeline(config)
        
        # Initialize only what's needed for querying
        pipeline.validate_config()
        pipeline.setup_database_connection()
        pipeline.setup_embeddings()
        pipeline.create_vectorstore()


        # Query the vector store
        results = pipeline.vectorstore.similarity_search_with_score(query, k=5)

        # Format results with extra metadata
        response_data = {
            'query': query,
            'results': []
        }

        for doc, score in results:
            document_name = doc.metadata.get('source', 'Unknown') if doc.metadata else 'Unknown'
            chunk_id = str(doc.metadata.get('chunk_id', 'N/A')) if doc.metadata else 'N/A'
            embedding_dim = doc.metadata.get('embedding_dim', 384) if doc.metadata else 384
            content_preview = doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content
            response_data['results'].append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'score': float(score),
                'document_name': document_name,
                'chunk_id': chunk_id,
                'embedding_dim': embedding_dim,
                'content_preview': content_preview
            })

        return jsonify({'success': True, 'data': response_data})

    except Exception as e:
        logger.error(f"Chat query failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/embeddings')
def api_get_embeddings():
    """API endpoint to get embeddings/chunks data."""
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 50))
        search_query = request.args.get('search', '').strip()

        offset = (page - 1) * per_page

        from ingestion import TextIngestionPipeline

        # Create pipeline instance
        config = DEFAULT_CONFIG.copy()
        config['input_dir'] = Path('./uploads')
        pipeline = TextIngestionPipeline(config)
        pipeline.validate_config()
        pipeline.setup_database_connection()

        if search_query:
            # Use similarity search
            pipeline.setup_embeddings()
            pipeline.create_vectorstore()
            chunks = pipeline.search_chunks(search_query, limit=per_page)
            total_chunks = len(chunks)  # Approximate for search
        else:
            # Get all chunks
            chunks = pipeline.get_all_chunks(limit=per_page, offset=offset)
            total_chunks = pipeline.get_chunk_count()

        return jsonify({
            'success': True,
            'chunks': chunks,
            'total': total_chunks,
            'page': page,
            'per_page': per_page,
            'total_pages': (total_chunks + per_page - 1) // per_page
        })

    except Exception as e:
        logger.error(f"Failed to get embeddings: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/embeddings/chunk/<chunk_id>', methods=['GET'])
def api_get_chunk_details(chunk_id):
    """API endpoint to get details of a specific chunk."""
    try:
        from ingestion import TextIngestionPipeline

        # Create pipeline instance
        config = DEFAULT_CONFIG.copy()
        config['input_dir'] = Path('./uploads')
        pipeline = TextIngestionPipeline(config)
        pipeline.validate_config()
        pipeline.setup_database_connection()

        chunk = pipeline.get_single_chunk(chunk_id)

        if chunk:
            return jsonify({'success': True, 'chunk': chunk})
        else:
            return jsonify({'success': False, 'error': f'Chunk {chunk_id} not found'}), 404

    except Exception as e:
        logger.error(f"Failed to get chunk {chunk_id}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/embeddings/chunk/<chunk_id>', methods=['DELETE'])
def api_delete_embedding(chunk_id):
    """API endpoint to delete a specific chunk."""
    try:
        from ingestion import TextIngestionPipeline

        # Create pipeline instance
        config = DEFAULT_CONFIG.copy()
        config['input_dir'] = Path('./uploads')
        pipeline = TextIngestionPipeline(config)
        pipeline.validate_config()
        pipeline.setup_database_connection()

        success = pipeline.delete_chunk(chunk_id)

        if success:
            return jsonify({'success': True, 'message': f'Chunk {chunk_id} deleted successfully'})
        else:
            return jsonify({'success': False, 'error': f'Chunk {chunk_id} not found'}), 404

    except Exception as e:
        logger.error(f"Failed to delete chunk {chunk_id}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/embeddings/search')
def api_search_embeddings():
    """API endpoint to search embeddings/chunks."""
    try:
        query = request.args.get('q', '').strip()
        if not query:
            return jsonify({'chunks': []})

        from ingestion import TextIngestionPipeline

        config = DEFAULT_CONFIG.copy()
        config['input_dir'] = Path('./uploads')
        pipeline = TextIngestionPipeline(config)
        pipeline.validate_config()
        pipeline.setup_database_connection()
        pipeline.setup_embeddings();
        pipeline.create_vectorstore();

        chunks = pipeline.search_chunks(query, limit=50);

        return jsonify({
            chunks: chunks
        });

    except Exception as e:
        logger.error(f"Failed to search chunks: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/catalog/stats')
def api_catalog_stats():
    """API endpoint for catalog statistics."""
    try:
        from ingestion import TextIngestionPipeline

        config = DEFAULT_CONFIG.copy()
        config['input_dir'] = Path('./uploads')
        pipeline = TextIngestionPipeline(config)
        pipeline.validate_config()
        pipeline.setup_database_connection()

        stats = pipeline.get_catalog_stats()

        return jsonify({
            'success': True,
            'stats': {
                'total_documents': stats.get('total_documents', 0),
                'total_chunks': stats.get('total_chunks', 0),
                'total_characters': stats.get('total_characters', 0),
                'avg_chunk_size': stats.get('avg_chunk_size', 0),
                'last_updated': datetime.now().isoformat()
            }
        })

    except Exception as e:
        logger.error(f"Failed to get catalog stats: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
@app.route('/api/catalog/documents')
def api_catalog_documents():
    """API endpoint for catalog documents."""
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))

        offset = (page - 1) * per_page

        from ingestion import TextIngestionPipeline

        config = DEFAULT_CONFIG.copy()
        config['input_dir'] = Path('./uploads')
        pipeline = TextIngestionPipeline(config)
        pipeline.validate_config()
        pipeline.setup_database_connection()

        documents = pipeline.get_catalog_documents(limit=per_page, offset=offset)

        return jsonify({
            'documents': documents,
            'page': page,
            'per_page': per_page
        })

    except Exception as e:
        logger.error(f"Failed to get catalog documents: {e}")
@app.route('/api/catalog/document/<path:source>/chunks')
def api_document_chunks(source):
    """API endpoint for document chunks."""
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 50))

        offset = (page - 1) * per_page

        from ingestion import TextIngestionPipeline

        config = DEFAULT_CONFIG.copy()
        config['input_dir'] = Path('./uploads')
        pipeline = TextIngestionPipeline(config)
        pipeline.validate_config()
        pipeline.setup_database_connection()

        chunks = pipeline.get_document_chunks(source, limit=per_page, offset=offset)

        return jsonify({
            'success': True,
            'chunks': chunks,
            'source': source,
            'page': page,
            'per_page': per_page
        })

    except Exception as e:
        logger.error(f"Failed to get document chunks: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
        logger.error(f"Failed to get document chunks: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/catalog/document/<path:source>', methods=['DELETE'])
def api_delete_document(source):
    """API endpoint to delete an entire document."""
    try:
        from ingestion import TextIngestionPipeline

        config = DEFAULT_CONFIG.copy()
        config['input_dir'] = Path('./uploads')
        pipeline = TextIngestionPipeline(config)
        pipeline.validate_config()
        pipeline.setup_database_connection()

        success = pipeline.delete_document(source)

        if success:
            return jsonify({'success': True, 'message': f'Document {source} deleted successfully'})
        else:
            return jsonify({'success': False, 'error': f'Document {source} not found'}), 404

    except Exception as e:
        logger.error(f"Failed to delete document {source}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/catalog/chunk/<chunk_id>', methods=['DELETE'])
def api_delete_chunk(chunk_id):
    """API endpoint to delete a specific chunk."""
    try:
        from ingestion import TextIngestionPipeline

        config = DEFAULT_CONFIG.copy()
        config['input_dir'] = Path('./uploads')
        pipeline = TextIngestionPipeline(config)
        pipeline.validate_config()
        pipeline.setup_database_connection()

        success = pipeline.delete_chunk(chunk_id)

        if success:
            return jsonify({'success': True, 'message': f'Chunk {chunk_id} deleted successfully'})
        else:
            return jsonify({'success': False, 'error': f'Chunk {chunk_id} not found'}), 404

    except Exception as e:
        logger.error(f"Failed to delete chunk {chunk_id}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    """Handle file too large error."""
    flash('File too large. Maximum size is 100MB.', 'error')
    return redirect(url_for('upload'))

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files."""
    return send_from_directory('static', filename)

def create_templates():
    """Create template directories and files."""
    template_dir = Path('templates')
    static_dir = Path('static')
    css_dir = static_dir / 'css'
    js_dir = static_dir / 'js'

    # Create directories
    template_dir.mkdir(exist_ok=True)
    static_dir.mkdir(exist_ok=True)
    css_dir.mkdir(exist_ok=True)
    js_dir.mkdir(exist_ok=True)

    # Create base template
    base_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Text Ingestion Interface{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <link href="{{ url_for('static', filename='css/style.css') }}" rel="stylesheet">
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container">
            <a class="navbar-brand" href="{{ url_for('index') }}">
                <i class="fas fa-upload"></i> Text Ingestion
            </a>
            <div class="navbar-nav ms-auto">
                <a class="nav-link" href="{{ url_for('index') }}">
                    <i class="fas fa-home"></i> Dashboard
                </a>
                <a class="nav-link" href="{{ url_for('upload') }}">
                    <i class="fas fa-upload"></i> Upload
                </a>
                <a class="nav-link" href="{{ url_for('catalog') }}">
                    <i class="fas fa-book"></i> Catalog
                </a>
                <a class="nav-link" href="{{ url_for('embeddings') }}">
                    <i class="fas fa-database"></i> Embeddings
                </a>
                <a class="nav-link" href="{{ url_for('chat') }}">
                    <i class="fas fa-comments"></i> Chat
                </a>
                <a class="nav-link" href="{{ url_for('jobs') }}">
                    <i class="fas fa-history"></i> Jobs
                </a>
                <a class="nav-link" href="{{ url_for('settings') }}">
                    <i class="fas fa-cog"></i> Settings
                </a>
            </div>
        </div>
    </nav>

    <div class="container mt-4">
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                    <div class="alert alert-{{ 'danger' if category == 'error' else category }} alert-dismissible fade show">
                        {{ message }}
                        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                    </div>
                {% endfor %}
            {% endif %}
        {% endwith %}

        {% block content %}{% endblock %}
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="{{ url_for('static', filename='js/app.js') }}"></script>
    {% block scripts %}{% endblock %}
</body>
</html>"""

    # Create index template
    index_html = """{% extends "base.html" %}

{% block title %}Dashboard - Text Ingestion Interface{% endblock %}

{% block content %}
<div class="row">
    <div class="col-md-8">
        <div class="card">
            <div class="card-header">
                <h5><i class="fas fa-tachometer-alt"></i> Dashboard</h5>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-4">
                        <div class="card bg-primary text-white">
                            <div class="card-body text-center">
                                <h3>{{ jobs|length }}</h3>
                                <p>Total Jobs</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card bg-success text-white">
                            <div class="card-body text-center">
                                <h3>{{ jobs|selectattr('status', 'equalto', 'completed')|list|length }}</h3>
                                <p>Completed</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card bg-warning text-white">
                            <div class="card-body text-center">
                                <h3>{{ jobs|selectattr('status', 'equalto', 'running')|list|length }}</h3>
                                <p>Running</p>
                            </div>
                        </div>
                    </div>
                </div>

                {% if active_job %}
                <div class="mt-4">
                    <h6>Active Job</h6>
                    <div class="progress">
                        <div class="progress-bar progress-bar-striped progress-bar-animated"
                             role="progressbar" style="width: 0%" id="activeJobProgress"></div>
                    </div>
                    <small class="text-muted" id="activeJobMessage">Initializing...</small>
                </div>
                {% endif %}
            </div>
        </div>
    </div>

    <div class="col-md-4">
        <div class="card">
            <div class="card-header">
                <h5><i class="fas fa-plus"></i> Quick Actions</h5>
            </div>
            <div class="card-body">
                <a href="{{ url_for('upload') }}" class="btn btn-primary btn-lg w-100 mb-2">
                    <i class="fas fa-upload"></i> Upload Files
                </a>
                <a href="{{ url_for('configure') }}" class="btn btn-outline-primary btn-lg w-100">
                    <i class="fas fa-cog"></i> Configure Ingestion
                </a>
            </div>
        </div>
    </div>
</div>

<div class="row mt-4">
    <div class="col-12">
        <div class="card">
            <div class="card-header">
                <h5><i class="fas fa-history"></i> Recent Jobs</h5>
            </div>
            <div class="card-body">
                {% if jobs %}
                <div class="table-responsive">
                    <table class="table table-striped">
                        <thead>
                            <tr>
                                <th>ID</th>
                                <th>Collection</th>
                                <th>Status</th>
                                <th>Created</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for job in jobs %}
                            <tr>
                                <td><code>{{ job.id[:8] }}</code></td>
                                <td>{{ job.config.collection_name }}</td>
                                <td>
                                    <span class="badge bg-{{ 'primary' if job.status == 'running' else 'success' if job.status == 'completed' else 'danger' if job.status == 'failed' else 'secondary' }}">
                                        {{ job.status }}
                                    </span>
                                </td>
                                <td>{{ job.created_at[:19] }}</td>
                                <td>
                                    <a href="{{ url_for('job_status', job_id=job.id) }}" class="btn btn-sm btn-outline-primary">
                                        <i class="fas fa-eye"></i> View
                                    </a>
                                </td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
                {% else %}
                <p class="text-muted">No jobs found. <a href="{{ url_for('upload') }}">Upload some files</a> to get started.</p>
                {% endif %}
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
{% if active_job %}
// Update active job progress
function updateActiveJobProgress() {
    fetch('/api/job/{{ active_job }}/status')
        .then(response => response.json())
        .then(data => {
            const progressBar = document.getElementById('activeJobProgress');
            const messageEl = document.getElementById('activeJobMessage');

            if (progressBar && data.progress !== undefined) {
                progressBar.style.width = data.progress + '%';
            }
            if (messageEl && data.message) {
                messageEl.textContent = data.message;
            }

            if (data.status === 'completed' || data.status === 'failed') {
                setTimeout(() => location.reload(), 2000);
            }
        })
        .catch(error => console.error('Error updating progress:', error));
}

setInterval(updateActiveJobProgress, 2000);
updateActiveJobProgress();
{% endif %}
</script>
{% endblock %}"""

    # Create upload template
    upload_html = """{% extends "base.html" %}

{% block title %}Upload Files - Text Ingestion Interface{% endblock %}

{% block content %}
<div class="row justify-content-center">
    <div class="col-md-8">
        <div class="card">
            <div class="card-header">
                <h5><i class="fas fa-upload"></i> Upload Text Files</h5>
            </div>
            <div class="card-body">
                <form method="post" enctype="multipart/form-data" id="uploadForm">
                    <div class="mb-3">
                        <label for="files" class="form-label">
                            Select Text Files <small class="text-muted">(Max 100MB total)</small>
                        </label>
                        <input type="file" class="form-control" id="files" name="files" multiple
                               accept=".txt,.md,.rst,.html,.xml,.json,.csv" required>
                        <div class="form-text">
                            Supported formats: .txt, .md, .rst, .html, .xml, .json, .csv
                        </div>
                    </div>

                    <div id="fileList" class="mb-3" style="display: none;">
                        <h6>Selected Files:</h6>
                        <ul id="selectedFiles" class="list-group"></ul>
                    </div>

                    <div class="d-grid gap-2">
                        <button type="submit" class="btn btn-primary btn-lg" id="uploadBtn">
                            <i class="fas fa-upload"></i> Upload & Configure
                        </button>
                        <a href="{{ url_for('index') }}" class="btn btn-outline-secondary">
                            <i class="fas fa-arrow-left"></i> Back to Dashboard
                        </a>
                    </div>
                </form>
            </div>
        </div>

        <div class="card mt-4">
            <div class="card-header">
                <h6><i class="fas fa-info-circle"></i> Upload Guidelines</h6>
            </div>
            <div class="card-body">
                <ul class="mb-0">
                    <li>Upload multiple files at once by holding Ctrl/Cmd while selecting</li>
                    <li>Files are temporarily stored and processed immediately</li>
                    <li>Maximum file size: 100MB per file</li>
                    <li>Supported formats will be automatically converted to text</li>
                    <li>After upload, you'll be taken to the configuration page</li>
                </ul>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
document.getElementById('files').addEventListener('change', function(e) {
    const files = Array.from(e.target.files);
    const fileList = document.getElementById('fileList');
    const selectedFiles = document.getElementById('selectedFiles');

    if (files.length > 0) {
        fileList.style.display = 'block';
        selectedFiles.innerHTML = '';

        files.forEach(file => {
            const li = document.createElement('li');
            li.className = 'list-group-item d-flex justify-content-between align-items-center';
            li.innerHTML = `
                ${file.name}
                <small class="text-muted">${(file.size / 1024).toFixed(1)} KB</small>
            `;
            selectedFiles.appendChild(li);
        });
    } else {
        fileList.style.display = 'none';
    }
});

document.getElementById('uploadForm').addEventListener('submit', function(e) {
    const btn = document.getElementById('uploadBtn');
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Uploading...';
});
</script>
{% endblock %}"""

    # Create configure template
    configure_html = """{% extends "base.html" %}

{% block title %}Configure Ingestion - Text Ingestion Interface{% endblock %}

{% block content %}
<div class="row justify-content-center">
    <div class="col-md-10">
        <div class="card">
            <div class="card-header">
                <h5><i class="fas fa-cog"></i> Configure Ingestion</h5>
            </div>
            <div class="card-body">
                {% if files %}
                <div class="alert alert-info">
                    <h6><i class="fas fa-files"></i> Files to Process:</h6>
                    <ul class="mb-0">
                        {% for file in files %}
                        <li>{{ file }}</li>
                        {% endfor %}
                    </ul>
                </div>
                {% endif %}

                <form method="post" id="configForm">
                    <div class="row">
                        <div class="col-md-6">
                            <h6 class="mb-3">Basic Settings</h6>

                            <div class="mb-3">
                                <label for="collection_name" class="form-label">Collection Name</label>
                                <input type="text" class="form-control" id="collection_name" name="collection_name"
                                       value="{{ config.collection_name }}" required>
                                <div class="form-text">Name for the vector database collection</div>
                            </div>

                            <div class="mb-3">
                                <label for="embedding_model" class="form-label">Embedding Model</label>
                                <select class="form-select" id="embedding_model" name="embedding_model">
                                    <option value="sentence-transformer" {{ 'selected' if config.embedding_model == 'sentence-transformer' }}>Sentence Transformer</option>
                                    <option value="openai" {{ 'selected' if config.embedding_model == 'openai' }}>OpenAI</option>
                                </select>
                            </div>

                            <div class="mb-3" id="model_name_group">
                                <label for="model_name" class="form-label">Model Name</label>
                                <input type="text" class="form-control" id="model_name" name="model_name"
                                       value="{{ config.model_name }}">
                            </div>
                        </div>

                        <div class="col-md-6">
                            <h6 class="mb-3">Processing Settings</h6>

                            <div class="row">
                                <div class="col-md-6">
                                    <div class="mb-3">
                                        <label for="chunk_size" class="form-label">Chunk Size</label>
                                        <input type="number" class="form-control" id="chunk_size" name="chunk_size"
                                               value="{{ config.chunk_size }}" min="100" max="5000">
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="mb-3">
                                        <label for="chunk_overlap" class="form-label">Chunk Overlap</label>
                                        <input type="number" class="form-control" id="chunk_overlap" name="chunk_overlap"
                                               value="{{ config.chunk_overlap }}" min="0" max="1000">
                                    </div>
                                </div>
                            </div>

                            <div class="mb-3">
                                <label for="batch_size" class="form-label">Batch Size</label>
                                <input type="number" class="form-control" id="batch_size" name="batch_size"
                                       value="{{ config.batch_size }}" min="1" max="500">
                                <div class="form-text">Number of documents to process at once</div>
                            </div>
                        </div>
                    </div>

                    <div class="row">
                        <div class="col-12">
                            <h6 class="mb-3">Database Settings</h6>
                        </div>
                    </div>

                    <div class="row">
                        <div class="col-md-6">
                            <div class="mb-3">
                                <label for="db_host" class="form-label">Host</label>
                                <input type="text" class="form-control" id="db_host" name="db_host"
                                       value="{{ config.database.host }}">
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="mb-3">
                                <label for="db_port" class="form-label">Port</label>
                                <input type="number" class="form-control" id="db_port" name="db_port"
                                       value="{{ config.database.port }}">
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="mb-3">
                                <label for="db_name" class="form-label">Database</label>
                                <input type="text" class="form-control" id="db_name" name="db_name"
                                       value="{{ config.database.database }}">
                            </div>
                        </div>
                    </div>

                    <div class="row">
                        <div class="col-md-6">
                            <div class="mb-3">
                                <label for="db_user" class="form-label">Username</label>
                                <input type="text" class="form-control" id="db_user" name="db_user"
                                       value="{{ config.database.user }}">
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="mb-3">
                                <label for="db_password" class="form-label">Password</label>
                                <input type="password" class="form-control" id="db_password" name="db_password"
                                       value="{{ config.database.password }}">
                            </div>
                        </div>
                    </div>

                    <div class="mb-3">
                        <div class="form-check">
                            <input class="form-check-input" type="checkbox" id="overwrite" name="overwrite"
                                   {{ 'checked' if config.overwrite }}>
                            <label class="form-check-label" for="overwrite">
                                Overwrite existing collection
                            </label>
                        </div>
                        <div class="form-check">
                            <input class="form-check-input" type="checkbox" id="verbose" name="verbose"
                                   {{ 'checked' if config.verbose }}>
                            <label class="form-check-label" for="verbose">
                                Verbose logging
                            </label>
                        </div>
                    </div>

                    <div class="d-grid gap-2">
                        <button type="submit" class="btn btn-success btn-lg" id="startBtn">
                            <i class="fas fa-play"></i> Start Ingestion
                        </button>
                        <a href="{{ url_for('upload') }}" class="btn btn-outline-secondary">
                            <i class="fas fa-arrow-left"></i> Back to Upload
                        </a>
                    </div>
                </form>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
document.getElementById('embedding_model').addEventListener('change', function(e) {
    const modelNameGroup = document.getElementById('model_name_group');
    const modelNameInput = document.getElementById('model_name');

    if (e.target.value === 'openai') {
        modelNameInput.value = 'text-embedding-ada-002';
        modelNameGroup.style.display = 'none';
    } else {
        modelNameInput.value = 'all-MiniLM-L6-v2';
        modelNameGroup.style.display = 'block';
    }
});

document.getElementById('configForm').addEventListener('submit', function(e) {
    const btn = document.getElementById('startBtn');
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Starting Ingestion...';
});
</script>
{% endblock %}"""

    # Create chat template
    chat_html = """{% extends "base.html" %}

{% block title %}Chat - Text Ingestion Interface{% endblock %}

{% block content %}
<div class="row justify-content-center">
    <div class="col-md-8">
        <div class="card">
            <div class="card-header">
                <h5><i class="fas fa-comments"></i> Chat with Your Documents</h5>
            </div>
            <div class="card-body">
                <div id="chat-messages" class="mb-3" style="height: 400px; overflow-y: auto; border: 1px solid #dee2e6; padding: 10px; border-radius: 5px;">
                    <div class="text-muted">Ask questions about your ingested documents...</div>
                </div>

                <form id="chat-form">
                    <div class="input-group">
                        <input type="text" class="form-control" id="chat-input" placeholder="Ask a question..." required>
                        <button class="btn btn-primary" type="submit">
                            <i class="fas fa-paper-plane"></i> Send
                        </button>
                    </div>
                </form>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
document.getElementById('chat-form').addEventListener('submit', function(e) {
    e.preventDefault();
    const input = document.getElementById('chat-input');
    const messages = document.getElementById('chat-messages');
    const query = input.value.trim();

    if (!query) return;

    // Add user message
    messages.innerHTML += `<div class="mb-2"><strong>You:</strong> ${query}</div>`;
    input.value = '';

    // Add loading message
    const loadingDiv = document.createElement('div');
    loadingDiv.id = 'loading';
    loadingDiv.innerHTML = '<div class="text-muted"><i>Searching documents...</i></div>';
    messages.appendChild(loadingDiv);
    messages.scrollTop = messages.scrollHeight;

    // Send request
    fetch('/api/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: query })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('loading').remove();

        if (data.success) {
            let responseHtml = '<div class="mb-3"><strong>Assistant:</strong></div>';
            data.data.results.forEach(result => {
                responseHtml += `
                    <div class="card mb-2">
                        <div class="card-body">
                            <p class="mb-1">${result.content}</p>
                            <small class="text-muted">Score: ${result.score.toFixed(3)}</small>
                        </div>
                    </div>
                `;
            });
            messages.innerHTML += responseHtml;
        } else {
            messages.innerHTML += `<div class="text-danger"><strong>Error:</strong> ${data.error}</div>`;
        }

        messages.scrollTop = messages.scrollHeight;
    })
    .catch(error => {
        document.getElementById('loading').remove();
        messages.innerHTML += `<div class="text-danger"><strong>Error:</strong> ${error.message}</div>`;
        messages.scrollTop = messages.scrollHeight;
    });
});
</script>
{% endblock %}"""

    # Create embeddings template
    embeddings_html = """{% extends "base.html" %}

{% block title %}Embeddings - Text Ingestion Interface{% endblock %}

{% block content %}
<div class="row">
    <div class="col-12">
        <div class="card">
            <div class="card-header d-flex justify-content-between align-items-center">
                <h5><i class="fas fa-database"></i> Document Chunks & Embeddings</h5>
                <div class="input-group" style="width: 300px;">
                    <input type="text" class="form-control" id="search-input" placeholder="Search chunks...">
                    <button class="btn btn-outline-secondary" type="button" id="search-btn">
                        <i class="fas fa-search"></i>
                    </button>
                </div>
            </div>
            <div class="card-body">
                <div class="table-responsive">
                    <table class="table table-striped table-hover" id="chunks-table">
                        <thead>
                            <tr>
                                <th>ID</th>
                                <th>Content Preview</th>
                                <th>Source</th>
                                <th>Chunk Size</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody id="chunks-tbody">
                            <tr>
                                <td colspan="5" class="text-center">
                                    <div class="spinner-border" role="status">
                                        <span class="visually-hidden">Loading...</span>
                                    </div>
                                    Loading chunks...
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>

                <nav aria-label="Chunks pagination" class="mt-3">
                    <ul class="pagination justify-content-center" id="pagination">
                    </ul>
                </nav>
            </div>
        </div>
    </div>
</div>

<!-- Chunk Details Modal -->
<div class="modal fade" id="chunkModal" tabindex="-1">
    <div class="modal-dialog modal-lg">
        <div class="modal-content">
            <div class="modal-header">
                <h5 class="modal-title">Chunk Details</h5>
                <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
            </div>
            <div class="modal-body">
                <div id="chunk-details">
                    <div class="mb-3">
                        <strong>ID:</strong> <code id="chunk-id"></code>
                    </div>
                    <div class="mb-3">
                        <strong>Content:</strong>
                        <div class="border p-2 bg-light" id="chunk-content" style="max-height: 300px; overflow-y: auto;"></div>
                    </div>
                    <div class="mb-3">
                        <strong>Metadata:</strong>
                        <pre id="chunk-metadata" class="bg-light p-2"></pre>
                    </div>
                    <div class="mb-3">
                        <strong>Embedding Preview:</strong>
                        <div class="border p-2 bg-light" id="chunk-embedding" style="max-height: 200px; overflow-y: auto; font-family: monospace; font-size: 12px;"></div>
                    </div>
                </div>
            </div>
            <div class="modal-footer">
                <button type="button" class="btn btn-danger" id="delete-chunk-btn">
                    <i class="fas fa-trash"></i> Delete Chunk
                </button>
                <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
let currentPage = 1;
let currentSearch = '';
const perPage = 20;

function loadChunks(page = 1, search = '') {
    const tbody = document.getElementById('chunks-tbody');
    tbody.innerHTML = `
        <tr>
            <td colspan="5" class="text-center">
                <div class="spinner-border" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                Loading chunks...
            </td>
        </tr>
    `;

    let url = `/api/embeddings?page=${page}&per_page=${perPage}`;
    if (search) {
        url += `&search=${encodeURIComponent(search)}`;
    }

    fetch(url)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                displayChunks(data.chunks);
                updatePagination(data.total_pages, page, search);
            } else {
                tbody.innerHTML = `
                    <tr>
                        <td colspan="5" class="text-center text-danger">
                            Error loading chunks: ${data.error}
                        </td>
                    </tr>
                `;
            }
        })
        .catch(error => {
            tbody.innerHTML = `
                <tr>
                    <td colspan="5" class="text-center text-danger">
                        Error loading chunks: ${error.message}
                    </td>
                </tr>
            `;
        });
}

function displayChunks(chunks) {
    const tbody = document.getElementById('chunks-tbody');

    if (chunks.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="5" class="text-center text-muted">
                    No chunks found.
                </td>
            </tr>
        `;
        return;
    }

    let html = '';
    chunks.forEach(chunk => {
        const source = chunk.metadata?.source || 'Unknown';
        const chunkSize = chunk.content?.length || 0;

        html += `
            <tr>
                <td><code>${chunk.id.substring(0, 8)}...</code></td>
                <td>
                    <div style="max-width: 400px; overflow: hidden; text-overflow: ellipsis;">
                        ${chunk.content_preview}
                    </div>
                </td>
                <td>${source}</td>
                <td>${chunkSize} chars</td>
                <td>
                    <button class="btn btn-sm btn-outline-primary me-1" onclick="viewChunk('${chunk.id}')">
                        <i class="fas fa-eye"></i> View
                    </button>
                    <button class="btn btn-sm btn-outline-danger" onclick="deleteChunk('${chunk.id}')">
                        <i class="fas fa-trash"></i> Delete
                    </button>
                </td>
            </tr>
        `;
    });

    tbody.innerHTML = html;
}

function updatePagination(totalPages, currentPage, search) {
    const pagination = document.getElementById('pagination');
    let html = '';

    if (totalPages > 1) {
        // Previous button
        if (currentPage > 1) {
            html += `<li class="page-item"><a class="page-link" href="#" onclick="loadChunks(${currentPage - 1}, '${search}')">Previous</a></li>`;
        } else {
            html += `<li class="page-item disabled"><span class="page-link">Previous</span></li>`;
        }

        // Page numbers
        for (let i = Math.max(1, currentPage - 2); i <= Math.min(totalPages, currentPage + 2); i++) {
            if (i === currentPage) {
                html += `<li class="page-item active"><span class="page-link">${i}</span></li>`;
            } else {
                html += `<li class="page-item"><a class="page-link" href="#" onclick="loadChunks(${i}, '${search}')">${i}</a></li>`;
            }
        }

        // Next button
        if (currentPage < totalPages) {
            html += `<li class="page-item"><a class="page-link" href="#" onclick="loadChunks(${currentPage + 1}, '${search}')">Next</a></li>`;
        } else {
            html += `<li class="page-item disabled"><span class="page-link">Next</span></li>`;
        }
    }

    pagination.innerHTML = html;
}

function viewChunk(chunkId) {
    // Fetch chunk details
    fetch(`/api/embeddings?page=1&per_page=1&search=${chunkId}`)
        .then(response => response.json())
        .then(data => {
            if (data.success && data.chunks.length > 0) {
                const chunk = data.chunks[0];

                document.getElementById('chunk-id').textContent = chunk.id;
                document.getElementById('chunk-content').textContent = chunk.content;
                document.getElementById('chunk-metadata').textContent = JSON.stringify(chunk.metadata, null, 2);

                if (chunk.embedding) {
                    const embeddingPreview = chunk.embedding.slice(0, 10).join(', ') + (chunk.embedding.length > 10 ? '...' : '');
                    document.getElementById('chunk-embedding').textContent = `[${embeddingPreview}] (${chunk.embedding.length} dimensions)`;
                } else {
                    document.getElementById('chunk-embedding').textContent = 'No embedding data available';
                }

                // Store chunk ID for deletion
                document.getElementById('delete-chunk-btn').setAttribute('data-chunk-id', chunk.id);

                const modal = new bootstrap.Modal(document.getElementById('chunkModal'));
                modal.show();
            }
        })
        .catch(error => {
            alert('Error loading chunk details: ' + error.message);
        });
}

function deleteChunk(chunkId) {
    if (!confirm('Are you sure you want to delete this chunk? This action cannot be undone.')) {
        return;
    }

    fetch(`/api/embeddings/${chunkId}`, {
        method: 'DELETE'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert('Chunk deleted successfully!');
            loadChunks(currentPage, currentSearch);
        } else {
            alert('Error deleting chunk: ' + data.error);
        }
    })
    .catch(error => {
        alert('Error deleting chunk: ' + error.message);
    });
}

// Event listeners
document.getElementById('search-btn').addEventListener('click', function() {
    currentSearch = document.getElementById('search-input').value.trim();
    currentPage = 1;
    loadChunks(currentPage, currentSearch);
});

document.getElementById('search-input').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        currentSearch = this.value.trim();
        currentPage = 1;
        loadChunks(currentPage, currentSearch);
    }
});

document.getElementById('delete-chunk-btn').addEventListener('click', function() {
    const chunkId = this.getAttribute('data-chunk-id');
    if (chunkId) {
        deleteChunk(chunkId);
        const modal = bootstrap.Modal.getInstance(document.getElementById('chunkModal'));
        modal.hide();
    }
});

// Load initial data
loadChunks();
</script>
{% endblock %}"""

    # Create catalog template
    catalog_html = """{% extends "base.html" %}

{% block title %}Document Catalog - Text Ingestion Interface{% endblock %}

{% block content %}
<div class="row">
    <div class="col-12">
        <!-- Catalog Statistics -->
        <div class="row mb-4" id="stats-row">
            <div class="col-md-3">
                <div class="card text-center">
                    <div class="card-body">
                        <div class="spinner-border" role="status">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                        <p class="card-text">Loading stats...</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card text-center">
                    <div class="card-body">
                        <div class="spinner-border" role="status">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                        <p class="card-text">Loading stats...</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card text-center">
                    <div class="card-body">
                        <div class="spinner-border" role="status">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                        <p class="card-text">Loading stats...</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card text-center">
                    <div class="card-body">
                        <div class="spinner-border" role="status">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                        <p class="card-text">Loading stats...</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- Documents List -->
        <div class="card">
            <div class="card-header d-flex justify-content-between align-items-center">
                <h5><i class="fas fa-file-alt"></i> Documents</h5>
                <div class="input-group" style="width: 300px;">
                    <input type="text" class="form-control" id="search-input" placeholder="Search documents...">
                    <button class="btn btn-outline-secondary" type="button" id="search-btn">
                        <i class="fas fa-search"></i>
                    </button>
                </div>
            </div>
            <div class="card-body">
                <div class="table-responsive">
                    <table class="table table-striped table-hover" id="documents-table">
                        <thead>
                            <tr>
                                <th>Filename</th>
                                <th>Chunks</th>
                                <th>Total Size</th>
                                <th>Avg Chunk Size</th>
                                <th>Last Ingested</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody id="documents-tbody">
                            <tr>
                                <td colspan="6" class="text-center">
                                    <div class="spinner-border" role="status">
                                        <span class="visually-hidden">Loading...</span>
                                    </div>
                                    Loading documents...
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>

                <nav aria-label="Documents pagination" class="mt-3">
                    <ul class="pagination justify-content-center" id="pagination">
                    </ul>
                </nav>
            </div>
        </div>
    </div>
</div>

<!-- Document Chunks Modal -->
<div class="modal fade" id="chunksModal" tabindex="-1" size="xl">
    <div class="modal-dialog modal-xl">
        <div class="modal-content">
            <div class="modal-header">
                <h5 class="modal-title"><i class="fas fa-list"></i> Chunks for <span id="modal-filename"></span></h5>
                <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
            </div>
            <div class="modal-body">
                <div class="mb-3">
                    <strong>Source:</strong> <code id="modal-source"></code>
                </div>

                <div class="table-responsive">
                    <table class="table table-sm table-striped">
                        <thead>
                            <tr>
                                <th style="width: 60px;">#</th>
                                <th>Content Preview</th>
                                <th style="width: 100px;">Size</th>
                                <th style="width: 120px;">Actions</th>
                            </tr>
                        </thead>
                        <tbody id="chunks-tbody">
                        </tbody>
                    </table>
                </div>

                <nav aria-label="Chunks pagination" class="mt-3">
                    <ul class="pagination pagination-sm justify-content-center" id="chunks-pagination">
                    </ul>
                </nav>
            </div>
            <div class="modal-footer">
                <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
            </div>
        </div>
    </div>
</div>

<!-- Chunk Details Modal -->
<div class="modal fade" id="chunkDetailModal" tabindex="-1">
    <div class="modal-dialog modal-lg">
        <div class="modal-content">
            <div class="modal-header">
                <h5 class="modal-title">Chunk Details</h5>
                <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
            </div>
            <div class="modal-body">
                <div class="mb-3">
                    <strong>ID:</strong> <code id="detail-chunk-id"></code>
                </div>
                <div class="mb-3">
                    <strong>Content:</strong>
                    <div class="border p-2 bg-light" id="detail-chunk-content" style="max-height: 400px; overflow-y: auto; white-space: pre-wrap;"></div>
                </div>
                <div class="mb-3">
                    <strong>Metadata:</strong>
                    <pre id="detail-chunk-metadata" class="bg-light p-2" style="max-height: 200px; overflow-y: auto;"></pre>
                </div>
            </div>
            <div class="modal-footer">
                <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
let currentPage = 1;
let currentSearch = '';
const perPage = 20;
let currentDocumentSource = '';
let currentChunksPage = 1;
const chunksPerPage = 50;

function loadCatalogStats() {
    fetch('/api/catalog/stats')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                updateStats(data.stats);
            }
        })
        .catch(error => console.error('Error loading stats:', error));
}

function updateStats(stats) {
    const statsRow = document.getElementById('stats-row');

    statsRow.innerHTML = `
        <div class="col-md-3">
            <div class="card text-center bg-primary text-white">
                <div class="card-body">
                    <h3>${stats.total_documents || 0}</h3>
                    <p class="card-text">Total Documents</p>
                </div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="card text-center bg-success text-white">
                <div class="card-body">
                    <h3>${stats.total_chunks || 0}</h3>
                    <p class="card-text">Total Chunks</p>
                </div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="card text-center bg-info text-white">
                <div class="card-body">
                    <h3>${formatNumber(stats.total_characters || 0)}</h3>
                    <p class="card-text">Total Characters</p>
                </div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="card text-center bg-warning text-white">
                <div class="card-body">
                    <h3>${stats.avg_chunk_size || 0}</h3>
                    <p class="card-text">Avg Chunk Size</p>
                </div>
            </div>
        </div>
    `;
}

function loadDocuments(page = 1, search = '') {
    const tbody = document.getElementById('documents-tbody');
    tbody.innerHTML = `
        <tr>
            <td colspan="6" class="text-center">
                <div class="spinner-border" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                Loading documents...
            </td>
        </tr>
    `;

    let url = `/api/catalog/documents?page=${page}&per_page=${perPage}`;
    if (search) {
        url += `&search=${encodeURIComponent(search)}`;
    }

    fetch(url)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                displayDocuments(data.documents);
                updatePagination(data.total_pages, page, search);
            } else {
                tbody.innerHTML = `
                    <tr>
                        <td colspan="6" class="text-center text-danger">
                            Error loading documents: ${data.error}
                        </td>
                    </tr>
                `;
            }
        })
        .catch(error => {
            tbody.innerHTML = `
                <tr>
                    <td colspan="6" class="text-center text-danger">
                        Error loading documents: ${error.message}
                    </td>
                </tr>
            `;
        });
}

function displayDocuments(documents) {
    const tbody = document.getElementById('documents-tbody');

    if (documents.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="6" class="text-center text-muted">
                    No documents found.
                </td>
            </tr>
        `;
        return;
    }

    let html = '';
    documents.forEach((doc, index) => {
        const lastIngested = new Date(doc.last_ingested).toLocaleDateString();

        html += `
            <tr>
                <td>
                    <strong>${doc.filename}</strong>
                    <br><small class="text-muted">${doc.source}</small>
                </td>
                <td><span class="badge bg-primary">${doc.chunk_count}</span></td>
                <td>${formatNumber(doc.total_chars)} chars</td>
                <td>${doc.avg_chunk_size} chars</td>
                <td>${lastIngested}</td>
                <td>
                    <button class="btn btn-sm btn-outline-primary" onclick="viewChunks('${doc.source}', '${doc.filename}')">
                        <i class="fas fa-list"></i> View Chunks
                    </button>
                </td>
            </tr>
        `;
    });

    tbody.innerHTML = html;
}

function updatePagination(totalPages, currentPage, search) {
    const pagination = document.getElementById('pagination');
    let html = '';

    if (totalPages > 1) {
        // Previous button
        if (currentPage > 1) {
            html += `<li class="page-item"><a class="page-link" href="#" onclick="loadDocuments(${currentPage - 1}, '${search}')">Previous</a></li>`;
        } else {
            html += `<li class="page-item disabled"><span class="page-link">Previous</span></li>`;
        }

        // Page numbers
        for (let i = Math.max(1, currentPage - 2); i <= Math.min(totalPages, currentPage + 2); i++) {
            if (i === currentPage) {
                html += `<li class="page-item active"><span class="page-link">${i}</span></li>`;
            } else {
                html += `<li class="page-item"><a class="page-link" href="#" onclick="loadDocuments(${i}, '${search}')">${i}</a></li>`;
            }
        }

        // Next button
        if (currentPage < totalPages) {
            html += `<li class="page-item"><a class="page-link" href="#" onclick="loadDocuments(${currentPage + 1}, '${search}')">Next</a></li>`;
        } else {
            html += `<li class="page-item disabled"><span class="page-link">Next</span></li>`;
        }
    }

    pagination.innerHTML = html;
}

function viewChunks(source, filename) {
    currentDocumentSource = source;
    currentChunksPage = 1;

    document.getElementById('modal-filename').textContent = filename;
    document.getElementById('modal-source').textContent = source;

    loadDocumentChunks(source, filename);
    const modal = new bootstrap.Modal(document.getElementById('chunksModal'));
    modal.show();
}

function loadDocumentChunks(source, filename, page = 1) {
    const tbody = document.getElementById('chunks-tbody');
    tbody.innerHTML = `
        <tr>
            <td colspan="4" class="text-center">
                <div class="spinner-border spinner-border-sm" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                Loading chunks...
            </td>
        </tr>
    `;

    fetch(`/api/catalog/document/${encodeURIComponent(source)}/chunks?page=${page}&per_page=${chunksPerPage}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                displayChunks(data.chunks, page);
            } else {
                tbody.innerHTML = `
                    <tr>
                        <td colspan="4" class="text-center text-danger">
                            Error loading chunks: ${data.error}
                        </td>
                    </tr>
                `;
            }
        })
        .catch(error => {
            tbody.innerHTML = `
                <tr>
                    <td colspan="4" class="text-center text-danger">
                        Error loading chunks: ${error.message}
                    </td>
                </tr>
            `;
        });
}

function displayChunks(chunks, page) {
    const tbody = document.getElementById('chunks-tbody');

    if (chunks.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="4" class="text-center text-muted">
                    No chunks found.
                </td>
            </tr>
        `;
        return;
    }

    let html = '';
    const startIndex = (page - 1) * chunksPerPage;

    chunks.forEach((chunk, index) => {
        const chunkNumber = startIndex + index + 1;
        html += `
            <tr>
                <td>${chunkNumber}</td>
                <td>
                    <div style="max-width: 500px; overflow: hidden; text-overflow: ellipsis;">
                        ${chunk.content_preview}
                    </div>
                </td>
                <td>${chunk.content_length} chars</td>
                <td>
                    <button class="btn btn-sm btn-outline-info" onclick="viewChunkDetail('${chunk.id}', '${chunk.content.replace(/'/g, "\\'").replace(/"/g, '\\"')}', '${JSON.stringify(chunk.metadata).replace(/'/g, "\\'")}')">
                        <i class="fas fa-eye"></i> View
                    </button>
                </td>
            </tr>
        `;
    });

    tbody.innerHTML = html;
}

function viewChunkDetail(chunkId, content, metadata) {
    document.getElementById('detail-chunk-id').textContent = chunkId;
    document.getElementById('detail-chunk-content').textContent = content;
    document.getElementById('detail-chunk-metadata').textContent = JSON.stringify(JSON.parse(metadata), null, 2);

    const modal = new bootstrap.Modal(document.getElementById('chunkDetailModal'));
    modal.show();
}

function formatNumber(num) {
    return num.toString().replace(/\\B(?=(\\d{3})+(?!\\d))/g, ",");
}

// Event listeners
document.getElementById('search-btn').addEventListener('click', function() {
    currentSearch = document.getElementById('search-input').value.trim();
    currentPage = 1;
    loadDocuments(currentPage, currentSearch);
});

document.getElementById('search-input').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        currentSearch = this.value.trim();
        currentPage = 1;
        loadDocuments(currentPage, currentSearch);
    }
});

// Load initial data
loadCatalogStats();
loadDocuments();
</script>
{% endblock %}"""
    
    # Write templates
    with open(template_dir / 'base.html', 'w') as f:
        f.write(base_html)
    with open(template_dir / 'index.html', 'w') as f:
        f.write(index_html)
    with open(template_dir / 'upload.html', 'w') as f:
        f.write(upload_html)
    with open(template_dir / 'configure.html', 'w') as f:
        f.write(configure_html)
    with open(template_dir / 'chat.html', 'w') as f:
        f.write(chat_html)
    with open(template_dir / 'embeddings.html', 'w') as f:
        f.write(embeddings_html)
    with open(template_dir / 'catalog.html', 'w') as f:
        f.write(catalog_html)

    # Create basic CSS
    css_content = """body {
    background-color: #f8f9fa;
}

.navbar-brand {
    font-weight: bold;
}

.card {
    box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
    border: 1px solid rgba(0, 0, 0, 0.125);
}

.card-header {
    background-color: #f8f9fa;
    border-bottom: 1px solid rgba(0, 0, 0, 0.125);
    font-weight: 600;
}

.btn {
    border-radius: 0.375rem;
}

.progress {
    height: 1.5rem;
    border-radius: 0.375rem;
}

.alert {
    border-radius: 0.375rem;
}

.table th {
    border-top: none;
    font-weight: 600;
}

.badge {
    font-size: 0.75em;
}

.form-text {
    font-size: 0.875em;
}

@media (max-width: 768px) {
    .container {
        padding-left: 15px;
        padding-right: 15px;
    }
}"""

    with open(css_dir / 'style.css', 'w') as f:
        f.write(css_content)

    # Create basic JS
    js_content = """// Common JavaScript functions
function showAlert(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

    const container = document.querySelector('.container');
    container.insertBefore(alertDiv, container.firstChild);

    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 5000);
}

// Auto-refresh functionality for status pages
function autoRefresh(interval = 5000) {
    setInterval(() => {
        location.reload();
    }, interval);
}

// Copy to clipboard functionality
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showAlert('Copied to clipboard!', 'success');
    }).catch(() => {
        showAlert('Failed to copy to clipboard', 'error');
    });
}

// Format file sizes
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}"""

    with open(js_dir / 'app.js', 'w') as f:
        f.write(js_content)

if __name__ == '__main__':
    # Create templates if they don't exist
    if not Path('templates').exists():
        print("Creating templates...")
        create_templates()

    # Run the app
    print("Starting web interface...")
    print("Visit http://localhost:5000 to access the interface")
    app.run(debug=True, host='0.0.0.0', port=5000)
