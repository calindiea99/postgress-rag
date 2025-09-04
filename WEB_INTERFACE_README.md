# Text Ingestion Web Interface

A modern web-based interface for the text ingestion pipeline that provides an easy-to-use graphical interface for ingesting text files into PostgreSQL vector databases.

## Features

### 🎯 **User-Friendly Interface**
- **Drag & Drop Upload**: Simple file upload with progress indicators
- **Visual Configuration**: Easy-to-use forms for all ingestion settings
- **Real-time Monitoring**: Live progress tracking and status updates
- **Job Management**: View, cancel, and monitor ingestion jobs
- **Responsive Design**: Works on desktop and mobile devices

### 🚀 **Core Functionality**
- **Multi-format Support**: Upload .txt, .md, .rst, .html, .xml, .json, .csv files
- **Flexible Configuration**: Customize chunk size, overlap, embedding models
- **Batch Processing**: Efficient processing of large file sets
- **Progress Tracking**: Real-time progress bars and status messages
- **Error Handling**: Comprehensive error reporting and recovery

### 📊 **Dashboard & Analytics**
- **Job Statistics**: Overview of completed, running, and failed jobs
- **Collection Management**: View and manage vector collections
- **Performance Metrics**: Processing speed and resource usage
- **History Tracking**: Complete job history with detailed logs

## Installation

### Prerequisites
- Python 3.8+
- PostgreSQL with pgvector extension
- All dependencies from `requirements.txt`

### Setup
```bash
# Install web interface dependencies
pip install -r web_requirements.txt

# Make sure your database is running
# Start the web interface
python web_interface.py
```

### Access
Once running, visit: **http://localhost:5000**

## Usage Guide

### 1. **Upload Files**
- Navigate to the **Upload** page
- Drag and drop files or click to browse
- Select multiple files at once
- Supported formats: .txt, .md, .rst, .html, .xml, .json, .csv

### 2. **Configure Ingestion**
- After upload, you'll be taken to the configuration page
- Set your preferred settings:
  - **Collection Name**: Name for the vector collection
  - **Embedding Model**: Choose between Sentence Transformers or OpenAI
  - **Chunk Settings**: Configure text splitting parameters
  - **Database Settings**: PostgreSQL connection details

### 3. **Monitor Progress**
- Start the ingestion job
- Monitor real-time progress on the dashboard
- View detailed job status and logs
- Cancel running jobs if needed

### 4. **View Results**
- Check job completion status
- View processing statistics
- Access ingested collections for RAG applications

## Configuration Options

### Basic Settings
- **Collection Name**: Identifier for the vector collection
- **Embedding Model**: Choose local (Sentence Transformers) or cloud (OpenAI)
- **Model Name**: Specific model to use for embeddings

### Processing Settings
- **Chunk Size**: Number of characters per text chunk (100-5000)
- **Chunk Overlap**: Overlap between chunks for context preservation
- **Batch Size**: Number of documents to process simultaneously

### Database Settings
- **Host**: PostgreSQL server address
- **Port**: Database port (default: 5432)
- **Database**: Database name
- **Credentials**: Username and password

## API Endpoints

The web interface provides REST API endpoints for integration:

```
GET  /api/job/<job_id>/status    # Get job status
POST /api/job/<job_id>/cancel    # Cancel running job
```

## File Structure

```
web_interface.py              # Main Flask application
templates/                    # HTML templates
├── base.html                 # Base template with navigation
├── index.html                # Dashboard page
├── upload.html               # File upload page
├── configure.html            # Configuration page
├── job_status.html           # Job monitoring page
├── jobs.html                 # Jobs history page
└── settings.html             # Settings page
static/                      # Static assets
├── css/
│   └── style.css            # Custom styles
└── js/
    └── app.js               # JavaScript utilities
web_requirements.txt         # Python dependencies
```

## Security Considerations

- **File Upload Limits**: 100MB maximum file size
- **Input Validation**: All inputs are validated server-side
- **Secure Headers**: Flask-WTF CSRF protection enabled
- **Session Management**: Secure session handling
- **Database Security**: Parameterized queries prevent SQL injection

## Troubleshooting

### Common Issues

1. **"Module not found" errors**
   - Install dependencies: `pip install -r web_requirements.txt`
   - Also install main dependencies: `pip install -r requirements.txt`

2. **Database connection failed**
   - Verify PostgreSQL is running
   - Check connection settings in configuration
   - Ensure pgvector extension is installed

3. **Upload fails**
   - Check file size (max 100MB)
   - Verify file format is supported
   - Check upload directory permissions

4. **High memory usage**
   - Reduce batch size in configuration
   - Process fewer files simultaneously
   - Monitor system resources

### Logs
- Application logs are written to `ingestion.log`
- Flask debug logs appear in console when running
- Job-specific logs are available in job status pages

## Development

### Running in Debug Mode
```bash
export FLASK_ENV=development
python web_interface.py
```

### Customizing Templates
- Templates use Jinja2 syntax
- Static files are served from `/static/` URL path
- Bootstrap 5 and Font Awesome are included by default

### Adding New Features
- Extend the `IngestionManager` class for new job types
- Add new routes in `web_interface.py`
- Create corresponding templates in `templates/`
- Add styles in `static/css/style.css`

## Integration

The web interface integrates seamlessly with:
- **Existing ingestion script**: Uses the same `ingestion.py` pipeline
- **Database setup**: Works with your existing PostgreSQL + pgvector setup
- **RAG applications**: Ingested data is immediately available for retrieval

## License

This web interface is part of the PostgreSQL RAG project and follows the same licensing terms.
