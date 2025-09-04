# Text Ingestion Script

This Python script provides a production-ready solution for ingesting text files into a PostgreSQL vector database using LangChain and pgvector.

## Features

- **Flexible Input**: Load text files from any directory with customizable file patterns
- **Smart Chunking**: Intelligent document splitting with configurable chunk size and overlap
- **Batch Processing**: Efficient batch processing for large document sets
- **Multiple Embeddings**: Support for both Sentence Transformers and OpenAI embeddings
- **Verification**: Built-in verification with test queries
- **Logging**: Comprehensive logging with file output option
- **Metadata Tracking**: Automatic metadata generation and storage

## Prerequisites

Make sure you have the required Python packages installed:

```bash
pip install -r requirements.txt
```

Key dependencies:
- langchain
- langchain-community
- langchain-openai
- langchain-text-splitters
- psycopg2-binary
- sentence-transformers

## Usage

### Basic Usage

```bash
# Ingest all .txt files from ./text_files directory
python ingestion.py --input-dir ./text_files --collection my_docs

# Or using the executable directly
./ingestion.py --input-dir ./text_files --collection my_docs
```

### Advanced Usage

```bash
# Use OpenAI embeddings (requires OPENAI_API_KEY)
python ingestion.py \
  --input-dir ./documents \
  --collection tech_docs \
  --model openai \
  --chunk-size 500 \
  --verbose

# Custom file patterns and batch processing
python ingestion.py \
  --input-dir ./data \
  --file-pattern "**/*.md" \
  --collection research \
  --batch-size 50 \
  --overwrite
```

### Command Line Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--input-dir` | `-i` | `./text_files` | Directory containing text files |
| `--collection` | `-c` | `text_ingestion_docs` | PGVector collection name |
| `--file-pattern` | `-p` | `**/*.txt` | Glob pattern for files to ingest |
| `--chunk-size` | | `1000` | Document chunk size in characters |
| `--chunk-overlap` | | `200` | Overlap between chunks |
| `--batch-size` | | `100` | Documents per batch |
| `--model` | | `sentence-transformer` | Embedding model (sentence-transformer/openai) |
| `--model-name` | | `all-MiniLM-L6-v2` | Sentence transformer model name |
| `--db-host` | | `localhost` | Database host |
| `--db-port` | | `5432` | Database port |
| `--db-name` | | `rag_db` | Database name |
| `--db-user` | | `rag_user` | Database user |
| `--db-password` | | `rag_password` | Database password |
| `--overwrite` | | `False` | Overwrite existing collection |
| `--verbose` | `-v` | `False` | Enable verbose logging |
| `--log-file` | | `False` | Save logs to file |

## Database Setup

Make sure your PostgreSQL database is running and has pgvector extension enabled. The script expects:

- Database: `rag_db` (configurable)
- User: `rag_user` (configurable)
- Password: `rag_password` (configurable)
- pgvector extension installed

## Output

The script generates:
- **Console output**: Progress and status information
- **Log file**: Detailed logs (when `--log-file` is used)
- **Metadata file**: JSON file with ingestion details
- **Vector store**: Documents stored in PGVector collection

## Examples

### Example 1: Basic text ingestion
```bash
python ingestion.py --input-dir ./my_texts --collection literature
```

### Example 2: Research papers with custom chunking
```bash
python ingestion.py \
  --input-dir ./research_papers \
  --file-pattern "**/*.pdf" \
  --collection research \
  --chunk-size 800 \
  --chunk-overlap 100 \
  --verbose
```

### Example 3: Using OpenAI embeddings
```bash
export OPENAI_API_KEY="your-api-key-here"
python ingestion.py \
  --input-dir ./documents \
  --collection ai_docs \
  --model openai \
  --overwrite
```

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure all dependencies are installed
2. **Database connection failed**: Check database credentials and pgvector extension
3. **No documents found**: Verify input directory and file patterns
4. **Memory issues**: Reduce batch size for large document sets

### Logs

Check the `ingestion.log` file (when using `--log-file`) for detailed error information.

## Integration

This script integrates seamlessly with your existing RAG applications. Use the same collection name in your query scripts to access the ingested documents.
