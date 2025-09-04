#!/usr/bin/env python3
"""
Text File Ingestion Script for LangChain and PGVector

This script ingests text files (.txt) into a PostgreSQL vector database
using LangChain and pgvector for RAG applications.

Usage:
    python ingestion.py --input-dir ./text_files --collection my_docs
    python ingestion.py --help

Author: Generated for PostgreSQL RAG project
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
import json
from datetime import datetime

# Third-party imports
import psycopg2
from langchain_community.vectorstores import PGVector
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_openai import OpenAIEmbeddings
# Use new CohereEmbeddings from langchain_cohere
try:
    from langchain_cohere import CohereEmbeddings
except ImportError:
    CohereEmbeddings = None


class TextIngestionPipeline:
    """Pipeline for ingesting text files into PGVector store."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.connection_string = None
        self.embeddings = None
        self.vectorstore = None

    def setup_logging(self):
        """Configure logging based on verbosity level."""
        level = logging.DEBUG if self.config['verbose'] else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('ingestion.log') if self.config['log_file'] else logging.NullHandler()
            ]
        )

    def validate_config(self):
        """Validate configuration parameters."""
        # Ensure input_dir is a Path object
        if isinstance(self.config['input_dir'], str):
            self.config['input_dir'] = Path(self.config['input_dir'])

        # Create input directory if it doesn't exist
        self.config['input_dir'].mkdir(parents=True, exist_ok=True)

        if not self.config['input_dir'].exists():
            raise ValueError(f"Input directory does not exist: {self.config['input_dir']}")

        if self.config['embedding_model'] == 'openai' and not os.getenv('OPENAI_API_KEY'):
            raise ValueError("OPENAI_API_KEY environment variable required for OpenAI embeddings")

    def setup_database_connection(self):
        """Setup database connection and test it."""
        db_config = self.config['database']

        self.connection_string = (
            f"postgresql+psycopg2://{db_config['user']}:"
            f"{db_config['password']}@{db_config['host']}:"
            f"{db_config['port']}/{db_config['database']}"
        )

        self.logger.info("Testing database connection...")
        try:
            conn = psycopg2.connect(**db_config)
            conn.close()
            self.logger.info("✅ Database connection successful")
        except Exception as e:
            raise ConnectionError(f"Database connection failed: {e}")

    def setup_embeddings(self):
        """Initialize embedding model."""
        self.logger.info(f"Initializing {self.config['embedding_model']} embeddings...")

        if self.config['embedding_model'] == 'openai':
            self.embeddings = OpenAIEmbeddings()
        elif self.config['embedding_model'] == 'sentence-transformer':
            self.embeddings = SentenceTransformerEmbeddings(
                model_name=self.config['model_name']
            )
        elif self.config['embedding_model'] == 'cohere':
            if CohereEmbeddings is None:
                raise ImportError("langchain-cohere is not installed. Please run 'pip install -U langchain-cohere'.")
            cohere_api_key = self.config.get('cohere_api_key') or os.getenv('COHERE_API_KEY')
            model_name = self.config.get('model_name', 'embed-multilingual-v3.0')
            user_agent = self.config.get('user_agent', 'postgres-rag-app/1.0')
            if not cohere_api_key:
                raise ValueError("COHERE_API_KEY environment variable or config required for Cohere embeddings")
            self.embeddings = CohereEmbeddings(
                model=model_name,
                cohere_api_key=cohere_api_key,
                user_agent=user_agent
            )
        else:
            raise ValueError(f"Unsupported embedding model: {self.config['embedding_model']}")

        # Test embeddings
        test_text = "This is a test document for embeddings."
        test_embedding = self.embeddings.embed_query(test_text)
        self.logger.info(f"✅ Embeddings initialized with dimension: {len(test_embedding)}")

    def create_vectorstore(self):
        """Create or connect to PGVector store."""
        self.logger.info(f"Creating PGVector store: {self.config['collection_name']}")

        try:
            self.vectorstore = PGVector(
                connection_string=self.connection_string,
                embedding_function=self.embeddings,
                collection_name=self.config['collection_name'],
                pre_delete_collection=self.config['overwrite']
            )
            self.logger.info("✅ PGVector store created successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to create vector store: {e}")

    def load_documents(self) -> List[Document]:
        """Load documents from input directory."""
        self.logger.info(f"Loading documents from: {self.config['input_dir']}")

        try:
            loader = DirectoryLoader(
                str(self.config['input_dir']),
                glob=self.config['file_pattern'],
                loader_cls=TextLoader,
                show_progress=self.config['verbose']
            )
            documents = loader.load()

            if not documents:
                self.logger.warning("No documents found in input directory")
                return []

            self.logger.info(f"✅ Loaded {len(documents)} documents")

            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    'ingestion_timestamp': datetime.now().isoformat(),
                    'collection': self.config['collection_name']
                })

            return documents

        except Exception as e:
            raise RuntimeError(f"Failed to load documents: {e}")

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        if not documents:
            return []

        self.logger.info("Splitting documents into chunks...")
        self.logger.info(f"Chunk size: {self.config['chunk_size']}, Overlap: {self.config['chunk_overlap']}")

        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config['chunk_size'],
                chunk_overlap=self.config['chunk_overlap'],
                separators=["\n\n", "\n", " ", ""],
                length_function=len
            )

            split_documents = text_splitter.split_documents(documents)

            # Statistics
            chunk_lengths = [len(doc.page_content) for doc in split_documents]
            self.logger.info(f"✅ Split {len(documents)} documents into {len(split_documents)} chunks")
            self.logger.info(f"Average chunk length: {sum(chunk_lengths) / len(chunk_lengths):.0f} characters")
            self.logger.info(f"Min chunk length: {min(chunk_lengths)} characters")
            self.logger.info(f"Max chunk length: {max(chunk_lengths)} characters")

            return split_documents

        except Exception as e:
            self.logger.error(f"Failed to split documents: {e}")
            return documents  # Return original documents as fallback

    def ingest_documents(self, documents: List[Document]):
        """Ingest documents into vector store."""
        if not documents:
            self.logger.warning("No documents to ingest")
            return

        self.logger.info(f"🚀 Starting ingestion of {len(documents)} documents...")

        try:
            batch_size = self.config['batch_size']
            total_added = 0

            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                self.vectorstore.add_documents(batch)
                total_added += len(batch)

                if self.config['verbose']:
                    self.logger.info(f"✅ Added batch {i//batch_size + 1}: {len(batch)} documents (Total: {total_added})")
                else:
                    print(f"Progress: {total_added}/{len(documents)} documents ingested", end='\r')

            print()  # New line after progress
            self.logger.info(f"🎉 Successfully ingested {total_added} documents!")
            self.logger.info(f"📚 Collection: {self.config['collection_name']}")
            self.logger.info(f"🤖 Embedding model: {self.embeddings.model_name}")

        except Exception as e:
            raise RuntimeError(f"Failed to ingest documents: {e}")

    def verify_ingestion(self):
        """Verify ingestion by running test queries."""
        self.logger.info("🔍 Verifying ingestion...")

        test_queries = [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "What is natural language processing?"
        ]

        for query in test_queries:
            try:
                results = self.vectorstore.similarity_search(query, k=2)
                self.logger.info(f"Query: '{query}' -> {len(results)} results")

                if self.config['verbose']:
                    for i, doc in enumerate(results, 1):
                        self.logger.debug(f"  {i}. {doc.metadata.get('source', 'Unknown')}: {doc.page_content[:50]}...")

            except Exception as e:
                self.logger.error(f"Query failed for '{query}': {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get collection statistics."""
        stats = {}

        try:
            db_config = self.config['database']
            conn = psycopg2.connect(**db_config)
            cursor = conn.cursor()

            # Get document count
            cursor.execute("""
                SELECT COUNT(*)
                FROM langchain_pg_embedding
                WHERE collection_id = (
                    SELECT uuid FROM langchain_pg_collection WHERE name = %s
                )
            """, (self.config['collection_name'],))

            doc_count = cursor.fetchone()[0]
            stats['document_count'] = doc_count

            # Get embedding dimension
            cursor.execute("""
                SELECT embedding
                FROM langchain_pg_embedding
                WHERE collection_id = (
                    SELECT uuid FROM langchain_pg_collection WHERE name = %s
                )
                LIMIT 1
            """, (self.config['collection_name'],))

            sample_embedding = cursor.fetchone()
            if sample_embedding:
                stats['embedding_dimension'] = len(sample_embedding[0])

            cursor.close()
            conn.close()

        except Exception as e:
            self.logger.error(f"Failed to get statistics: {e}")

        return stats

    def get_all_chunks(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Get all chunks from the vector store."""
        chunks = []

        try:
            db_config = self.config['database']
            conn = psycopg2.connect(**db_config)
            cursor = conn.cursor()

            # Get chunks with metadata
            cursor.execute("""
                SELECT
                    e.uuid,
                    e.document,
                    e.cmetadata,
                    e.embedding
                FROM langchain_pg_embedding e
                WHERE e.collection_id = (
                    SELECT uuid FROM langchain_pg_collection WHERE name = %s
                )
                ORDER BY e.uuid
                LIMIT %s OFFSET %s
            """, (self.config['collection_name'], limit, offset))

            rows = cursor.fetchall()

            for row in rows:
                chunk_id, document, metadata, embedding = row
                
                # Extract document name from metadata
                document_name = 'Unknown'
                if metadata and 'source' in metadata:
                    document_name = metadata['source']
                
                # Get embedding dimension and handle different formats
                embedding_dim = 384  # Default embedding dimension
                embedding_data = None
                
                if embedding:
                    if hasattr(embedding, 'tolist'):
                        # It's a numpy array
                        embedding_data = embedding.tolist()
                        embedding_dim = len(embedding_data)
                    elif isinstance(embedding, str):
                        # It's stored as a string, try to parse it
                        try:
                            # If it's a string representation of a list/array
                            if embedding.startswith('[') and embedding.endswith(']'):
                                import ast
                                embedding_data = ast.literal_eval(embedding)
                                embedding_dim = len(embedding_data) if isinstance(embedding_data, list) else 384
                            else:
                                # Just use default dimension
                                embedding_dim = 384
                        except:
                            # If parsing fails, use default
                            embedding_dim = 384
                    else:
                        # Unknown format, use default
                        embedding_dim = 384
                
                chunks.append({
                    'id': str(chunk_id),
                    'document_name': document_name,
                    'content': document,
                    'metadata': metadata or {},
                    'embedding': embedding_data,
                    'embedding_dim': embedding_dim,
                    'content_preview': document[:200] + '...' if len(document) > 200 else document
                })

            cursor.close()
            conn.close()

        except Exception as e:
            self.logger.error(f"Failed to get chunks: {e}")

        return chunks

    def get_single_chunk(self, chunk_id: str) -> Dict[str, Any]:
        """Get a single chunk by its ID."""
        try:
            db_config = self.config['database']
            conn = psycopg2.connect(**db_config)
            cursor = conn.cursor()

            # Get chunk with metadata
            cursor.execute("""
                SELECT
                    e.uuid,
                    e.document,
                    e.cmetadata,
                    e.embedding
                FROM langchain_pg_embedding e
                WHERE e.collection_id = (
                    SELECT uuid FROM langchain_pg_collection WHERE name = %s
                ) AND e.uuid = %s
            """, (self.config['collection_name'], chunk_id))

            row = cursor.fetchone()

            if not row:
                return None

            chunk_uuid, document, metadata, embedding = row
            
            # Extract document name from metadata
            document_name = 'Unknown'
            if metadata and 'source' in metadata:
                document_name = metadata['source']
            
            # Get embedding dimension and handle different formats
            embedding_dim = 384  # Default embedding dimension
            embedding_data = None
            
            if embedding:
                if hasattr(embedding, 'tolist'):
                    # It's a numpy array
                    embedding_data = embedding.tolist()
                    embedding_dim = len(embedding_data)
                elif isinstance(embedding, str):
                    # It's stored as a string, try to parse it
                    try:
                        # If it's a string representation of a list/array
                        if embedding.startswith('[') and embedding.endswith(']'):
                            import ast
                            embedding_data = ast.literal_eval(embedding)
                            embedding_dim = len(embedding_data) if isinstance(embedding_data, list) else 384
                        else:
                            # Just use default dimension
                            embedding_dim = 384
                    except:
                        # If parsing fails, use default
                        embedding_dim = 384
                else:
                    # Unknown format, use default
                    embedding_dim = 384
            
            chunk_data = {
                'id': str(chunk_uuid),
                'document_name': document_name,
                'content': document,
                'metadata': metadata or {},
                'embedding': embedding_data,
                'embedding_dim': embedding_dim,
                'content_preview': document[:200] + '...' if len(document) > 200 else document
            }

            cursor.close()
            conn.close()

            return chunk_data

        except Exception as e:
            self.logger.error(f"Failed to get chunk {chunk_id}: {e}")
            return None

    def get_chunk_count(self) -> int:
        """Get total number of chunks in the collection."""
        try:
            db_config = self.config['database']
            conn = psycopg2.connect(**db_config)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT COUNT(*)
                FROM langchain_pg_embedding
                WHERE collection_id = (
                    SELECT uuid FROM langchain_pg_collection WHERE name = %s
                )
            """, (self.config['collection_name'],))

            count = cursor.fetchone()[0]

            cursor.close()
            conn.close()

            return count

        except Exception as e:
            self.logger.error(f"Failed to get chunk count: {e}")
            return 0

    def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a chunk by its UUID."""
        try:
            db_config = self.config['database']
            conn = psycopg2.connect(**db_config)
            cursor = conn.cursor()

            # Delete the chunk
            cursor.execute("""
                DELETE FROM langchain_pg_embedding
                WHERE uuid = %s AND collection_id = (
                    SELECT uuid FROM langchain_pg_collection WHERE name = %s
                )
            """, (chunk_id, self.config['collection_name']))

            deleted = cursor.rowcount > 0
            conn.commit()

            cursor.close()
            conn.close()

            if deleted:
                self.logger.info(f"Deleted chunk {chunk_id}")
            else:
                self.logger.warning(f"Chunk {chunk_id} not found")

            return deleted

        except Exception as e:
            self.logger.error(f"Failed to delete chunk {chunk_id}: {e}")
            return False

    def search_chunks(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search chunks using similarity search."""
        try:
            if not self.vectorstore:
                self.validate_config()
                self.setup_database_connection()
                self.setup_embeddings()
                self.create_vectorstore()

            results = self.vectorstore.similarity_search_with_score(query, k=limit)

            chunks = []
            for doc, score in results:
                # Extract document name from metadata
                document_name = 'Unknown'
                if doc.metadata and 'source' in doc.metadata:
                    document_name = doc.metadata['source']
                
                chunks.append({
                    'id': str(doc.metadata.get('chunk_id', 'N/A')) if doc.metadata else 'N/A',
                    'document_name': document_name,
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': float(score),
                    'embedding_dim': 384,  # Default embedding dimension
                    'content_preview': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content
                })

            return chunks

        except Exception as e:
            self.logger.error(f"Failed to search chunks: {e}")
            return []

    def get_catalog_documents(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Get all unique documents (by source) from the vector store."""
        documents = []

        try:
            db_config = self.config['database']
            conn = psycopg2.connect(**db_config)
            cursor = conn.cursor()

            # Get unique documents by source with metadata (get one sample per source)
            cursor.execute("""
                SELECT DISTINCT ON (e.cmetadata->>'source')
                    e.cmetadata->>'source' as source,
                    COUNT(*) OVER (PARTITION BY e.cmetadata->>'source') as chunk_count,
                    SUM(LENGTH(e.document)) OVER (PARTITION BY e.cmetadata->>'source') as total_chars,
                    MIN(e.cmetadata->>'ingestion_timestamp') OVER (PARTITION BY e.cmetadata->>'source') as first_ingested,
                    MAX(e.cmetadata->>'ingestion_timestamp') OVER (PARTITION BY e.cmetadata->>'source') as last_ingested,
                    e.cmetadata as full_metadata
                FROM langchain_pg_embedding e
                WHERE e.collection_id = (
                    SELECT uuid FROM langchain_pg_collection WHERE name = %s
                )
                AND e.cmetadata->>'source' IS NOT NULL
                ORDER BY e.cmetadata->>'source', e.cmetadata->>'ingestion_timestamp' DESC
                LIMIT %s OFFSET %s
            """, (self.config['collection_name'], limit, offset))

            rows = cursor.fetchall()

            for row in rows:
                source, chunk_count, total_chars, first_ingested, last_ingested, full_metadata = row
                documents.append({
                    'filename': source.split('/')[-1] if '/' in source else source,
                    'upload_date': last_ingested or 'N/A',
                    'chunk_count': chunk_count,
                    'size': f"{total_chars} chars",
                    'source': source,
                    'total_chars': total_chars,
                    'avg_chunk_size': total_chars // chunk_count if chunk_count > 0 else 0,
                    'first_ingested': first_ingested,
                    'last_ingested': last_ingested,
                    'metadata': full_metadata or {}
                })

            cursor.close()
            conn.close()

        except Exception as e:
            self.logger.error(f"Failed to get catalog documents: {e}")

        return documents

    def delete_document(self, source: str) -> bool:
        """Delete all chunks for a specific document source."""
        try:
            db_config = self.config['database']
            conn = psycopg2.connect(**db_config)
            cursor = conn.cursor()

            # Delete all chunks for this source
            cursor.execute("""
                DELETE FROM langchain_pg_embedding
                WHERE collection_id = (
                    SELECT uuid FROM langchain_pg_collection WHERE name = %s
                )
                AND cmetadata->>'source' = %s
            """, (self.config['collection_name'], source))

            deleted_count = cursor.rowcount
            conn.commit()

            cursor.close()
            conn.close()

            if deleted_count > 0:
                self.logger.info(f"Deleted {deleted_count} chunks for document {source}")
                return True
            else:
                self.logger.warning(f"No chunks found for document {source}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to delete document {source}: {e}")
            return False
        """Get all chunks for a specific document source."""
        chunks = []

        try:
            db_config = self.config['database']
            conn = psycopg2.connect(**db_config)
            cursor = conn.cursor()

            # Get chunks for this source
            cursor.execute("""
                SELECT
                    e.uuid,
                    e.document,
                    e.cmetadata,
                    LENGTH(e.document) as content_length
                FROM langchain_pg_embedding e
                WHERE e.collection_id = (
                    SELECT uuid FROM langchain_pg_collection WHERE name = %s
                )
                AND e.cmetadata->>'source' = %s
                ORDER BY e.uuid
                LIMIT %s OFFSET %s
            """, (self.config['collection_name'], source, limit, offset))

            rows = cursor.fetchall()

            for row in rows:
                chunk_id, document, metadata, content_length = row
                chunks.append({
                    'id': str(chunk_id),
                    'content': document,
                    'metadata': metadata or {},
                    'content_length': content_length,
                    'content_preview': document[:200] + '...' if len(document) > 200 else document
                })

            cursor.close()
            conn.close()

        except Exception as e:
            self.logger.error(f"Failed to get document chunks: {e}")

        return chunks

    def get_catalog_stats(self) -> Dict[str, Any]:
        """Get catalog statistics."""
        stats = {}

        try:
            db_config = self.config['database']
            conn = psycopg2.connect(**db_config)
            cursor = conn.cursor()

            # Total documents
            cursor.execute("""
                SELECT COUNT(DISTINCT e.cmetadata->>'source')
                FROM langchain_pg_embedding e
                WHERE e.collection_id = (
                    SELECT uuid FROM langchain_pg_collection WHERE name = %s
                )
                AND e.cmetadata->>'source' IS NOT NULL
            """, (self.config['collection_name'],))

            stats['total_documents'] = cursor.fetchone()[0]

            # Total chunks
            cursor.execute("""
                SELECT COUNT(*)
                FROM langchain_pg_embedding
                WHERE collection_id = (
                    SELECT uuid FROM langchain_pg_collection WHERE name = %s
                )
            """, (self.config['collection_name'],))

            stats['total_chunks'] = cursor.fetchone()[0]

            # Total characters
            cursor.execute("""
                SELECT SUM(LENGTH(e.document))
                FROM langchain_pg_embedding e
                WHERE e.collection_id = (
                    SELECT uuid FROM langchain_pg_collection WHERE name = %s
                )
            """, (self.config['collection_name'],))

            result = cursor.fetchone()[0]
            stats['total_characters'] = result if result else 0

            # Average chunk size
            if stats['total_chunks'] > 0:
                stats['avg_chunk_size'] = stats['total_characters'] // stats['total_chunks']
            else:
                stats['avg_chunk_size'] = 0

            cursor.close()
            conn.close()

        except Exception as e:
            self.logger.error(f"Failed to get catalog stats: {e}")

        return stats

    def save_metadata(self):
        """Save ingestion metadata to file."""
        # Ensure input_dir is a Path object
        input_dir = Path(self.config['input_dir']) if isinstance(self.config['input_dir'], str) else self.config['input_dir']

        metadata = {
            "ingestion_timestamp": datetime.now().isoformat(),
            "collection_name": self.config['collection_name'],
            "embedding_model": self.config['embedding_model'],
            "model_name": self.config['model_name'],
            "chunk_size": self.config['chunk_size'],
            "chunk_overlap": self.config['chunk_overlap'],
            "input_directory": str(input_dir),
            "file_pattern": self.config['file_pattern'],
            "batch_size": self.config['batch_size']
        }

        metadata_file = input_dir / f"{self.config['collection_name']}_metadata.json"
        try:
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            self.logger.info(f"📋 Metadata saved to: {metadata_file}")
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {e}")

    def run(self):
        """Run the complete ingestion pipeline."""
        try:
            self.logger.info("🚀 Starting text ingestion pipeline...")

            # Setup
            self.validate_config()
            self.setup_database_connection()
            self.setup_embeddings()
            self.create_vectorstore()

            # Process documents
            documents = self.load_documents()
            split_documents = self.split_documents(documents)
            self.ingest_documents(split_documents)

            # Verify and cleanup
            self.verify_ingestion()
            stats = self.get_statistics()
            self.save_metadata()

            self.logger.info("✅ Ingestion pipeline completed successfully!")
            self.logger.info(f"📊 Final statistics: {stats}")

        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {e}")
            sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest text files into PGVector store for RAG applications",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingestion.py --input-dir ./text_files --collection my_docs
  python ingestion.py --input-dir ./docs --collection tech_docs --model openai
  python ingestion.py --input-dir ./data --collection research --chunk-size 500 --verbose
        """
    )

    # Input/Output options
    parser.add_argument('--input-dir', '-i', type=Path, default=Path('./text_files'),
                       help='Directory containing text files (default: ./text_files)')
    parser.add_argument('--collection', '-c', dest='collection_name', default='text_ingestion_docs',
                       help='Name of the PGVector collection (default: text_ingestion_docs)')
    parser.add_argument('--file-pattern', '-p', default='**/*.txt',
                       help='Glob pattern for files to ingest (default: **/*.txt)')

    # Processing options
    parser.add_argument('--chunk-size', type=int, default=1000,
                       help='Size of document chunks in characters (default: 1000)')
    parser.add_argument('--chunk-overlap', type=int, default=200,
                       help='Overlap between chunks in characters (default: 200)')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Number of documents to process in each batch (default: 100)')

    # Embedding options
    parser.add_argument('--model', dest='embedding_model', choices=['sentence-transformer', 'openai', 'cohere'],
                       default='sentence-transformer', help='Embedding model to use (sentence-transformer, openai, cohere; default: sentence-transformer)')
    parser.add_argument('--model-name', default='all-MiniLM-L6-v2',
                       help='Model name for sentence transformer (default: all-MiniLM-L6-v2)')

    # Database options
    parser.add_argument('--db-host', default='localhost', help='Database host (default: localhost)')
    parser.add_argument('--db-port', type=int, default=5432, help='Database port (default: 5432)')
    parser.add_argument('--db-name', default='rag_db', help='Database name (default: rag_db)')
    parser.add_argument('--db-user', default='rag_user', help='Database user (default: rag_user)')
    parser.add_argument('--db-password', default='rag_password', help='Database password (default: rag_password)')

    # Control options
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing collection if it exists')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--log-file', action='store_true',
                       help='Save logs to file (ingestion.log)')

    args = parser.parse_args()

    # Build configuration
    config = {
        'input_dir': Path(args.input_dir),
        'collection_name': args.collection_name,
        'file_pattern': args.file_pattern,
        'chunk_size': args.chunk_size,
        'chunk_overlap': args.chunk_overlap,
        'batch_size': args.batch_size,
        'embedding_model': args.embedding_model,
        'model_name': args.model_name,
        'database': {
            'host': args.db_host,
            'port': args.db_port,
            'database': args.db_name,
            'user': args.db_user,
            'password': args.db_password
        },
        'overwrite': args.overwrite,
        'verbose': args.verbose,
        'log_file': args.log_file
    }

    # Create input directory if it doesn't exist
    config['input_dir'].mkdir(exist_ok=True)

    # Run pipeline
    pipeline = TextIngestionPipeline(config)
    pipeline.setup_logging()
    pipeline.run()


if __name__ == "__main__":
    main()
