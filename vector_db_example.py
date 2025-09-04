#!/usr/bin/env python3
"""
PostgreSQL Vector Database Connection Example
This script demonstrates how to connect to the PostgreSQL database
and work with vector embeddings using pgvector.
"""

import psycopg2
import numpy as np
from typing import List, Dict, Any
import json

class VectorDatabase:
    def __init__(self, host="localhost", port="5432", database="rag_db",
                 user="rag_user", password="rag_password"):
        self.conn_params = {
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "password": password
        }
        self.conn = None

    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(**self.conn_params)
            print("✅ Connected to PostgreSQL database")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False

    def disconnect(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("✅ Database connection closed")

    def create_sample_data(self):
        """Create sample vector data for testing"""
        if not self.conn:
            print("❌ Not connected to database")
            return

        try:
            cursor = self.conn.cursor()

            # Generate random embeddings
            embeddings = []
            for i in range(5):
                # Create random 1536-dimensional vector (OpenAI ada-002 size)
                embedding = np.random.rand(1536).tolist()
                embeddings.append(embedding)

            # Insert sample documents with embeddings
            sample_docs = [
                "This is a sample document about machine learning",
                "Another document discussing artificial intelligence",
                "A third document about natural language processing",
                "Document four covers computer vision topics",
                "The fifth document is about data science"
            ]

            for i, (doc, embedding) in enumerate(zip(sample_docs, embeddings)):
                cursor.execute("""
                    INSERT INTO document_embeddings (content, embedding, metadata)
                    VALUES (%s, %s::VECTOR, %s)
                """, (doc, embedding, {"id": i+1, "type": "sample"}))

            self.conn.commit()
            print("✅ Sample vector data created successfully")

        except Exception as e:
            print(f"❌ Error creating sample data: {e}")
            self.conn.rollback()

    def search_similar(self, query_embedding: List[float], limit: int = 5):
        """Search for similar documents using vector similarity"""
        if not self.conn:
            print("❌ Not connected to database")
            return []

        try:
            cursor = self.conn.cursor()

            # Convert embedding to PostgreSQL vector format
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'

            cursor.execute("""
                SELECT content, metadata,
                       1 - (embedding <=> %s::VECTOR) as similarity
                FROM document_embeddings
                ORDER BY embedding <=> %s::VECTOR
                LIMIT %s
            """, (embedding_str, embedding_str, limit))

            results = cursor.fetchall()
            return results

        except Exception as e:
            print(f"❌ Error searching: {e}")
            return []

    def get_table_info(self):
        """Get information about vector tables"""
        if not self.conn:
            print("❌ Not connected to database")
            return

        try:
            cursor = self.conn.cursor()

            # Check vector tables
            cursor.execute("""
                SELECT table_name, table_schema
                FROM information_schema.tables
                WHERE table_name IN ('sample_data', 'document_embeddings')
                AND table_schema = 'public'
            """)

            tables = cursor.fetchall()
            print("📊 Vector Tables:")
            for table in tables:
                print(f"  - {table[1]}.{table[0]}")

            # Check vector extension
            cursor.execute("""
                SELECT name, default_version, installed_version
                FROM pg_available_extensions
                WHERE name = 'vector'
            """)

            extension = cursor.fetchone()
            if extension:
                print(f"\n🔧 pgvector Extension: {extension[2]} (available: {extension[1]})")
            else:
                print("\n⚠️  pgvector extension not found")

        except Exception as e:
            print(f"❌ Error getting table info: {e}")


def main():
    """Main function demonstrating vector database operations"""
    print("🚀 PostgreSQL Vector Database Example")
    print("=" * 50)

    # Initialize database connection
    db = VectorDatabase()

    # Connect to database
    if not db.connect():
        return

    try:
        # Get database information
        db.get_table_info()

        print("\n" + "=" * 50)

        # Create sample data
        print("📝 Creating sample vector data...")
        db.create_sample_data()

        print("\n" + "=" * 50)

        # Perform similarity search
        print("🔍 Performing similarity search...")
        query_embedding = np.random.rand(1536).tolist()  # Random query vector

        results = db.search_similar(query_embedding, limit=3)

        if results:
            print("📋 Top similar documents:")
            for i, (content, metadata, similarity) in enumerate(results, 1):
                print(".4f")
        else:
            print("❌ No results found")

    finally:
        # Always close the connection
        db.disconnect()


if __name__ == "__main__":
    main()
