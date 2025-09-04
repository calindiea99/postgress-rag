# Quick Connection Guide

## Setup Python Environment
```bash
# Activate the virtual environment
./activate_env.sh

# Or manually:
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"
pyenv activate rag-env
```

## Start Services
```bash
cd /home/calin-unix/Projects/postgress-rag
docker-compose up -d
```

## Access pgAdmin
- **URL**: http://localhost:5050
- **Email**: admin@rag.com
- **Password**: admin_password

## Register PostgreSQL Server in pgAdmin
- **Host name/address**: `postgres` (use service name, not localhost)
- **Port**: `5432`
- **Maintenance database**: `rag_db`
- **Username**: `rag_user`
- **Password**: `rag_password`

## Vector Data Storage
Vector embeddings are stored in:
- **`sample_data.embedding`**: VECTOR(1536) column for general embeddings
- **`document_embeddings.embedding`**: Dedicated table for document vectors with IVFFlat index

## Example Vector Queries
```sql
-- Find similar documents
SELECT content, 1 - (embedding <=> '[0.1, 0.2, 0.3]'::VECTOR) as similarity
FROM document_embeddings
ORDER BY embedding <=> '[0.1, 0.2, 0.3]'::VECTOR
LIMIT 5;
```

## Python Database Connection
```python
import psycopg2
import numpy as np

# Connect to database
conn = psycopg2.connect(
    host="localhost",
    port="5432",
    database="rag_db",
    user="rag_user",
    password="rag_password"
)

# Example: Insert vector embedding
embedding = np.random.rand(1536).tolist()
cursor = conn.cursor()
cursor.execute("""
    INSERT INTO document_embeddings (content, embedding, metadata)
    VALUES (%s, %s::VECTOR, %s)
""", ("Sample document", embedding, {"source": "python"}))

conn.commit()
cursor.close()
conn.close()
```

## Direct Database Connection
```bash
psql -h localhost -p 5432 -U rag_user -d rag_db
```

## Stop Services
```bash
docker-compose down
```
