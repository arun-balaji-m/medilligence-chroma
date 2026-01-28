"""
ChromaDB database client and configuration
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
import os
from app.config import settings

# Disable ChromaDB telemetry completely
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

# Ensure ChromaDB directory exists
os.makedirs(settings.CHROMA_PERSIST_DIRECTORY, exist_ok=True)

# ChromaDB Client (Persistent) - Disable telemetry
chroma_client = chromadb.PersistentClient(
    path=settings.CHROMA_PERSIST_DIRECTORY,
    settings=ChromaSettings(
        anonymized_telemetry=False,
        allow_reset=True
    )
)

# Get or create collection
registry_collection = chroma_client.get_or_create_collection(
    name=settings.CHROMA_COLLECTION_NAME,
    metadata={"description": "Database table schemas and metadata"}
)

# Embedding Model (Local)
print(f"🔄 Loading embedding model: {settings.EMBEDDING_MODEL}")
embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
print(f"✅ Embedding model loaded successfully")


def health_check() -> bool:
    """
    Check if ChromaDB is accessible and working
    """
    try:
        # Try to get collection count
        registry_collection.count()
        return True
    except Exception as e:
        print(f"❌ ChromaDB health check failed: {e}")
        return False