"""
ChromaDB Service - Standalone Vector Database API
Provides REST endpoints for managing ChromaDB collections and embeddings
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
from typing import List, Dict, Optional
import os

from app.database import (
    chroma_client,
    registry_collection,
    embedding_model,
    health_check
)
from app.models import (
    AddDocumentRequest,
    QueryRequest,
    QueryResponse,
    TableSchemaRequest,
    HealthResponse,
    CollectionInfo
)
from app.registry import (
    initialize_registry,
    add_table_to_registry,
    list_registered_tables,
    _create_document_from_schema
)
from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    print("🚀 Starting ChromaDB Service...")
    print(f"📁 ChromaDB Directory: {settings.CHROMA_PERSIST_DIRECTORY}")
    print(f"🔧 Collection Name: {settings.CHROMA_COLLECTION_NAME}")

    # Initialize registry on startup
    initialize_registry()

    yield

    print("👋 Shutting down ChromaDB Service...")


app = FastAPI(
    title="ChromaDB Service",
    description="Standalone ChromaDB vector database service with REST API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# HEALTH & INFO ENDPOINTS
# ==========================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "service": "ChromaDB Service",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    db_healthy = health_check()

    return HealthResponse(
        status="healthy" if db_healthy else "unhealthy",
        chroma_connected=db_healthy,
        collection_name=settings.CHROMA_COLLECTION_NAME,
        document_count=registry_collection.count() if db_healthy else 0
    )


@app.get("/info", response_model=CollectionInfo)
async def collection_info():
    """Get collection information"""
    try:
        count = registry_collection.count()
        return CollectionInfo(
            collection_name=settings.CHROMA_COLLECTION_NAME,
            document_count=count,
            embedding_model=settings.EMBEDDING_MODEL,
            persist_directory=settings.CHROMA_PERSIST_DIRECTORY
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get collection info: {str(e)}"
        )


# ==========================================
# REGISTRY ENDPOINTS
# ==========================================

@app.get("/registry/tables", response_model=List[str])
async def get_registered_tables():
    """Get list of all registered table names"""
    try:
        tables = list_registered_tables()
        return tables
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get registered tables: {str(e)}"
        )


@app.post("/registry/table", status_code=status.HTTP_201_CREATED)
async def add_table(request: TableSchemaRequest):
    """Add a new table schema to the registry"""
    try:
        success = add_table_to_registry(request.table_schema)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to add table to registry"
            )

        return {
            "message": f"Table '{request.table_schema['table_name']}' added successfully",
            "table_name": request.table_schema['table_name']
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add table: {str(e)}"
        )


@app.delete("/registry/table/{table_name}", status_code=status.HTTP_200_OK)
async def delete_table(table_name: str):
    """Delete a table from the registry"""
    try:
        registry_collection.delete(ids=[f"table_{table_name}"])
        return {
            "message": f"Table '{table_name}' deleted successfully",
            "table_name": table_name
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete table: {str(e)}"
        )


@app.post("/registry/reinitialize", status_code=status.HTTP_200_OK)
async def reinitialize_registry():
    """Reinitialize the registry (clear and reload all tables)"""
    try:
        # Clear existing registry
        registry_collection.delete(
            where={"table_name": {"$ne": ""}}
        )

        # Reinitialize
        success = initialize_registry()

        return {
            "message": "Registry reinitialized successfully",
            "initialized": success,
            "table_count": registry_collection.count()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reinitialize registry: {str(e)}"
        )


# ==========================================
# QUERY ENDPOINTS
# ==========================================

@app.post("/query", response_model=QueryResponse)
async def query_registry(request: QueryRequest):
    """Query the registry for relevant tables"""
    try:
        # Generate embedding for the query
        query_embedding = embedding_model.encode(request.query).tolist()

        # Query ChromaDB
        results = registry_collection.query(
            query_embeddings=[query_embedding],
            n_results=request.n_results
        )

        # Format response
        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        distances = results.get('distances', [[]])[0]

        return QueryResponse(
            query=request.query,
            results=[
                {
                    "document": doc,
                    "metadata": meta,
                    "distance": dist
                }
                for doc, meta, dist in zip(documents, metadatas, distances)
            ]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}"
        )


# ==========================================
# DOCUMENT MANAGEMENT ENDPOINTS
# ==========================================

@app.post("/documents/add", status_code=status.HTTP_201_CREATED)
async def add_document(request: AddDocumentRequest):
    """Add a custom document to the collection"""
    try:
        # Generate embedding
        embedding = embedding_model.encode(request.document).tolist()

        # Add to collection
        registry_collection.add(
            documents=[request.document],
            embeddings=[embedding],
            metadatas=[request.metadata] if request.metadata else None,
            ids=[request.id]
        )

        return {
            "message": "Document added successfully",
            "id": request.id
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add document: {str(e)}"
        )


@app.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    """Get a document by ID"""
    try:
        result = registry_collection.get(ids=[doc_id])

        if not result['documents']:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with id '{doc_id}' not found"
            )

        return {
            "id": doc_id,
            "document": result['documents'][0],
            "metadata": result['metadatas'][0] if result['metadatas'] else None
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get document: {str(e)}"
        )


@app.delete("/documents/{doc_id}", status_code=status.HTTP_200_OK)
async def delete_document(doc_id: str):
    """Delete a document by ID"""
    try:
        registry_collection.delete(ids=[doc_id])
        return {
            "message": f"Document '{doc_id}' deleted successfully",
            "id": doc_id
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )


# ==========================================
# ADMIN ENDPOINTS
# ==========================================

@app.post("/admin/reset", status_code=status.HTTP_200_OK)
async def reset_collection():
    """Reset the entire collection (use with caution!)"""
    try:
        # Get all IDs
        all_data = registry_collection.get()
        if all_data['ids']:
            registry_collection.delete(ids=all_data['ids'])

        return {
            "message": "Collection reset successfully",
            "documents_deleted": len(all_data['ids']) if all_data['ids'] else 0
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset collection: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=settings.DEBUG
    )