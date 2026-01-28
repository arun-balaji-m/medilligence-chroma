"""
Pydantic models for request and response validation
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any


class QueryRequest(BaseModel):
    """Request model for querying the registry"""
    query: str = Field(..., description="Natural language query")
    n_results: int = Field(default=3, ge=1, le=20, description="Number of results to return")


class QueryResult(BaseModel):
    """Individual query result"""
    document: str
    metadata: Dict[str, Any]
    distance: float


class QueryResponse(BaseModel):
    """Response model for query results"""
    query: str
    results: List[QueryResult]


class TableSchemaRequest(BaseModel):
    """Request model for adding a table schema"""
    table_schema: Dict[str, Any] = Field(..., description="Complete table schema dictionary")


class AddDocumentRequest(BaseModel):
    """Request model for adding a custom document"""
    id: str = Field(..., description="Unique document identifier")
    document: str = Field(..., description="Document text content")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata")


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    chroma_connected: bool
    collection_name: str
    document_count: int


class CollectionInfo(BaseModel):
    """Response model for collection information"""
    collection_name: str
    document_count: int
    embedding_model: str
    persist_directory: str