"""
ChromaDB Service Client
Easy-to-use client library for integrating with the ChromaDB service
"""

import requests
from typing import List, Dict, Any, Optional


class ChromaDBClient:
    """Client for interacting with ChromaDB Service"""

    def __init__(self, base_url: str, timeout: int = 30):
        """
        Initialize the client

        Args:
            base_url: Base URL of the ChromaDB service (e.g., https://your-service.onrender.com)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout

    def health_check(self) -> Dict[str, Any]:
        """Check if the service is healthy"""
        response = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information"""
        response = requests.get(f"{self.base_url}/info", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def query_registry(self, query: str, n_results: int = 3) -> Dict[str, Any]:
        """
        Query the registry for relevant tables

        Args:
            query: Natural language query
            n_results: Number of results to return (1-20)

        Returns:
            Query results with documents, metadata, and distances
        """
        response = requests.post(
            f"{self.base_url}/query",
            json={"query": query, "n_results": n_results},
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def list_tables(self) -> List[str]:
        """Get list of all registered table names"""
        response = requests.get(f"{self.base_url}/registry/tables", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def add_table(self, table_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add a new table to the registry

        Args:
            table_schema: Complete table schema dictionary

        Returns:
            Success response with table name
        """
        response = requests.post(
            f"{self.base_url}/registry/table",
            json={"table_schema": table_schema},
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def delete_table(self, table_name: str) -> Dict[str, Any]:
        """
        Delete a table from the registry

        Args:
            table_name: Name of the table to delete

        Returns:
            Success response
        """
        response = requests.delete(
            f"{self.base_url}/registry/table/{table_name}",
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def reinitialize_registry(self) -> Dict[str, Any]:
        """Reinitialize the entire registry"""
        response = requests.post(
            f"{self.base_url}/registry/reinitialize",
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def add_document(
            self,
            doc_id: str,
            document: str,
            metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add a custom document to the collection

        Args:
            doc_id: Unique document identifier
            document: Document text content
            metadata: Optional metadata dictionary

        Returns:
            Success response with document ID
        """
        response = requests.post(
            f"{self.base_url}/documents/add",
            json={
                "id": doc_id,
                "document": document,
                "metadata": metadata
            },
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def get_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Get a document by ID

        Args:
            doc_id: Document identifier

        Returns:
            Document with metadata
        """
        response = requests.get(
            f"{self.base_url}/documents/{doc_id}",
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Delete a document by ID

        Args:
            doc_id: Document identifier

        Returns:
            Success response
        """
        response = requests.delete(
            f"{self.base_url}/documents/{doc_id}",
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def reset_collection(self) -> Dict[str, Any]:
        """Reset the entire collection (use with caution!)"""
        response = requests.post(
            f"{self.base_url}/admin/reset",
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()


# Example Usage
if __name__ == "__main__":
    # Initialize client
    client = ChromaDBClient("http://localhost:8000")

    # Health check
    print("Health Check:", client.health_check())

    # Query registry
    results = client.query_registry("find patient medication records", n_results=2)
    print("\nQuery Results:")
    for result in results['results']:
        print(f"  - {result['metadata']['table_name']} (distance: {result['distance']:.4f})")

    # List all tables
    tables = client.list_tables()
    print(f"\nRegistered Tables ({len(tables)}):", tables)