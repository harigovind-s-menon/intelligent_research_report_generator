"""Vector store using PostgreSQL with pgvector extension.

This module provides:
- Storage of documents with embeddings
- Semantic similarity search
- Document deduplication by URL
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np
from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker

from src.config import get_settings
from src.models.embeddings import get_embedding_model, embed_text

logger = logging.getLogger(__name__)

settings = get_settings()

# Create engine
engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class Document(Base):
    """Document with embedding for semantic search."""

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    url = Column(String(2048), unique=True, index=True, nullable=False)
    title = Column(String(512))
    content = Column(Text)
    snippet = Column(Text)
    # Note: embedding column is added via raw SQL due to pgvector type
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Document(id={self.id}, url='{self.url[:50]}...')>"


def init_vector_store():
    """Initialize the vector store tables and extensions."""
    with engine.connect() as conn:
        # Enable pgvector extension
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
        logger.info("pgvector extension enabled")

    # Create base tables
    Base.metadata.create_all(bind=engine)

    # Add vector column if it doesn't exist
    embedding_dim = get_embedding_model().dimension
    with engine.connect() as conn:
        # Check if column exists
        result = conn.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'documents' AND column_name = 'embedding'
        """))
        if not result.fetchone():
            conn.execute(text(f"""
                ALTER TABLE documents 
                ADD COLUMN embedding vector({embedding_dim})
            """))
            conn.commit()
            logger.info(f"Added embedding column with dimension {embedding_dim}")

        # Create index for fast similarity search
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS documents_embedding_idx 
            ON documents 
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
        """))
        conn.commit()
        logger.info("Vector index created")


class VectorStore:
    """Vector store for document storage and semantic search."""

    def __init__(self):
        """Initialize the vector store."""
        self.embedding_model = get_embedding_model()

    def add_document(
        self,
        url: str,
        title: str,
        content: str,
        snippet: Optional[str] = None,
    ) -> Optional[int]:
        """
        Add a document to the vector store.

        Args:
            url: Document URL (unique identifier)
            title: Document title
            content: Document content (used for embedding)
            snippet: Short snippet/summary

        Returns:
            Document ID if successful, None if duplicate
        """
        db = SessionLocal()
        try:
            # Check if document already exists
            existing = db.execute(
                text("SELECT id FROM documents WHERE url = :url"),
                {"url": url}
            ).fetchone()

            if existing:
                logger.debug(f"Document already exists: {url[:50]}...")
                return existing[0]

            # Generate embedding from content
            text_to_embed = f"{title}. {content[:2000]}"  # Limit content length
            embedding = self.embedding_model.embed(text_to_embed)
            # Convert to string format for pgvector
            embedding_str = "[" + ",".join(str(x) for x in embedding.tolist()) + "]" 

            # Insert document with embedding
            # Using CAST() instead of :: to avoid SQLAlchemy parameter binding issues
            result = db.execute(
                text("""
                    INSERT INTO documents (url, title, content, snippet, embedding, created_at, updated_at)
                    VALUES (:url, :title, :content, :snippet, CAST(:embedding AS vector), NOW(), NOW())
                    RETURNING id
                """),
                {
                    "url": url,
                    "title": title,
                    "content": content[:10000],  # Limit content size
                    "snippet": snippet[:1000] if snippet else None,
                    "embedding": embedding_str,
                }
            )
            db.commit()

            doc_id = result.fetchone()[0]
            logger.debug(f"Added document {doc_id}: {title[:50]}...")
            return doc_id

        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            db.rollback()
            return None
        finally:
            db.close()

    def add_documents(
        self,
        documents: list[dict],
    ) -> list[int]:
        """
        Add multiple documents to the vector store.

        Args:
            documents: List of dicts with url, title, content, snippet

        Returns:
            List of document IDs
        """
        ids = []
        for doc in documents:
            doc_id = self.add_document(
                url=doc["url"],
                title=doc["title"],
                content=doc["content"],
                snippet=doc.get("snippet"),
            )
            if doc_id:
                ids.append(doc_id)
        return ids

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_similarity: float = 0.5,
    ) -> list[dict]:
        """
        Search for similar documents.

        Args:
            query: Search query
            top_k: Maximum number of results
            min_similarity: Minimum similarity threshold (0-1)

        Returns:
            List of dicts with id, url, title, snippet, similarity
        """
        db = SessionLocal()
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.embed(query)
            # Convert to string format for pgvector
            embedding_str = "[" + ",".join(str(x) for x in query_embedding.tolist()) + "]" 

            # Search using cosine similarity
            # Note: pgvector uses distance, so we convert to similarity
            # Using CAST() instead of :: to avoid SQLAlchemy parameter binding issues
            result = db.execute(
                text("""
                    SELECT 
                        id, url, title, snippet,
                        1 - (embedding <=> CAST(:embedding AS vector)) as similarity
                    FROM documents
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> CAST(:embedding AS vector)
                    LIMIT :limit
                """),
                {
                    "embedding": embedding_str,
                    "limit": top_k,
                }
            )

            documents = []
            for row in result:
                if row.similarity >= min_similarity:
                    documents.append({
                        "id": row.id,
                        "url": row.url,
                        "title": row.title,
                        "snippet": row.snippet,
                        "similarity": float(row.similarity),
                    })

            logger.info(f"Found {len(documents)} similar documents for query")
            return documents

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
        finally:
            db.close()

    def get_document(self, doc_id: int) -> Optional[dict]:
        """Get a document by ID."""
        db = SessionLocal()
        try:
            result = db.execute(
                text("SELECT id, url, title, content, snippet FROM documents WHERE id = :id"),
                {"id": doc_id}
            ).fetchone()

            if result:
                return {
                    "id": result.id,
                    "url": result.url,
                    "title": result.title,
                    "content": result.content,
                    "snippet": result.snippet,
                }
            return None
        finally:
            db.close()

    def get_document_by_url(self, url: str) -> Optional[dict]:
        """Get a document by URL."""
        db = SessionLocal()
        try:
            result = db.execute(
                text("SELECT id, url, title, content, snippet FROM documents WHERE url = :url"),
                {"url": url}
            ).fetchone()

            if result:
                return {
                    "id": result.id,
                    "url": result.url,
                    "title": result.title,
                    "content": result.content,
                    "snippet": result.snippet,
                }
            return None
        finally:
            db.close()

    def count(self) -> int:
        """Get total number of documents."""
        db = SessionLocal()
        try:
            result = db.execute(text("SELECT COUNT(*) FROM documents")).fetchone()
            return result[0]
        finally:
            db.close()

    def delete_document(self, doc_id: int) -> bool:
        """Delete a document by ID."""
        db = SessionLocal()
        try:
            db.execute(text("DELETE FROM documents WHERE id = :id"), {"id": doc_id})
            db.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            db.rollback()
            return False
        finally:
            db.close()


# Singleton instance
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Get the singleton vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


if __name__ == "__main__":
    # Test the vector store
    logging.basicConfig(level=logging.INFO)

    print("Initializing vector store...")
    init_vector_store()

    store = VectorStore()
    print(f"Current document count: {store.count()}")

    # Add test documents
    test_docs = [
        {
            "url": "https://example.com/quantum1",
            "title": "Quantum Computing Breakthrough",
            "content": "Scientists have achieved a major breakthrough in quantum computing, demonstrating quantum supremacy with a new 1000-qubit processor.",
            "snippet": "Major quantum computing breakthrough achieved",
        },
        {
            "url": "https://example.com/ai1",
            "title": "AI Language Models",
            "content": "Large language models continue to improve, with new architectures enabling better reasoning and longer context windows.",
            "snippet": "LLMs improving with new architectures",
        },
        {
            "url": "https://example.com/climate1",
            "title": "Climate Change Report",
            "content": "New climate report shows accelerating warming trends and calls for immediate action to reduce emissions.",
            "snippet": "Climate report shows accelerating warming",
        },
    ]

    print("\nAdding test documents...")
    ids = store.add_documents(test_docs)
    print(f"Added {len(ids)} documents")

    # Search
    print("\nSearching for 'quantum computing advances'...")
    results = store.search("quantum computing advances", top_k=5)
    for doc in results:
        print(f"  {doc['similarity']:.3f}: {doc['title']}")

    print(f"\nFinal document count: {store.count()}")
