"""Document data structure for RAG pipeline.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import uuid


@dataclass
class Document:
    """Represents a piece of text with associated metadata."""
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    embedding: Optional[List[float]] = None

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary."""
        return {
            "id": self.id,
            "page_content": self.page_content,
            "metadata": self.metadata,
            "embedding": self.embedding,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """Create document from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            page_content=data.get("page_content", ""),
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding", None),
        )

    def __repr__(self) -> str:
        preview = (self.page_content[:60] + "...") if len(self.page_content) > 60 else self.page_content
        preview_clean = preview.replace("\n", " ")
        return f"Document(id={self.id[:8]}, source='{self.metadata.get('source', 'unknown')}', content='{preview_clean}')"

