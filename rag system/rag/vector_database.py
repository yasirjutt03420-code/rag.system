"""Vector Database module providing persistent, multi-collection vector storage and search.
Supports cosine similarity, L2 distance, dot product, metadata filtering, and disk persistence.
"""
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from rag.document import Document


class VectorDatabase:
    """Persistent, multi-collection Vector Database engine."""

    def __init__(
        self,
        persist_directory: Union[str, Path] = "./vector_store",
        collection_name: str = "default",
        distance_metric: str = "cosine",
    ):
        """Initialize Vector Database.

        Args:
            persist_directory: Directory where collections are stored on disk.
            collection_name: Active collection name.
            distance_metric: 'cosine', 'dot', or 'l2'
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.distance_metric = distance_metric.lower()

        # In-memory storage for current collection
        self.ids: List[str] = []
        self.documents: List[Document] = []
        self.vectors: Optional[np.ndarray] = None  # Shape: (N, D)

        # Load existing collection if present
        self.load_collection(collection_name)

    @property
    def collection_dir(self) -> Path:
        return self.persist_directory / self.collection_name

    def list_collections(self) -> List[str]:
        """List all collection names found in persist directory."""
        if not self.persist_directory.exists():
            return []
        cols = [
            d.name
            for d in self.persist_directory.iterdir()
            if d.is_dir() and (d / "metadata.json").exists()
        ]
        if self.collection_name not in cols:
            cols.append(self.collection_name)
        return sorted(list(set(cols)))

    def switch_collection(self, collection_name: str):
        """Switch to another collection, saving current state first."""
        self.save()
        self.collection_name = collection_name
        self.load_collection(collection_name)

    def add_documents(
        self,
        documents: List[Document],
        embeddings: Union[np.ndarray, List[List[float]]],
    ) -> List[str]:
        """Add documents and corresponding embedding vectors to the active collection.

        Args:
            documents: List of Document objects
            embeddings: Matrix of shape (len(documents), dimension)

        Returns:
            List of added document IDs
        """
        if not documents:
            return []

        emb_matrix = np.asarray(embeddings, dtype=np.float32)
        if len(documents) != len(emb_matrix):
            raise ValueError(
                f"Count mismatch: {len(documents)} documents vs {len(emb_matrix)} embeddings"
            )

        # Normalize if using cosine metric
        if self.distance_metric == "cosine":
            norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            emb_matrix = emb_matrix / norms

        added_ids = []
        for doc in documents:
            self.ids.append(doc.id)
            self.documents.append(doc)
            added_ids.append(doc.id)

        if self.vectors is None or len(self.vectors) == 0:
            self.vectors = emb_matrix
        else:
            self.vectors = np.vstack([self.vectors, emb_matrix])

        # Auto-persist
        self.save()
        return added_ids

    def search(
        self,
        query_vector: Union[np.ndarray, List[float]],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """Search collection for most similar documents given a query vector.

        Args:
            query_vector: 1D query embedding vector
            top_k: Maximum number of results to return
            score_threshold: Minimum similarity threshold (for cosine/dot) or max distance (for l2)
            filter_metadata: Optional dict of key-value pairs required in doc.metadata

        Returns:
            List of (Document, score) tuples sorted by descending relevance
        """
        if self.vectors is None or len(self.documents) == 0:
            return []

        q_vec = np.asarray(query_vector, dtype=np.float32)
        if q_vec.ndim > 1:
            q_vec = q_vec.flatten()

        # Normalize query vector for cosine similarity
        if self.distance_metric == "cosine":
            norm = np.linalg.norm(q_vec)
            if norm > 0:
                q_vec = q_vec / norm

        # Compute similarity scores
        if self.distance_metric in ("cosine", "dot"):
            # Dot product against all stored vectors
            scores = np.dot(self.vectors, q_vec)
        elif self.distance_metric == "l2":
            # Euclidean distance (lower is better, convert to inverse score)
            dists = np.linalg.norm(self.vectors - q_vec, axis=1)
            scores = 1.0 / (1.0 + dists)
        else:
            scores = np.dot(self.vectors, q_vec)

        # Apply candidate filtering
        candidate_indices = range(len(self.documents))
        if filter_metadata:
            candidate_indices = [
                idx for idx in candidate_indices
                if all(self.documents[idx].metadata.get(k) == v for k, v in filter_metadata.items())
            ]

        if not candidate_indices:
            return []

        # Pair candidates with scores
        scored_pairs = [(self.documents[i], float(scores[i])) for i in candidate_indices]

        # Apply score threshold if specified
        if score_threshold is not None:
            scored_pairs = [pair for pair in scored_pairs if pair[1] >= score_threshold]

        # Sort descending by score
        scored_pairs.sort(key=lambda x: x[1], reverse=True)

        return scored_pairs[:top_k]

    def save(self):
        """Persist vectors and metadata to disk."""
        self.collection_dir.mkdir(parents=True, exist_ok=True)

        # Save vectors
        vectors_file = self.collection_dir / "vectors.npz"
        if self.vectors is not None and len(self.vectors) > 0:
            np.savez_compressed(vectors_file, vectors=self.vectors)
        elif vectors_file.exists():
            vectors_file.unlink()

        # Save metadata and documents
        meta_file = self.collection_dir / "metadata.json"
        data = {
            "collection_name": self.collection_name,
            "distance_metric": self.distance_metric,
            "count": len(self.documents),
            "documents": [doc.to_dict() for doc in self.documents],
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_collection(self, collection_name: str) -> bool:
        """Load collection data from disk."""
        col_dir = self.persist_directory / collection_name
        meta_file = col_dir / "metadata.json"
        vectors_file = col_dir / "vectors.npz"

        self.ids = []
        self.documents = []
        self.vectors = None

        if not meta_file.exists():
            return False

        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.distance_metric = data.get("distance_metric", self.distance_metric)
            docs_data = data.get("documents", [])
            for d in docs_data:
                doc = Document.from_dict(d)
                self.documents.append(doc)
                self.ids.append(doc.id)

            if vectors_file.exists():
                npz = np.load(vectors_file)
                self.vectors = npz["vectors"]

            return True
        except Exception as e:
            print(f"Error loading collection {collection_name}: {e}")
            return False

    def clear(self):
        """Clear all documents and vectors in the current collection."""
        self.ids = []
        self.documents = []
        self.vectors = None
        self.save()

    def delete_collection(self, collection_name: Optional[str] = None):
        """Delete an entire collection directory."""
        col = collection_name or self.collection_name
        target_dir = self.persist_directory / col
        if target_dir.exists():
            import shutil
            shutil.rmtree(target_dir)

        if col == self.collection_name:
            self.ids = []
            self.documents = []
            self.vectors = None

    def get_stats(self) -> Dict[str, Any]:
        """Return diagnostic metrics about the active collection."""
        dim = self.vectors.shape[1] if self.vectors is not None and len(self.vectors) > 0 else 0
        sources = set(doc.metadata.get("source", "unknown") for doc in self.documents)
        return {
            "collection_name": self.collection_name,
            "document_count": len(self.documents),
            "vector_dimension": dim,
            "distance_metric": self.distance_metric,
            "unique_sources": len(sources),
            "sources_list": sorted(list(sources)),
            "persist_path": str(self.collection_dir.resolve()),
        }

