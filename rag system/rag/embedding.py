"""Embedding Engine supporting SentenceTransformers, OpenAI, Gemini, and TF-IDF fallbacks.
Includes in-memory LRU caching and vector normalization.
"""
import hashlib
import json
import os
from typing import Dict, List, Optional, Union
import numpy as np
import requests


class EmbeddingEngine:
    """Manages text embeddings across multiple backends."""

    def __init__(
        self,
        model_type: str = "sentence-transformers",
        model_name: str = "all-MiniLM-L6-v2",
        api_key: Optional[str] = None,
        cache_enabled: bool = True,
    ):
        """Initialize EmbeddingEngine.

        Args:
            model_type: 'sentence-transformers', 'openai', 'gemini', or 'tfidf'
            model_name: Model identifier (e.g. 'all-MiniLM-L6-v2', 'text-embedding-3-small')
            api_key: API Key for cloud providers
            cache_enabled: Whether to cache text -> embedding vector
        """
        self.model_type = model_type.lower()
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, np.ndarray] = {}
        self._model = None
        self._vector_dim: Optional[int] = None

        self._init_backend()

    def _init_backend(self):
        """Lazy load or initialize the underlying embedding model."""
        if self.model_type == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer
                # Load SentenceTransformer model
                self._model = SentenceTransformer(self.model_name)
                # Compute test vector to determine dimensionality
                test_emb = self._model.encode(["test"], normalize_embeddings=True)
                self._vector_dim = test_emb.shape[1]
            except Exception as e:
                print(f"Failed to load SentenceTransformer ({e}). Falling back to TF-IDF.")
                self.model_type = "tfidf"
                self._init_tfidf()

        elif self.model_type == "tfidf":
            self._init_tfidf()

        elif self.model_type == "openai":
            self._vector_dim = 1536 if "ada" in self.model_name else 1536

        elif self.model_type == "gemini":
            self._vector_dim = 768

    def _init_tfidf(self):
        """Fallback TF-IDF vectorizer with SVD dimension reduction."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        self._tfidf_vectorizer = TfidfVectorizer(max_features=1000)
        self._dim_reducer = None
        self._vector_dim = 384
        self._is_tfidf_fitted = False

    @property
    def vector_dimension(self) -> int:
        return self._vector_dim or 384

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Generate normalized embedding vectors for a list of document strings.

        Returns:
            2D numpy array of shape (len(texts), vector_dimension)
        """
        if not texts:
            return np.empty((0, self.vector_dimension), dtype=np.float32)

        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        if self.cache_enabled:
            for i, text in enumerate(texts):
                key = self._cache_key(text)
                if key in self._cache:
                    results[i] = self._cache[key]
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)
        else:
            uncached_indices = list(range(len(texts)))
            uncached_texts = list(texts)

        if uncached_texts:
            computed_vectors = self._embed_batch(uncached_texts)
            for idx, vec in zip(uncached_indices, computed_vectors):
                results[idx] = vec
                if self.cache_enabled:
                    self._cache[self._cache_key(texts[idx])] = vec

        matrix = np.array(results, dtype=np.float32)
        return self._normalize(matrix)

    def embed_query(self, query: str) -> np.ndarray:
        """Generate normalized embedding vector for a single query string.

        Returns:
            1D numpy array of shape (vector_dimension,)
        """
        if self.cache_enabled:
            key = self._cache_key(query)
            if key in self._cache:
                return self._cache[key]

        vecs = self.embed_documents([query])
        vec = vecs[0]
        if self.cache_enabled:
            self._cache[self._cache_key(query)] = vec
        return vec

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """Internal dispatch for batch embedding."""
        if self.model_type == "sentence-transformers" and self._model is not None:
            embeddings = self._model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return np.asarray(embeddings, dtype=np.float32)

        elif self.model_type == "openai":
            return self._embed_openai(texts)

        elif self.model_type == "gemini":
            return self._embed_gemini(texts)

        else:
            return self._embed_tfidf(texts)

    def _embed_openai(self, texts: List[str]) -> np.ndarray:
        """Call OpenAI embeddings endpoint via HTTP."""
        if not self.api_key:
            raise ValueError("OpenAI API key is missing. Set OPENAI_API_KEY or pass api_key.")
        url = "https://api.openai.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name or "text-embedding-3-small",
            "input": texts,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()["data"]
        # Ensure ordered by index
        data.sort(key=lambda x: x["index"])
        vectors = [item["embedding"] for item in data]
        return np.array(vectors, dtype=np.float32)

    def _embed_gemini(self, texts: List[str]) -> np.ndarray:
        """Call Google Gemini text-embedding endpoint via REST."""
        if not self.api_key:
            raise ValueError("Gemini API key is missing. Set GEMINI_API_KEY or pass api_key.")
        
        vectors = []
        for text in texts:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={self.api_key}"
            payload = {
                "model": "models/text-embedding-004",
                "content": {"parts": [{"text": text}]},
            }
            resp = requests.post(url, json=payload, timeout=20)
            resp.raise_for_status()
            val = resp.json().get("embedding", {}).get("values", [])
            vectors.append(val)
        return np.array(vectors, dtype=np.float32)

    def _embed_tfidf(self, texts: List[str]) -> np.ndarray:
        """Deterministic TF-IDF embedding fallback with hashed pseudo-projection."""
        from sklearn.feature_extraction.text import HashingVectorizer
        hasher = HashingVectorizer(n_features=self.vector_dimension, norm="l2")
        sparse = hasher.transform(texts)
        return sparse.toarray().astype(np.float32)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        """L2 Normalize rows of a 2D matrix or a 1D vector."""
        if vectors.ndim == 1:
            norm = np.linalg.norm(vectors)
            return vectors if norm == 0 else vectors / norm
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

