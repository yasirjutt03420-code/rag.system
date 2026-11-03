"""Retriever module providing Dense, BM25 Lexical, Hybrid (RRF), and MMR retrieval strategies.
"""
from collections import Counter
import math
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from rag.document import Document
from rag.embedding import EmbeddingEngine
from rag.vector_database import VectorDatabase


class SimpleBM25:
    """Lightweight, zero-dependency BM25 scoring implementation."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.doc_lens: List[int] = []
        self.doc_freqs: Dict[str, int] = {}
        self.term_freqs: List[Counter] = []

    def fit(self, texts: List[str]):
        """Build BM25 term frequency index from texts."""
        self.corpus_size = len(texts)
        if self.corpus_size == 0:
            return

        self.doc_lens = []
        self.doc_freqs = {}
        self.term_freqs = []

        total_len = 0
        for text in texts:
            tokens = self._tokenize(text)
            self.doc_lens.append(len(tokens))
            total_len += len(tokens)
            tf = Counter(tokens)
            self.term_freqs.append(tf)

            for term in tf.keys():
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        self.avg_doc_len = total_len / max(1, self.corpus_size)

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def score(self, query: str) -> List[float]:
        """Compute BM25 score of all indexed documents for query."""
        if self.corpus_size == 0:
            return []

        q_tokens = self._tokenize(query)
        scores = [0.0] * self.corpus_size

        for term in q_tokens:
            if term not in self.doc_freqs:
                continue

            df = self.doc_freqs[term]
            idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)

            for i in range(self.corpus_size):
                tf = self.term_freqs[i].get(term, 0)
                doc_len = self.doc_lens[i]
                num = tf * (self.k1 + 1.0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / max(1e-6, self.avg_doc_len)))
                scores[i] += idf * (num / denom)

        return scores


class DenseRetriever:
    """Retrieves documents using dense vector similarity."""

    def __init__(self, vector_db: VectorDatabase, embedding_engine: EmbeddingEngine):
        self.vector_db = vector_db
        self.embedding_engine = embedding_engine

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """Retrieve top_k documents semantically related to query."""
        q_vec = self.embedding_engine.embed_query(query)
        return self.vector_db.search(
            query_vector=q_vec,
            top_k=top_k,
            score_threshold=score_threshold,
            filter_metadata=filter_metadata,
        )


class HybridRetriever:
    """Combines Dense Vector Retrieval and Sparse BM25 Keyword Matching via Reciprocal Rank Fusion."""

    def __init__(
        self,
        vector_db: VectorDatabase,
        embedding_engine: EmbeddingEngine,
        alpha: float = 0.5,
        rrf_k: int = 60,
    ):
        """Initialize Hybrid Retriever.

        Args:
            vector_db: Vector database
            embedding_engine: Embedding model
            alpha: Weight for dense vs lexical scoring (0.0 = pure BM25, 1.0 = pure Dense)
            rrf_k: Constant for Reciprocal Rank Fusion smoothing
        """
        self.vector_db = vector_db
        self.embedding_engine = embedding_engine
        self.alpha = alpha
        self.rrf_k = rrf_k
        self.bm25 = SimpleBM25()
        self._rebuild_bm25_index()

    def _rebuild_bm25_index(self):
        texts = [doc.page_content for doc in self.vector_db.documents]
        self.bm25.fit(texts)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """Hybrid search blending dense vector and lexical BM25 rankings."""
        if not self.vector_db.documents:
            return []

        # Ensure BM25 has same count as vector DB
        if self.bm25.corpus_size != len(self.vector_db.documents):
            self._rebuild_bm25_index()

        # 1. Dense retrieval
        q_vec = self.embedding_engine.embed_query(query)
        dense_results = self.vector_db.search(
            query_vector=q_vec,
            top_k=len(self.vector_db.documents),
            filter_metadata=filter_metadata,
        )

        dense_rank_map = {doc.id: rank for rank, (doc, _) in enumerate(dense_results)}
        dense_score_map = {doc.id: score for doc, score in dense_results}

        # 2. BM25 scoring
        bm25_scores = self.bm25.score(query)
        scored_bm25 = [
            (self.vector_db.documents[i], bm25_scores[i])
            for i in range(len(self.vector_db.documents))
        ]
        if filter_metadata:
            scored_bm25 = [
                pair for pair in scored_bm25
                if all(pair[0].metadata.get(k) == v for k, v in filter_metadata.items())
            ]
        scored_bm25.sort(key=lambda x: x[1], reverse=True)
        bm25_rank_map = {doc.id: rank for rank, (doc, _) in enumerate(scored_bm25)}

        # 3. Reciprocal Rank Fusion (RRF)
        combined_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        all_docs = self.vector_db.documents
        for doc in all_docs:
            doc_id = doc.id
            doc_map[doc_id] = doc

            # RRF formula: sum( 1 / (k + rank) )
            dense_rank = dense_rank_map.get(doc_id, len(all_docs))
            bm25_rank = bm25_rank_map.get(doc_id, len(all_docs))

            dense_rrf = 1.0 / (self.rrf_k + dense_rank + 1)
            bm25_rrf = 1.0 / (self.rrf_k + bm25_rank + 1)

            fused_score = (self.alpha * dense_rrf) + ((1.0 - self.alpha) * bm25_rrf)
            combined_scores[doc_id] = fused_score

        # Sort and take top_k
        ranked_doc_ids = sorted(combined_scores.keys(), key=lambda did: combined_scores[did], reverse=True)
        
        # Normalize top score to range ~ 0.0 - 1.0 for consistency
        max_score = max(combined_scores.values()) if combined_scores else 1.0
        results = [
            (doc_map[did], combined_scores[did] / max(1e-6, max_score))
            for did in ranked_doc_ids[:top_k]
        ]
        return results


class MMRRetriever:
    """Maximal Marginal Relevance (MMR) retriever for diverse, non-redundant search results."""

    def __init__(
        self,
        vector_db: VectorDatabase,
        embedding_engine: EmbeddingEngine,
        lambda_mult: float = 0.6,
    ):
        """Initialize MMR Retriever.

        Args:
            vector_db: Vector store
            embedding_engine: Embedding engine
            lambda_mult: Diversity factor between 0.0 (maximum diversity) and 1.0 (pure similarity).
        """
        self.vector_db = vector_db
        self.embedding_engine = embedding_engine
        self.lambda_mult = lambda_mult

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        fetch_k: int = 20,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """Run MMR algorithm to select diverse, relevant documents."""
        if not self.vector_db.documents or self.vector_db.vectors is None:
            return []

        fetch_k = max(top_k, min(fetch_k, len(self.vector_db.documents)))
        q_vec = self.embedding_engine.embed_query(query)

        # Get initial candidate pool
        candidates = self.vector_db.search(
            query_vector=q_vec,
            top_k=fetch_k,
            filter_metadata=filter_metadata,
        )
        if not candidates:
            return []

        cand_docs = [doc for doc, _ in candidates]
        cand_indices = [self.vector_db.ids.index(doc.id) for doc in cand_docs]
        cand_vectors = self.vector_db.vectors[cand_indices]

        # Normalize
        q_norm = q_vec / max(1e-6, np.linalg.norm(q_vec))
        doc_norms = cand_vectors / np.maximum(1e-6, np.linalg.norm(cand_vectors, axis=1, keepdims=True))

        # Query similarities: shape (fetch_k,)
        sim_to_query = np.dot(doc_norms, q_norm)

        selected_indices: List[int] = []
        remaining_indices: List[int] = list(range(len(cand_docs)))

        # Greedily pick first best
        first_pick = int(np.argmax(sim_to_query))
        selected_indices.append(first_pick)
        remaining_indices.remove(first_pick)

        while len(selected_indices) < min(top_k, len(cand_docs)):
            best_mmr = -float("inf")
            best_idx = remaining_indices[0]

            for idx in remaining_indices:
                # Sim to query
                relevance = sim_to_query[idx]
                
                # Max similarity to already selected documents
                redundancy = max(
                    np.dot(doc_norms[idx], doc_norms[sel]) for sel in selected_indices
                )

                mmr_score = self.lambda_mult * relevance - (1.0 - self.lambda_mult) * redundancy
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = idx

            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        return [(cand_docs[i], float(sim_to_query[i])) for i in selected_indices]

