"""Reranker module providing Cross-Encoder and Hybrid Semantic-Lexical Reranking.
Evaluates query-passage pairs to reorder candidate documents by deep relevance.
"""
from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from rag.document import Document


@dataclass
class RerankResult:
    """Detailed result of a single document after reranking."""
    document: Document
    initial_rank: int
    reranked_rank: int
    initial_score: float
    rerank_score: float
    rank_change: int  # positive means moved up, negative means moved down

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.document.id,
            "source": self.document.metadata.get("source", "unknown"),
            "initial_rank": self.initial_rank,
            "reranked_rank": self.reranked_rank,
            "initial_score": round(self.initial_score, 4),
            "rerank_score": round(self.rerank_score, 4),
            "rank_change": self.rank_change,
            "page_content": self.document.page_content,
        }


class Reranker:
    """Base interface for rerankers."""

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Document, float]],
        top_n: Optional[int] = None,
    ) -> List[RerankResult]:
        raise NotImplementedError


class CrossEncoderReranker(Reranker):
    """Cross-Encoder & Hybrid Reranker.
    
    Evaluates query-passage pairs using deep semantic & lexical cross-matching.
    Can use SentenceTransformers CrossEncoder or high-precision multi-signal cross-scoring.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        use_neural_model: bool = False,
    ):
        self.model_name = model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.use_neural_model = use_neural_model
        self._model = None
        self._is_cross_encoder_loaded = False
        if self.use_neural_model:
            self._init_model()

    def _init_model(self):
        try:
            from sentence_transformers import CrossEncoder
            # Try to load local cached model first
            try:
                self._model = CrossEncoder(self.model_name, local_files_only=True)
                self._is_cross_encoder_loaded = True
            except Exception:
                # If not local, load standard
                self._model = CrossEncoder(self.model_name)
                self._is_cross_encoder_loaded = True
        except Exception as e:
            print(f"CrossEncoder '{self.model_name}' could not be loaded ({e}). Using FastHybridReranker.")
            self._is_cross_encoder_loaded = False

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Document, float]],
        top_n: Optional[int] = None,
    ) -> List[RerankResult]:
        """Rerank retrieved candidates by cross-attention relevance to query."""
        if not candidates:
            return []

        top_n = top_n or len(candidates)
        top_n = min(top_n, len(candidates))

        docs = [pair[0] for pair in candidates]
        initial_scores = [pair[1] for pair in candidates]

        if self._is_cross_encoder_loaded and self._model is not None:
            try:
                pairs = [[query, doc.page_content] for doc in docs]
                raw_scores = self._model.predict(pairs)
                # Convert logits to sigmoid probabilities if necessary
                if isinstance(raw_scores, np.ndarray):
                    # Sigmoid for stability if range is unbound
                    rerank_scores = 1.0 / (1.0 + np.exp(-raw_scores))
                    rerank_scores = [float(s) for s in rerank_scores]
                else:
                    rerank_scores = [float(s) for s in raw_scores]
            except Exception as e:
                print(f"Error during CrossEncoder inference: {e}. Falling back.")
                rerank_scores = self._heuristic_score(query, docs, initial_scores)
        else:
            rerank_scores = self._heuristic_score(query, docs, initial_scores)

        # Pair up data
        indexed_items = []
        for i in range(len(docs)):
            indexed_items.append({
                "doc": docs[i],
                "initial_rank": i,
                "initial_score": initial_scores[i],
                "rerank_score": rerank_scores[i],
            })

        # Sort descending by rerank_score
        indexed_items.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Build RerankResult list
        results: List[RerankResult] = []
        for new_rank, item in enumerate(indexed_items[:top_n]):
            init_rank = item["initial_rank"]
            rank_change = init_rank - new_rank  # e.g., was rank 3, now rank 0 -> change +3
            results.append(
                RerankResult(
                    document=item["doc"],
                    initial_rank=init_rank,
                    reranked_rank=new_rank,
                    initial_score=item["initial_score"],
                    rerank_score=item["rerank_score"],
                    rank_change=rank_change,
                )
            )

        return results

    def _heuristic_score(
        self,
        query: str,
        docs: List[Document],
        initial_scores: List[float],
    ) -> List[float]:
        """High-precision lexical-semantic cross-scoring fallback.
        
        Evaluates exact phrase matching, query term coverage, position of first match,
        and blend with initial vector similarity score.
        """
        q_clean = query.lower()
        q_tokens = set(re.findall(r"\w+", q_clean))
        if not q_tokens:
            return initial_scores

        scores = []
        for doc, init_score in zip(docs, initial_scores):
            content = doc.page_content.lower()
            doc_tokens = re.findall(r"\w+", content)
            doc_token_set = set(doc_tokens)

            # 1. Query Term Coverage (recall of query terms in chunk)
            matched_terms = q_tokens.intersection(doc_token_set)
            term_coverage = len(matched_terms) / len(q_tokens)

            # 2. Exact phrase bonus
            exact_phrase_bonus = 0.25 if q_clean in content else 0.0

            # 3. Term frequency density
            tf_count = sum(doc_tokens.count(t) for t in matched_terms)
            tf_density = min(0.2, tf_count / max(1, len(doc_tokens)))

            # 4. Proximity / early occurrence bonus
            early_bonus = 0.0
            for term in matched_terms:
                pos = content.find(term)
                if pos != -1 and pos < 200:
                    early_bonus = 0.1
                    break

            # Blend with initial vector score (normalized ~ 0.0 to 1.0)
            base_score = float(init_score)
            rerank_score = (
                0.40 * base_score +
                0.35 * term_coverage +
                exact_phrase_bonus +
                tf_density +
                early_bonus
            )
            scores.append(round(rerank_score, 4))

        return scores
