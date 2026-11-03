"""RAG Evaluation and Benchmarking Framework.
Evaluates the RAG Triad (Faithfulness, Answer Relevance, Context Relevance)
along with Ground Truth Answer Correctness, Context Recall, and Composite Quality.
"""
from dataclasses import dataclass, field
import json
import math
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from rag.document import Document


# Common english stopwords for token-level lexical analysis
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "which", "this", "that", "these", "those", "then", "just", "so", "than",
    "such", "both", "through", "about", "for", "is", "of", "while", "during",
    "to", "from", "in", "out", "on", "off", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will",
    "don", "should", "now", "are", "was", "were", "been", "being", "have",
    "has", "had", "having", "do", "does", "did", "doing", "would", "could"
}


def _tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words."""
    return [w for w in re.findall(r"\w+", text.lower()) if len(w) > 1]


def _content_words(text: str) -> List[str]:
    """Filter out stopwords to focus on content-bearing keywords."""
    return [w for w in _tokenize(text) if w not in STOPWORDS]


def _compute_f1(pred_tokens: List[str], ref_tokens: List[str]) -> Tuple[float, float, float]:
    """Compute token-level precision, recall, and F1."""
    if not pred_tokens or not ref_tokens:
        return 0.0, 0.0, 0.0
    pred_set = set(pred_tokens)
    ref_set = set(ref_tokens)
    common = pred_set.intersection(ref_set)
    if not common:
        return 0.0, 0.0, 0.0
    prec = len(common) / len(pred_set)
    rec = len(common) / len(ref_set)
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


@dataclass
class MetricScore:
    """A single evaluation metric score with reasoning."""
    name: str
    score: float  # Normalized 0.0 to 1.0
    passed: bool
    explanation: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "score": round(self.score, 3),
            "passed": self.passed,
            "explanation": self.explanation,
            "details": self.details,
        }


@dataclass
class RAGEvaluationResult:
    """Complete evaluation report for a single RAG query-response-context triplet."""
    query: str
    answer: str
    retrieved_contexts: List[str]
    ground_truth: Optional[str] = None
    faithfulness: MetricScore = field(default_factory=lambda: MetricScore("Faithfulness", 0.0, False, ""))
    answer_relevance: MetricScore = field(default_factory=lambda: MetricScore("Answer Relevance", 0.0, False, ""))
    context_relevance: MetricScore = field(default_factory=lambda: MetricScore("Context Relevance", 0.0, False, ""))
    context_recall: Optional[MetricScore] = None
    answer_correctness: Optional[MetricScore] = None
    composite_score: float = 0.0
    latency_ms: float = 0.0
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "query": self.query,
            "composite_score": round(self.composite_score, 3),
            "composite_percentage": f"{round(self.composite_score * 100, 1)}%",
            "latency_ms": round(self.latency_ms, 2),
            "metrics": {
                "faithfulness": self.faithfulness.to_dict(),
                "answer_relevance": self.answer_relevance.to_dict(),
                "context_relevance": self.context_relevance.to_dict(),
            },
            "recommendations": self.recommendations,
        }
        if self.context_recall:
            data["metrics"]["context_recall"] = self.context_recall.to_dict()
        if self.answer_correctness:
            data["metrics"]["answer_correctness"] = self.answer_correctness.to_dict()
        return data


@dataclass
class EvaluationReport:
    """Aggregated evaluation benchmark report over multiple queries."""
    total_queries: int
    mean_composite_score: float
    mean_faithfulness: float
    mean_answer_relevance: float
    mean_context_relevance: float
    mean_context_recall: Optional[float] = None
    mean_answer_correctness: Optional[float] = None
    results: List[RAGEvaluationResult] = field(default_factory=list)
    weak_queries: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "mean_composite_score": round(self.mean_composite_score, 3),
            "mean_composite_percentage": f"{round(self.mean_composite_score * 100, 1)}%",
            "mean_faithfulness": round(self.mean_faithfulness, 3),
            "mean_answer_relevance": round(self.mean_answer_relevance, 3),
            "mean_context_relevance": round(self.mean_context_relevance, 3),
            "mean_context_recall": round(self.mean_context_recall, 3) if self.mean_context_recall is not None else None,
            "mean_answer_correctness": round(self.mean_answer_correctness, 3) if self.mean_answer_correctness is not None else None,
            "weak_queries_count": len(self.weak_queries),
        }


class RAGEvaluator:
    """Evaluator engine for RAG pipelines implementing the RAG Triad and benchmark metrics.
    
    Metrics:
    1. Faithfulness (Groundedness): Answer claims supported by retrieved context
    2. Answer Relevance: Answer addresses the query's topic and intent
    3. Context Relevance: Retrieved chunks contain information pertinent to query
    4. Context Recall (optional): Retrieved context contains ground-truth knowledge
    5. Answer Correctness (optional): Answer aligns with reference ground truth
    """

    def __init__(
        self,
        faithfulness_threshold: float = 0.70,
        answer_relevance_threshold: float = 0.70,
        context_relevance_threshold: float = 0.60,
    ):
        self.faithfulness_threshold = faithfulness_threshold
        self.answer_relevance_threshold = answer_relevance_threshold
        self.context_relevance_threshold = context_relevance_threshold

    def evaluate(
        self,
        query: str,
        answer: str,
        retrieved_contexts: Union[List[str], List[Document]],
        ground_truth: Optional[str] = None,
    ) -> RAGEvaluationResult:
        """Evaluate a single query-answer-context triplet."""
        t0 = time.time()

        # Normalize contexts to list of strings
        context_strings: List[str] = []
        for ctx in retrieved_contexts:
            if isinstance(ctx, Document):
                context_strings.append(ctx.page_content)
            elif isinstance(ctx, str):
                context_strings.append(ctx)

        # 1. Evaluate Faithfulness
        faith_metric = self._eval_faithfulness(answer, context_strings)

        # 2. Evaluate Answer Relevance
        rel_metric = self._eval_answer_relevance(query, answer)

        # 3. Evaluate Context Relevance
        ctx_metric = self._eval_context_relevance(query, context_strings)

        # 4. Optional Context Recall
        rec_metric = None
        if ground_truth:
            rec_metric = self._eval_context_recall(ground_truth, context_strings)

        # 5. Optional Answer Correctness
        corr_metric = None
        if ground_truth:
            corr_metric = self._eval_answer_correctness(answer, ground_truth)

        # 6. Composite Score (Weighted average of RAG Triad)
        w_faith = 0.40
        w_ans_rel = 0.35
        w_ctx_rel = 0.25
        composite = (
            w_faith * faith_metric.score
            + w_ans_rel * rel_metric.score
            + w_ctx_rel * ctx_metric.score
        )
        if corr_metric:
            # If ground truth exists, include correctness in composite
            composite = 0.30 * faith_metric.score + 0.25 * rel_metric.score + 0.20 * ctx_metric.score + 0.25 * corr_metric.score

        # 7. Generate diagnostic recommendations
        recommendations = []
        if faith_metric.score < self.faithfulness_threshold:
            recommendations.append(
                "⚠️ Low Faithfulness detected: The answer makes statements not firmly supported by context. Tighten prompt constraints or reduce LLM temperature."
            )
        if rel_metric.score < self.answer_relevance_threshold:
            recommendations.append(
                "⚠️ Low Answer Relevance: The response drifts from the user's specific query. Check PromptBuilder query injection."
            )
        if ctx_metric.score < self.context_relevance_threshold:
            recommendations.append(
                "⚠️ Low Context Relevance: Retrieved chunks contain noisy distractors. Consider increasing top_k for retriever or enabling Cross-Encoder Reranker."
            )
        if not recommendations:
            recommendations.append("✅ Excellent RAG response: Grounded in evidence, highly relevant, and cleanly attributed.")

        latency = (time.time() - t0) * 1000

        return RAGEvaluationResult(
            query=query,
            answer=answer,
            retrieved_contexts=context_strings,
            ground_truth=ground_truth,
            faithfulness=faith_metric,
            answer_relevance=rel_metric,
            context_relevance=ctx_metric,
            context_recall=rec_metric,
            answer_correctness=corr_metric,
            composite_score=composite,
            latency_ms=latency,
            recommendations=recommendations,
        )

    def _eval_faithfulness(self, answer: str, context_strings: List[str]) -> MetricScore:
        """Measure what fraction of answer statements are supported by the context."""
        if not context_strings:
            return MetricScore(
                name="Faithfulness",
                score=0.0,
                passed=False,
                explanation="No context was retrieved to ground the answer.",
                details={"supported_claims": 0, "total_claims": 0},
            )

        # Clean answer and split into sentence-level claims
        clean_ans = re.sub(r"\[Document \d+[^\]]*\]", "", answer)  # Remove citation tags
        clean_ans = re.sub(r"\[Doc \d+[^\]]*\]", "", clean_ans)
        clean_ans = re.sub(r"\[\^?\d+\]", "", clean_ans)
        clean_ans = re.sub(r"###.*?\n", "", clean_ans)  # Remove markdown headers
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_ans) if len(s.strip()) > 15]

        # Ignore generic boilerplate disclaimers from claim count
        boilerplate_patterns = [
            r"not enough information",
            r"note: generated using",
            r"please upload documents",
        ]
        claims = []
        for s in sentences:
            if not any(re.search(bp, s, re.IGNORECASE) for bp in boilerplate_patterns):
                claims.append(s)

        if not claims:
            # If model stated it lacks info, that is 100% faithful to the lack of context!
            if "not enough information" in answer.lower():
                return MetricScore(
                    name="Faithfulness",
                    score=1.0,
                    passed=True,
                    explanation="Accurately recognized lack of information in documents without hallucinating.",
                    details={"supported_claims": 0, "total_claims": 0, "abstention": True},
                )
            return MetricScore(
                name="Faithfulness",
                score=0.85,
                passed=True,
                explanation="Answer contained high-level response with no contestable factual claims.",
                details={"supported_claims": 0, "total_claims": 0},
            )

        combined_context = " ".join(context_strings).lower()
        context_tokens = set(_content_words(combined_context))

        supported_count = 0
        claim_details = []

        for claim in claims:
            c_words = _content_words(claim)
            if not c_words:
                continue

            overlap = len([w for w in c_words if w in context_tokens])
            overlap_ratio = overlap / max(1, len(c_words))

            # A claim is supported if >= 55% of its core content words exist in context
            is_supported = overlap_ratio >= 0.55
            if is_supported:
                supported_count += 1
            claim_details.append({
                "claim": claim[:100] + ("..." if len(claim) > 100 else ""),
                "overlap_ratio": round(overlap_ratio, 2),
                "supported": is_supported,
            })

        faith_score = supported_count / max(1, len(claims))
        passed = faith_score >= self.faithfulness_threshold

        explanation = (
            f"{supported_count} of {len(claims)} key claims verified against retrieved documents ({round(faith_score * 100)}%)."
        )

        return MetricScore(
            name="Faithfulness",
            score=faith_score,
            passed=passed,
            explanation=explanation,
            details={"supported_claims": supported_count, "total_claims": len(claims), "claims": claim_details},
        )

    def _eval_answer_relevance(self, query: str, answer: str) -> MetricScore:
        """Measure if the answer addresses the specific user query."""
        q_words = set(_content_words(query))
        a_words = set(_content_words(answer))

        if not q_words:
            return MetricScore(
                name="Answer Relevance",
                score=1.0,
                passed=True,
                explanation="Query consisted only of generic terms.",
            )

        # Key term recall in answer
        common = q_words.intersection(a_words)
        term_recall = len(common) / max(1, len(q_words))

        # Check for non-empty substantive answer
        answer_substantive = len(_tokenize(answer)) >= 10
        completeness = 1.0 if answer_substantive else 0.4

        score = (0.7 * term_recall) + (0.3 * completeness)
        score = min(1.0, max(0.0, score))
        passed = score >= self.answer_relevance_threshold

        return MetricScore(
            name="Answer Relevance",
            score=score,
            passed=passed,
            explanation=f"Answer covered {len(common)}/{len(q_words)} core query terms ({round(score * 100)}% relevance).",
            details={"query_terms": list(q_words), "matched_terms": list(common)},
        )

    def _eval_context_relevance(self, query: str, context_strings: List[str]) -> MetricScore:
        """Measure what fraction of retrieved chunks are relevant to the query."""
        if not context_strings:
            return MetricScore(
                name="Context Relevance",
                score=0.0,
                passed=False,
                explanation="No context chunks available for evaluation.",
            )

        q_words = set(_content_words(query))
        if not q_words:
            return MetricScore(
                name="Context Relevance",
                score=1.0,
                passed=True,
                explanation="Empty content in query.",
            )

        relevant_chunks = 0
        chunk_scores = []

        for ctx in context_strings:
            c_words = set(_content_words(ctx))
            overlap = len(q_words.intersection(c_words))
            ratio = overlap / max(1, len(q_words))
            chunk_scores.append(round(ratio, 2))
            if ratio >= 0.25:  # At least 25% query terms present
                relevant_chunks += 1

        precision = relevant_chunks / max(1, len(context_strings))
        mean_ratio = sum(chunk_scores) / max(1, len(chunk_scores))

        # Blended context relevance score
        score = (0.6 * precision) + (0.4 * min(1.0, mean_ratio * 1.5))
        passed = score >= self.context_relevance_threshold

        return MetricScore(
            name="Context Relevance",
            score=score,
            passed=passed,
            explanation=f"{relevant_chunks}/{len(context_strings)} retrieved chunks contain key query concepts.",
            details={"chunk_overlap_ratios": chunk_scores, "relevant_chunks": relevant_chunks},
        )

    def _eval_context_recall(self, ground_truth: str, context_strings: List[str]) -> MetricScore:
        """Measure if retrieved context captures all facts in ground truth."""
        gt_words = set(_content_words(ground_truth))
        if not gt_words:
            return MetricScore(name="Context Recall", score=1.0, passed=True, explanation="Ground truth is empty.")

        combined_context = " ".join(context_strings).lower()
        ctx_words = set(_content_words(combined_context))

        recalled = gt_words.intersection(ctx_words)
        recall_score = len(recalled) / max(1, len(gt_words))
        passed = recall_score >= 0.70

        return MetricScore(
            name="Context Recall",
            score=recall_score,
            passed=passed,
            explanation=f"Retrieved context contains {len(recalled)}/{len(gt_words)} ground-truth concepts.",
            details={"recalled_count": len(recalled), "total_gt_words": len(gt_words)},
        )

    def _eval_answer_correctness(self, answer: str, ground_truth: str) -> MetricScore:
        """Measure answer correctness against reference ground truth."""
        ans_tokens = _content_words(answer)
        gt_tokens = _content_words(ground_truth)

        prec, rec, f1 = _compute_f1(ans_tokens, gt_tokens)
        passed = f1 >= 0.50

        return MetricScore(
            name="Answer Correctness",
            score=f1,
            passed=passed,
            explanation=f"F1 token match against reference ground truth is {round(f1 * 100, 1)}%.",
            details={"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3)},
        )

    def evaluate_benchmark(
        self,
        rag_pipeline,
        test_dataset: List[Dict[str, str]],
    ) -> EvaluationReport:
        """Run batch evaluation on a benchmark test dataset.

        Args:
            rag_pipeline: RAGPipeline instance
            test_dataset: List of dicts, each with 'query' and optional 'ground_truth'
        """
        results: List[RAGEvaluationResult] = []
        weak_queries = []

        for item in test_dataset:
            query = item["query"]
            gt = item.get("ground_truth")

            # Execute pipeline
            rag_resp = rag_pipeline.query(query_text=query)

            # Evaluate response
            contexts = [r.document.page_content for r in rag_resp.reranked_chunks] or [
                d.page_content for d, _ in rag_resp.retrieved_chunks
            ]
            eval_res = self.evaluate(
                query=query,
                answer=rag_resp.answer,
                retrieved_contexts=contexts,
                ground_truth=gt,
            )
            results.append(eval_res)

            if eval_res.composite_score < 0.65:
                weak_queries.append({
                    "query": query,
                    "score": round(eval_res.composite_score, 3),
                    "reason": eval_res.recommendations[0] if eval_res.recommendations else "Low overall quality",
                })

        # Calculate averages
        n = max(1, len(results))
        mean_comp = sum(r.composite_score for r in results) / n
        mean_faith = sum(r.faithfulness.score for r in results) / n
        mean_ans_rel = sum(r.answer_relevance.score for r in results) / n
        mean_ctx_rel = sum(r.context_relevance.score for r in results) / n

        # Optional metrics
        recalls = [r.context_recall.score for r in results if r.context_recall is not None]
        mean_rec = sum(recalls) / len(recalls) if recalls else None

        corrs = [r.answer_correctness.score for r in results if r.answer_correctness is not None]
        mean_corr = sum(corrs) / len(corrs) if corrs else None

        return EvaluationReport(
            total_queries=len(results),
            mean_composite_score=mean_comp,
            mean_faithfulness=mean_faith,
            mean_answer_relevance=mean_ans_rel,
            mean_context_relevance=mean_ctx_rel,
            mean_context_recall=mean_rec,
            mean_answer_correctness=mean_corr,
            results=results,
            weak_queries=weak_queries,
        )


# Pre-packaged default RAG benchmark dataset based on sample_ai_rag_guide.txt
DEFAULT_RAG_BENCHMARK = [
    {
        "query": "What are the primary benefits of RAG over vanilla LLMs?",
        "ground_truth": (
            "RAG provides hallucination reduction by grounding models in verifiable documents, "
            "up-to-date knowledge without expensive retraining, traceable citations, and data privacy."
        ),
    },
    {
        "query": "What is the difference between a dense bi-encoder retriever and a cross-encoder reranker?",
        "ground_truth": (
            "A dense bi-encoder embeds queries and documents separately for fast vector search, "
            "while a cross-encoder evaluates query and document pairs simultaneously with full self-attention for deeper semantic precision."
        ),
    },
    {
        "query": "Why is chunk overlap important when splitting documents in RAG?",
        "ground_truth": (
            "Chunk overlap preserves semantic context across split boundaries, preventing critical sentences or ideas from being severed."
        ),
    },
    {
        "query": "How does Reciprocal Rank Fusion (RRF) combine dense and sparse BM25 retrieval?",
        "ground_truth": (
            "Reciprocal Rank Fusion blends dense vector and BM25 candidate ranks by summing inverted ranks "
            "weighted by a constant k, giving balanced priority to both keyword and semantic matches."
        ),
    },
]

