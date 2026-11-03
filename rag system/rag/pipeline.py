"""RAG Pipeline Coordinator: Connects DataLoader, Splitter, Embedder, VectorDB,
Retriever, Reranker, PromptBuilder, LLM, and Evaluator.
Provides end-to-end ingestion, querying, prompt engineering, and diagnostic evaluation.
"""
from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from rag.data_loader import DataLoader
from rag.data_splitter import DataSplitter
from rag.document import Document
from rag.embedding import EmbeddingEngine
from rag.evaluation import DEFAULT_RAG_BENCHMARK, EvaluationReport, RAGEvaluationResult, RAGEvaluator
from rag.generator import Generator
from rag.llm import BaseLLM, LLMResponse
from rag.prompt_builder import PromptBuilder, PromptPayload
from rag.reranker import CrossEncoderReranker, RerankResult, Reranker
from rag.retriever import DenseRetriever, HybridRetriever, MMRRetriever
from rag.vector_database import VectorDatabase


@dataclass
class RAGResponse:
    """Rich response containing generated answer, retrieved chunks, reranked chunks,
    prompt payload, LLM telemetry, and optional RAG evaluation metrics.
    """
    query: str
    answer: str
    retrieved_chunks: List[Tuple[Document, float]]  # (doc, initial_score)
    reranked_chunks: List[RerankResult]             # RerankResult objects
    sources: List[str]
    stats: Dict[str, Any] = field(default_factory=dict)
    prompt_payload: Optional[PromptPayload] = None
    llm_telemetry: Optional[Dict[str, Any]] = None
    evaluation: Optional[RAGEvaluationResult] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "query": self.query,
            "answer": self.answer,
            "sources": self.sources,
            "stats": self.stats,
            "retrieved_count": len(self.retrieved_chunks),
            "reranked_count": len(self.reranked_chunks),
            "llm_telemetry": self.llm_telemetry,
        }
        if self.prompt_payload:
            data["prompt"] = self.prompt_payload.to_dict()
        if self.evaluation:
            data["evaluation"] = self.evaluation.to_dict()
        return data


class RAGPipeline:
    """Complete Retrieval-Augmented Generation Pipeline with Evaluation."""

    def __init__(
        self,
        persist_directory: str = "./vector_store",
        collection_name: str = "default",
        embedding_model_type: str = "sentence-transformers",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        generator_provider: str = "local_extractive",
        generator_model: Optional[str] = None,
        api_key: Optional[str] = None,
        reranker_enabled: bool = True,
        prompt_builder: Optional[PromptBuilder] = None,
        evaluator: Optional[RAGEvaluator] = None,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.reranker_enabled = reranker_enabled

        # 1. Initialize Vector Database
        self.vector_db = VectorDatabase(
            persist_directory=persist_directory,
            collection_name=collection_name,
        )

        # 2. Initialize Embedding Engine
        self.embedding_engine = EmbeddingEngine(
            model_type=embedding_model_type,
            model_name=embedding_model_name,
            api_key=api_key,
        )

        # 3. Initialize Retrievers
        self.dense_retriever = DenseRetriever(self.vector_db, self.embedding_engine)
        self.hybrid_retriever = HybridRetriever(self.vector_db, self.embedding_engine)
        self.mmr_retriever = MMRRetriever(self.vector_db, self.embedding_engine)

        # 4. Initialize Reranker
        self.reranker: Reranker = CrossEncoderReranker()

        # 5. Initialize Prompt Builder
        self.prompt_builder = prompt_builder or PromptBuilder(template_type="grounded_qa")

        # 6. Initialize Generator (which wires PromptBuilder and BaseLLM)
        self.generator = Generator(
            provider=generator_provider,
            model=generator_model,
            api_key=api_key,
            prompt_builder=self.prompt_builder,
        )

        # 7. Initialize Evaluator
        self.evaluator = evaluator or RAGEvaluator()

    @property
    def llm(self) -> BaseLLM:
        """Direct access to the underlying LLM client."""
        return self.generator.llm

    def switch_collection(self, collection_name: str):
        """Switch active vector database collection."""
        self.collection_name = collection_name
        self.vector_db.switch_collection(collection_name)
        # Update retrievers
        self.dense_retriever = DenseRetriever(self.vector_db, self.embedding_engine)
        self.hybrid_retriever = HybridRetriever(self.vector_db, self.embedding_engine)
        self.mmr_retriever = MMRRetriever(self.vector_db, self.embedding_engine)

    def ingest_documents(
        self,
        documents: List[Document],
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> Dict[str, Any]:
        """Split documents into chunks, compute embeddings, and insert into Vector DB."""
        start_time = time.time()
        splitter = DataSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(documents)

        if not chunks:
            return {"status": "empty", "chunks_added": 0, "time_ms": 0}

        # Compute embeddings for all chunks
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embedding_engine.embed_documents(texts)

        # Store in vector database
        added_ids = self.vector_db.add_documents(chunks, embeddings)

        # Update hybrid retriever index
        self.hybrid_retriever._rebuild_bm25_index()

        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        return {
            "status": "success",
            "original_documents": len(documents),
            "chunks_added": len(chunks),
            "total_collection_chunks": len(self.vector_db.documents),
            "time_ms": elapsed_ms,
        }

    def ingest_files(
        self,
        file_inputs: List[Any],
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> Dict[str, Any]:
        """Load and ingest multiple files."""
        all_docs = []
        for file_in in file_inputs:
            docs = DataLoader.load_file(file_in)
            all_docs.extend(docs)

        return self.ingest_documents(all_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def ingest_text(
        self,
        text: str,
        source_name: str = "direct_input",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> Dict[str, Any]:
        """Ingest raw text into active collection."""
        docs = DataLoader.load_text(text, source_name=source_name)
        return self.ingest_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def query(
        self,
        query_text: str,
        top_k: int = 8,
        top_n_rerank: int = 4,
        retrieval_strategy: str = "hybrid",
        use_reranker: Optional[bool] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        evaluate: bool = False,
        ground_truth: Optional[str] = None,
    ) -> RAGResponse:
        """Run full RAG pipeline: Query -> Retrieve (top_k) -> Rerank (top_n) -> Prompt Build -> LLM -> Evaluate.

        Args:
            query_text: The user's question
            top_k: Number of candidate chunks to retrieve initially
            top_n_rerank: Number of chunks to select after reranking
            retrieval_strategy: 'hybrid', 'dense', or 'mmr'
            use_reranker: Override instance reranker_enabled
            chat_history: Previous conversational messages
            evaluate: Whether to automatically run RAGEvaluator on this query
            ground_truth: Optional expected reference answer for correctness evaluation

        Returns:
            RAGResponse object
        """
        t0 = time.time()
        apply_reranker = self.reranker_enabled if use_reranker is None else use_reranker

        # Phase 1: Retrieval
        t_ret_start = time.time()
        strat = retrieval_strategy.lower()
        if strat == "dense":
            retrieved_candidates = self.dense_retriever.retrieve(query_text, top_k=top_k)
        elif strat == "mmr":
            retrieved_candidates = self.mmr_retriever.retrieve(query_text, top_k=top_k)
        else:
            # Default: hybrid
            retrieved_candidates = self.hybrid_retriever.retrieve(query_text, top_k=top_k)
        t_ret_ms = round((time.time() - t_ret_start) * 1000, 2)

        # Phase 2: Reranking
        t_rerank_start = time.time()
        rerank_results: List[RerankResult] = []
        if apply_reranker and retrieved_candidates:
            rerank_results = self.reranker.rerank(
                query=query_text,
                candidates=retrieved_candidates,
                top_n=top_n_rerank,
            )
            final_docs = [r.document for r in rerank_results]
        else:
            # Bypass reranker: take top_n_rerank directly from retriever
            final_docs = [doc for doc, _ in retrieved_candidates[:top_n_rerank]]
            for idx, (doc, score) in enumerate(retrieved_candidates[:top_n_rerank]):
                rerank_results.append(
                    RerankResult(
                        document=doc,
                        initial_rank=idx,
                        reranked_rank=idx,
                        initial_score=score,
                        rerank_score=score,
                        rank_change=0,
                    )
                )
        t_rerank_ms = round((time.time() - t_rerank_start) * 1000, 2)

        # Phase 3: Prompt Building
        t_prompt_start = time.time()
        prompt_payload = self.prompt_builder.build_prompt(
            query=query_text,
            documents=final_docs,
            chat_history=chat_history,
        )
        t_prompt_ms = round((time.time() - t_prompt_start) * 1000, 2)

        # Phase 4: Generation via LLM
        t_gen_start = time.time()
        answer, llm_resp = self.generator.generate_detailed(
            query=query_text,
            documents=final_docs,
            chat_history=chat_history,
        )
        t_gen_ms = round((time.time() - t_gen_start) * 1000, 2)

        # Phase 5: Optional Evaluation
        eval_result = None
        t_eval_ms = 0.0
        if evaluate:
            t_eval_start = time.time()
            eval_result = self.evaluator.evaluate(
                query=query_text,
                answer=answer,
                retrieved_contexts=final_docs,
                ground_truth=ground_truth,
            )
            t_eval_ms = round((time.time() - t_eval_start) * 1000, 2)

        total_ms = round((time.time() - t0) * 1000, 2)

        # Extract unique sources
        sources = list(
            dict.fromkeys(
                doc.metadata.get("source") or doc.metadata.get("filename") or "Unknown"
                for doc in final_docs
            )
        )

        stats = {
            "retrieval_strategy": retrieval_strategy,
            "retrieved_candidates_count": len(retrieved_candidates),
            "reranked_final_count": len(final_docs),
            "reranker_applied": apply_reranker,
            "retrieval_ms": t_ret_ms,
            "rerank_ms": t_rerank_ms,
            "prompt_ms": t_prompt_ms,
            "generate_ms": t_gen_ms,
            "eval_ms": t_eval_ms,
            "total_ms": total_ms,
        }

        return RAGResponse(
            query=query_text,
            answer=answer,
            retrieved_chunks=retrieved_candidates,
            reranked_chunks=rerank_results,
            sources=sources,
            stats=stats,
            prompt_payload=prompt_payload,
            llm_telemetry=llm_resp.to_dict(),
            evaluation=eval_result,
        )

    def evaluate_benchmark(
        self,
        test_dataset: Optional[List[Dict[str, str]]] = None,
    ) -> EvaluationReport:
        """Run batch benchmark evaluation across the pipeline."""
        dataset = test_dataset or DEFAULT_RAG_BENCHMARK
        return self.evaluator.evaluate_benchmark(self, dataset)
