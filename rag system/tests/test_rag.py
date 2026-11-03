"""Comprehensive tests for RAG components:
DataLoader, DataSplitter, EmbeddingEngine, VectorDatabase, Retriever, Reranker,
PromptBuilder, LLM layer, Evaluation framework, and Pipeline.
"""
import os
import shutil
import tempfile
import unittest
import numpy as np

from rag.document import Document
from rag.data_loader import DataLoader
from rag.data_splitter import DataSplitter
from rag.embedding import EmbeddingEngine
from rag.vector_database import VectorDatabase
from rag.retriever import DenseRetriever, HybridRetriever, MMRRetriever
from rag.reranker import CrossEncoderReranker
from rag.prompt_builder import PromptBuilder, PromptPayload
from rag.llm import BaseLLM, LLMResponse, MockLLM, LocalExtractiveLLM, get_llm
from rag.evaluation import RAGEvaluator, RAGEvaluationResult, DEFAULT_RAG_BENCHMARK
from rag.pipeline import RAGPipeline


class TestRAGComponents(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_01_document_creation(self):
        doc = Document(page_content="Test content", metadata={"source": "test.txt"})
        self.assertIsNotNone(doc.id)
        self.assertEqual(doc.page_content, "Test content")
        self.assertEqual(doc.metadata["source"], "test.txt")

        # Dict conversion
        d = doc.to_dict()
        doc2 = Document.from_dict(d)
        self.assertEqual(doc.id, doc2.id)
        self.assertEqual(doc.page_content, doc2.page_content)

    def test_02_data_loader(self):
        # Test text loading
        text = "Artificial Intelligence is transforming industries worldwide."
        docs = DataLoader.load_text(text, source_name="ai_intro.txt")
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].metadata["source"], "ai_intro.txt")
        self.assertIn("Artificial Intelligence", docs[0].page_content)

        # Test writing and loading file
        test_file = os.path.join(self.test_dir, "sample.txt")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("Line 1\nLine 2\nLine 3")
        file_docs = DataLoader.load_file(test_file)
        self.assertEqual(len(file_docs), 1)
        self.assertIn("Line 2", file_docs[0].page_content)

    def test_03_data_splitter(self):
        text = (
            "Paragraph one introduces neural networks and deep learning.\n\n"
            "Paragraph two covers vector embeddings and similarity search.\n\n"
            "Paragraph three discusses cross-encoder reranking algorithms."
        )
        doc = Document(page_content=text, metadata={"source": "doc1.txt"})
        splitter = DataSplitter(chunk_size=90, chunk_overlap=15)
        chunks = splitter.split_documents([doc])

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.page_content), 120)  # reasonable margin
            self.assertEqual(chunk.metadata["source"], "doc1.txt")
            self.assertIn("chunk_index", chunk.metadata)

    def test_04_embedding_engine(self):
        engine = EmbeddingEngine(model_type="sentence-transformers", model_name="all-MiniLM-L6-v2")
        texts = [
            "Vector databases store high-dimensional embeddings.",
            "Cross-encoders score query and document together.",
        ]
        embs = engine.embed_documents(texts)
        self.assertEqual(embs.shape[0], 2)
        self.assertEqual(embs.shape[1], 384)

        # Check normalization (norm should be ~ 1.0)
        norm = np.linalg.norm(embs[0])
        self.assertAlmostEqual(norm, 1.0, places=4)

        # Query embedding
        q_emb = engine.embed_query("What is a vector database?")
        self.assertEqual(len(q_emb), 384)

    def test_05_vector_database(self):
        db = VectorDatabase(persist_directory=self.test_dir, collection_name="test_col")
        engine = EmbeddingEngine(model_type="sentence-transformers", model_name="all-MiniLM-L6-v2")

        docs = [
            Document(page_content="Apples, oranges, and bananas are fruits.", metadata={"category": "food"}),
            Document(page_content="Python, Rust, and C++ are programming languages.", metadata={"category": "code"}),
            Document(page_content="Dogs, wolves, and foxes are canines.", metadata={"category": "animal"}),
        ]
        embs = engine.embed_documents([d.page_content for d in docs])
        added_ids = db.add_documents(docs, embs)
        self.assertEqual(len(added_ids), 3)

        # Query vector search
        q_vec = engine.embed_query("What are some fruits?")
        results = db.search(q_vec, top_k=2)
        self.assertGreaterEqual(len(results), 1)
        best_doc, best_score = results[0]
        self.assertIn("fruits", best_doc.page_content.lower())

        # Test persistence
        db2 = VectorDatabase(persist_directory=self.test_dir, collection_name="test_col")
        self.assertEqual(len(db2.documents), 3)
        self.assertEqual(db2.vectors.shape, (3, 384))

    def test_06_retrievers(self):
        db = VectorDatabase(persist_directory=self.test_dir, collection_name="retriever_col")
        engine = EmbeddingEngine(model_type="sentence-transformers", model_name="all-MiniLM-L6-v2")

        docs = [
            Document(page_content="Vector databases like Chroma, FAISS, and Milvus index embeddings.", metadata={"source": "d1"}),
            Document(page_content="BM25 is a term-frequency based sparse lexical retriever algorithm.", metadata={"source": "d2"}),
            Document(page_content="Rerankers re-score candidate documents using cross-attention.", metadata={"source": "d3"}),
        ]
        embs = engine.embed_documents([d.page_content for d in docs])
        db.add_documents(docs, embs)

        # Dense retriever
        dense = DenseRetriever(db, engine)
        dense_res = dense.retrieve("BM25 algorithm", top_k=1)
        self.assertEqual(len(dense_res), 1)

        # Hybrid retriever
        hybrid = HybridRetriever(db, engine)
        hybrid_res = hybrid.retrieve("BM25 algorithm", top_k=2)
        self.assertGreaterEqual(len(hybrid_res), 1)
        self.assertIn("BM25", hybrid_res[0][0].page_content)

        # MMR retriever
        mmr = MMRRetriever(db, engine)
        mmr_res = mmr.retrieve("embeddings", top_k=2)
        self.assertEqual(len(mmr_res), 2)

    def test_07_reranker(self):
        reranker = CrossEncoderReranker()
        docs = [
            Document(page_content="The weather today is cloudy with light rain."),
            Document(page_content="A retriever fetches candidate chunks from a vector database."),
            Document(page_content="The capital of France is Paris."),
        ]
        candidates = [(docs[0], 0.3), (docs[1], 0.6), (docs[2], 0.2)]
        query = "How do retrievers fetch candidate chunks?"

        reranked = reranker.rerank(query, candidates, top_n=2)
        self.assertEqual(len(reranked), 2)
        # The retriever chunk should be ranked #1
        self.assertIn("retriever", reranked[0].document.page_content.lower())
        self.assertGreater(reranked[0].rerank_score, reranked[1].rerank_score)

    def test_08_end_to_end_pipeline(self):
        pipeline = RAGPipeline(
            persist_directory=self.test_dir,
            collection_name="pipeline_col",
            generator_provider="local_extractive",
            reranker_enabled=True,
        )

        text = (
            "Retrieval-Augmented Generation (RAG) combines search retrieval with generative AI. "
            "It eliminates hallucinations and provides grounded citations. "
            "Vector databases store embeddings and support nearest neighbor queries."
        )
        res = pipeline.ingest_text(text, source_name="rag_overview.txt", chunk_size=150, chunk_overlap=30)
        self.assertEqual(res["status"], "success")
        self.assertGreater(res["chunks_added"], 0)

        # Query with evaluation
        response = pipeline.query(
            "What does RAG combine and what does it eliminate?",
            top_k=3,
            top_n_rerank=2,
            evaluate=True,
        )
        self.assertIsNotNone(response.answer)
        self.assertGreater(len(response.reranked_chunks), 0)
        self.assertIn("rag_overview.txt", response.sources)
        self.assertIn("total_ms", response.stats)
        self.assertIsNotNone(response.prompt_payload)
        self.assertIsNotNone(response.llm_telemetry)
        self.assertIsNotNone(response.evaluation)
        self.assertGreater(response.evaluation.composite_score, 0.5)

    def test_09_prompt_builder(self):
        docs = [
            Document(page_content="BM25 is a lexical keyword search algorithm.", metadata={"source": "bm25.pdf", "page": 4}),
            Document(page_content="Dense embeddings represent semantic vectors.", metadata={"source": "vector.txt", "chunk_index": 1}),
        ]
        builder = PromptBuilder(template_type="grounded_qa", citation_style="bracketed")
        payload = builder.build_prompt(
            query="Explain BM25 search",
            documents=docs,
            chat_history=[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}],
        )
        self.assertIn("Explain BM25 search", payload.user_prompt)
        self.assertIn("BM25 is a lexical keyword", payload.context_text)
        self.assertIn("[Document 1 | bm25.pdf, p.4]", payload.context_text)
        self.assertEqual(payload.template_type, "grounded_qa")

        # Test numeric citation style
        builder_num = PromptBuilder(citation_style="numeric")
        ctx_num = builder_num.format_context(docs)
        self.assertIn("[1] (Source: bm25.pdf)", ctx_num)

        # Test conversational template
        builder_conv = PromptBuilder(template_type="conversational")
        payload_conv = builder_conv.build_prompt("Follow-up question", docs, chat_history=[{"role": "user", "content": "What is AI?"}])
        self.assertIn("What is AI?", payload_conv.user_prompt)

    def test_10_llm_abstraction(self):
        # Test Mock LLM
        mock_llm = MockLLM(canned_response="This is grounded in context [Doc 1].")
        resp = mock_llm.generate("Test query")
        self.assertEqual(resp.content, "This is grounded in context [Doc 1].")
        self.assertEqual(resp.provider, "mock")
        self.assertGreater(resp.total_tokens, 0)

        # Test LocalExtractiveLLM
        extractive_llm = LocalExtractiveLLM()
        doc = Document(page_content="Cross-encoders score query and passage jointly for high reranking precision.")
        prompt_b = PromptBuilder()
        payload = prompt_b.build_prompt("What do cross-encoders score?", [doc])
        resp_ext = extractive_llm.generate(payload, documents=[doc], query="What do cross-encoders score?")
        self.assertIn("Cross-encoders score", resp_ext.content)
        self.assertEqual(resp_ext.provider, "local_extractive")
        self.assertGreater(resp_ext.completion_tokens, 0)

        # Test get_llm factory
        llm = get_llm("local_extractive")
        self.assertIsInstance(llm, LocalExtractiveLLM)

    def test_11_evaluation_framework(self):
        evaluator = RAGEvaluator()
        query = "What is a vector database?"
        answer = "A vector database stores high-dimensional embeddings for nearest-neighbor similarity search [Doc 1]."
        contexts = [
            "Vector databases store high-dimensional vector embeddings and execute fast nearest-neighbor similarity search.",
            "Rerankers re-score candidate documents.",
        ]
        ground_truth = "A vector database indexes and stores high-dimensional embedding vectors for nearest neighbor search."

        eval_result = evaluator.evaluate(
            query=query,
            answer=answer,
            retrieved_contexts=contexts,
            ground_truth=ground_truth,
        )

        self.assertIsInstance(eval_result, RAGEvaluationResult)
        self.assertGreaterEqual(eval_result.faithfulness.score, 0.70)
        self.assertGreaterEqual(eval_result.answer_relevance.score, 0.70)
        self.assertGreater(eval_result.context_relevance.score, 0.30)
        self.assertIsNotNone(eval_result.answer_correctness)
        self.assertGreater(eval_result.answer_correctness.score, 0.50)
        self.assertGreater(eval_result.composite_score, 0.60)
        self.assertGreater(len(eval_result.recommendations), 0)


if __name__ == "__main__":
    unittest.main()
