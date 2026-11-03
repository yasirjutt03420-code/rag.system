# 🧠 Modular RAG Chat Board & Pipeline Inspector

An end-to-end, production-grade **Retrieval-Augmented Generation (RAG)** system featuring an interactive **Streamlit Chat Board**, a real-time **Pipeline Step Inspector**, and an automated **RAG Triad Evaluation & Benchmarking Dashboard**.

Built with 10 modular components:
1. **Data Loader** (`rag/data_loader.py`): Multi-format document parser (PDF, DOCX, PPTX, XLSX, CSV, TXT, MD, JSON, Python Code, and Web URLs).
2. **Data Splitter** (`rag/data_splitter.py`): Recursive character text splitter with semantic separator hierarchy and configurable chunk overlap.
3. **Embedding Engine** (`rag/embedding.py`): Dense vector embeddings powered by SentenceTransformers (`all-MiniLM-L6-v2`), OpenAI (`text-embedding-3-small`), Gemini, or TF-IDF.
4. **Vector Database** (`rag/vector_database.py`): Multi-collection persistent vector storage supporting Cosine Similarity, Dot Product, and L2 distance with disk serialization.
5. **Retriever** (`rag/retriever.py`): Hybrid Retrieval (Dense Vector + Sparse BM25 Lexical with Reciprocal Rank Fusion) and MMR (Maximal Marginal Relevance).
6. **Reranker** (`rag/reranker.py`): Cross-Encoder and Multi-signal Semantic-Lexical Reranker that scores query-document pairs to reorder candidates.
7. **Prompt Builder** (`rag/prompt_builder.py`): Dynamic prompt construction supporting built-in templates (Strict Grounded QA, Conversational Assistant, Executive Summary, Few-Shot), citation formatting (`[Document | Source, p.X]`, `[1]`, `[^1]`), and context length limits.
8. **Unified LLM Layer** (`rag/llm.py`): Pluggable client supporting OpenAI, Google Gemini, Groq, Ollama, and a zero-dependency Local Extractive Synthesizer with standardized response telemetry.
9. **Generator** (`rag/generator.py`): Connects PromptBuilder and LLM client to synthesize grounded answers with source citations.
10. **Evaluation Framework** (`rag/evaluation.py`): Comprehensive RAG Triad evaluator (Faithfulness / Groundedness, Answer Relevance, Context Relevance / Precision, Context Recall, Ground Truth Match, and Composite Quality Score) with batch benchmarking.

---

## 🏗️ Architecture Workflow

```
[Documents: PDF/DOCX/TXT/Web]
              │
              ▼
      [1. Data Loader]
              │
              ▼
     [2. Data Splitter] (Recursive Chunking with Overlap)
              │
              ▼
    [3. Embedding Engine] (Dense Vectors, e.g. 384-dim)
              │
              ▼
    [4. Vector Database] (Persistent Collections & Indexing)
              │
   ┌──────────┴──────────────────────────┐
   │                                     │
   ▼                                     ▼
[User Query] ───────────────► [5. Hybrid Retriever] (Dense + BM25 RRF)
                                         │  (Top-K Candidates)
                                         ▼
                                  [6. Reranker] (Cross-Attention Rescoring)
                                         │  (Top-N Refined Passages)
                                         ▼
                               [7. Prompt Builder] (Templates, Citations, Context)
                                         │  (Prompt Payload & Messages)
                                         ▼
                               [8. Unified LLM Layer] (Gemini / OpenAI / Groq / Local)
                                         │
                                         ▼
                               [9. Cited Answer & Telemetry]
                                         │
                                         ▼
                              [10. RAG Triad Evaluator]
                        (Faithfulness • Relevance • Precision)
```

---

## 🚀 Quickstart

### 1. Launch the Streamlit Chat Board & Inspector
```bash
python -m streamlit run app.py
```
Open your browser at **`http://localhost:8501`**.

### 2. Run Automated Unit & Integration Tests
Verify all 11 components:
```bash
python -m unittest tests/test_rag.py -v
```

---

## 🧩 Pipeline Components

### 1. Data Loader (`rag/data_loader.py`)
- Automatically detects and extracts text from:
  - **PDF** (`.pdf`) via `pypdf` (extracts text page-by-page with page metadata).
  - **Word** (`.docx`, `.doc`) via `python-docx` (extracts paragraphs and tables).
  - **Presentations** (`.pptx`, `.ppt`) via `python-pptx` (slide-by-slide extraction).
  - **Spreadsheets** (`.xlsx`, `.xls`, `.csv`) via `pandas` / `openpyxl`.
  - **Text & Code** (`.txt`, `.md`, `.json`, `.py`, `.html`, `.log`).
  - **Web URLs**: Scrapes and cleans web pages.

### 2. Data Splitter (`rag/data_splitter.py`)
- Recursive Character Text Splitting with delimiter hierarchy: `["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""]`.
- Configurable `chunk_size` and `chunk_overlap`.
- Preserves chunk metadata: `chunk_index`, `total_chunks`, `chunk_char_count`, `parent_id`.

### 3. Embedding Engine (`rag/embedding.py`)
- Local `sentence-transformers` (`all-MiniLM-L6-v2` - 384 dimensions, normalized).
- Cloud Embeddings: OpenAI (`text-embedding-3-small`) & Google Gemini (`text-embedding-004`).
- Deterministic TF-IDF fallback for zero-network testing.
- In-memory LRU cache to prevent duplicate vector calculations.

### 4. Vector Database (`rag/vector_database.py`)
- Multi-collection management (create, switch, delete, list).
- Persistent storage: vectors in `vectors.npz` and metadata in `metadata.json`.
- Similarity metrics: Cosine Similarity, Dot Product, Euclidean Distance.

### 5. Retriever (`rag/retriever.py`)
- **DenseRetriever**: Semantic vector similarity search.
- **BM25Retriever**: Token-level lexical BM25 search.
- **HybridRetriever**: Blends Dense Vector + BM25 using Reciprocal Rank Fusion:
  $$\text{RRF Score}(d) = \sum_{m \in \{dense, bm25\}} \frac{w_m}{k + \text{rank}_m(d)}$$
- **MMRRetriever**: Maximal Marginal Relevance balancing relevance vs redundancy.

### 6. Reranker (`rag/reranker.py`)
- Cross-attention rescoring of query-document pairs $(Query, Passage)$.
- Tracks initial vs reranked rank, score differentials, and rank shifts ($\Delta rank$).

### 7. Prompt Builder (`rag/prompt_builder.py`)
- Predefined prompt templates:
  - `grounded_qa`: Strict anti-hallucination instruction with source attribution.
  - `conversational`: Multi-turn chat context with pronoun resolution.
  - `executive_summary`: High-level takeaway, bulleted findings, and implications.
  - `few_shot`: Injected few-shot grounded citation demonstrations.
  - `custom`: User-editable string template with variable placeholders (`{query}`, `{context}`).
- Citation styles: `bracketed` (`[Document 1 | source, p.2]`), `numeric` (`[1]`), `footnote` (`[^1]`).
- Automatic context window length truncation.

### 8. Unified LLM Layer (`rag/llm.py`)
- Standardized `BaseLLM` interface and `LLMResponse` (latency, tokens, finish reason).
- Providers:
  - **Local Extractive Synthesizer**: Offline, instant, grounded sentence synthesis with citations.
  - **Google Gemini**: `gemini-1.5-flash`, `gemini-2.0-flash`.
  - **OpenAI**: `gpt-4o-mini`, `gpt-4o`.
  - **Groq**: `llama-3.3-70b-versatile`, `mixtral-8x7b-32768`.
  - **Ollama**: Local models (`llama3`, `mistral`, `phi3`).

### 9. RAG Evaluation Framework (`rag/evaluation.py`)
- Implements the complete **RAG Triad**:
  1. **Faithfulness / Groundedness**: Fraction of claims in the answer verified against retrieved context passages.
  2. **Answer Relevance**: Measure of how directly the generated answer addresses the user query.
  3. **Context Relevance / Precision**: Proportion of retrieved chunks containing query-pertinent facts.
  4. **Context Recall & Answer Correctness**: Token F1 match against ground-truth reference answers.
  5. **Composite Score**: Harmonic weighted quality index with actionable tuning recommendations.
- Automated batch benchmarking with pre-packaged reference evaluation sets.

---

## 💻 Python Code Usage Example

```python
from rag import RAGPipeline, DataLoader, PromptBuilder, RAGEvaluator

# 1. Initialize Pipeline with Prompt Builder & Local LLM
pipeline = RAGPipeline(
    collection_name="my_knowledge_base",
    embedding_model_name="all-MiniLM-L6-v2",
    reranker_enabled=True,
    generator_provider="local_extractive", # or "openai", "gemini", "groq"
)

# 2. Ingest Documents
docs = DataLoader.load_file("data/sample_ai_rag_guide.txt")
pipeline.ingest_documents(docs, chunk_size=500, chunk_overlap=60)

# 3. Query with Real-Time Evaluation
response = pipeline.query(
    query_text="What is the role of a Reranker in RAG?",
    top_k=8,
    top_n_rerank=4,
    retrieval_strategy="hybrid",
    evaluate=True, # Runs RAG Triad Evaluator automatically
)

# 4. Inspect Answer, Sources, & Telemetry
print("Generated Answer:\n", response.answer)
print("\nSources Used:", response.sources)
print("Pipeline Latency:", response.stats["total_ms"], "ms")

# 5. Inspect RAG Evaluation Metrics
if response.evaluation:
    ev = response.evaluation
    print(f"\n🛡️ Faithfulness: {ev.faithfulness.score * 100:.1f}% ({ev.faithfulness.explanation})")
    print(f"🎯 Answer Relevance: {ev.answer_relevance.score * 100:.1f}%")
    print(f"🔍 Context Relevance: {ev.context_relevance.score * 100:.1f}%")
    print(f"🏆 Composite Quality: {ev.composite_score * 100:.1f}%")

# 6. Run Batch Benchmark Suite
report = pipeline.evaluate_benchmark()
print(f"\nBenchmark Mean Composite: {report.mean_composite_score * 100:.1f}%")
```

---

## 📂 Project Structure

```
rag system/
├── app.py                     # Streamlit Chat Board, Ingestion, Step Inspector, & Evaluation Dashboard
├── requirements.txt           # Project dependencies
├── README.md                  # Comprehensive Documentation
├── data/
│   └── sample_ai_rag_guide.txt# Comprehensive sample RAG knowledge base
├── rag/
│   ├── __init__.py
│   ├── document.py            # Document dataclass & serialization
│   ├── data_loader.py         # Multi-format Document Loader (PDF, DOCX, PPTX, XLSX, CSV, TXT, Web)
│   ├── data_splitter.py       # Recursive text chunking with overlap & metadata
│   ├── embedding.py           # SentenceTransformers, OpenAI, Gemini, TF-IDF
│   ├── vector_database.py     # Persistent Vector DB with multi-collections & metadata filter
│   ├── retriever.py           # Dense, BM25, Hybrid RRF, and MMR retrievers
│   ├── reranker.py            # Cross-Encoder & Hybrid RRF Rerankers
│   ├── prompt_builder.py      # Templated prompt builder, citation styles, & few-shot injection
│   ├── llm.py                 # Unified LLM abstraction (OpenAI, Gemini, Groq, Ollama, Local Extractive)
│   ├── generator.py           # LLM Synthesizer connecting PromptBuilder & LLM Client
│   ├── evaluation.py          # RAG Triad Evaluator (Faithfulness, Relevance, Context, Benchmarking)
│   └── pipeline.py            # End-to-end RAG orchestrator with diagnostics & evaluation
├── tests/
│   └── test_rag.py            # Complete 11-stage unit and integration test suite
└── vector_store/              # Persistent on-disk vector database collections
```
