"""RAG Chat Board & Pipeline Inspector: Interactive Dashboard.
Features: Multi-format Data Loader, Recursive Splitter, Dense/Hybrid/MMR Retriever,
Cross-Encoder Reranker, Prompt Builder, Unified LLM Client, and RAG Triad Evaluator.
"""
import os
import sys
import time
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

# Add current workspace to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from rag.document import Document
from rag.data_loader import DataLoader
from rag.data_splitter import DataSplitter
from rag.embedding import EmbeddingEngine
from rag.vector_database import VectorDatabase
from rag.retriever import DenseRetriever, HybridRetriever, MMRRetriever
from rag.reranker import CrossEncoderReranker
from rag.prompt_builder import PromptBuilder, BUILTIN_TEMPLATES, PromptPayload
from rag.llm import BaseLLM, LLMResponse, get_llm
from rag.evaluation import RAGEvaluator, RAGEvaluationResult, EvaluationReport, DEFAULT_RAG_BENCHMARK
from rag.generator import Generator
from rag.pipeline import RAGPipeline, RAGResponse

# Page configuration
st.set_page_config(
    page_title="Modular RAG Chat Board & Inspector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1E88E5, #7E57C2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .badge-metric {
        background-color: #f0f4f8;
        border: 1px solid #d9e2ec;
        border-radius: 8px;
        padding: 6px 12px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #243b53;
        display: inline-block;
        margin-right: 6px;
        margin-bottom: 8px;
    }
    .eval-badge {
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 6px;
        margin-top: 6px;
    }
    .eval-pass {
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #86efac;
    }
    .eval-warn {
        background-color: #fef9c3;
        color: #854d0e;
        border: 1px solid #fde047;
    }
    .eval-fail {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fca5a5;
    }
    .chunk-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #3b82f6;
        border-radius: 6px;
        padding: 12px;
        margin-bottom: 10px;
        font-size: 0.9rem;
    }
    .rerank-card-up {
        border-left: 4px solid #10b981;
        background-color: #f0fdf4;
    }
    .rerank-card-down {
        border-left: 4px solid #ef4444;
        background-color: #fef2f2;
    }
    .rerank-card-same {
        border-left: 4px solid #6b7280;
        background-color: #f9fafb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_rag_pipeline(
    persist_dir: str,
    collection_name: str,
    embedding_model_name: str,
    reranker_enabled: bool,
    cache_version: str = "v2.1",
):
    """Cache the core pipeline components to avoid reloading models on every interaction."""
    return RAGPipeline(
        persist_directory=persist_dir,
        collection_name=collection_name,
        embedding_model_name=embedding_model_name,
        reranker_enabled=reranker_enabled,
    )


# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "active_collection" not in st.session_state:
    st.session_state.active_collection = "default"

if "last_response" not in st.session_state:
    st.session_state.last_response = None

if "benchmark_report" not in st.session_state:
    st.session_state.benchmark_report = None

PERSIST_DIR = os.path.abspath("./vector_store")
os.makedirs(PERSIST_DIR, exist_ok=True)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.title("⚙️ RAG Control Panel")

    # 1. Collection Management
    st.subheader("📚 Knowledge Collections")
    existing_collections = VectorDatabase(PERSIST_DIR, "default").list_collections()
    if not existing_collections:
        existing_collections = ["default"]

    selected_collection = st.selectbox(
        "Active Collection",
        options=existing_collections,
        index=existing_collections.index(st.session_state.active_collection) if st.session_state.active_collection in existing_collections else 0,
    )
    if selected_collection != st.session_state.active_collection:
        st.session_state.active_collection = selected_collection
        st.rerun()

    new_col_name = st.text_input("➕ Create New Collection", placeholder="e.g. enterprise_docs")
    if st.button("Create Collection") and new_col_name:
        clean_name = new_col_name.strip().replace(" ", "_").lower()
        st.session_state.active_collection = clean_name
        st.success(f"Created '{clean_name}'")
        st.rerun()

    st.markdown("---")

    # 2. Text Splitting Parameters
    st.subheader("🧩 Chunking Parameters")
    chunk_size = st.slider("Chunk Size (characters)", min_value=150, max_value=1500, value=500, step=50)
    chunk_overlap = st.slider("Chunk Overlap (characters)", min_value=0, max_value=300, value=60, step=10)

    st.markdown("---")

    # 3. Retrieval & Reranker
    st.subheader("🔍 Retrieval & Reranker")
    retrieval_strategy = st.selectbox(
        "Retrieval Strategy",
        options=["Hybrid (BM25 + Dense RRF)", "Dense Vector Similarity", "MMR (Maximal Marginal Relevance)"],
        index=0,
    )
    strategy_code = "hybrid"
    if "Dense" in retrieval_strategy:
        strategy_code = "dense"
    elif "MMR" in retrieval_strategy:
        strategy_code = "mmr"

    top_k = st.slider("Retriever Candidates (Top-K)", min_value=2, max_value=20, value=8, step=1)
    use_reranker = st.toggle("Enable Reranker", value=True, help="Re-scores top candidate chunks with cross-attention / hybrid rank fusion.")
    top_n = st.slider("Reranked Output (Top-N)", min_value=1, max_value=10, value=4, step=1)

    st.markdown("---")

    # 4. Prompt Builder Configuration
    st.subheader("📝 Prompt Builder")
    template_options = {
        "grounded_qa": "Strict Grounded Q&A",
        "conversational": "Conversational Assistant",
        "executive_summary": "Executive Summary & Insights",
        "few_shot": "Few-Shot Grounded QA",
        "custom": "Custom Prompt Template",
    }
    selected_template_key = st.selectbox(
        "Prompt Template",
        options=list(template_options.keys()),
        format_func=lambda k: template_options[k],
        index=0,
    )

    citation_style = st.selectbox(
        "Citation Style",
        options=["bracketed", "numeric", "footnote"],
        format_func=lambda s: {
            "bracketed": "[Document N | Source, p.X]",
            "numeric": "Numbered [1], [2]",
            "footnote": "Footnote [^1], [^2]",
        }[s],
        index=0,
    )

    custom_system_override = None
    with st.expander("🛠️ Custom System Prompt", expanded=False):
        custom_system_override = st.text_area(
            "System Instruction",
            value=BUILTIN_TEMPLATES[selected_template_key]["system_instruction"],
            height=120,
        )

    st.markdown("---")

    # 5. LLM Generator
    st.subheader("🤖 LLM Inference")
    llm_provider = st.selectbox(
        "LLM Provider",
        options=["Local Extractive (Offline / Fast)", "Google Gemini", "OpenAI", "Groq", "Ollama (Local Server)"],
        index=0,
    )
    provider_code = "local_extractive"
    if "Gemini" in llm_provider:
        provider_code = "gemini"
    elif "OpenAI" in llm_provider:
        provider_code = "openai"
    elif "Groq" in llm_provider:
        provider_code = "groq"
    elif "Ollama" in llm_provider:
        provider_code = "ollama"

    api_key = ""
    model_name = None
    if provider_code in ["openai", "gemini", "groq"]:
        api_key = st.text_input(f"{llm_provider} API Key", type="password", help="API keys are kept in-memory only.")
        if provider_code == "openai":
            model_name = st.selectbox("OpenAI Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
        elif provider_code == "gemini":
            model_name = st.selectbox("Gemini Model", ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"])
        elif provider_code == "groq":
            model_name = st.selectbox("Groq Model", ["llama-3.3-70b-versatile", "mixtral-8x7b-32768", "llama3-8b-8192"])
    elif provider_code == "ollama":
        model_name = st.text_input("Ollama Model Name", value="llama3")

    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    st.markdown("---")

    # 6. Evaluation Configuration
    st.subheader("⚖️ RAG Evaluation")
    enable_realtime_eval = st.toggle(
        "⚡ Real-Time RAG Triad Evaluation",
        value=True,
        help="Evaluates Faithfulness, Answer Relevance, and Context Relevance for every query response.",
    )

    st.markdown("---")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_response = None
            st.rerun()
    with col_btn2:
        if st.button("🗑️ Reset DB", use_container_width=True):
            db = VectorDatabase(PERSIST_DIR, st.session_state.active_collection)
            db.clear()
            st.success("Cleared collection!")
            st.rerun()


# Initialize active pipeline
pipeline = get_rag_pipeline(
    persist_dir=PERSIST_DIR,
    collection_name=st.session_state.active_collection,
    embedding_model_name="all-MiniLM-L6-v2",
    reranker_enabled=use_reranker,
)
if pipeline.collection_name != st.session_state.active_collection:
    pipeline.switch_collection(st.session_state.active_collection)

# Defensive synchronization in case of cached resource transition
if not hasattr(pipeline, "prompt_builder") or pipeline.prompt_builder is None:
    pipeline.prompt_builder = PromptBuilder(template_type="grounded_qa")
if not hasattr(pipeline, "evaluator") or pipeline.evaluator is None:
    pipeline.evaluator = RAGEvaluator()
if not hasattr(pipeline.generator, "prompt_builder") or pipeline.generator.prompt_builder is None:
    pipeline.generator.prompt_builder = pipeline.prompt_builder

# Synchronize PromptBuilder settings
pipeline.prompt_builder.set_template(selected_template_key)
pipeline.prompt_builder.citation_style = citation_style
if custom_system_override:
    pipeline.prompt_builder.system_instruction = custom_system_override

# Synchronize LLM dynamic settings
pipeline.generator.provider = provider_code
pipeline.generator.temperature = temperature
if api_key:
    pipeline.generator.api_key = api_key
if model_name:
    pipeline.generator.model = model_name

# Retrieve collection stats
stats = pipeline.vector_db.get_stats()

# ----------------- MAIN HEADER -----------------
st.markdown('<div class="main-header">🧠 Modular RAG Chat Board & Inspector</div>', unsafe_allow_html=True)
st.caption("End-to-End RAG Architecture: Loader • Splitter • Embeddings • Vector DB • Retriever • Reranker • Prompt Builder • LLM • Evaluation")

# Badges Header
badge_html = f"""
<div>
    <span class="badge-metric">📁 Collection: <b>{stats['collection_name']}</b></span>
    <span class="badge-metric">📊 Indexed Chunks: <b>{stats['document_count']}</b></span>
    <span class="badge-metric">📐 Vector Dim: <b>{stats['vector_dimension']}d</b></span>
    <span class="badge-metric">⚡ Retriever: <b>{retrieval_strategy.split('(')[0].strip()} (k={top_k})</b></span>
    <span class="badge-metric">🎯 Reranker: <b>{'ON (n=' + str(top_n) + ')' if use_reranker else 'OFF'}</b></span>
    <span class="badge-metric">📝 Prompt: <b>{template_options[selected_template_key]}</b></span>
    <span class="badge-metric">🤖 LLM: <b>{pipeline.generator.provider.replace('_', ' ').title()}</b></span>
    <span class="badge-metric">⚖️ Eval: <b>{'Active' if enable_realtime_eval else 'Off'}</b></span>
</div>
"""
st.markdown(badge_html, unsafe_allow_html=True)

# ----------------- MAIN TABS -----------------
tab_chat, tab_ingest, tab_inspector, tab_evaluation = st.tabs([
    "💬 Interactive Chat Board",
    "📁 Data Ingestion Board",
    "🔬 Pipeline Step Inspector",
    "⚖️ RAG Evaluation & Benchmarks",
])

# ============================================================
# TAB 1: INTERACTIVE CHAT BOARD
# ============================================================
with tab_chat:
    col_chat, col_telemetry = st.columns([1.6, 1.0])

    with col_chat:
        st.subheader("💬 Chat with your Knowledge Base")

        # Quick Prompt suggestions
        st.write("💡 *Quick sample prompts:*")
        col_p1, col_p2, col_p3 = st.columns(3)
        sample_prompt = None
        with col_p1:
            if st.button("What is RAG architecture?", use_container_width=True):
                sample_prompt = "What is RAG architecture and what are its key benefits?"
        with col_p2:
            if st.button("Why use a Reranker?", use_container_width=True):
                sample_prompt = "Why is a Reranker essential and how does it differ from a bi-encoder retriever?"
        with col_p3:
            if st.button("Explain Chunking & Overlap", use_container_width=True):
                sample_prompt = "Explain chunk size, chunk overlap, and recursive character text splitting."

        # Display conversational chat history
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if "sources" in msg and msg["sources"]:
                        st.caption(f"📚 Sources: {', '.join(msg['sources'])}")

                    # Show real-time evaluation chip if available
                    if "eval" in msg and msg["eval"]:
                        ev = msg["eval"]
                        f_pct = round(ev["metrics"]["faithfulness"]["score"] * 100)
                        a_pct = round(ev["metrics"]["answer_relevance"]["score"] * 100)
                        c_pct = round(ev["metrics"]["context_relevance"]["score"] * 100)
                        comp_pct = ev["composite_percentage"]

                        f_cls = "eval-pass" if f_pct >= 70 else "eval-warn"
                        a_cls = "eval-pass" if a_pct >= 70 else "eval-warn"
                        c_cls = "eval-pass" if c_pct >= 60 else "eval-warn"

                        st.markdown(
                            f"""
                            <div>
                                <span class="eval-badge {f_cls}">🛡️ Faithfulness: {f_pct}%</span>
                                <span class="eval-badge {a_cls}">🎯 Relevance: {a_pct}%</span>
                                <span class="eval-badge {c_cls}">🔍 Precision: {c_pct}%</span>
                                <span class="eval-badge eval-pass">🏆 Composite: {comp_pct}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

        # Chat Input
        user_query = st.chat_input("Ask any question about your ingested documents...")
        prompt_to_run = sample_prompt or user_query

        if prompt_to_run:
            st.session_state.messages.append({"role": "user", "content": prompt_to_run})
            with st.chat_message("user"):
                st.markdown(prompt_to_run)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving candidates, reranking, and synthesizing answer..."):
                    history_payload = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[:-1]
                    ]
                    response = pipeline.query(
                        query_text=prompt_to_run,
                        top_k=top_k,
                        top_n_rerank=top_n,
                        retrieval_strategy=strategy_code,
                        use_reranker=use_reranker,
                        chat_history=history_payload,
                        evaluate=enable_realtime_eval,
                    )
                    st.markdown(response.answer)
                    if response.sources:
                        st.caption(f"📚 Sources: {', '.join(response.sources)}")

                    eval_dict = response.evaluation.to_dict() if response.evaluation else None
                    if eval_dict:
                        f_pct = round(eval_dict["metrics"]["faithfulness"]["score"] * 100)
                        a_pct = round(eval_dict["metrics"]["answer_relevance"]["score"] * 100)
                        c_pct = round(eval_dict["metrics"]["context_relevance"]["score"] * 100)
                        st.markdown(
                            f"""
                            <div>
                                <span class="eval-badge eval-pass">🛡️ Faithfulness: {f_pct}%</span>
                                <span class="eval-badge eval-pass">🎯 Relevance: {a_pct}%</span>
                                <span class="eval-badge eval-pass">🔍 Precision: {c_pct}%</span>
                                <span class="eval-badge eval-pass">🏆 Composite: {eval_dict['composite_percentage']}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response.answer,
                        "sources": response.sources,
                        "eval": eval_dict,
                    })
                    st.session_state.last_response = response

            st.rerun()

    with col_telemetry:
        st.subheader("📊 Retrieval & Reranker Telemetry")
        last_resp = st.session_state.last_response

        if last_resp is None:
            st.info("Ask a question in the chat to see real-time retrieval, reranker, and evaluation diagnostics.")
        else:
            stats_dict = last_resp.stats
            # Timing breakdown metrics
            t1, t2, t3, t4 = st.columns(4)
            t1.metric("Retriever", f"{stats_dict.get('retrieval_ms', 0)} ms")
            t2.metric("Reranker", f"{stats_dict.get('rerank_ms', 0)} ms")
            t3.metric("Prompt", f"{stats_dict.get('prompt_ms', 0)} ms")
            t4.metric("LLM Gen", f"{stats_dict.get('generate_ms', 0)} ms")

            st.markdown(f"**Total Query Latency:** `{stats_dict.get('total_ms', 0)} ms`")

            # LLM Token & Generation Stats
            if last_resp.llm_telemetry:
                llm_meta = last_resp.llm_telemetry
                u = llm_meta.get("usage", {})
                st.caption(f"🤖 **Model:** `{llm_meta.get('provider')}/{llm_meta.get('model')}` | **Tokens:** Prompt `{u.get('prompt_tokens', 0)}`, Output `{u.get('completion_tokens', 0)}`")

            # Reranker Shift Comparison Table
            st.markdown("#### 🎯 Reranker Ranking Shifts")
            if last_resp.reranked_chunks:
                table_rows = []
                for item in last_resp.reranked_chunks:
                    doc = item.document
                    src = doc.metadata.get("source") or doc.metadata.get("filename") or "doc"
                    p_num = doc.metadata.get("page")
                    loc = f"{src} (p.{p_num})" if p_num else src
                    
                    shift = item.rank_change
                    if shift > 0:
                        shift_str = f"▲ +{shift}"
                    elif shift < 0:
                        shift_str = f"▼ {shift}"
                    else:
                        shift_str = "— 0"

                    table_rows.append({
                        "Initial Rank": f"#{item.initial_rank + 1}",
                        "Reranked Rank": f"#{item.reranked_rank + 1}",
                        "Rank Shift": shift_str,
                        "Initial Score": f"{item.initial_score:.3f}",
                        "Rerank Score": f"{item.rerank_score:.3f}",
                        "Source": loc,
                    })
                st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

            # Expandable Source Passages
            with st.expander("📖 Inspect Top Reranked Passages", expanded=True):
                for i, r in enumerate(last_resp.reranked_chunks):
                    doc = r.document
                    src = doc.metadata.get("source", "Unknown")
                    card_class = "rerank-card-up" if r.rank_change > 0 else ("rerank-card-down" if r.rank_change < 0 else "rerank-card-same")
                    st.markdown(
                        f"""
                        <div class="chunk-card {card_class}">
                            <b>Rank #{i+1}</b> | <b>Source:</b> {src} | <b>Score:</b> {r.rerank_score:.3f} | <b>Shift:</b> {r.rank_change:+d}<br>
                            <p style="margin-top:6px; color:#334155;">{doc.page_content}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


# ============================================================
# TAB 2: DATA INGESTION BOARD
# ============================================================
with tab_ingest:
    st.subheader("📁 Ingest Documents into Vector Database")
    st.markdown("Upload files, enter raw text, or load the built-in comprehensive RAG guide to test instantly.")

    col_ing1, col_ing2 = st.columns([1.2, 1.0])

    with col_ing1:
        st.markdown("#### 1. Multi-Format File Uploader")
        uploaded_files = st.file_uploader(
            "Upload documents (PDF, DOCX, PPTX, XLSX, CSV, TXT, MD, JSON, Code)",
            type=["pdf", "docx", "doc", "pptx", "xlsx", "csv", "txt", "md", "json", "py"],
            accept_multiple_files=True,
        )

        if st.button("🚀 Ingest Uploaded Files", disabled=not uploaded_files, use_container_width=True):
            with st.spinner("Parsing files, chunking, and embedding..."):
                res = pipeline.ingest_files(
                    uploaded_files,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                st.success(f"✅ Successfully ingested {res['original_documents']} files into {res['chunks_added']} chunks in {res['time_ms']} ms!")
                st.rerun()

        st.markdown("---")
        st.markdown("#### 2. Direct Text / URL Ingestion")
        raw_text_input = st.text_area("Paste Raw Text or Notes", height=130, placeholder="Paste articles, meeting notes, policies, or technical docs here...")
        raw_source_name = st.text_input("Source Document Name", value="manual_notes.txt")

        if st.button("📥 Ingest Pasted Text", disabled=not raw_text_input.strip()):
            with st.spinner("Chunking and generating embeddings..."):
                res = pipeline.ingest_text(
                    raw_text_input,
                    source_name=raw_source_name,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                st.success(f"✅ Ingested text into {res['chunks_added']} chunks in {res['time_ms']} ms!")
                st.rerun()

    with col_ing2:
        st.markdown("#### 3. Quick-Start Sample Knowledge Base")
        st.info(
            "Want to test immediately without uploading? Load our pre-packaged **Comprehensive Guide to RAG Architecture** "
            "which covers loaders, splitters, embeddings, vector databases, retrievers, prompt builders, and evaluation."
        )

        sample_file_path = os.path.abspath("data/sample_ai_rag_guide.txt")
        if st.button("⚡ Load Sample RAG Guide (1-Click)", use_container_width=True):
            if os.path.exists(sample_file_path):
                with st.spinner("Indexing sample RAG guide..."):
                    docs = DataLoader.load_file(sample_file_path)
                    res = pipeline.ingest_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    st.success(f"✅ Ingested sample guide into {res['chunks_added']} chunks in {res['time_ms']} ms!")
                    st.rerun()
            else:
                st.error("Sample guide file not found.")

        # Collection status overview
        st.markdown("---")
        st.markdown("#### 📊 Current Collection Inventory")
        cur_stats = pipeline.vector_db.get_stats()
        st.write(f"**Collection:** `{cur_stats['collection_name']}`")
        st.write(f"**Total Stored Chunks:** `{cur_stats['document_count']}`")
        st.write(f"**Embedding Dimensions:** `{cur_stats['vector_dimension']} dims`")
        st.write(f"**Indexed Sources:** {cur_stats['unique_sources']}")

        if cur_stats["sources_list"]:
            with st.expander("📄 View Indexed File List", expanded=True):
                for src in cur_stats["sources_list"]:
                    st.markdown(f"- 📄 `{src}`")


# ============================================================
# TAB 3: PIPELINE STEP INSPECTOR
# ============================================================
with tab_inspector:
    st.subheader("🔬 Step-by-Step RAG Pipeline Inspector")
    st.caption("Visually inspect each layer: Loader -> Splitter -> Embedding -> Vector DB -> Retriever -> Reranker -> Prompt Builder")

    step_tab1, step_tab2, step_tab3, step_tab4, step_tab5 = st.tabs([
        "1️⃣ Data Loader & Splitter",
        "2️⃣ Embeddings & Vector DB",
        "3️⃣ Retriever Candidates",
        "4️⃣ Reranker Comparative Matrix",
        "5️⃣ Prompt Builder & Context Payload",
    ])

    # 1. DATA LOADER & SPLITTER
    with step_tab1:
        st.markdown("#### 🧩 Text Chunking and Metadata Inspection")
        if not pipeline.vector_db.documents:
            st.warning("No documents currently in the vector database. Ingest documents in Tab 2 to inspect them here.")
        else:
            all_docs = pipeline.vector_db.documents
            st.write(f"Total chunks in database: **{len(all_docs)}**")

            lengths = [len(d.page_content) for d in all_docs]
            df_lens = pd.DataFrame({"Chunk Length (Chars)": lengths})
            st.bar_chart(df_lens.head(50))

            selected_idx = st.slider("Select Chunk Index to Inspect", 0, len(all_docs) - 1, 0) if len(all_docs) > 1 else 0
            target_doc = all_docs[selected_idx]

            col_meta, col_content = st.columns([1, 2])
            with col_meta:
                st.markdown("**Chunk Metadata:**")
                st.json(target_doc.metadata)
                st.write(f"**Chunk ID:** `{target_doc.id}`")
            with col_content:
                st.markdown("**Chunk Text Content:**")
                st.info(target_doc.page_content)

    # 2. EMBEDDINGS & VECTOR DB
    with step_tab2:
        st.markdown("#### 📐 Embedding Vector Space Inspection")
        if pipeline.vector_db.vectors is None or len(pipeline.vector_db.vectors) == 0:
            st.warning("No vectors loaded yet.")
        else:
            vecs = pipeline.vector_db.vectors
            st.write(f"Vector Matrix Shape: **{vecs.shape[0]} chunks × {vecs.shape[1]} dimensions**")
            st.write(f"Embedding Engine: **{pipeline.embedding_engine.model_name}**")

            if len(vecs) > 0:
                inspect_vec_idx = st.selectbox("Select Vector to Inspect", list(range(min(len(vecs), 20))))
                if inspect_vec_idx is not None and inspect_vec_idx < len(vecs):
                    sample_vec = vecs[inspect_vec_idx]
                    v_norm = float(np.linalg.norm(sample_vec))
                    st.metric("Vector L2 Norm", f"{v_norm:.4f}")
                    st.markdown("**Vector Dimensions (First 32 values):**")
                    st.line_chart(sample_vec[:32])

    # 3. RETRIEVER CANDIDATES
    with step_tab3:
        st.markdown("#### 🔍 Test Retriever Independent of Generation")
        test_q = st.text_input("Test Retrieval Query", value="How does cosine similarity work in vector search?")
        test_k = st.slider("Retrieval Candidate Count", min_value=1, max_value=15, value=5, key="test_k_slider")

        if st.button("Run Retriever Test"):
            if not pipeline.vector_db.documents:
                st.warning("Please ingest documents first.")
            else:
                dense_hits = pipeline.dense_retriever.retrieve(test_q, top_k=test_k)
                hybrid_hits = pipeline.hybrid_retriever.retrieve(test_q, top_k=test_k)

                col_dense, col_hybrid = st.columns(2)
                with col_dense:
                    st.markdown("##### ⚡ Dense Vector Hits")
                    for r, (d, s) in enumerate(dense_hits):
                        st.markdown(f"**#{r+1} (Score: {s:.3f})** - `{d.metadata.get('source')}`")
                        st.caption(d.page_content[:180] + "...")

                with col_hybrid:
                    st.markdown("##### 🔀 Hybrid (Dense + BM25) Hits")
                    for r, (d, s) in enumerate(hybrid_hits):
                        st.markdown(f"**#{r+1} (Score: {s:.3f})** - `{d.metadata.get('source')}`")
                        st.caption(d.page_content[:180] + "...")

    # 4. RERANKER COMPARATIVE MATRIX
    with step_tab4:
        st.markdown("#### 🎯 Before Reranker vs. After Reranker Side-by-Side")
        cmp_query = st.text_input("Query for Comparative Reranking", value="What is the role of chunk overlap in text splitting?")
        cmp_k = st.slider("Initial Candidates", min_value=3, max_value=12, value=6, key="cmp_k_slider")

        if st.button("Run Comparative Rerank Analysis"):
            if not pipeline.vector_db.documents:
                st.warning("Please ingest documents first.")
            else:
                candidates = pipeline.hybrid_retriever.retrieve(cmp_query, top_k=cmp_k)
                reranked = pipeline.reranker.rerank(cmp_query, candidates, top_n=cmp_k)

                col_before, col_after = st.columns(2)
                with col_before:
                    st.markdown("##### 📥 Initial Retriever Output (Before Rerank)")
                    for i, (doc, score) in enumerate(candidates):
                        st.markdown(
                            f"""
                            <div class="chunk-card">
                                <b>Rank #{i+1}</b> | Initial Score: <code>{score:.3f}</code><br>
                                <small>Source: {doc.metadata.get('source')}</small><br>
                                <p style="margin-top:4px;">{doc.page_content[:180]}...</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                with col_after:
                    st.markdown("##### 🎯 Reranked Output (After Rerank)")
                    for item in reranked:
                        shift = item.rank_change
                        badge_color = "#10b981" if shift > 0 else ("#ef4444" if shift < 0 else "#6b7280")
                        badge_symbol = f"+{shift}" if shift > 0 else f"{shift}"
                        st.markdown(
                            f"""
                            <div class="chunk-card" style="border-left-color: {badge_color};">
                                <b>Rank #{item.reranked_rank + 1}</b> (Was #{item.initial_rank + 1}) | 
                                <span style="background:{badge_color}; color:white; padding:2px 6px; border-radius:4px; font-size:0.75rem;">Shift: {badge_symbol}</span><br>
                                Rerank Score: <code>{item.rerank_score:.3f}</code> | Initial: <code>{item.initial_score:.3f}</code><br>
                                <small>Source: {item.document.metadata.get('source')}</small><br>
                                <p style="margin-top:4px;">{item.document.page_content[:180]}...</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    # 5. PROMPT BUILDER & CONTEXT PAYLOAD
    with step_tab5:
        st.markdown("#### 📝 Prompt Builder & Context Payload Inspector")
        st.markdown("Examine the exact formatted context, citations, and system instruction sent to the LLM.")

        last_resp = st.session_state.last_response
        if last_resp is None or last_resp.prompt_payload is None:
            st.info("Run a query in the chat tab to inspect the constructed prompt payload here.")
        else:
            payload = last_resp.prompt_payload
            st.write(f"**Template Type:** `{payload.template_type}`")
            st.write(f"**Context Length:** `{len(payload.context_text)} characters`")

            col_p_sys, col_p_usr = st.columns(2)
            with col_p_sys:
                st.markdown("##### 🛡️ Active System Prompt")
                st.code(payload.system_prompt, language="markdown")

            with col_p_usr:
                st.markdown("##### 👤 User Prompt with Formatted Context & Citations")
                st.code(payload.user_prompt, language="markdown")

            with st.expander("📄 View Formatted Context Block Separately"):
                st.code(payload.context_text)


# ============================================================
# TAB 4: RAG EVALUATION & BENCHMARKS
# ============================================================
with tab_evaluation:
    st.subheader("⚖️ RAG Evaluation & Benchmarking Dashboard")
    st.caption("Measure and optimize your RAG system using the RAG Triad: Faithfulness, Answer Relevance, and Context Relevance.")

    eval_sub1, eval_sub2, eval_sub3 = st.tabs([
        "📊 Current Query Diagnostics",
        "🚀 Automated Benchmark Suite",
        "🧪 Interactive Metric Sandbox",
    ])

    # SUB-TAB 1: CURRENT QUERY DIAGNOSTICS
    with eval_sub1:
        st.markdown("#### 🔬 Detailed Triad Analysis of Last Chat Response")
        last_resp = st.session_state.last_response

        if last_resp is None or last_resp.evaluation is None:
            st.info("Ask a question in the chat with 'Real-Time RAG Evaluation' enabled to see the comprehensive diagnostic report.")
        else:
            ev: RAGEvaluationResult = last_resp.evaluation
            
            # Big Metric Cards
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("🛡️ Faithfulness", f"{round(ev.faithfulness.score * 100)}%", delta="Passed" if ev.faithfulness.passed else "Warning")
            m2.metric("🎯 Answer Relevance", f"{round(ev.answer_relevance.score * 100)}%", delta="Passed" if ev.answer_relevance.passed else "Warning")
            m3.metric("🔍 Context Relevance", f"{round(ev.context_relevance.score * 100)}%", delta="Passed" if ev.context_relevance.passed else "Warning")
            m4.metric("🏆 Composite Score", f"{round(ev.composite_score * 100, 1)}%", delta=f"Lat: {ev.latency_ms:.1f}ms")

            # Chart Representation
            chart_df = pd.DataFrame({
                "Metric": ["Faithfulness", "Answer Relevance", "Context Relevance", "Composite Score"],
                "Score (%)": [
                    round(ev.faithfulness.score * 100, 1),
                    round(ev.answer_relevance.score * 100, 1),
                    round(ev.context_relevance.score * 100, 1),
                    round(ev.composite_score * 100, 1),
                ]
            }).set_index("Metric")
            st.bar_chart(chart_df)

            # Recommendations
            st.markdown("##### 💡 Diagnostic Recommendations")
            for rec in ev.recommendations:
                st.markdown(f"- {rec}")

            # Sentence-by-sentence Claim Verification Table
            st.markdown("##### 🔎 Claim-by-Claim Faithfulness Verification")
            claims = ev.faithfulness.details.get("claims", [])
            if claims:
                claim_table = []
                for idx, c in enumerate(claims, start=1):
                    claim_table.append({
                        "#": idx,
                        "Claim Sentence": c["claim"],
                        "Context Overlap": f"{round(c['overlap_ratio'] * 100)}%",
                        "Grounded in Evidence": "✅ Supported" if c["supported"] else "❌ Ungrounded",
                    })
                st.dataframe(pd.DataFrame(claim_table), use_container_width=True, hide_index=True)
            else:
                st.caption("No contestable factual claims detected or model acknowledged lack of context.")

    # SUB-TAB 2: AUTOMATED BENCHMARK SUITE
    with eval_sub2:
        st.markdown("#### 🚀 Batch RAG Benchmark Suite")
        st.write(
            "Execute automated benchmark queries against the active knowledge base to assess overall system performance "
            "across multiple dimensions including Ground-Truth Answer Match."
        )

        col_b_info, col_b_run = st.columns([2, 1])
        with col_b_info:
            st.write(f"**Loaded Benchmark Dataset:** {len(DEFAULT_RAG_BENCHMARK)} verified RAG architectural questions with reference ground truths.")
        with col_b_run:
            run_bench = st.button("🏁 Run Full Benchmark Suite", use_container_width=True)

        if run_bench:
            with st.spinner("Executing benchmark queries, evaluating RAG Triad and ground truth matches..."):
                report: EvaluationReport = pipeline.evaluate_benchmark(DEFAULT_RAG_BENCHMARK)
                st.session_state.benchmark_report = report
                st.success("✅ Benchmark evaluation finished!")

        report = st.session_state.benchmark_report
        if report:
            st.markdown("---")
            st.markdown("### 📈 Aggregated Benchmark Performance")

            bm1, bm2, bm3, bm4, bm5 = st.columns(5)
            bm1.metric("Composite Score", f"{round(report.mean_composite_score * 100, 1)}%")
            bm2.metric("Mean Faithfulness", f"{round(report.mean_faithfulness * 100, 1)}%")
            bm3.metric("Mean Answer Rel.", f"{round(report.mean_answer_relevance * 100, 1)}%")
            bm4.metric("Mean Context Rel.", f"{round(report.mean_context_relevance * 100, 1)}%")
            if report.mean_answer_correctness is not None:
                bm5.metric("Ground Truth Match", f"{round(report.mean_answer_correctness * 100, 1)}%")

            # Detailed results table
            st.markdown("#### 📋 Query Benchmark Results Breakdown")
            table_rows = []
            for r in report.results:
                corr_str = f"{round(r.answer_correctness.score * 100)}%" if r.answer_correctness else "N/A"
                table_rows.append({
                    "Query": r.query,
                    "Composite": f"{round(r.composite_score * 100)}%",
                    "Faithfulness": f"{round(r.faithfulness.score * 100)}%",
                    "Relevance": f"{round(r.answer_relevance.score * 100)}%",
                    "Context Rel.": f"{round(r.context_relevance.score * 100)}%",
                    "GT Match": corr_str,
                    "Status": "✅ Pass" if r.composite_score >= 0.65 else "⚠️ Needs Tune",
                })
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    # SUB-TAB 3: INTERACTIVE METRIC SANDBOX
    with eval_sub3:
        st.markdown("#### 🧪 Test Any Query, Answer, & Context Triplet")
        sandbox_q = st.text_input("Query", value="What is the role of a reranker in RAG?")
        sandbox_ans = st.text_area(
            "Answer to Evaluate",
            value="A reranker scores query and candidate document passages together using deep cross-attention, eliminating false positives from initial bi-encoder vector search [Doc 1].",
            height=80,
        )
        sandbox_ctx = st.text_area(
            "Context Chunks (separate multiple chunks with double newline)",
            value="A cross-encoder reranker evaluates the query and passage simultaneously through transformer self-attention layers to score relevance precisely.",
            height=100,
        )
        sandbox_gt = st.text_input("Optional Reference Ground Truth", value="A reranker rescores candidate documents using cross-attention.")

        if st.button("⚡ Evaluate Sandbox Input"):
            evaluator = RAGEvaluator()
            chunks = [c.strip() for c in sandbox_ctx.split("\n\n") if c.strip()]
            res = evaluator.evaluate(
                query=sandbox_q,
                answer=sandbox_ans,
                retrieved_contexts=chunks,
                ground_truth=sandbox_gt if sandbox_gt.strip() else None,
            )

            sb1, sb2, sb3, sb4 = st.columns(4)
            sb1.metric("Faithfulness", f"{round(res.faithfulness.score * 100)}%")
            sb2.metric("Answer Relevance", f"{round(res.answer_relevance.score * 100)}%")
            sb3.metric("Context Relevance", f"{round(res.context_relevance.score * 100)}%")
            sb4.metric("Composite Quality", f"{round(res.composite_score * 100, 1)}%")

            st.write(f"**Faithfulness Explanation:** {res.faithfulness.explanation}")
            st.write(f"**Answer Relevance Explanation:** {res.answer_relevance.explanation}")
            st.write(f"**Context Relevance Explanation:** {res.context_relevance.explanation}")
            if res.answer_correctness:
                st.write(f"**Ground Truth Correctness:** {res.answer_correctness.explanation}")
