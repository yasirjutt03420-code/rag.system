"""Prompt Builder module for RAG pipelines.
Provides templated prompt construction, few-shot examples, dynamic context formatting,
citation styling, and role instructions.
"""
from dataclasses import dataclass, field
import json
import re
from typing import Any, Dict, List, Optional, Union

from rag.document import Document


# Pre-defined system instructions
DEFAULT_SYSTEM_INSTRUCTION = """You are an accurate, helpful, and grounded AI Assistant.
Your task is to answer user questions using ONLY the facts present in the provided Context.
Guidelines:
1. Base your answer strictly on the facts present in the Context. Do not invent or extrapolate information.
2. If the context does not contain enough information to answer the question, clearly state:
   "I don't have enough information in the provided documents to answer this question."
3. Always include specific bracketed citations referencing the exact source documents (e.g. [Document 1 | source.pdf, p.3]).
4. Maintain a clear, professional, and well-structured markdown response."""

CONVERSATIONAL_INSTRUCTION = """You are a conversational knowledge assistant with access to curated document context.
Your goal is to answer questions collaboratively, naturally, and precisely using the provided context while remembering previous turns.
Guidelines:
1. Prioritize context facts over parametric memory.
2. If the user asks a follow-up question, resolve pronouns and references using conversational history.
3. Cite your sources clearly at the end of key statements or paragraphs.
4. If the context lacks the answer, politely inform the user and suggest what related information is available."""

EXECUTIVE_SUMMARY_INSTRUCTION = """You are an executive knowledge synthesizer.
Analyze the provided document context and formulate an executive-level summary and direct answers.
Guidelines:
1. Start with a 1-2 sentence high-level executive takeaway.
2. Use bulleted key findings with bold lead-ins.
3. Cite sources for every key finding.
4. Conclude with actionable implications or next steps mentioned in the context."""


# Pre-defined prompt templates
BUILTIN_TEMPLATES = {
    "grounded_qa": {
        "name": "Strict Grounded Q&A",
        "description": "Anti-hallucination grounded answering with strict source attribution.",
        "system_instruction": DEFAULT_SYSTEM_INSTRUCTION,
        "user_template": (
            "Context Information:\n"
            "---------------------\n"
            "{context}\n"
            "---------------------\n\n"
            "Question: {query}\n\n"
            "Provide a grounded, cited answer using only the context above."
        ),
    },
    "conversational": {
        "name": "Conversational Assistant",
        "description": "Multi-turn conversational assistant with context integration.",
        "system_instruction": CONVERSATIONAL_INSTRUCTION,
        "user_template": (
            "Context Information:\n"
            "---------------------\n"
            "{context}\n"
            "---------------------\n\n"
            "{chat_history_section}"
            "User Question: {query}\n\n"
            "Answer the question helpfully and cite references from the context."
        ),
    },
    "executive_summary": {
        "name": "Executive Summary & Insights",
        "description": "Structured executive brief with key findings and citations.",
        "system_instruction": EXECUTIVE_SUMMARY_INSTRUCTION,
        "user_template": (
            "Document Evidence:\n"
            "=====================\n"
            "{context}\n"
            "=====================\n\n"
            "Inquiry: {query}\n\n"
            "Synthesize an executive summary with key findings and precise citations."
        ),
    },
    "few_shot": {
        "name": "Few-Shot Grounded QA",
        "description": "Includes high-quality few-shot examples demonstrating proper citation formatting.",
        "system_instruction": DEFAULT_SYSTEM_INSTRUCTION,
        "user_template": (
            "Here are examples of how to answer using provided context:\n\n"
            "{few_shot_examples}\n\n"
            "Now answer the new inquiry using the provided context:\n\n"
            "Context:\n"
            "---------------------\n"
            "{context}\n"
            "---------------------\n\n"
            "Question: {query}\n"
            "Answer:"
        ),
    },
    "custom": {
        "name": "Custom User Template",
        "description": "User-defined prompt template with variable substitution.",
        "system_instruction": DEFAULT_SYSTEM_INSTRUCTION,
        "user_template": (
            "Context:\n{context}\n\n"
            "Question: {query}\n\n"
            "Answer:"
        ),
    },
}

# Default few-shot demonstration
DEFAULT_FEW_SHOT_EXAMPLES = [
    {
        "context": "[Doc 1 | architecture.pdf, p.2]\nRAG combines dense vector retrieval with an LLM generator to eliminate hallucinations.",
        "query": "How does RAG prevent hallucinations?",
        "answer": "RAG prevents hallucinations by grounding the LLM's parametric responses in external verifiable documents [Doc 1 | architecture.pdf, p.2].",
    },
    {
        "context": "[Doc 1 | reranking.txt]\nCross-encoders accept query and document simultaneously into self-attention layers to score relevance.",
        "query": "What is the key mechanism of a cross-encoder reranker?",
        "answer": "A cross-encoder processes both the query and document together through joint self-attention layers to capture fine-grained semantic interactions [Doc 1 | reranking.txt].",
    },
]


@dataclass
class PromptPayload:
    """The formatted result of PromptBuilder, ready to feed into any LLM."""
    system_prompt: str
    user_prompt: str
    full_prompt: str
    messages: List[Dict[str, str]]
    context_text: str
    template_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "full_prompt": self.full_prompt,
            "messages": self.messages,
            "context_length": len(self.context_text),
            "template_type": self.template_type,
            "metadata": self.metadata,
        }


class PromptBuilder:
    """Configurable Prompt Construction Engine for RAG Systems.
    
    Features:
    - Multiple prompt templates (Grounded QA, Conversational, Executive, Few-Shot, Custom)
    - Flexible citation formatting (bracketed, numeric, footnote, source-only)
    - Context length constraints / windowing
    - Chat history integration
    - Few-shot example injection
    - Custom variable placeholders
    """

    def __init__(
        self,
        template_type: str = "grounded_qa",
        system_instruction: Optional[str] = None,
        custom_user_template: Optional[str] = None,
        citation_style: str = "bracketed",
        max_context_chars: int = 12000,
        include_metadata: bool = True,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
    ):
        """Initialize PromptBuilder.

        Args:
            template_type: 'grounded_qa', 'conversational', 'executive_summary', 'few_shot', or 'custom'
            system_instruction: Custom override for the system prompt
            custom_user_template: Custom string template using {context}, {query}, etc.
            citation_style: 'bracketed' ([Doc 1 | source, p.2]), 'numeric' ([1]), or 'footnote' ([^1])
            max_context_chars: Truncation threshold for retrieved context
            include_metadata: Whether to include metadata (page, chunk_index) in document headers
            few_shot_examples: List of example dicts with keys (context, query, answer)
        """
        self.template_type = template_type if template_type in BUILTIN_TEMPLATES else "grounded_qa"
        self.system_instruction = system_instruction
        self.custom_user_template = custom_user_template
        self.citation_style = citation_style.lower()
        self.max_context_chars = max_context_chars
        self.include_metadata = include_metadata
        self.few_shot_examples = few_shot_examples or DEFAULT_FEW_SHOT_EXAMPLES

    def set_template(self, template_type: str, custom_template: Optional[str] = None):
        """Switch prompt template style."""
        if template_type in BUILTIN_TEMPLATES:
            self.template_type = template_type
        if custom_template:
            self.custom_user_template = custom_template
            self.template_type = "custom"

    def format_context(self, documents: List[Document]) -> str:
        """Format a list of retrieved Document objects into a structured context string."""
        if not documents:
            return "No relevant context documents were found."

        formatted_blocks = []
        for i, doc in enumerate(documents, start=1):
            src = doc.metadata.get("source") or doc.metadata.get("filename") or "document"
            page = doc.metadata.get("page")
            chunk_idx = doc.metadata.get("chunk_index")

            if self.citation_style == "numeric":
                header = f"[{i}] (Source: {src})"
            elif self.citation_style == "footnote":
                header = f"[^{i}] (Source: {src})"
            else:
                # Default bracketed citation
                details = []
                if self.include_metadata:
                    if page is not None:
                        details.append(f"p.{page}")
                    if chunk_idx is not None:
                        details.append(f"chk.{chunk_idx}")
                meta_suffix = f", {', '.join(details)}" if details else ""
                header = f"[Document {i} | {src}{meta_suffix}]"

            content = doc.page_content.strip()
            formatted_blocks.append(f"{header}\n{content}")

        full_context = "\n\n".join(formatted_blocks)

        # Enforce max context length limit
        if len(full_context) > self.max_context_chars:
            full_context = full_context[: self.max_context_chars] + "\n\n[... Context truncated for length ...]"

        return full_context

    def _format_few_shots(self) -> str:
        """Format few-shot examples into text block."""
        blocks = []
        for idx, ex in enumerate(self.few_shot_examples, start=1):
            block = (
                f"### Example {idx}:\n"
                f"Context:\n{ex.get('context', '')}\n"
                f"Question: {ex.get('query', '')}\n"
                f"Answer: {ex.get('answer', '')}"
            )
            blocks.append(block)
        return "\n\n".join(blocks)

    def _format_chat_history(self, chat_history: Optional[List[Dict[str, str]]]) -> str:
        """Format multi-turn chat history into a conversational context section."""
        if not chat_history:
            return ""

        lines = ["Recent Conversation History:"]
        for msg in chat_history[-6:]:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "").strip()
            # Truncate very long past turns
            if len(content) > 300:
                content = content[:300] + "..."
            lines.append(f"- {role}: {content}")
        lines.append("\n")
        return "\n".join(lines)

    def build_prompt(
        self,
        query: str,
        documents: List[Document],
        chat_history: Optional[List[Dict[str, str]]] = None,
        custom_variables: Optional[Dict[str, Any]] = None,
    ) -> PromptPayload:
        """Construct the prompt payload from query, documents, and chat history.

        Args:
            query: User question
            documents: Retrieved & reranked context documents
            chat_history: Prior conversational messages
            custom_variables: Extra placeholders for custom templates

        Returns:
            PromptPayload containing system prompt, user prompt, messages, and raw text
        """
        template_spec = BUILTIN_TEMPLATES.get(self.template_type, BUILTIN_TEMPLATES["grounded_qa"])

        # Determine system prompt
        system_prompt = self.system_instruction or template_spec["system_instruction"]

        # Format context text
        context_text = self.format_context(documents)

        # Determine user template string
        if self.template_type == "custom" and self.custom_user_template:
            user_template_str = self.custom_user_template
        else:
            user_template_str = template_spec["user_template"]

        # Prepare variables
        vars_dict = {
            "query": query.strip(),
            "context": context_text,
            "chat_history_section": self._format_chat_history(chat_history),
            "few_shot_examples": self._format_few_shots(),
            "document_count": len(documents),
            "citation_style": self.citation_style,
        }
        if custom_variables:
            vars_dict.update(custom_variables)

        # Substitute variables safely (handling missing keys gracefully)
        user_prompt = user_template_str
        for key, val in vars_dict.items():
            user_prompt = user_prompt.replace(f"{{{key}}}", str(val))

        # Build messages payload for chat models
        messages = [{"role": "system", "content": system_prompt}]
        if chat_history:
            for turn in chat_history[-6:]:
                messages.append({"role": turn.get("role", "user"), "content": turn.get("content", "")})
        messages.append({"role": "user", "content": user_prompt})

        full_prompt = f"System:\n{system_prompt}\n\nUser:\n{user_prompt}"

        return PromptPayload(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            full_prompt=full_prompt,
            messages=messages,
            context_text=context_text,
            template_type=self.template_type,
            metadata={
                "documents_count": len(documents),
                "citation_style": self.citation_style,
                "context_char_length": len(context_text),
            },
        )

