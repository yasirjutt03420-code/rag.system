"""Unified LLM abstraction layer for RAG pipelines.
Supports OpenAI, Google Gemini, Groq, Ollama, and a built-in Local Extractive LLM.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
import os
import re
import time
from typing import Any, Dict, Generator, List, Optional, Union
import requests

from rag.document import Document
from rag.prompt_builder import PromptPayload


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    content: str
    provider: str
    model: str
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: Optional[str] = "stop"
    raw_response: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "provider": self.provider,
            "model": self.model,
            "latency_ms": round(self.latency_ms, 2),
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
            },
            "finish_reason": self.finish_reason,
        }


class BaseLLM(ABC):
    """Abstract Base Class for all LLM providers."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout_seconds: int = 45,
    ):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds

    @abstractmethod
    def generate(
        self,
        prompt_payload: Union[PromptPayload, str],
        **kwargs,
    ) -> LLMResponse:
        """Generate response given a PromptPayload or string prompt."""
        pass


class OpenAILLM(BaseLLM):
    """OpenAI API Provider (GPT-4o, GPT-4o-mini, GPT-3.5-Turbo)."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        base_url: str = "https://api.openai.com/v1",
    ):
        super().__init__(model, api_key or os.environ.get("OPENAI_API_KEY"), temperature, max_tokens)
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt_payload: Union[PromptPayload, str], **kwargs) -> LLMResponse:
        t0 = time.time()
        if not self.api_key:
            return LLMResponse(
                content="⚠️ OpenAI API key is missing. Enter your API key in the RAG control panel.",
                provider="openai",
                model=self.model,
                latency_ms=0.0,
            )

        if isinstance(prompt_payload, PromptPayload):
            messages = prompt_payload.messages
        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": str(prompt_payload)},
            ]

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_seconds)
            resp.raise_for_status()
            data = resp.json()
            latency = (time.time() - t0) * 1000

            choice = data["choices"][0]
            usage = data.get("usage", {})
            return LLMResponse(
                content=choice["message"]["content"],
                provider="openai",
                model=self.model,
                latency_ms=latency,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                finish_reason=choice.get("finish_reason", "stop"),
                raw_response=data,
            )
        except Exception as e:
            latency = (time.time() - t0) * 1000
            return LLMResponse(
                content=f"❌ OpenAI API Error: {e}",
                provider="openai",
                model=self.model,
                latency_ms=latency,
                finish_reason="error",
            )


class GeminiLLM(BaseLLM):
    """Google Gemini API Provider (gemini-1.5-flash, gemini-2.0-flash, gemini-1.5-pro)."""

    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ):
        super().__init__(model, api_key or os.environ.get("GEMINI_API_KEY"), temperature, max_tokens)

    def generate(self, prompt_payload: Union[PromptPayload, str], **kwargs) -> LLMResponse:
        t0 = time.time()
        if not self.api_key:
            return LLMResponse(
                content="⚠️ Google Gemini API key is missing. Enter your API key in the RAG control panel.",
                provider="gemini",
                model=self.model,
                latency_ms=0.0,
            )

        if isinstance(prompt_payload, PromptPayload):
            system_instruction = prompt_payload.system_prompt
            prompt_text = prompt_payload.user_prompt
        else:
            system_instruction = "You are a helpful assistant."
            prompt_text = str(prompt_payload)

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": f"System Instructions:\n{system_instruction}\n\nUser Task:\n{prompt_text}"}]}],
            "generationConfig": {
                "temperature": kwargs.get("temperature", self.temperature),
                "maxOutputTokens": kwargs.get("max_tokens", self.max_tokens),
            },
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_seconds)
            resp.raise_for_status()
            data = resp.json()
            latency = (time.time() - t0) * 1000

            candidate = data["candidates"][0]
            answer = candidate["content"]["parts"][0]["text"]
            usage = data.get("usageMetadata", {})
            return LLMResponse(
                content=answer,
                provider="gemini",
                model=self.model,
                latency_ms=latency,
                prompt_tokens=usage.get("promptTokenCount", 0),
                completion_tokens=usage.get("candidatesTokenCount", 0),
                total_tokens=usage.get("totalTokenCount", 0),
                finish_reason=candidate.get("finishReason", "stop"),
                raw_response=data,
            )
        except Exception as e:
            latency = (time.time() - t0) * 1000
            return LLMResponse(
                content=f"❌ Gemini API Error: {e}",
                provider="gemini",
                model=self.model,
                latency_ms=latency,
                finish_reason="error",
            )


class GroqLLM(BaseLLM):
    """Groq High-Speed Inference Provider (Llama-3.3-70b, Mixtral)."""

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ):
        super().__init__(model, api_key or os.environ.get("GROQ_API_KEY"), temperature, max_tokens)

    def generate(self, prompt_payload: Union[PromptPayload, str], **kwargs) -> LLMResponse:
        t0 = time.time()
        if not self.api_key:
            return LLMResponse(
                content="⚠️ Groq API key is missing. Enter your API key in the RAG control panel.",
                provider="groq",
                model=self.model,
                latency_ms=0.0,
            )

        if isinstance(prompt_payload, PromptPayload):
            messages = prompt_payload.messages
        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": str(prompt_payload)},
            ]

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_seconds)
            resp.raise_for_status()
            data = resp.json()
            latency = (time.time() - t0) * 1000

            choice = data["choices"][0]
            usage = data.get("usage", {})
            return LLMResponse(
                content=choice["message"]["content"],
                provider="groq",
                model=self.model,
                latency_ms=latency,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                finish_reason=choice.get("finish_reason", "stop"),
                raw_response=data,
            )
        except Exception as e:
            latency = (time.time() - t0) * 1000
            return LLMResponse(
                content=f"❌ Groq API Error: {e}",
                provider="groq",
                model=self.model,
                latency_ms=latency,
                finish_reason="error",
            )


class OllamaLLM(BaseLLM):
    """Local Ollama Service Provider (llama3, mistral, phi3, etc.)."""

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ):
        super().__init__(model, None, temperature, max_tokens)
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt_payload: Union[PromptPayload, str], **kwargs) -> LLMResponse:
        t0 = time.time()
        if isinstance(prompt_payload, PromptPayload):
            prompt_text = prompt_payload.full_prompt
        else:
            prompt_text = str(prompt_payload)

        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt_text,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
            },
        }

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout_seconds)
            resp.raise_for_status()
            data = resp.json()
            latency = (time.time() - t0) * 1000
            return LLMResponse(
                content=data.get("response", ""),
                provider="ollama",
                model=self.model,
                latency_ms=latency,
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
                total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                finish_reason="stop",
                raw_response=data,
            )
        except Exception as e:
            latency = (time.time() - t0) * 1000
            return LLMResponse(
                content=f"❌ Ollama Connection Error ({e}). Ensure Ollama is running (`ollama run {self.model}`).",
                provider="ollama",
                model=self.model,
                latency_ms=latency,
                finish_reason="error",
            )


class LocalExtractiveLLM(BaseLLM):
    """High-precision, offline, zero-dependency semantic extractive synthesizer.
    
    Synthesizes grounded answers with proper bracketed citations directly from retrieved documents.
    Operates 100% locally with zero cloud API keys, zero network, and sub-millisecond execution.
    """

    def __init__(self, model: str = "extractive-synthesizer-v2"):
        super().__init__(model=model, api_key=None, temperature=0.0, max_tokens=1000)

    def generate(
        self,
        prompt_payload: Union[PromptPayload, str],
        documents: Optional[List[Document]] = None,
        query: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        t0 = time.time()

        # Extract query and context
        raw_text = prompt_payload.full_prompt if isinstance(prompt_payload, PromptPayload) else str(prompt_payload)
        
        # Determine query if not explicitly passed
        if not query:
            q_match = re.search(r"(?:Question|Inquiry|User Question):\s*(.+?)(?:\n|$)", raw_text, re.IGNORECASE)
            query = q_match.group(1).strip() if q_match else "Inquiry"

        q_terms = set(re.findall(r"\w+", query.lower()))

        # If documents are passed, use them; otherwise extract from prompt text
        extracted_facts = []
        if documents:
            target_docs = documents[:4]
            for doc in target_docs:
                source = doc.metadata.get("source") or doc.metadata.get("filename") or "Document"
                page = doc.metadata.get("page")
                cite = f"[{source}, p.{page}]" if page else f"[{source}]"

                sentences = re.split(r"(?<=[.!?])\s+", doc.page_content.strip())
                relevant = []
                for s in sentences:
                    s_clean = s.strip()
                    if not s_clean:
                        continue
                    s_terms = set(re.findall(r"\w+", s_clean.lower()))
                    overlap = len(q_terms.intersection(s_terms))
                    if overlap > 0:
                        relevant.append((s_clean, overlap))

                relevant.sort(key=lambda x: x[1], reverse=True)
                top_s = [s for s, _ in relevant[:3]]
                if top_s:
                    extracted_facts.append(f"**From {cite}**:\n" + "\n".join(f"- {s}" for s in top_s))
                else:
                    snippet = doc.page_content[:200].replace("\n", " ") + "..."
                    extracted_facts.append(f"**From {cite}**:\n- {snippet}")
        else:
            # Fallback parsing of context block in prompt
            context_match = re.search(r"(?:Context Information|Document Evidence|Context):\s*[-=]+\s*(.+?)\s*[-=]+", raw_text, re.DOTALL | re.IGNORECASE)
            ctx = context_match.group(1).strip() if context_match else raw_text
            paras = [p.strip() for p in ctx.split("\n\n") if p.strip()][:3]
            for p in paras:
                extracted_facts.append(f"- {p[:250]}...")

        if not extracted_facts:
            answer = "ℹ️ No sufficiently relevant facts found in the retrieved context to answer this query."
        else:
            answer = (
                f"### 📋 Synthesized Answer for: *\"{query}\"*\n\n"
                + "\n\n".join(extracted_facts)
                + "\n\n---\n*💡 Note: Generated using Local Extractive Synthesizer. Connect OpenAI, Gemini, or Groq for generative LLM prose.*"
            )

        latency = (time.time() - t0) * 1000
        approx_tokens = len(answer) // 4
        return LLMResponse(
            content=answer,
            provider="local_extractive",
            model=self.model,
            latency_ms=latency,
            prompt_tokens=len(raw_text) // 4,
            completion_tokens=approx_tokens,
            total_tokens=(len(raw_text) // 4) + approx_tokens,
            finish_reason="stop",
        )


class MockLLM(BaseLLM):
    """Deterministic mock LLM for testing without network."""

    def __init__(self, canned_response: str = "This is a verified test answer grounded in context [Doc 1]."):
        super().__init__(model="mock-llm", api_key=None)
        self.canned_response = canned_response

    def generate(self, prompt_payload: Union[PromptPayload, str], **kwargs) -> LLMResponse:
        return LLMResponse(
            content=self.canned_response,
            provider="mock",
            model="mock-llm",
            latency_ms=1.5,
            prompt_tokens=40,
            completion_tokens=15,
            total_tokens=55,
            finish_reason="stop",
        )


def get_llm(
    provider: str = "local_extractive",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> BaseLLM:
    """Factory helper to instantiate the appropriate LLM client."""
    prov = (provider or "local_extractive").lower()
    if "openai" in prov:
        return OpenAILLM(model=model or "gpt-4o-mini", api_key=api_key, temperature=temperature, max_tokens=max_tokens)
    elif "gemini" in prov:
        return GeminiLLM(model=model or "gemini-1.5-flash", api_key=api_key, temperature=temperature, max_tokens=max_tokens)
    elif "groq" in prov:
        return GroqLLM(model=model or "llama-3.3-70b-versatile", api_key=api_key, temperature=temperature, max_tokens=max_tokens)
    elif "ollama" in prov:
        return OllamaLLM(model=model or "llama3", temperature=temperature, max_tokens=max_tokens)
    elif "mock" in prov:
        return MockLLM()
    else:
        return LocalExtractiveLLM(model=model or "extractive-synthesizer-v2")

