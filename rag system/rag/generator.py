"""Generator module for LLM response synthesis with retrieved context and citations.
Integrates PromptBuilder for prompt engineering and BaseLLM clients for unified inference.
"""
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from rag.document import Document
from rag.llm import BaseLLM, LLMResponse, get_llm
from rag.prompt_builder import PromptBuilder, PromptPayload, DEFAULT_SYSTEM_INSTRUCTION


class Generator:
    """Orchestrates generation of answers using LLM backends and PromptBuilder."""

    def __init__(
        self,
        provider: str = "local_extractive",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1000,
        prompt_builder: Optional[PromptBuilder] = None,
        llm: Optional[BaseLLM] = None,
    ):
        """Initialize Generator.

        Args:
            provider: 'openai', 'gemini', 'groq', 'ollama', or 'local_extractive'
            model: Model name
            api_key: API Key for cloud providers
            system_prompt: System instructions
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            prompt_builder: Optional PromptBuilder instance
            llm: Optional pre-configured BaseLLM instance
        """
        self._provider = (provider or "local_extractive").lower()
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY") or os.environ.get("GROQ_API_KEY")
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_INSTRUCTION

        # Determine model
        if model:
            self._model = model
        elif self._provider == "openai":
            self._model = "gpt-4o-mini"
        elif self._provider == "gemini":
            self._model = "gemini-1.5-flash"
        elif self._provider == "groq":
            self._model = "llama-3.3-70b-versatile"
        elif self._provider == "ollama":
            self._model = "llama3"
        else:
            self._model = "extractive-synthesizer-v2"

        # Initialize PromptBuilder
        self.prompt_builder = prompt_builder or PromptBuilder(
            template_type="grounded_qa",
            system_instruction=self._system_prompt,
        )

        # Initialize LLM instance
        self.llm = llm or self._build_llm()

    def _build_llm(self) -> BaseLLM:
        return get_llm(
            provider=self._provider,
            model=self._model,
            api_key=self._api_key,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

    # Dynamic property synchronization for interactive UI updates
    @property
    def provider(self) -> str:
        return self._provider

    @provider.setter
    def provider(self, val: str):
        self._provider = val.lower()
        self.llm = self._build_llm()

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, val: str):
        self._model = val
        self.llm = self._build_llm()

    @property
    def api_key(self) -> Optional[str]:
        return self._api_key

    @api_key.setter
    def api_key(self, val: Optional[str]):
        self._api_key = val
        self.llm = self._build_llm()

    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, val: float):
        self._temperature = val
        self.llm.temperature = val

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @max_tokens.setter
    def max_tokens(self, val: int):
        self._max_tokens = val
        self.llm.max_tokens = val

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, val: str):
        self._system_prompt = val
        self.prompt_builder.system_instruction = val

    def format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents into a context block with source tags."""
        return self.prompt_builder.format_context(documents)

    def generate(
        self,
        query: str,
        documents: List[Document],
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Generate response given user query and retrieved documents."""
        ans, _ = self.generate_detailed(query, documents, chat_history)
        return ans

    def generate_detailed(
        self,
        query: str,
        documents: List[Document],
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Tuple[str, LLMResponse]:
        """Generate response and return both text and full LLMResponse telemetry."""
        # 1. Build prompt payload using PromptBuilder
        payload: PromptPayload = self.prompt_builder.build_prompt(
            query=query,
            documents=documents,
            chat_history=chat_history,
        )

        # 2. Invoke LLM client
        llm_resp = self.llm.generate(
            prompt_payload=payload,
            documents=documents,
            query=query,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        return llm_resp.content, llm_resp
