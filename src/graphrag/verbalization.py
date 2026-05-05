"""graphrag/verbalization.py

This module implements natural-language verbalization of serialized
graph content using large language models.

Given a serialized subgraph representation and a query, the module
produces a fluent natural-language answer.
"""

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureChatPromptExecutionSettings
from semantic_kernel.connectors.ai.ollama import OllamaTextCompletion, OllamaTextPromptExecutionSettings
from graphrag.utils.ollama_launcher import ensure_ollama_running
import asyncio
import nest_asyncio
import os

nest_asyncio.apply()

# intern variable to reuse the generator
_generators_cache = {}

_LLM_SERVICES = {}

_ollama_initialized = False

# one global loop
_LOOP = asyncio.new_event_loop()
asyncio.set_event_loop(_LOOP)

class NLGenerator:
    """
    Wrapper class for language-model-based verbalization and query generation.

    This class abstracts over different LLM backends (e.g., Ollama, Azure OpenAI)
    and provides a unified interface for generating
    verbalized answers from serialized graph content.
    """
    def __init__(
        self,
        model_type: str,  # "openai" / "ollama"
        model: str | None = None,
        api_key: str | None = None,
        endpoint: str | None = None,
        max_tokens: int = 200,
        temperature: float = 0.0
    ):
        self.kernel = Kernel()
        self.model_type = model_type.lower()
        self.max_tokens = max_tokens
        self.temperature = temperature

        if self.model_type == "openai":
            self.model = model or "gpt-4"
            self.llm_service = AzureChatCompletion(
                ai_model_id=self.model,
                api_key=api_key
            )
            self.settings =  AzureChatPromptExecutionSettings(
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
        elif self.model_type == "ollama":
            self.model = model or "llama3:8b"
            endpoint = endpoint or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
            key = (endpoint, model)
            if key not in _LLM_SERVICES:
                _LLM_SERVICES[key] = OllamaTextCompletion(ai_model_id=model, host=endpoint)

            self.llm_service = _LLM_SERVICES[key]

            self.settings = OllamaTextPromptExecutionSettings(
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")


    async def _generate_async(self, question: str, answer: str) -> str:
        # Prompt designed to map structured graph-derived content to a
        # natural-language sentence, without introducing external knowledge.
        prompt = f"""
        You will receive a question and an answer.
        Verbalize the answer as a clear, natural-sounding, complete English sentence.
        Question: {question}
        Answer: {answer}
        Return only the verbalized answer.
        """

        result = await self.llm_service.get_text_content(prompt=prompt, settings=self.settings)
        if result is None:
            return ""

        # different attributes for each model type
        if self.model_type == "ollama":
            return result.text if hasattr(result, "text") else str(result)
        elif self.model_type == "openai":
            return result.content if hasattr(result, "content") else str(result)
        return ""

    def generate_verbalization(self, question, answer):
        return _LOOP.run_until_complete(self._generate_async(question, answer))

    async def _generate_question_async(self, query_type: str, components: dict) -> str:
        """
        Generate a natural language question from a subgraph description.
        Works for arbitrary graphs.
        """

        labels = ", ".join(components.get("labels", []))

        edges_desc = []
        for e in components.get("edges", []):
            from_labels = ", ".join(e.get("from", []))
            to_labels = ", ".join(e.get("to", []))
            rel = e.get("relation")
            edges_desc.append(f"{from_labels} -[{rel}]-> {to_labels}")

        edges_text = "\n".join(edges_desc)

        props_text = ""
        if components.get("node_properties"):
            props_text = f"\nKnown node properties: {components['node_properties']}"

        prompt = f"""
    You are given a graph substructure extracted from a knowledge graph.

    Graph labels involved:
    {labels}

    Relationships:
    {edges_text}
    {props_text}

    Task:
    Generate ONE concise, natural-sounding English question
    that could be answered by exactly this graph structure.

    Rules:
    - Do NOT say "connected to"
    - Use the actual relationship names
    - Do NOT mention Cypher, graphs, or databases
    - Do NOT add explanations

    Return ONLY the question.
    """

        result = await self.llm_service.get_text_content(
            prompt=prompt,
            settings=self.settings
        )

        if result is None:
            return ""

        if self.model_type == "ollama":
            return result.text if hasattr(result, "text") else str(result)
        elif self.model_type == "openai":
            return result.content if hasattr(result, "content") else str(result)

        return str(result)

    def generate_question(self, query_type, components):
        return _LOOP.run_until_complete(self._generate_question_async(query_type, components))

def generate_verbalization(answer, question, model, model_type="ollama",api_key=None, endpoint=None) -> str:
    if model_type.lower() == "ollama":
        global _ollama_initialized
        if not _ollama_initialized:
            ensure_ollama_running()
    key = (model_type, model, api_key, endpoint)
    if key not in _generators_cache:
        _generators_cache[key] = NLGenerator(
            model_type=model_type,
            model=model,
            api_key=api_key,
            endpoint=endpoint
        )
    generator = _generators_cache[key]
    return generator.generate_verbalization(question, answer)

def generate_question(query_type: str, components: dict, model, model_type="ollama",api_key=None, endpoint=None) -> str:
    if model_type.lower() == "ollama":
        global _ollama_initialized
        if not _ollama_initialized:
            ensure_ollama_running()
    key = (model_type, model, api_key, endpoint)
    if key not in _generators_cache:
        _generators_cache[key] = NLGenerator(
            model_type=model_type,
            model=model,
            api_key=api_key,
            endpoint=endpoint
        )
    generator = _generators_cache[key]
    return generator.generate_question(query_type, components)

