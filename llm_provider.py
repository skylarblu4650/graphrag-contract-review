"""
Central factory for LLM, embedding, and Semantic Kernel services.

Configure via environment variables:

  LLM_PROVIDER        : openai | anthropic | gemini | local  (default: openai)
  EMBEDDING_PROVIDER  : openai | gemini | local              (default: mirrors LLM_PROVIDER;
                        anthropic has no embedding model so it defaults to local)

  LLM_MODEL_ID        : model name for text generation       (provider-specific defaults below)
  EMBEDDING_MODEL_ID  : model name for embeddings            (provider-specific defaults below)
  EMBEDDING_DIMENSIONS: int, output dimensions of the embedding model (auto-detected when unset)

  OPENAI_API_KEY      : OpenAI API key
  ANTHROPIC_API_KEY   : Anthropic API key
  GOOGLE_API_KEY      : Google / Gemini API key
  LOCAL_BASE_URL      : base URL of the OpenAI-compatible server, e.g.
                        http://192.168.1.50:11434/v1   (Ollama)
                        http://localhost:1234/v1        (LM Studio)
  LOCAL_API_KEY       : API key for the local server (default: "local")

Default models per provider:
  openai    LLM: gpt-4o              embeddings: text-embedding-3-small (1536 dims)
  anthropic LLM: claude-3-5-sonnet-20241022  embeddings: all-MiniLM-L6-v2 via SentenceTransformer (384 dims)
  gemini    LLM: gemini-1.5-pro      embeddings: text-embedding-004 (768 dims)
  local     LLM: llama3.2            embeddings: nomic-embed-text   (768 dims)
"""

import os
from enum import Enum
from typing import Any


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    LOCAL = "local"


_DEFAULT_DIMS: dict[Provider, int] = {
    Provider.OPENAI: 1536,
    Provider.ANTHROPIC: 384,
    Provider.GEMINI: 768,
    Provider.LOCAL: 768,
}

_GEMINI_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"


def get_provider() -> Provider:
    val = os.getenv("LLM_PROVIDER", "openai").lower()
    try:
        return Provider(val)
    except ValueError:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{val}'. Valid options: openai, anthropic, gemini, local"
        )


def get_embedding_provider() -> Provider:
    val = os.getenv("EMBEDDING_PROVIDER", "").lower()
    if val:
        try:
            return Provider(val)
        except ValueError:
            raise ValueError(
                f"Unknown EMBEDDING_PROVIDER '{val}'. Valid options: openai, gemini, local"
            )
    p = get_provider()
    # Anthropic has no embedding model; fall back to local sentence transformers
    return Provider.LOCAL if p == Provider.ANTHROPIC else p


def get_embedding_dimensions() -> int:
    from_env = os.getenv("EMBEDDING_DIMENSIONS")
    if from_env:
        return int(from_env)
    return _DEFAULT_DIMS.get(get_embedding_provider(), 1536)


# ---------------------------------------------------------------------------
# Semantic Kernel service
# ---------------------------------------------------------------------------

def get_sk_service(service_id: str):
    """Return the correct Semantic Kernel chat-completion service for the active provider."""
    provider = get_provider()
    model_id = os.getenv("LLM_MODEL_ID")

    if provider == Provider.OPENAI:
        from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
        return OpenAIChatCompletion(
            ai_model_id=model_id or "gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            service_id=service_id,
        )

    if provider == Provider.ANTHROPIC:
        from semantic_kernel.connectors.ai.anthropic import AnthropicChatCompletion
        return AnthropicChatCompletion(
            ai_model_id=model_id or "claude-3-5-sonnet-20241022",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            service_id=service_id,
        )

    if provider == Provider.GEMINI:
        from semantic_kernel.connectors.ai.google.google_ai import GoogleAIChatCompletion
        return GoogleAIChatCompletion(
            gemini_model_id=model_id or "gemini-1.5-pro",
            api_key=os.getenv("GOOGLE_API_KEY"),
            service_id=service_id,
        )

    if provider == Provider.LOCAL:
        from openai import AsyncOpenAI
        from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
        async_client = AsyncOpenAI(
            api_key=os.getenv("LOCAL_API_KEY", "local"),
            base_url=os.getenv("LOCAL_BASE_URL", "http://localhost:11434/v1"),
        )
        return OpenAIChatCompletion(
            ai_model_id=model_id or "llama3.2",
            async_client=async_client,
            service_id=service_id,
        )

    raise ValueError(f"Unhandled provider: {provider}")


# ---------------------------------------------------------------------------
# neo4j-graphrag LLM  (used for Text2Cypher)
# ---------------------------------------------------------------------------

def get_graphrag_llm():
    """Return a neo4j_graphrag LLM instance for the active provider."""
    provider = get_provider()
    model_id = os.getenv("LLM_MODEL_ID")

    if provider == Provider.OPENAI:
        from neo4j_graphrag.llm import OpenAILLM
        return OpenAILLM(
            model_name=model_id or "gpt-4o",
            model_params={"temperature": 0},
        )

    if provider == Provider.ANTHROPIC:
        from neo4j_graphrag.llm import AnthropicLLM
        return AnthropicLLM(
            model_name=model_id or "claude-3-5-sonnet-20241022",
            model_params={"temperature": 0, "max_tokens": 4096},
        )

    if provider == Provider.GEMINI:
        # Use Google's OpenAI-compatible endpoint so we reuse OpenAILLM
        from neo4j_graphrag.llm import OpenAILLM
        return OpenAILLM(
            model_name=model_id or "gemini-1.5-pro",
            model_params={"temperature": 0},
            base_url=_GEMINI_OPENAI_BASE,
            api_key=os.getenv("GOOGLE_API_KEY"),
        )

    if provider == Provider.LOCAL:
        from neo4j_graphrag.llm import OpenAILLM
        return OpenAILLM(
            model_name=model_id or "llama3.2",
            model_params={"temperature": 0},
            base_url=os.getenv("LOCAL_BASE_URL", "http://localhost:11434/v1"),
            api_key=os.getenv("LOCAL_API_KEY", "local"),
        )

    raise ValueError(f"Unhandled provider: {provider}")


# ---------------------------------------------------------------------------
# neo4j-graphrag embedder
# ---------------------------------------------------------------------------

def get_graphrag_embedder():
    """Return a neo4j_graphrag embedder for the active embedding provider."""
    provider = get_embedding_provider()
    model_id = os.getenv("EMBEDDING_MODEL_ID")

    if provider == Provider.OPENAI:
        from neo4j_graphrag.embeddings import OpenAIEmbeddings
        return OpenAIEmbeddings(model=model_id or "text-embedding-3-small")

    if provider == Provider.GEMINI:
        from neo4j_graphrag.embeddings import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=model_id or "text-embedding-004",
            base_url=_GEMINI_OPENAI_BASE,
            api_key=os.getenv("GOOGLE_API_KEY"),
        )

    if provider == Provider.LOCAL:
        local_base = os.getenv("LOCAL_BASE_URL", "")
        if local_base:
            from neo4j_graphrag.embeddings import OpenAIEmbeddings
            return OpenAIEmbeddings(
                model=model_id or "nomic-embed-text",
                base_url=local_base,
                api_key=os.getenv("LOCAL_API_KEY", "local"),
            )
        # No LOCAL_BASE_URL → fall through to SentenceTransformer
        from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
        return SentenceTransformerEmbeddings(model=model_id or "all-MiniLM-L6-v2")

    # ANTHROPIC or LOCAL without a base URL: use SentenceTransformer
    from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
    return SentenceTransformerEmbeddings(model=model_id or "all-MiniLM-L6-v2")
