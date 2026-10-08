"""
Factory tạo LLM và Embeddings cho 5 providers: openai, gemini, anthropic, ollama, openrouter.

Cách dùng:
    from utils.llm_factory import get_llm, get_embeddings

    llm        = get_llm()            # dùng PROVIDER từ .env
    embeddings = get_embeddings()     # dùng PROVIDER từ .env

    llm_gemini = get_llm("gemini")    # chỉ định provider cụ thể
"""
import sys
import time
import re
from pathlib import Path
from langchain_core.rate_limiters import InMemoryRateLimiter

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Rate limiter an toàn cho Gemini Free Tier (15 RPM -> 0.2 req/s ~ 12 RPM)
_gemini_rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.2,
    check_every_n_seconds=0.1,
    max_bucket_size=1,
)


def get_llm(provider: str = None, temperature: float = 0.0):
    """
    Trả về BaseChatModel tương ứng với provider được chọn.

    Args:
        provider    : "openai" | "gemini" | "anthropic" | "ollama" | "openrouter"
                      Mặc định: đọc PROVIDER từ .env (config.PROVIDER)
        temperature : độ ngẫu nhiên (0.0 = tất định, 1.0 = sáng tạo)

    Returns:
        BaseChatModel instance sẵn sàng sử dụng

    Raises:
        ValueError nếu provider không hợp lệ
        ImportError nếu package tương ứng chưa được cài đặt
    """
    provider = (provider or config.PROVIDER).lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        kwargs = {
            "model": config.OPENAI_MODEL,
            "api_key": config.OPENAI_API_KEY,
            "temperature": temperature,
        }
        if config.OPENAI_BASE_URL:
            kwargs["base_url"] = config.OPENAI_BASE_URL
        return ChatOpenAI(**kwargs)

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=temperature,
            rate_limiter=_gemini_rate_limiter,
            max_retries=10,
        )

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=config.ANTHROPIC_MODEL,
            api_key=config.ANTHROPIC_API_KEY,
            temperature=temperature,
        )

    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=temperature,
        )

    elif provider == "openrouter":
        # OpenRouter dùng OpenAI-compatible API
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.OPENROUTER_MODEL,
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL,
            temperature=temperature,
        )

    else:
        raise ValueError(
            f"Provider không hợp lệ: '{provider}'. "
            "Chọn một trong: openai, gemini, anthropic, ollama, openrouter"
        )


def get_embeddings(provider: str = None):
    """
    Trả về Embeddings instance tương ứng với provider được chọn.

    Lưu ý quan trọng:
        - Anthropic KHÔNG có Embeddings API → tự động fallback về OpenAI embeddings
        - OpenRouter cũng dùng OpenAI embeddings (không có API embeddings riêng)
        - Ollama cần model embedding riêng (mặc định: nomic-embed-text)
          Cài đặt: ollama pull nomic-embed-text

    Args:
        provider: "openai" | "gemini" | "anthropic" | "ollama" | "openrouter"
                  Mặc định: đọc PROVIDER từ .env

    Returns:
        Embeddings instance sẵn sàng sử dụng
    """
    provider = (provider or config.PROVIDER).lower()

    if provider in ("openai", "openrouter"):
        from langchain_openai import OpenAIEmbeddings
        api_key = config.OPENAI_API_KEY or config.OPENROUTER_API_KEY
        base_url = config.OPENAI_BASE_URL or (
            config.OPENROUTER_BASE_URL if provider == "openrouter" else None
        )
        kwargs = {
            "model": config.OPENAI_EMBEDDING_MODEL,
            "api_key": api_key,
            "check_embedding_ctx_length": False,
            "tiktoken_enabled": False,
        }
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAIEmbeddings(**kwargs)

    elif provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        class SafeGoogleEmbeddings(GoogleGenerativeAIEmbeddings):
            """
            Embeddings có batching (40 chunks) và tự động chờ khi gặp 429 RESOURCE_EXHAUSTED.
            Tránh vượt giới hạn 100 requests/phút của Gemini free tier.
            """
            def embed_documents(self, texts: list[str], **kwargs) -> list[list[float]]:
                batch_size = 40
                results = []
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    for attempt in range(6):
                        try:
                            res = super().embed_documents(batch, **kwargs)
                            results.extend(res)
                            break
                        except Exception as e:
                            err_msg = str(e)
                            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                                match = re.search(r"retry in (\d+\.?\d*)s", err_msg)
                                wait_time = float(match.group(1)) + 2.0 if match else 25.0
                                print(f"⚠️ [RateLimit] Gặp 429, tự động chờ {wait_time:.1f}s trước khi thử lại ({attempt + 1}/6)...")
                                time.sleep(wait_time)
                            else:
                                raise e
                    if i + batch_size < len(texts):
                        time.sleep(4)
                return results

            def embed_query(self, text: str, **kwargs) -> list[float]:
                for attempt in range(6):
                    try:
                        return super().embed_query(text, **kwargs)
                    except Exception as e:
                        err_msg = str(e)
                        if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                            match = re.search(r"retry in (\d+\.?\d*)s", err_msg)
                            wait_time = float(match.group(1)) + 2.0 if match else 20.0
                            print(f"⚠️ [RateLimit] Gặp 429 khi embed query, chờ {wait_time:.1f}s...")
                            time.sleep(wait_time)
                        else:
                            raise e

        return SafeGoogleEmbeddings(
            model=config.GEMINI_EMBEDDING_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            rate_limiter=_gemini_rate_limiter,
        )

    elif provider == "anthropic":
        # Anthropic không cung cấp Embeddings API → dùng OpenAI thay thế
        print("⚠️  Anthropic không có Embeddings API — đang dùng OpenAI embeddings thay thế.")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=config.OPENAI_EMBEDDING_MODEL,
            api_key=config.OPENAI_API_KEY,
        )

    elif provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            model=config.OLLAMA_EMBEDDING_MODEL,
            base_url=config.OLLAMA_BASE_URL,
        )

    else:
        raise ValueError(
            f"Provider không hợp lệ: '{provider}'. "
            "Chọn một trong: openai, gemini, anthropic, ollama, openrouter"
        )
