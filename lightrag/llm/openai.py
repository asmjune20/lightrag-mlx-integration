from ..utils import verbose_debug, VERBOSE_DEBUG
import sys
import os
import logging

if sys.version_info < (3, 9):
    from typing import AsyncIterator
else:
    from collections.abc import AsyncIterator
import pipmaster as pm  # Pipmaster for dynamic library install

# install specific modules
if not pm.is_installed("openai"):
    pm.install("openai")

from openai import (
    AsyncOpenAI,
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from ..utils import (
    logger,
    verbose_debug,
    compute_args_hash,
    safe_unicode_decode,
    wrap_embedding_func_with_attrs,
    handle_cache,
    save_to_cache,
    CacheData,
)
from lightrag.types import GPTKeywordExtractionFormat
from lightrag.api import __api_version__
from ..base import BaseKVStorage

import numpy as np
from typing import Any, Union

from dotenv import load_dotenv

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)


class InvalidResponseError(Exception):
    """Custom exception class for triggering retry mechanism"""

    pass


def create_openai_async_client(
    api_key: str | None = None,
    base_url: str | None = None,
    client_configs: dict[str, Any] = None,
) -> AsyncOpenAI:
    """Create an AsyncOpenAI client with the given configuration.

    Args:
        api_key: OpenAI API key. If None, uses the OPENAI_API_KEY environment variable.
        base_url: Base URL for the OpenAI API. If None, uses the default OpenAI API URL.
        client_configs: Additional configuration options for the AsyncOpenAI client.
            These will override any default configurations but will be overridden by
            explicit parameters (api_key, base_url).

    Returns:
        An AsyncOpenAI client instance.
    """
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_BINDING_API_KEY")
        if not api_key:
             raise ValueError("API key must be provided either as an argument or as an environment variable (OPENAI_API_KEY or LLM_BINDING_API_KEY)")

    # If api_key is 'not-needed', set it to None to avoid sending it in the header.
    if api_key == "not-needed":
        api_key = None
        
    default_headers = {
        "User-Agent": f"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_8) LightRAG/{__api_version__}",
        "Content-Type": "application/json",
    }

    if client_configs is None:
        client_configs = {}

    # Create a merged config dict with precedence: explicit params > client_configs > defaults
    merged_configs = {
        **client_configs,
        "default_headers": default_headers,
        "api_key": api_key,
    }

    if base_url is not None:
        merged_configs["base_url"] = base_url
    else:
        merged_configs["base_url"] = os.environ.get("LLM_BINDING_HOST") or os.environ.get(
            "OPENAI_API_BASE", "https://api.openai.com/v1"
        )
    logger.info(f"Creating OpenAI client with base_url: {merged_configs.get('base_url')}")
    return AsyncOpenAI(**merged_configs)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(APIConnectionError),
)
async def openai_complete_if_cache(
    prompt: str,
    hashing_kv: BaseKVStorage,
    model: str,
    system_prompt: str | None = None,
    history_messages: list[dict] | None = None,
    stream: bool = False,
    **kwargs,
) -> str | AsyncIterator[str]:
    # Extract client configuration parameters
    api_key = kwargs.pop("api_key", None)
    base_url = kwargs.pop("base_url", None)
    client_configs = kwargs.pop("client_configs", None)
    
    # Check cache first (without client config parameters)
    args_hash = compute_args_hash(
        model, prompt, system_prompt, history_messages, **kwargs
    )
    cached_response, _, _, _ = await handle_cache(
        hashing_kv, args_hash, prompt, model, "completion"
    )
    if cached_response:
        if stream:

            async def stream_generator():
                yield cached_response

            return stream_generator()
        return cached_response

    # Prepare messages for the API call
    # MLX/Mistral models require alternating user/assistant roles
    # Combine system prompt with user prompt for compatibility
    messages = []
    if history_messages:
        messages.extend(history_messages)
    
    # Combine system prompt and user prompt into a single user message
    if system_prompt:
        combined_prompt = f"{system_prompt}\n\n{prompt}"
    else:
        combined_prompt = prompt
    
    messages.append({"role": "user", "content": combined_prompt})

    # Create OpenAI client with the extracted parameters
    client = create_openai_async_client(api_key, base_url, client_configs)
    
    try:
        # Use chat completions endpoint for MLX server compatibility
        response = await client.chat.completions.create(
            model=model, messages=messages, stream=stream, **kwargs
        )
    except Exception as e:
        logger.error(
            f"OpenAI API Call Failed,\nModel: {model},\nParams: {kwargs}, Got: {e.__class__.__name__}"
        )
        raise e

    if stream:

        async def _stream_generator():
            nonlocal response
            full_response = ""
            try:
                async for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content:
                        full_response += content
                        yield content
            finally:
                if hashing_kv.global_config.get("enable_llm_cache"):
                    await save_to_cache(
                        hashing_kv,
                        CacheData(
                            args_hash=args_hash,
                            content=full_response,
                            prompt=prompt,
                            cache_type="completion",
                        ),
                    )

        return _stream_generator()
    else:
        full_response = response.choices[0].message.content
        if hashing_kv.global_config.get("enable_llm_cache"):
            await save_to_cache(
                hashing_kv,
                CacheData(
                    args_hash=args_hash,
                    content=full_response,
                    prompt=prompt,
                    cache_type="completion",
                ),
            )
        return full_response


async def openai_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    keyword_extraction=False,
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
    if history_messages is None:
        history_messages = []
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        kwargs["response_format"] = "json"
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    return await openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gpt_4o_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    keyword_extraction=False,
    **kwargs,
) -> str:
    if history_messages is None:
        history_messages = []
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        kwargs["response_format"] = GPTKeywordExtractionFormat
    return await openai_complete_if_cache(
        "gpt-4o",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gpt_4o_mini_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    keyword_extraction=False,
    **kwargs,
) -> str:
    if history_messages is None:
        history_messages = []
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        kwargs["response_format"] = GPTKeywordExtractionFormat
    return await openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def nvidia_openai_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    keyword_extraction=False,
    **kwargs,
) -> str:
    if history_messages is None:
        history_messages = []
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    result = await openai_complete_if_cache(
        "nvidia/llama-3.1-nemotron-70b-instruct",  # context length 128k
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        base_url="https://integrate.api.nvidia.com/v1",
        **kwargs,
    )
    if keyword_extraction:  # TODO: use JSON API
        return locate_json_string_body_from_string(result)
    return result


@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=(
        retry_if_exception_type(RateLimitError)
        | retry_if_exception_type(APIConnectionError)
        | retry_if_exception_type(APITimeoutError)
    ),
)
async def openai_embed(
    texts: list[str],
    model: str = "text-embedding-3-small",
    base_url: str = None,
    api_key: str = None,
    client_configs: dict[str, Any] = None,
) -> np.ndarray:
    """Embed texts using OpenAI's API."""
    openai_async_client = create_openai_async_client(api_key, base_url, client_configs)
    try:
        response = await openai_async_client.embeddings.create(input=texts, model=model)
        return np.array([embedding.embedding for embedding in response.data])
    except Exception as e:
        logger.error(f"OpenAI Embedding Failed, error: {e}")
        raise e


async def openai_completion(prompt: str, **kwargs: Any) -> AsyncIterator[str] | str:
    """
    Generate completions using an OpenAI-compatible endpoint.
    Handles both streaming and non-streaming responses.
    """
    params = kwargs.copy()
    api_key = params.pop("api_key", None)
    base_url = params.pop("base_url", None)
    client_configs = params.pop("client_configs", None)
    stream = params.get("stream", False)
    system_prompt = params.pop("system_prompt", None)

    # For completion endpoints, we combine the system prompt and the user prompt
    if system_prompt:
        # A common format for system + user prompt
        prompt = f"System: {system_prompt}\n\nUser: {prompt}"

    # Add the final combined prompt to the parameters
    params["prompt"] = prompt

    openai_async_client = create_openai_async_client(
        api_key, base_url, client_configs
    )

    # Log parameters before making the call
    logger.info(f"OpenAI API Call Params: {params}")
    try:
        response = await openai_async_client.completions.create(**params)
    except Exception as e:
        logger.error(
            f"OpenAI API Call Failed,\nModel: {params.get('model')},\nParams: {params}, Got: {e.__class__.__name__}"
        )
        raise e

    if stream:
        async def _stream_generator():
            nonlocal response
            try:
                async for chunk in response:
                    content = chunk.choices[0].text
                    if content:
                        yield content
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                raise

        return _stream_generator()
    else:
        return response.choices[0].text


# TODO: can we use a base client and extend it for different providers?
class OpenAIClient:
    """
    A client for interacting with the OpenAI API, providing methods for text completion and embedding.
    """
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        client_configs: dict[str, Any] = None,
    ):
        self.openai_async_client = create_openai_async_client(
            api_key=api_key, base_url=base_url, client_configs=client_configs
        )

    async def complete(self, prompt: str, stream: bool = False, **kwargs) -> str | AsyncIterator[str]:
        try:
            response = await self.openai_async_client.completions.create(
                model=kwargs["hashing_kv"].global_config["llm_model_name"],
                prompt=prompt,
                stream=stream,
                **kwargs
            )
        except Exception as e:
            logger.error(
                f"OpenAI API Call Failed,\nModel: {kwargs['hashing_kv'].global_config['llm_model_name']},\nParams: {kwargs}, Got: {e.__class__.__name__}"
            )
            raise e

        if stream:
            async def _stream_generator():
                nonlocal response
                full_response = ""
                try:
                    async for chunk in response:
                        content = chunk.choices[0].text
                        if content:
                            full_response += content
                            yield content
                finally:
                    pass
            return _stream_generator()
        else:
            return response.choices[0].text

    async def embed(self, texts: list[str], model: str = "text-embedding-3-small") -> np.ndarray:
        try:
            response = await self.openai_async_client.embeddings.create(input=texts, model=model)
            return np.array([item.embedding for item in response.data])
        except Exception as e:
            logger.error(
                f"OpenAI API Call Failed,\nModel: {model},\nGot: {e.__class__.__name__}"
            )
            raise e
