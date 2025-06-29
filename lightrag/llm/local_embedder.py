import pipmaster as pm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)
import numpy as np
from typing import List, Dict, Any
import logging
from lightrag.utils import wrap_embedding_func_with_attrs, logger
import asyncio

# Dynamically install sentence-transformers and its dependencies if not present
# sentence-transformers requires torch
for lib in ["torch", "sentence-transformers"]:
    if not pm.is_installed(lib):
        logger.info(f"Installing {lib} for local embeddings...")
        pm.install(lib)

from sentence_transformers import SentenceTransformer

# Global cache for sentence transformer models to avoid reloading them
_model_cache: Dict[str, SentenceTransformer] = {}


def get_sentence_transformer_model(model_name: str) -> SentenceTransformer:
    """
    Loads a SentenceTransformer model from cache or downloads it.
    """
    if model_name not in _model_cache:
        logger.info(f"Loading local embedding model: {model_name}. This may take a while...")
        _model_cache[model_name] = SentenceTransformer(model_name)
        logger.info(f"Finished loading model: {model_name}")
    return _model_cache[model_name]


# A default model for sentence transformers
DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_DIM = 384
DEFAULT_MAX_TOKEN_SIZE = 512 # The model's max sequence length


@wrap_embedding_func_with_attrs(embedding_dim=DEFAULT_DIM, max_token_size=DEFAULT_MAX_TOKEN_SIZE)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def local_embed(
    texts: List[str],
    model: str = DEFAULT_MODEL,
    **kwargs: Any,
) -> np.ndarray:
    """
    Generates embeddings for a list of texts using a local SentenceTransformer model.

    Args:
        texts: List of texts to embed.
        model: The name of the SentenceTransformer model to use.

    Returns:
        A numpy array of embeddings.
    """
    try:
        transformer_model = get_sentence_transformer_model(model)
        # Run the synchronous, CPU-bound encode method in a separate thread
        # to avoid blocking the asyncio event loop.
        embeddings = await asyncio.to_thread(
            transformer_model.encode, texts, show_progress_bar=False, **kwargs
        )
        return np.array(embeddings)
    except Exception as e:
        logger.error(f"Failed to generate local embeddings with model {model}: {e}")
        raise 