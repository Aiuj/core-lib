"""
Utility functions for handling embedding dimension normalization.

This module provides functions to normalize embeddings of different dimensions
to match the expected dimensions defined in database schemas (PostgreSQL and OpenSearch).
"""

from typing import List, Optional
import numpy as np
from core_lib.tracing.logger import get_module_logger

logger = get_module_logger()


# Keywords/patterns that indicate Matryoshka support
MATRYOSHKA_PATTERNS = [
    "matryoshka",
    "mrl",
    "adaptive",
    "flexible-dim",
    "multi-dimension",
    "variable-dim",
]


def normalize_embedding_dimension(
    embedding: List[float],
    target_dimension: int,
    method: str = "truncate_or_pad"
) -> List[float]:
    """
    Normalize an embedding vector to match the target dimension.
    
    Args:
        embedding: The input embedding vector
        target_dimension: The desired output dimension
        method: Normalization method. Options:
            - "truncate_or_pad": Truncate if too long, pad with zeros if too short
            - "interpolate": Use linear interpolation to resize (preserves more information)
            - "pca_approximate": Simple dimensionality reduction (requires numpy)
    
    Returns:
        List[float]: Normalized embedding vector of length target_dimension
        
    Raises:
        ValueError: If method is not supported
    """
    if not embedding:
        logger.warning("Empty embedding provided, returning zero vector")
        return [0.0] * target_dimension
    
    current_dimension = len(embedding)
    
    # No normalization needed
    if current_dimension == target_dimension:
        return embedding
    
    logger.debug(
        f"Normalizing embedding from dimension {current_dimension} to {target_dimension} "
        f"using method '{method}'"
    )
    
    if method == "truncate_or_pad":
        return _truncate_or_pad(embedding, target_dimension)
    elif method == "interpolate":
        return _interpolate(embedding, target_dimension)
    elif method == "pca_approximate":
        return _pca_approximate(embedding, target_dimension)
    else:
        raise ValueError(f"Unsupported normalization method: {method}")


def _truncate_or_pad(embedding: List[float], target_dimension: int) -> List[float]:
    """
    Simple truncation or padding with zeros.
    
    - If embedding is longer than target: truncate to target length
    - If embedding is shorter than target: pad with zeros
    """
    current_dimension = len(embedding)
    
    if current_dimension > target_dimension:
        # Truncate
        normalized = embedding[:target_dimension]
        logger.debug(f"Truncated embedding from {current_dimension} to {target_dimension}")
    else:
        # Pad with zeros
        normalized = embedding + [0.0] * (target_dimension - current_dimension)
        logger.debug(f"Padded embedding from {current_dimension} to {target_dimension}")
    
    return normalized


def _interpolate(embedding: List[float], target_dimension: int) -> List[float]:
    """
    Use linear interpolation to resize the embedding vector.
    
    This method preserves more information than simple truncation by
    sampling the embedding space uniformly.
    """
    try:
        current_dimension = len(embedding)
        
        # Create indices for interpolation
        x_original = np.linspace(0, 1, current_dimension)
        x_target = np.linspace(0, 1, target_dimension)
        
        # Interpolate
        embedding_array = np.array(embedding, dtype=np.float32)
        interpolated = np.interp(x_target, x_original, embedding_array)
        
        # Renormalize to unit length (L2 norm) to preserve vector properties
        norm = np.linalg.norm(interpolated)
        if norm > 0:
            interpolated = interpolated / norm
        
        logger.debug(
            f"Interpolated embedding from {current_dimension} to {target_dimension} "
            f"with L2 renormalization"
        )
        
        return interpolated.tolist()
        
    except Exception as e:
        logger.warning(f"Interpolation failed: {e}. Falling back to truncate_or_pad")
        return _truncate_or_pad(embedding, target_dimension)


def _pca_approximate(embedding: List[float], target_dimension: int) -> List[float]:
    """
    Approximate PCA-style dimensionality reduction.
    
    For dimension reduction (current > target), this uses a simple weighted
    averaging approach. For expansion (current < target), falls back to padding.
    
    Note: This is a simplified approach and not true PCA. For production use
    with large dimension changes, consider using proper PCA with sklearn.
    """
    try:
        current_dimension = len(embedding)
        
        if current_dimension <= target_dimension:
            # For expansion, use interpolation instead
            return _interpolate(embedding, target_dimension)
        
        # For reduction: group dimensions and average them
        embedding_array = np.array(embedding, dtype=np.float32)
        
        # Calculate how many source dimensions map to each target dimension
        chunk_size = current_dimension / target_dimension
        
        reduced = []
        for i in range(target_dimension):
            start_idx = int(i * chunk_size)
            end_idx = int((i + 1) * chunk_size)
            
            # Average the chunk
            chunk_mean = np.mean(embedding_array[start_idx:end_idx])
            reduced.append(float(chunk_mean))
        
        # Renormalize to unit length
        reduced_array = np.array(reduced, dtype=np.float32)
        norm = np.linalg.norm(reduced_array)
        if norm > 0:
            reduced_array = reduced_array / norm
        
        logger.debug(
            f"Applied PCA-approximate reduction from {current_dimension} to {target_dimension}"
        )
        
        return reduced_array.tolist()
        
    except Exception as e:
        logger.warning(f"PCA-approximate failed: {e}. Falling back to truncate_or_pad")
        return _truncate_or_pad(embedding, target_dimension)


def is_matryoshka_model(model_name: str) -> bool:
    """
    Check if a model supports Matryoshka Representation Learning.
    
    Uses the models database and fuzzy matching to identify models that support MRL,
    which allows for efficient dimension reduction without significant quality loss.
    
    Args:
        model_name: The name or identifier of the embedding model
    
    Returns:
        bool: True if the model supports Matryoshka representation
        
    Example:
        >>> is_matryoshka_model("nomic-embed-text-v1.5")
        True
        >>> is_matryoshka_model("text-embedding-3-large")
        True
        >>> is_matryoshka_model("text-embedding-ada-002")
        False
    """
    if not model_name:
        return False
    
    # First check the models database
    from .models_database import supports_matryoshka as db_supports_matryoshka
    
    if db_supports_matryoshka(model_name):
        return True
    
    # Fallback: Check for Matryoshka-related patterns in the name
    model_lower = model_name.lower().strip()
    for pattern in MATRYOSHKA_PATTERNS:
        if pattern in model_lower:
            return True
    
    return False


def get_best_normalization_method(
    model_name: Optional[str] = None,
    current_dimension: Optional[int] = None,
    target_dimension: Optional[int] = None
) -> str:
    """
    Determine the best normalization method based on the model and dimension change.
    
    This function intelligently selects the optimal normalization strategy:
    - For Matryoshka models: prefers truncate_or_pad (preserves trained dimensions)
    - For non-Matryoshka models: prefers interpolate (preserves more information)
    - For large dimension increases: always uses interpolate
    - For small changes: considers PCA-approximate
    
    Args:
        model_name: Name or identifier of the embedding model (optional)
        current_dimension: Current embedding dimension (optional)
        target_dimension: Target embedding dimension (optional)
    
    Returns:
        str: Recommended normalization method ("truncate_or_pad", "interpolate", or "pca_approximate")
        
    Example:
        >>> get_best_normalization_method("nomic-embed-text-v1.5", 768, 512)
        'truncate_or_pad'
        >>> get_best_normalization_method("text-embedding-ada-002", 1536, 512)
        'interpolate'
        >>> get_best_normalization_method(current_dimension=1024, target_dimension=1536)
        'interpolate'
    """
    # If model is specified, check if it supports Matryoshka
    if model_name:
        is_mrl = is_matryoshka_model(model_name)
        
        # For Matryoshka models, truncate_or_pad is optimal
        # because these models are explicitly trained to maintain quality at various dimensions
        if is_mrl:
            logger.debug(
                f"Model '{model_name}' supports Matryoshka representation. "
                "Recommending 'truncate_or_pad' method."
            )
            return "truncate_or_pad"
    
    # If we have dimension information, make dimension-aware decisions
    if current_dimension is not None and target_dimension is not None:
        dimension_ratio = target_dimension / current_dimension if current_dimension > 0 else 0
        
        # No change needed
        if current_dimension == target_dimension:
            return "truncate_or_pad"  # Doesn't matter, but this is fastest

        # Hosted text-embedding models are not generic transformer vectors;
        # interpolation is the stable fallback when shortening them.
        hosted_embedding_model = bool(
            model_name and model_name.lower().startswith("text-embedding-")
        )
        if hosted_embedding_model:
            return "interpolate"
        
        # Large dimension increase (>20%) - interpolation is better
        if dimension_ratio > 1.2:
            logger.debug(
                f"Large dimension increase detected ({current_dimension} -> {target_dimension}). "
                "Recommending 'interpolate' method."
            )
            return "interpolate"
        
        # Large dimension reduction (>50%) without Matryoshka - use PCA-approximate
        if dimension_ratio <= (2 / 3) and not (model_name and is_matryoshka_model(model_name)):
            logger.debug(
                f"Large dimension reduction detected ({current_dimension} -> {target_dimension}) "
                "for non-Matryoshka model. Recommending 'pca_approximate' method."
            )
            return "pca_approximate"
        
        # Moderate dimension reduction - interpolation generally works well
        if dimension_ratio < 1.0:
            logger.debug(
                f"Moderate dimension change detected ({current_dimension} -> {target_dimension}). "
                "Recommending 'interpolate' method."
            )
            return "interpolate"
    
    # Default: interpolate for general use (best balance of quality and flexibility)
    logger.debug("Using default recommendation: 'interpolate' method.")
    return "interpolate"


def normalize_embeddings_batch(
    embeddings: List[List[float]],
    target_dimension: int,
    method: str = "truncate_or_pad"
) -> List[List[float]]:
    """
    Normalize a batch of embedding vectors to the target dimension.
    
    Args:
        embeddings: List of embedding vectors
        target_dimension: The desired output dimension
        method: Normalization method (see normalize_embedding_dimension)
    
    Returns:
        List of normalized embedding vectors
    """
    normalized = []
    for i, embedding in enumerate(embeddings):
        try:
            normalized_embedding = normalize_embedding_dimension(
                embedding, target_dimension, method
            )
            normalized.append(normalized_embedding)
        except Exception as e:
            logger.error(f"Failed to normalize embedding at index {i}: {e}")
            # Fallback to zero vector
            normalized.append([0.0] * target_dimension)
    
    return normalized


def get_target_dimension_for_storage(storage_type: str, index_name: str = None) -> int:
    """
    Get the target embedding dimension for a specific storage type.
    
    Args:
        storage_type: Either "postgresql" or "opensearch"
        index_name: For OpenSearch, specify the index name (e.g., "qa_pairs", "document_chunks")
    
    Returns:
        int: Target dimension for the storage system
        
    Raises:
        ValueError: If storage_type or index_name is invalid
    """
    from .embeddings_config import embeddings_settings
    
    if storage_type == "postgresql":
        # PostgreSQL uses the dimension from settings
        return embeddings_settings.embedding_dimension
    
    elif storage_type == "opensearch":
        # OpenSearch has different dimensions per index based on the schema
        if index_name == "qa_pairs":
            # From opensearch_qa_pairs_enhanced_mapping.json
            return 512
        elif index_name == "document_chunks":
            # From opensearch_document_chunks_mapping.json
            return 1024
        else:
            raise ValueError(
                f"Unknown OpenSearch index name: {index_name}. "
                "Valid options: 'qa_pairs', 'document_chunks'"
            )
    
    else:
        raise ValueError(
            f"Unknown storage_type: {storage_type}. "
            "Valid options: 'postgresql', 'opensearch'"
        )


def ensure_embedding_compatibility(
    embedding: List[float],
    storage_type: str,
    index_name: str = None,
    method: str = "interpolate"
) -> List[float]:
    """
    Convenience function to ensure an embedding is compatible with the target storage.
    
    This function automatically determines the target dimension and normalizes
    the embedding accordingly.
    
    Args:
        embedding: The input embedding vector
        storage_type: Either "postgresql" or "opensearch"
        index_name: For OpenSearch, specify the index name
        method: Normalization method (default: "interpolate" for better quality)
    
    Returns:
        List[float]: Normalized embedding ready for storage
        
    Example:
        >>> embedding = [0.1, 0.2, ...] # 768-dimensional embedding
        >>> normalized = ensure_embedding_compatibility(
        ...     embedding, "opensearch", "qa_pairs"
        ... )
        >>> len(normalized)  # Will be 512 for qa_pairs index
        512
    """
    target_dimension = get_target_dimension_for_storage(storage_type, index_name)
    return normalize_embedding_dimension(embedding, target_dimension, method)


def ensure_embeddings_batch_compatibility(
    embeddings: List[List[float]],
    storage_type: str,
    index_name: str = None,
    method: str = "interpolate"
) -> List[List[float]]:
    """
    Batch version of ensure_embedding_compatibility.
    
    Args:
        embeddings: List of embedding vectors
        storage_type: Either "postgresql" or "opensearch"
        index_name: For OpenSearch, specify the index name
        method: Normalization method (default: "interpolate" for better quality)
    
    Returns:
        List of normalized embeddings ready for storage
    """
    target_dimension = get_target_dimension_for_storage(storage_type, index_name)
    return normalize_embeddings_batch(embeddings, target_dimension, method)
