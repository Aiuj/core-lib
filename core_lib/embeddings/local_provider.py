"""Local embedding client implementation using HuggingFace models."""
import time
import math
from typing import List, Optional, Union, Dict, Any
import os

try:
    from sentence_transformers import SentenceTransformer
    import torch
    _sentence_transformers_available = True
except ImportError:
    SentenceTransformer = None
    torch = None
    _sentence_transformers_available = False

try:
    from transformers import AutoTokenizer, AutoModel
    import torch.nn.functional as F
    _transformers_available = True
except ImportError:
    AutoTokenizer = None
    AutoModel = None
    F = None
    _transformers_available = False

from .embeddings_config import embeddings_settings
from .base import BaseEmbeddingClient, EmbeddingGenerationError
from .models import EmbeddingResponse
from core_lib.tracing.logger import get_module_logger
from core_lib.tracing.service_usage import log_embedding_usage

logger = get_module_logger()


class LocalEmbeddingClient(BaseEmbeddingClient):
    """Client for generating embeddings using local HuggingFace models."""

    def __init__(
        self,
        model: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        use_l2_norm: bool = True,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = False,
        use_sentence_transformers: bool = True,
    ):
        """Initialize local embedding client.
        
        Args:
            model: Model name from HuggingFace (e.g., 'sentence-transformers/all-MiniLM-L6-v2')
            embedding_dim: Target embedding dimension
            use_l2_norm: Whether to apply L2 normalization
            device: Device to run the model on ('cpu', 'cuda', 'auto')
            cache_dir: Directory to cache downloaded models
            trust_remote_code: Whether to trust remote code when loading models
            use_sentence_transformers: Whether to use sentence-transformers library (if available)
        """
        super().__init__(model=model, embedding_dim=embedding_dim, use_l2_norm=use_l2_norm)
        
        # Set default model if not provided
        if not self.model:
            self.model = "sentence-transformers/all-MiniLM-L6-v2"
        
        # Device configuration
        if device is None or device == 'auto':
            if torch and torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/huggingface/transformers")
        self.trust_remote_code = trust_remote_code
        self.use_sentence_transformers = use_sentence_transformers
        
        # Initialize the model
        self._model = None
        self._tokenizer = None
        self._load_model()
        
        logger.debug(f"Initialized LocalEmbeddingClient with model={self.model}, device={self.device}")

    def _load_model(self):
        """Load the embedding model."""
        try:
            if self.use_sentence_transformers and _sentence_transformers_available:
                self._load_sentence_transformer()
            elif _transformers_available:
                self._load_transformers_model()
            else:
                raise ImportError(
                    "Neither sentence-transformers nor transformers library is available. "
                    "Install with: pip install sentence-transformers or pip install transformers torch"
                )
        except Exception as e:
            logger.error(f"Failed to load model {self.model}: {e}")
            raise EmbeddingGenerationError(f"Model loading failed: {e}")

    def _load_sentence_transformer(self):
        """Load model using sentence-transformers library."""
        logger.info(f"Loading model {self.model} using sentence-transformers")
        self._model = SentenceTransformer(
            self.model,
            device=self.device,
            cache_folder=self.cache_dir,
            trust_remote_code=self.trust_remote_code
        )

    def _load_transformers_model(self):
        """Load model using transformers library."""
        logger.info(f"Loading model {self.model} using transformers")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model,
            cache_dir=self.cache_dir,
            trust_remote_code=self.trust_remote_code
        )
        self._model = AutoModel.from_pretrained(
            self.model,
            cache_dir=self.cache_dir,
            trust_remote_code=self.trust_remote_code
        )
        self._model.to(self.device)
        self._model.eval()

    def _generate_embedding_raw(self, texts: List[str]) -> List[List[float]]:
        """Generate raw embeddings using the local model."""
        start_time = time.time()
        
        try:
            if self.use_sentence_transformers and isinstance(self._model, SentenceTransformer):
                embeddings = self._generate_with_sentence_transformers(texts)
            else:
                embeddings = self._generate_with_transformers(texts)
            
            self.embedding_time_ms = (time.time() - start_time) * 1000
            logger.debug(f"Generated {len(embeddings)} embeddings in {self.embedding_time_ms:.2f}ms")
            
            # Sanitize embeddings: replace any NaN or Inf values with 0
            # This can happen with some models under certain conditions
            embeddings = self._sanitize_embeddings(embeddings)
            
            # Log service usage to OpenTelemetry/OpenSearch
            try:
                # Local models don't have explicit token counts, estimate from text length
                # Rough estimate: ~4 chars per token for English text
                estimated_tokens = sum(len(text) // 4 for text in texts)
                
                log_embedding_usage(
                    provider="local",
                    model=self.model,
                    input_tokens=estimated_tokens,
                    num_texts=len(texts),
                    embedding_dim=self.embedding_dim or len(embeddings[0]) if embeddings else None,
                    latency_ms=self.embedding_time_ms,
                    host="local",
                )
            except Exception as e:
                logger.warning(f"Failed to log embedding usage: {e}")
            
            return embeddings
            
        except Exception as e:
            self.embedding_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Error generating embeddings with local model: {e}")
            raise EmbeddingGenerationError(f"Local embedding generation failed: {e}")

    def _generate_with_sentence_transformers(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using sentence-transformers."""
        embeddings = self._model.encode(texts, convert_to_tensor=False, normalize_embeddings=False)
        return embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings

    def _generate_with_transformers(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using transformers."""
        if not torch:
            raise ImportError("torch is required for transformers-based embeddings")
        
        # Tokenize inputs
        encoded_input = self._tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        )
        
        # Move to device
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        
        # Generate embeddings
        with torch.no_grad():
            model_output = self._model(**encoded_input)
            
            # Use mean pooling strategy
            embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
            
        return embeddings.cpu().numpy().tolist()

    def _mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling to get sentence embeddings."""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def health_check(self) -> bool:
        """Check if the local model is loaded and accessible."""
        try:
            # Try to generate a simple embedding to test the model
            test_embeddings = self._generate_embedding_raw(["test"])
            return len(test_embeddings) > 0 and len(test_embeddings[0]) > 0
        except Exception as e:
            logger.warning(f"Local model health check failed: {e}")
            return False

    def _sanitize_embeddings(self, embeddings: List[List[float]]) -> List[List[float]]:
        """Sanitize embeddings by replacing NaN and Inf values with 0.
        
        Local models can occasionally return NaN or Inf values due to numerical issues
        in certain configurations or inputs. This method detects and replaces such
        values to prevent JSON serialization errors.
        
        Args:
            embeddings: List of embedding vectors potentially containing NaN/Inf values
            
        Returns:
            Sanitized embeddings with NaN/Inf values replaced by 0
        """
        sanitized = []
        had_invalid = False
        
        for vec in embeddings:
            sanitized_vec = []
            for val in vec:
                # Handle both float and numpy float types
                try:
                    if math.isnan(val) or math.isinf(val):
                        sanitized_vec.append(0.0)
                        had_invalid = True
                    else:
                        sanitized_vec.append(float(val))
                except (TypeError, ValueError):
                    # If value can't be checked, try to convert to float
                    sanitized_vec.append(float(val))
            sanitized.append(sanitized_vec)
        
        if had_invalid:
            logger.warning(
                f"Detected and sanitized NaN/Inf values in {len(embeddings)} embeddings. "
                "This may indicate numerical instability in the model. "
                "Consider checking your inputs or model configuration."
            )
        
        return sanitized

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        info = {
            'model_name': self.model,
            'device': self.device,
            'library': 'sentence-transformers' if self.use_sentence_transformers else 'transformers',
        }
        
        if hasattr(self._model, 'get_sentence_embedding_dimension'):
            info['embedding_dimension'] = self._model.get_sentence_embedding_dimension()
        elif self.embedding_dim:
            info['embedding_dimension'] = self.embedding_dim
            
        return info

    def get_popular_models(self) -> List[str]:
        """Get list of popular local embedding models."""
        return [
            'sentence-transformers/all-MiniLM-L6-v2',
            'sentence-transformers/all-mpnet-base-v2',
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            'BAAI/bge-small-en-v1.5',
            'BAAI/bge-base-en-v1.5',
            'BAAI/bge-large-en-v1.5',
            'microsoft/DialoGPT-medium',
            'sentence-transformers/multi-qa-MiniLM-L6-cos-v1',
        ]

    def unload_model(self):
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model unloaded successfully")