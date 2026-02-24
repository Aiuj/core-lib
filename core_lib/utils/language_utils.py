"""
language_utils.py
Utilities for language manipulation, including detection and future NLP features.

This module provides utilities for:
- Language detection with secure input handling
- Text cropping while preserving word and sentence boundaries
"""

from fast_langdetect import detect, LangDetectConfig, LangDetector
import logging
import re
from typing import List, Dict, Optional

# Module-level detector configured with max_input_length=200
# Use model="lite" to always use the bundled 0.89MB model (lid.176.ftz).
# The default "auto" mode tries the 126MB large model first and only falls
# back on MemoryError — on a fresh container this triggers a 126MB download
# on the first detection call, blocking the request for many seconds.
_LANG_DETECTOR_CONFIG = None
DETECTOR = None
try:
    _LANG_DETECTOR_CONFIG = LangDetectConfig(max_input_length=200, model="lite")
    DETECTOR = LangDetector(_LANG_DETECTOR_CONFIG)
    # Provide a module-level detect wrapper that prefers the configured detector
    def _module_detect(text, **kwargs):
        return DETECTOR.detect(text, **kwargs)
    detect = _module_detect
except Exception:
    # If initialization fails, leave the imported `detect` as-is
    DETECTOR = None


class LanguageUtils:
    """Utilities for language detection and text manipulation.
    
    This class provides static methods for language detection using fast-langdetect,
    text cropping with word/sentence boundary preservation, and multi-text language
    analysis. All methods are static and don't require instantiation.
    """
    
    @staticmethod
    def crop_text_preserve_words(
        text: str, 
        max_length: int = 200, 
        prefer_sentences: bool = True, 
        min_word_boundary: Optional[int] = None
    ) -> str:
        """
        Crops text to a specified length while preserving word boundaries and optionally sentence boundaries.
        
        Args:
            text (str): The input text to crop.
            max_length (int): Maximum length of the cropped text. Default is 500.
            prefer_sentences (bool): If True, tries to crop at sentence boundaries first. Default is True.
            min_word_boundary (int): Minimum length threshold for word boundary breaking. 
                                   If None, defaults to 80% of max_length.
        
        Returns:
            str: The cropped text, preserving word/sentence boundaries when possible.
            
        Raises:
            ValueError: If input is not a string or max_length is not positive.
        """
        if not isinstance(text, str):
            raise ValueError("Input text must be a string.")
        
        if not isinstance(max_length, int) or max_length <= 0:
            raise ValueError("max_length must be a positive integer.")
        
        # Set default min_word_boundary if not provided
        if min_word_boundary is None:
            min_word_boundary = int(max_length * 0.8)
        
        # If text is already short enough, return as-is
        if len(text) <= max_length:
            return text
        
        cropped_text = text
        
        # Try to crop at sentence boundary if prefer_sentences is True
        if prefer_sentences:
            sentence_endings = ['. ', '! ', '? ']
            first_sentence_end = None
            
            for ending in sentence_endings:
                pos = text.find(ending)
                if pos != -1:
                    if first_sentence_end is None or pos < first_sentence_end:
                        first_sentence_end = pos + len(ending.rstrip())  # Include punctuation but not space
            
            # Use first sentence if it's within the limit and we found one
            if first_sentence_end is not None and first_sentence_end <= max_length:
                cropped_text = text[:first_sentence_end].strip()
                return cropped_text
        
        # Crop to max_length
        cropped_text = text[:max_length].strip()
        
        # Try to break at word boundary if we're above the minimum threshold
        last_space = cropped_text.rfind(' ')
        if last_space > min_word_boundary:
            cropped_text = cropped_text[:last_space].strip()
        
        return cropped_text

    @staticmethod
    def _normalize_detector_output(raw) -> List[Dict[str, Optional[float]]]:
        """
        Normalize various detector return shapes into a list of dicts.
        
        Converts detector output from various formats (dict, list, tuple, string) into 
        a consistent format: [{'lang': <code>, 'score': <float|None>}, ...]
        
        Args:
            raw: Raw output from language detector in any supported format
            
        Returns:
            List of normalized language detection results
            
        Examples:
            >>> _normalize_detector_output({'lang': 'en', 'score': 0.95})
            [{'lang': 'en', 'score': 0.95}]
            
            >>> _normalize_detector_output([('en', 0.95), ('fr', 0.03)])
            [{'lang': 'en', 'score': 0.95}, {'lang': 'fr', 'score': 0.03}]
        """
        # Handle single dict result
        if isinstance(raw, dict):
            lang = raw.get("lang") or raw.get("language")
            if lang is not None:
                return [{"lang": lang, "score": raw.get("score")}]
            return []

        # Handle plain string result
        if isinstance(raw, str):
            return [{"lang": raw, "score": None}]

        # Handle iterable results (list/tuple)
        if isinstance(raw, (list, tuple)):
            results = []
            for item in raw:
                # Handle tuple/list like ('en', 0.95)
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    lang = item[0]
                    score = item[1] if len(item) > 1 else None
                    results.append({"lang": lang, "score": score})
                # Handle dict like {'lang': 'en', 'score': 0.95}
                elif isinstance(item, dict):
                    lang = item.get("lang") or item.get("language")
                    if lang is not None:
                        results.append({"lang": lang, "score": item.get("score")})
                # Handle plain string items
                elif isinstance(item, str):
                    results.append({"lang": item, "score": None})
            return results

        # Fallback: stringify unknown types
        try:
            return [{"lang": str(raw), "score": None}]
        except Exception:
            raise RuntimeError(f"Unexpected return type from language detector: {type(raw)}")

    # New helper to centralize preprocessing for detection methods
    @staticmethod
    def _preprocess_text_for_detection(
        text: str, 
        max_length: int = 500, 
        min_word_boundary: Optional[int] = 400
    ) -> str:
        """
        Validate and preprocess text for language detection:
        - Ensure it's a non-empty string
        - Trim and remove newlines
        - Collapse whitespace
        - Crop using crop_text_preserve_words
        - Ensure minimum length (>=3) after processing
        
        Args:
            text: Input text to preprocess
            max_length: Maximum length after cropping
            min_word_boundary: Minimum length threshold for word boundary breaking
            
        Returns:
            Preprocessed text ready for language detection
            
        Raises:
            ValueError: If input is not a string, empty, or too short after processing
        """
        if not isinstance(text, str):
            raise ValueError("Input to detect_language must be a string.")
        text = text.strip()
        if not text:
            raise ValueError("Input text cannot be empty after trimming whitespace.")
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        # If no explicit max_length provided, prefer the module-level detector's setting
        if max_length is None:
            max_length = 200
        # Use detector's configured max_input_length when available
        try:
            detector_cfg_max = _LANG_DETECTOR_CONFIG.max_input_length if _LANG_DETECTOR_CONFIG is not None else None
            if detector_cfg_max is not None:
                max_length = detector_cfg_max
        except Exception:
            pass

        # Default min_word_boundary to 80% of effective max_length when not provided
        if min_word_boundary is None:
            min_word_boundary = int(max_length * 0.8)

        text = LanguageUtils.crop_text_preserve_words(text, max_length=max_length, prefer_sentences=True, min_word_boundary=min_word_boundary)
        if len(text) < 3:
            raise ValueError("Input text is too short for reliable language detection (minimum 3 characters after processing).")
        return text

    @staticmethod
    def detect_languages(text: str, min_confidence: float = 0.5) -> List[Dict[str, Optional[float]]]:
        """
        Detects languages and returns a list of candidates with scores >= min_confidence.
        Only candidates with a numeric score are considered for thresholding.
        """
        # Reuse centralized preprocessing
        text = LanguageUtils._preprocess_text_for_detection(text, max_length=500, min_word_boundary=400)

        # Use module-level `detect` wrapper so tests can patch it.
        raw = detect(text)
        normalized = LanguageUtils._normalize_detector_output(raw)

        # Filter by min_confidence: only keep entries with numeric score >= min_confidence
        result: List[Dict[str, Optional[float]]] = []
        for entry in normalized:
            score = entry.get("score")
            if isinstance(score, (int, float)) and score >= min_confidence:
                result.append(entry)

        # Optionally sort by score descending (highest confidence first)
        result.sort(key=lambda e: e.get("score", 0.0), reverse=True)
        return result

    @staticmethod
    def detect_language(text: str) -> Dict[str, Optional[float]]:
        """
        Detects the single best language candidate for the given text.
        
        Args:
            text: Input text to analyze (will be preprocessed)
            
        Returns:
            Dictionary with 'lang' (language code) and 'score' (confidence) keys.
            Example: {'lang': 'en', 'score': 0.95}
            
        Raises:
            ValueError: If text is invalid or too short
            RuntimeError: If language detector returns no results
            
        Examples:
            >>> LanguageUtils.detect_language("Hello world")
            {'lang': 'en', 'score': 0.95}
            
            >>> LanguageUtils.detect_language("Bonjour le monde")
            {'lang': 'fr', 'score': 0.98}
        """
        # Reuse centralized preprocessing
        text = LanguageUtils._preprocess_text_for_detection(text, max_length=500, min_word_boundary=400)

        # Use module-level `detect` wrapper so tests can patch it.
        raw = detect(text)
        normalized = LanguageUtils._normalize_detector_output(raw)

        # Choose best candidate by numerical score (None treated as 0.0)
        if not normalized:
            raise RuntimeError("Language detector returned no results.")

        def score_val(d: Dict[str, Optional[float]]) -> float:
            s = d.get("score")
            return float(s) if isinstance(s, (int, float)) else 0.0

        best = max(normalized, key=score_val)
        return {"lang": best.get("lang"), "score": best.get("score")}

    @staticmethod
    def detect_language_safe(text: Optional[str], min_length: int = 20) -> Optional[str]:
        """
        Safely detect the language of text, returning None on any error.
        
        This is a convenience wrapper around detect_language that handles edge cases
        gracefully without raising exceptions. Useful for scenarios where language
        detection is optional and failures should not interrupt the main workflow.
        
        Args:
            text: Input text to analyze (can be None or any type)
            min_length: Minimum text length required for detection (default: 20)
            
        Returns:
            ISO 639-1 language code (e.g., 'en', 'fr', 'de') or None if:
            - text is None, not a string, or too short
            - detection fails for any reason
            
        Examples:
            >>> LanguageUtils.detect_language_safe("Hello, how are you today?")
            'en'
            
            >>> LanguageUtils.detect_language_safe(None)
            None
            
            >>> LanguageUtils.detect_language_safe("Hi")  # Too short
            None
        """
        # Validate input type
        if not isinstance(text, str):
            return None
        
        # Check minimum length after stripping
        stripped = text.strip()
        if len(stripped) < min_length:
            return None
        
        try:
            result = LanguageUtils.detect_language(text)
            if result and result.get('lang'):
                lang_code = result['lang']
                if isinstance(lang_code, str):
                    return lang_code.lower()
        except (ValueError, RuntimeError):
            # Expected errors from detect_language (invalid/short text, no results)
            pass
        except Exception:
            # Unexpected errors - log but don't propagate
            try:
                from core_lib.tracing.logger import get_module_logger
                get_module_logger().debug("Unexpected error during language detection")
            except Exception:
                pass
        
        return None

    @staticmethod
    def detect_most_common_language(texts: List[str], min_confidence: float = 0.5) -> Optional[str]:
        """
        Detect the most common language across multiple text samples.

        Args:
            texts (List[str]): Iterable of text samples to analyze.
            min_confidence (float): Minimum numeric confidence threshold for per-sample
                candidates (passed to `detect_languages`). Default is 0.5.

        Returns:
            Optional[str]: The most common language code across samples, or `None`
                if no reliable detections were made.
        """
        if not texts:
            return None

        detections: List[str] = []
        for t in texts:
            if not isinstance(t, str):
                continue
            s = t.strip()
            # skip very short samples that aren't useful for detection
            if len(s) < 10:
                continue

            try:
                candidates = LanguageUtils.detect_languages(s, min_confidence=min_confidence)
            except Exception:
                # Avoid failing the whole operation for one bad sample
                from core_lib.tracing.logger import get_module_logger
                get_module_logger().warning("Language detection failed for a sample; skipping.")
                continue

            if candidates:
                # candidates are sorted by confidence; take the top candidate's lang
                top = candidates[0]
                lang = top.get("lang")
                if isinstance(lang, str):
                    detections.append(lang)

        if not detections:
            return None

        from collections import Counter

        most_common = Counter(detections).most_common(1)
        return most_common[0][0] if most_common else None


def detect_language_code(text: str, min_length: int = 5) -> Optional[str]:
    """Detect the language of *text* and return its ISO 639-1 code.

    A module-level convenience wrapper around
    :meth:`LanguageUtils.detect_language_safe` for callers that just need a
    plain language code string without instantiating the class.

    Args:
        text: Input text to analyse.
        min_length: Minimum character length required for detection (default 5).

    Returns:
        ISO 639-1 language code (e.g. ``'fr'``, ``'en'``) or ``None`` if the
        text is too short or detection is inconclusive.
    """
    return LanguageUtils.detect_language_safe(text, min_length=min_length)
