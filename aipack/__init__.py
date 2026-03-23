from __future__ import annotations
"""aipack — AI context compressor for local LLMs."""

from .compress import compress_text, compress_file, CompressResult, PROFILES, PASS_REGISTRY

__version__ = "0.1.0"
__all__ = [
    "compress_text",
    "compress_file",
    "CompressResult",
    "PROFILES",
    "PASS_REGISTRY",
]
