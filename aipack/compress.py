"""
aipack.compress
───────────────
Core compression algorithms for AI model context files.
All algorithms are lossy-aware and tuned for LLM tokenizers (BPE-family).

Compatible: Python 3.7+.  Zero mandatory dependencies.
"""

from __future__ import annotations  # enables e.g. list[str] on 3.7-3.9

import re
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

# ── Token estimator ──────────────────────────────────────────────────────────

def estimate_tokens(text):
    # type: (str) -> int
    """Fast BPE token estimate: ~4 chars/token Latin, ~2 for CJK."""
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    latin = len(text) - cjk
    return max(1, (latin // 4) + (cjk // 2))


# ── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class CompressResult:
    original_bytes:    int
    compressed_bytes:  int
    original_tokens:   int
    compressed_tokens: int
    techniques: Dict[str, float] = field(default_factory=dict)
    passes:     List[str]        = field(default_factory=list)

    @property
    def ratio(self):
        # type: () -> float
        if self.original_bytes == 0:
            return 0.0
        return 1.0 - self.compressed_bytes / float(self.original_bytes)

    @property
    def tokens_saved(self):
        # type: () -> int
        return self.original_tokens - self.compressed_tokens

    def summary(self):
        # type: () -> str
        lines = [
            "  Original   : {:>10,} bytes  ({:,} tokens)".format(
                self.original_bytes, self.original_tokens),
            "  Compressed : {:>10,} bytes  ({:,} tokens)".format(
                self.compressed_bytes, self.compressed_tokens),
            "  Saved      : {:>10,} bytes  ({:,} tokens  {:.1f}%)".format(
                self.original_bytes - self.compressed_bytes,
                self.tokens_saved,
                self.ratio * 100),
            "  Passes     : {}".format(", ".join(self.passes) if self.passes else "none"),
        ]
        return "\n".join(lines)


# ── Individual passes ────────────────────────────────────────────────────────

def _pass_whitespace(text):
    # type: (str) -> Tuple[str, float]
    """Collapse redundant whitespace and blank lines."""
    before = len(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    saved = max(0.0, (before - len(text)) / max(before, 1))
    return text, saved


def _pass_stopwords_light(text):
    # type: (str) -> Tuple[str, float]
    """Remove low-information filler phrases common in LLM system prompts."""
    fillers = [
        r"\bplease\b\s*",
        r"\bkindly\b\s*",
        r"\bfeel free to\b\s*",
        r"\bdon't hesitate to\b\s*",
        r"\bas an AI(?: language model)?\b,?\s*",
        r"\bI want you to\b\s*",
        r"\byour (?:task|job|role|goal) is to\b\s*",
        r"\bmake sure (?:to|that)\b\s*",
        r"\bit is (?:important|essential|critical) (?:that|to)\b\s*",
        r"\bof course\b,?\s*",
        r"\bcertainly\b,?\s*",
        r"\babsolutely\b,?\s*",
        r"\bwithout a doubt\b,?\s*",
    ]
    before = len(text)
    for pat in fillers:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t]+", " ", text)
    saved = max(0.0, (before - len(text)) / max(before, 1))
    return text, saved


def _pass_dedup_lines(text):
    # type: (str) -> Tuple[str, float]
    """Remove exactly-duplicate lines, keeping first occurrence."""
    before = len(text)
    seen = set()
    out = []
    for line in text.splitlines():
        key = line.strip()
        if key and key in seen:
            continue
        seen.add(key)
        out.append(line)
    result = "\n".join(out)
    saved = max(0.0, (before - len(result)) / max(before, 1))
    return result, saved


def _pass_dedup_sentences(text):
    # type: (str) -> Tuple[str, float]
    """Remove near-duplicate sentences (exact match after normalisation)."""
    before = len(text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    seen = set()
    out = []
    for s in sentences:
        key = re.sub(r"\s+", " ", s.lower().strip())
        if key and key in seen:
            continue
        seen.add(key)
        out.append(s)
    result = " ".join(out)
    saved = max(0.0, (before - len(result)) / max(before, 1))
    return result, saved


def _pass_json_minify(text):
    # type: (str) -> Tuple[str, float]
    """Minify JSON if the content is valid JSON."""
    before = len(text)
    try:
        obj = json.loads(text)
        result = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        saved = max(0.0, (before - len(result)) / max(before, 1))
        return result, saved
    except (json.JSONDecodeError, ValueError):
        return text, 0.0


def _pass_jsonl_dedup(text):
    # type: (str) -> Tuple[str, float]
    """Deduplicate JSONL lines by content hash."""
    before = len(text)
    seen = set()
    out = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        h = hashlib.md5(stripped.encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        out.append(line)
    result = "\n".join(out)
    saved = max(0.0, (before - len(result)) / max(before, 1))
    return result, saved


def _pass_markdown_trim(text):
    # type: (str) -> Tuple[str, float]
    """Compact markdown: collapse deep headers, remove HRs and HTML comments."""
    before = len(text)
    text = re.sub(r"^#{4,}\s*", "## ", text, flags=re.MULTILINE)
    text = re.sub(r"^-{3,}$", "", text, flags=re.MULTILINE)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"\n{3,}", "\n\n", text)
    saved = max(0.0, (before - len(text)) / max(before, 1))
    return text, saved


def _pass_code_comments(text):
    # type: (str) -> Tuple[str, float]
    """Strip single-line comments from Python/JS/C-family source files."""
    before = len(text)
    text = re.sub(r"(?<!#)#(?!!)(?!.*coding)[ \t]+.*", "", text)   # Python #
    text = re.sub(r"[ \t]*//(?!/).*", "", text)                     # C/JS //
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)          # /* block */
    text = re.sub(r"\n{3,}", "\n\n", text)
    saved = max(0.0, (before - len(text)) / max(before, 1))
    return text, saved


def _pass_llama3_special(text):
    # type: (str) -> Tuple[str, float]
    """Remove empty turns in Llama-3 chat-format files."""
    before = len(text)
    text = re.sub(
        r"<\|start_header_id\|>[^|]+<\|end_header_id\|>\s*<\|eot_id\|>",
        "",
        text,
    )
    saved = max(0.0, (before - len(text)) / max(before, 1))
    return text, saved


def _pass_chatml_special(text):
    # type: (str) -> Tuple[str, float]
    """Remove empty turns in ChatML format (<|im_start|> / <|im_end|>)."""
    before = len(text)
    text = re.sub(
        r"<\|im_start\|>[^\n]*\n\s*<\|im_end\|>",
        "",
        text,
    )
    saved = max(0.0, (before - len(text)) / max(before, 1))
    return text, saved


# ── Pass registry ─────────────────────────────────────────────────────────────

PASS_REGISTRY = {
    "whitespace":      _pass_whitespace,
    "stopwords":       _pass_stopwords_light,
    "dedup_lines":     _pass_dedup_lines,
    "dedup_sentences": _pass_dedup_sentences,
    "json_minify":     _pass_json_minify,
    "jsonl_dedup":     _pass_jsonl_dedup,
    "markdown_trim":   _pass_markdown_trim,
    "code_comments":   _pass_code_comments,
    "llama3_special":  _pass_llama3_special,
    "chatml_special":  _pass_chatml_special,
}  # type: Dict[str, Callable[[str], Tuple[str, float]]]


# ── Profiles ──────────────────────────────────────────────────────────────────

PROFILES = {
    "safe": {
        "passes": ["whitespace", "dedup_lines"],
        "description": "Minimal lossless-ish. Only removes exact duplicates and whitespace.",
    },
    "semantic": {
        "passes": ["whitespace", "stopwords", "dedup_lines", "dedup_sentences"],
        "description": "Default. Removes filler phrases and duplicate sentences.",
    },
    "markdown": {
        "passes": ["whitespace", "markdown_trim", "dedup_lines"],
        "description": "Optimised for .md files: compacts headers, strips HRs and HTML comments.",
    },
    "json": {
        "passes": ["json_minify"],
        "description": "Minify valid JSON.",
    },
    "jsonl": {
        "passes": ["jsonl_dedup", "whitespace"],
        "description": "Deduplicate JSONL training records by hash.",
    },
    "code": {
        "passes": ["whitespace", "code_comments", "dedup_lines"],
        "description": "Strip comments from Python / JS / C-family files.",
    },
    "llama3": {
        "passes": [
            "whitespace", "llama3_special", "stopwords",
            "dedup_lines", "dedup_sentences",
        ],
        "description": "Full pipeline for Llama-3 chat-format files.",
    },
    "chatml": {
        "passes": [
            "whitespace", "chatml_special", "stopwords",
            "dedup_lines", "dedup_sentences",
        ],
        "description": "Full pipeline for ChatML-format files (Mistral, Phi, Qwen, etc.).",
    },
    "extreme": {
        "passes": [
            "whitespace", "stopwords", "dedup_lines", "dedup_sentences",
            "markdown_trim", "code_comments", "llama3_special", "chatml_special",
        ],
        "description": "All passes. Maximum token reduction, minor fidelity trade-off.",
    },
}  # type: Dict[str, dict]


# ── Main compressor ───────────────────────────────────────────────────────────

def compress_text(text, profile="semantic", custom_passes=None):
    # type: (str, str, Optional[List[str]]) -> Tuple[str, CompressResult]
    """
    Compress *text* using the named profile or an explicit pass list.
    Returns (compressed_text, CompressResult).
    """
    if custom_passes is not None:
        passes = custom_passes
    else:
        if profile not in PROFILES:
            raise ValueError(
                "Unknown profile '{}'. Available: {}".format(
                    profile, ", ".join(PROFILES))
            )
        passes = PROFILES[profile]["passes"]

    for p in passes:
        if p not in PASS_REGISTRY:
            raise ValueError(
                "Unknown pass '{}'. Available: {}".format(
                    p, ", ".join(PASS_REGISTRY))
            )

    original_bytes  = len(text.encode("utf-8"))
    original_tokens = estimate_tokens(text)
    techniques = {}  # type: Dict[str, float]
    result = text
    applied = []     # type: List[str]

    for pass_name in passes:
        fn = PASS_REGISTRY[pass_name]
        new_text, saved_fraction = fn(result)
        if saved_fraction > 0.0001:
            techniques[pass_name] = round(saved_fraction * 100, 2)
            applied.append(pass_name)
        result = new_text

    stats = CompressResult(
        original_bytes=original_bytes,
        compressed_bytes=len(result.encode("utf-8")),
        original_tokens=original_tokens,
        compressed_tokens=estimate_tokens(result),
        techniques=techniques,
        passes=applied,
    )
    return result, stats


def compress_file(src, dest=None, profile="semantic", custom_passes=None, encoding="utf-8"):
    # type: (Path, Optional[Path], str, Optional[List[str]], str) -> CompressResult
    """
    Read *src*, compress, write to *dest*.
    Default dest: src.stem + '.compressed' + src.suffix.
    """
    text = src.read_text(encoding=encoding, errors="replace")
    compressed, stats = compress_text(text, profile=profile, custom_passes=custom_passes)
    if dest is None:
        dest = src.with_name(src.stem + ".compressed" + src.suffix)
    dest.write_text(compressed, encoding=encoding)
    return stats
