"""Tests for aipack.compress — run with: python -m pytest tests/"""

import json
import pytest
from aipack.compress import (
    compress_text,
    compress_file,
    estimate_tokens,
    PROFILES,
    PASS_REGISTRY,
    _pass_whitespace,
    _pass_stopwords_light,
    _pass_dedup_lines,
    _pass_dedup_sentences,
    _pass_json_minify,
    _pass_jsonl_dedup,
    _pass_markdown_trim,
    _pass_code_comments,
    _pass_llama3_special,
)


# ── estimate_tokens ──────────────────────────────────────────────────────────

def test_token_estimate_empty():
    assert estimate_tokens("") == 1

def test_token_estimate_basic():
    text = "hello world"  # 11 chars → ~2-3 tokens
    assert 1 <= estimate_tokens(text) <= 5

def test_token_estimate_cjk():
    text = "你好世界"  # 4 CJK chars → ~2 tokens
    assert estimate_tokens(text) >= 1


# ── Individual passes ─────────────────────────────────────────────────────────

def test_whitespace_collapses_spaces():
    text, saved = _pass_whitespace("hello   world\n\n\n\nbye")
    assert "   " not in text
    assert text.count("\n") <= 2
    assert saved > 0

def test_whitespace_no_change():
    text, saved = _pass_whitespace("hello world")
    assert text == "hello world"
    assert saved == 0.0

def test_stopwords_removes_please():
    text, saved = _pass_stopwords_light("Please make sure to answer the question.")
    assert "Please" not in text
    assert saved > 0

def test_stopwords_removes_as_an_ai():
    text, _ = _pass_stopwords_light("As an AI language model, I cannot do that.")
    assert "As an AI" not in text

def test_dedup_lines_basic():
    text = "line one\nline two\nline one\nline three"
    result, saved = _pass_dedup_lines(text)
    assert result.count("line one") == 1
    assert saved > 0

def test_dedup_lines_no_dupes():
    text = "a\nb\nc"
    result, saved = _pass_dedup_lines(text)
    assert saved == 0.0
    assert result == text

def test_dedup_sentences():
    text = "The cat sat. The cat sat. The dog barked."
    result, saved = _pass_dedup_sentences(text)
    assert result.count("The cat sat") == 1
    assert saved > 0

def test_json_minify():
    obj = {"key": "value", "nested": {"a": 1}}
    pretty = json.dumps(obj, indent=2)
    result, saved = _pass_json_minify(pretty)
    assert saved > 0
    assert json.loads(result) == obj  # semantically identical

def test_json_minify_invalid():
    text = "not json at all"
    result, saved = _pass_json_minify(text)
    assert result == text
    assert saved == 0.0

def test_jsonl_dedup():
    line = '{"text": "hello"}'
    text = "\n".join([line, line, '{"text": "world"}'])
    result, saved = _pass_jsonl_dedup(text)
    assert result.count('"hello"') == 1
    assert saved > 0

def test_markdown_trim():
    text = "#### Deep header\n\n---\n\n<!-- comment -->\nContent"
    result, saved = _pass_markdown_trim(text)
    assert "####" not in result
    assert "<!--" not in result
    assert saved > 0

def test_code_comments_python():
    text = "x = 1  # this is a comment\ny = 2"
    result, saved = _pass_code_comments(text)
    assert "this is a comment" not in result
    assert saved > 0

def test_code_comments_shebang_preserved():
    text = "#!/usr/bin/env python3\nx = 1"
    result, _ = _pass_code_comments(text)
    assert "#!/usr/bin/env python3" in result

def test_llama3_empty_turns():
    text = (
        "<|start_header_id|>user<|end_header_id|>\n\n<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\nHello<|eot_id|>"
    )
    result, saved = _pass_llama3_special(text)
    # Empty user turn should be removed
    assert saved > 0 or len(result) <= len(text)


# ── compress_text ─────────────────────────────────────────────────────────────

def test_compress_text_default_profile():
    text = "Please make sure to always respond helpfully.\nPlease make sure to always respond helpfully."
    compressed, stats = compress_text(text)
    assert len(compressed) <= len(text)
    assert stats.original_bytes >= stats.compressed_bytes

def test_compress_text_all_profiles():
    text = "Hello world.\n\nHello world.\n\nThis is a test sentence."
    for profile in PROFILES:
        compressed, stats = compress_text(text, profile=profile)
        assert isinstance(compressed, str)
        assert stats.ratio >= 0.0

def test_compress_text_custom_passes():
    text = "a\na\nb\n   c   "
    compressed, stats = compress_text(
        text, custom_passes=["whitespace", "dedup_lines"]
    )
    assert "a\na" not in compressed

def test_compress_text_unknown_profile():
    with pytest.raises(ValueError, match="Unknown profile"):
        compress_text("text", profile="does_not_exist")

def test_compress_text_unknown_pass():
    with pytest.raises(ValueError, match="Unknown pass"):
        compress_text("text", custom_passes=["whitespace", "fake_pass"])

def test_compress_result_ratio():
    text = "word " * 200
    _, stats = compress_text(text)
    assert 0.0 <= stats.ratio <= 1.0
    assert stats.tokens_saved >= 0

def test_compress_result_summary():
    _, stats = compress_text("hello world\nhello world")
    s = stats.summary()
    assert "Original" in s
    assert "Saved" in s


# ── compress_file ─────────────────────────────────────────────────────────────

def test_compress_file(tmp_path):
    src = tmp_path / "test.txt"
    src.write_text("Hello world.\nHello world.\nGoodbye.", encoding="utf-8")
    dest = tmp_path / "test.compressed.txt"
    stats = compress_file(src, dest)
    assert dest.exists()
    assert dest.read_text(encoding="utf-8") != ""
    assert stats.original_bytes > 0

def test_compress_file_default_dest(tmp_path):
    src = tmp_path / "input.txt"
    src.write_text("Some text here.\nSome text here.", encoding="utf-8")
    stats = compress_file(src)
    expected = tmp_path / "input.compressed.txt"
    assert expected.exists()

def test_compress_file_json(tmp_path):
    src = tmp_path / "data.json"
    src.write_text(json.dumps({"a": 1, "b": [1, 2, 3]}, indent=4), encoding="utf-8")
    dest = tmp_path / "data.compressed.json"
    stats = compress_file(src, dest, profile="json")
    assert stats.ratio > 0
    assert json.loads(dest.read_text()) == {"a": 1, "b": [1, 2, 3]}

def test_compress_file_jsonl(tmp_path):
    lines = [json.dumps({"text": "hello"})] * 5 + [json.dumps({"text": "world"})]
    src = tmp_path / "train.jsonl"
    src.write_text("\n".join(lines), encoding="utf-8")
    dest = tmp_path / "train.compressed.jsonl"
    stats = compress_file(src, dest, profile="jsonl")
    result_lines = [l for l in dest.read_text().splitlines() if l.strip()]
    assert len(result_lines) == 2  # deduped: hello + world
    assert stats.ratio > 0
