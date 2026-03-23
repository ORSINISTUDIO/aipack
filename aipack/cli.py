from __future__ import annotations
"""
aipack  —  AI context compressor for local LLMs
================================================
Compatible: Python 3.7+.  Zero mandatory dependencies.

Usage
-----
  aipack compress <file> [options]
  aipack batch    <dir>  [options]
  aipack info     <file>
  aipack profiles

Options
-------
  -p, --profile   <name>   Compression profile  [default: semantic]
  -o, --output    <path>   Output file or directory
  -P, --passes    <a,b,c>  Explicit pass list (overrides profile)
  -e, --encoding  <enc>    File encoding         [default: utf-8]
  -q, --quiet              Suppress progress output
  -s, --stats              Print per-pass breakdown
      --overwrite          Overwrite input file in place
      --dry-run            Show stats without writing output
"""


import sys
import argparse
import textwrap
import time
from pathlib import Path

from .compress import (
    compress_file,
    compress_text,
    PROFILES,
    PASS_REGISTRY,
    estimate_tokens,
)

# ── ANSI colours (no third-party dep) ───────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[32m"
BLUE   = "\033[34m"
CYAN   = "\033[36m"
YELLOW = "\033[33m"
RED    = "\033[31m"
WHITE  = "\033[97m"


def _no_color() -> bool:
    import os
    return not sys.stdout.isatty() or os.environ.get("NO_COLOR")


def c(color: str, text: str) -> str:
    if _no_color():
        return text
    return f"{color}{text}{RESET}"


# ── Banner ────────────────────────────────────────────────────────────────────

BANNER = r"""
  █████╗ ██╗██████╗  █████╗  ██████╗██╗  ██╗
 ██╔══██╗██║██╔══██╗██╔══██╗██╔════╝██║ ██╔╝
 ███████║██║██████╔╝███████║██║     █████╔╝ 
 ██╔══██║██║██╔═══╝ ██╔══██║██║     ██╔═██╗ 
 ██║  ██║██║██║     ██║  ██║╚██████╗██║  ██╗
 ╚═╝  ╚═╝╚═╝╚═╝     ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝
 AI context compressor for local LLMs  v0.1.0
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _bar(ratio: float, width: int = 28) -> str:
    filled = int(ratio * width)
    bar = "█" * filled + "░" * (width - filled)
    return bar


def _print_result(stats, path: Path | None = None, show_passes: bool = False) -> None:
    ratio = stats.ratio
    bar   = _bar(ratio)
    color = GREEN if ratio > 0.3 else BLUE if ratio > 0.1 else YELLOW

    if path:
        print(c(BOLD + WHITE, f"\n  {path.name}"))

    print(
        f"  {c(DIM, 'Original')}   {_human(stats.original_bytes):>10}  "
        f"{c(DIM, str(stats.original_tokens) + ' tok')}"
    )
    print(
        f"  {c(DIM, 'Compressed')} {_human(stats.compressed_bytes):>10}  "
        f"{c(DIM, str(stats.compressed_tokens) + ' tok')}"
    )
    print(
        f"  {c(DIM, 'Saved')}      "
        f"{c(color, _human(stats.original_bytes - stats.compressed_bytes)):>10}  "
        f"{c(color, str(stats.tokens_saved) + ' tok')}"
    )
    print(
        f"\n  {c(color, bar)}  "
        f"{c(BOLD + color, f'{ratio*100:.1f}%')}"
    )

    if show_passes and stats.techniques:
        print(f"\n  {c(DIM, 'Passes breakdown:')}")
        for pass_name, pct in sorted(
            stats.techniques.items(), key=lambda x: -x[1]
        ):
            mini_bar = _bar(pct / 100, 16)
            print(
                f"    {pass_name:<20} {c(CYAN, mini_bar)}  "
                f"{c(DIM, f'{pct:.1f}%')}"
            )


# ── Commands ─────────────────────────────────────────────────────────────────

def cmd_profiles(args: argparse.Namespace) -> int:
    print(c(BOLD, "\n  Available profiles\n"))
    for name, info in PROFILES.items():
        passes = ", ".join(info["passes"])
        print(f"  {c(BOLD + CYAN, f'{name:<12}')}{info['description']}")
        print(f"  {' ' * 12}{c(DIM, 'passes: ' + passes)}\n")

    print(c(BOLD, "  Available passes\n"))
    for name in PASS_REGISTRY:
        print(f"    {c(CYAN, name)}")
    print()
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    path = Path(args.file)
    if not path.exists():
        print(c(RED, f"  File not found: {path}"), file=sys.stderr)
        return 1
    text = path.read_text(encoding=args.encoding, errors="replace")
    tokens = estimate_tokens(text)
    lines  = text.count("\n") + 1
    words  = len(text.split())
    print(c(BOLD, f"\n  {path.name}"))
    print(f"  Size    : {_human(len(text.encode()))}")
    print(f"  Lines   : {lines:,}")
    print(f"  Words   : {words:,}")
    print(f"  ~Tokens : {tokens:,}  (BPE estimate)")
    print()
    return 0


def cmd_compress(args: argparse.Namespace) -> int:
    src = Path(args.file)
    if not src.exists():
        print(c(RED, f"  File not found: {src}"), file=sys.stderr)
        return 1

    profile      = args.profile
    custom_passes = [p.strip() for p in args.passes.split(",")] if args.passes else None

    if not args.quiet:
        print(c(DIM, f"\n  Compressing {src.name} "
                     f"[profile: {custom_passes or profile}] ..."))

    t0 = time.perf_counter()

    if args.dry_run:
        text = src.read_text(encoding=args.encoding, errors="replace")
        _, stats = compress_text(text, profile=profile, custom_passes=custom_passes)
        dest = None
    else:
        if args.overwrite:
            dest = src
        elif args.output:
            dest = Path(args.output)
        else:
            dest = src.with_name(src.stem + ".compressed" + src.suffix)

        stats = compress_file(
            src, dest,
            profile=profile,
            custom_passes=custom_passes,
            encoding=args.encoding,
        )

    elapsed = time.perf_counter() - t0

    if not args.quiet:
        _print_result(stats, src, show_passes=args.stats)
        if dest and not args.dry_run:
            print(c(DIM, f"\n  Written to {dest}  ({elapsed*1000:.0f} ms)\n"))
        elif args.dry_run:
            print(c(YELLOW, "\n  Dry run — no file written.\n"))

    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    directory  = Path(args.dir)
    if not directory.is_dir():
        print(c(RED, f"  Not a directory: {directory}"), file=sys.stderr)
        return 1

    profile      = args.profile
    custom_passes = [p.strip() for p in args.passes.split(",")] if args.passes else None
    exts         = {e.strip() for e in args.ext.split(",")} if args.ext else {
        ".txt", ".md", ".json", ".jsonl", ".csv", ".py",
        ".js", ".ts", ".yaml", ".yml", ".toml", ".xml",
    }

    files = [f for f in directory.rglob("*") if f.is_file() and f.suffix in exts]

    if not files:
        print(c(YELLOW, f"  No matching files in {directory}"))
        return 0

    if not args.quiet:
        print(c(BOLD, f"\n  Batch compress  {directory}  ({len(files)} files)\n"))

    total_in  = 0
    total_out = 0
    total_tok_in  = 0
    total_tok_out = 0
    errors = 0

    out_dir = Path(args.output) if args.output else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for f in sorted(files):
        try:
            if out_dir:
                dest = out_dir / f.name
            elif args.overwrite:
                dest = f
            else:
                dest = f.with_name(f.stem + ".compressed" + f.suffix)

            if args.dry_run:
                text = f.read_text(encoding=args.encoding, errors="replace")
                _, stats = compress_text(
                    text, profile=profile, custom_passes=custom_passes
                )
            else:
                stats = compress_file(
                    f, dest,
                    profile=profile,
                    custom_passes=custom_passes,
                    encoding=args.encoding,
                )

            total_in      += stats.original_bytes
            total_out     += stats.compressed_bytes
            total_tok_in  += stats.original_tokens
            total_tok_out += stats.compressed_tokens

            if not args.quiet:
                ratio = stats.ratio
                color = GREEN if ratio > 0.3 else BLUE if ratio > 0.1 else DIM
                print(
                    f"  {c(DIM, f.name):<44} "
                    f"{c(color, f'{ratio*100:5.1f}%')}  "
                    f"{c(DIM, _human(stats.original_bytes))} -> "
                    f"{c(DIM, _human(stats.compressed_bytes))}"
                )
        except Exception as exc:
            errors += 1
            print(c(RED, f"  ERROR {f.name}: {exc}"), file=sys.stderr)

    if not args.quiet:
        ratio = max(0.0, 1 - total_out / max(total_in, 1))
        print(c(BOLD, f"\n  Total"))
        print(
            f"  {_human(total_in)} -> {_human(total_out)}  "
            f"{c(GREEN if ratio > 0.2 else BLUE, f'{ratio*100:.1f}% saved')}  "
            f"|  {total_tok_in:,} -> {total_tok_out:,} tokens"
        )
        if errors:
            print(c(RED, f"  {errors} file(s) failed"))
        if args.dry_run:
            print(c(YELLOW, "  Dry run — no files written."))
        print()

    return 0 if not errors else 1


# ── Argument parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aipack",
        description="AI context compressor for local LLMs (Llama 3, Mistral, …)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          aipack compress system_prompt.txt
          aipack compress data.jsonl -p jsonl --overwrite
          aipack compress context.md  -p llama3 -o out/context.md -s
          aipack batch ./data -p extreme -o ./data_compressed
          aipack batch ./data --ext .jsonl,.txt --dry-run
          aipack profiles
          aipack info huge_context.txt
        """),
    )
    sub = parser.add_subparsers(dest="command")

    # compress
    p_compress = sub.add_parser("compress", help="Compress a single file")
    p_compress.add_argument("file")
    p_compress.add_argument("-p", "--profile",  default="semantic")
    p_compress.add_argument("-o", "--output",   default=None)
    p_compress.add_argument("-P", "--passes",   default=None,
                             metavar="PASS1,PASS2")
    p_compress.add_argument("-e", "--encoding", default="utf-8")
    p_compress.add_argument("-q", "--quiet",    action="store_true")
    p_compress.add_argument("-s", "--stats",    action="store_true")
    p_compress.add_argument("--overwrite",      action="store_true")
    p_compress.add_argument("--dry-run",        action="store_true")

    # batch
    p_batch = sub.add_parser("batch", help="Compress all files in a directory")
    p_batch.add_argument("dir")
    p_batch.add_argument("-p", "--profile",   default="semantic")
    p_batch.add_argument("-o", "--output",    default=None)
    p_batch.add_argument("-P", "--passes",    default=None,
                          metavar="PASS1,PASS2")
    p_batch.add_argument("-e", "--encoding",  default="utf-8")
    p_batch.add_argument("-q", "--quiet",     action="store_true")
    p_batch.add_argument("-s", "--stats",     action="store_true")
    p_batch.add_argument("--ext",             default=None,
                          metavar=".txt,.md,…")
    p_batch.add_argument("--overwrite",       action="store_true")
    p_batch.add_argument("--dry-run",         action="store_true")

    # info
    p_info = sub.add_parser("info", help="Show stats for a file without compressing")
    p_info.add_argument("file")
    p_info.add_argument("-e", "--encoding", default="utf-8")

    # profiles
    sub.add_parser("profiles", help="List available profiles and passes")

    return parser


# ── Entry ─────────────────────────────────────────────────────────────────────

def main() -> None:
    if not _no_color():
        print(c(CYAN, BANNER))

    parser = build_parser()
    args   = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "compress": cmd_compress,
        "batch":    cmd_batch,
        "info":     cmd_info,
        "profiles": cmd_profiles,
    }

    sys.exit(dispatch[args.command](args))


if __name__ == "__main__":
    main()
