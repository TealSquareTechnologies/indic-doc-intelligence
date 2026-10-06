"""Command line access to the same pipeline the API serves."""

from __future__ import annotations

import argparse
import mimetypes
import sys
from pathlib import Path

from app.core.errors import AppError
from app.core.logging import configure_logging
from app.db.database import init_database
from app.documents import service
from app.documents.service import classify_upload, run_pipeline
from app.rag.chunking import ChunkStrategy
from app.rag.pipeline import ask
from app.translate.translator import translate

RULE = "-" * 68


def _print_section(title: str, body: str) -> None:
    print(f"\n{title}\n{RULE}\n{body.strip() or '(empty)'}")


def _read_file(path_text: str) -> tuple[Path, bytes, str | None]:
    path = Path(path_text).expanduser()
    if not path.is_file():
        raise SystemExit(f"File not found: {path}")
    content_type, _ = mimetypes.guess_type(path.name)
    return path, path.read_bytes(), content_type


def command_recognise(args: argparse.Namespace) -> int:
    """Recognise a file and print the result without storing it."""
    path, data, content_type = _read_file(args.path)
    result = run_pipeline(
        data,
        source_kind=classify_upload(content_type, path.name),
        language=args.language,
        apply_correction=not args.no_correct,
        apply_translation=args.translate,
        target_language=args.target_language,
    )

    languages = ", ".join(f"{item.name} ({item.share:.0%})" for item in result.detected)
    print(f"\nFile:        {path.name}")
    print(f"Languages:   {languages or 'none detected'}")
    print(f"Confidence:  {result.confidence:.0%}")
    print(f"Elapsed:     {result.elapsed_ms} ms")

    _print_section("Raw text", result.raw_text)
    if result.correction_applied:
        _print_section("Corrected text", result.corrected_text)
    if result.translated_text:
        _print_section(f"Translated ({args.target_language})", result.translated_text)
    for note in result.notes:
        print(f"\nNote: {note}")
    return 0


def command_add(args: argparse.Namespace) -> int:
    """Process a file and store it so it can be searched."""
    init_database()
    path, data, content_type = _read_file(args.path)
    row = service.create_document(
        filename=path.name,
        content_type=content_type,
        data=data,
        language=args.language,
        apply_correction=not args.no_correct,
        apply_translation=args.translate,
        target_language=args.target_language,
        index=True,
        strategy=ChunkStrategy(args.strategy),
    )
    print(f"Stored document {row['id']}: {row['filename']} ({row['chunk_count']} chunks)")
    return 0


def command_list(args: argparse.Namespace) -> int:
    """List stored documents."""
    init_database()
    rows = service.list_documents(limit=args.limit)
    if not rows:
        print("No documents stored yet.")
        return 0

    print(f"{'ID':>4}  {'STATUS':<10} {'CHUNKS':>6}  FILENAME")
    for row in rows:
        print(f"{row['id']:>4}  {row['status']:<10} {row['chunk_count']:>6}  {row['filename']}")
    return 0


def command_delete(args: argparse.Namespace) -> int:
    """Remove a stored document."""
    init_database()
    service.delete_document(args.document_id)
    print(f"Deleted document {args.document_id}")
    return 0


def command_ask(args: argparse.Namespace) -> int:
    """Answer a question from the stored documents."""
    init_database()
    answer = ask(args.question, document_ids=args.document, top_k=args.top_k)

    _print_section("Answer", answer.answer)
    print(f"\nBest similarity: {answer.best_score:.4f}   Grounded: {answer.grounded}")

    if answer.sources and args.show_sources:
        for position, source in enumerate(answer.sources, start=1):
            preview = source.text[:300] + ("..." if len(source.text) > 300 else "")
            _print_section(
                f"Source {position} (document {source.document_id}, score {source.score:.4f})",
                preview,
            )
    return 0


def command_translate(args: argparse.Namespace) -> int:
    """Translate text supplied on the command line or through standard input."""
    text = args.text or sys.stdin.read()
    result = translate(text, target_language=args.target_language)
    _print_section(f"Translation ({result.target_language})", result.text)
    if not result.applied and result.detail:
        print(f"\nNote: {result.detail}")
    return 0


def command_serve(args: argparse.Namespace) -> int:
    """Start the API server and the browser interface."""
    import uvicorn

    from app.core.config import get_settings

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=args.host or settings.host,
        port=args.port or settings.port,
        reload=args.reload,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Define the command line surface."""
    parser = argparse.ArgumentParser(
        prog="indic-doc",
        description="Recognise, correct, translate and question multilingual documents.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_pipeline_options(target: argparse.ArgumentParser) -> None:
        target.add_argument("-l", "--language", help="Language string such as 'kan+eng'")
        target.add_argument("--no-correct", action="store_true", help="Skip model correction")
        target.add_argument("--translate", action="store_true", help="Also translate the text")
        target.add_argument("--target-language", default="English", help="Translation target")

    recognise = subparsers.add_parser("ocr", help="Recognise a file without storing it")
    recognise.add_argument("path", help="Path to an image or text file")
    add_pipeline_options(recognise)
    recognise.set_defaults(handler=command_recognise)

    add = subparsers.add_parser("add", help="Process a file and store it for searching")
    add.add_argument("path", help="Path to an image or text file")
    add_pipeline_options(add)
    add.add_argument(
        "--strategy",
        choices=[item.value for item in ChunkStrategy],
        default=ChunkStrategy.WORDS.value,
        help="How the document is divided before indexing",
    )
    add.set_defaults(handler=command_add)

    listing = subparsers.add_parser("list", help="List stored documents")
    listing.add_argument("--limit", type=int, default=100)
    listing.set_defaults(handler=command_list)

    removal = subparsers.add_parser("delete", help="Remove a stored document")
    removal.add_argument("document_id", type=int)
    removal.set_defaults(handler=command_delete)

    question = subparsers.add_parser("ask", help="Ask a question about stored documents")
    question.add_argument("question")
    question.add_argument(
        "-d", "--document", type=int, action="append", help="Restrict to this document id"
    )
    question.add_argument("-k", "--top-k", type=int, default=None)
    question.add_argument("--show-sources", action="store_true", help="Print retrieved chunks")
    question.set_defaults(handler=command_ask)

    translation = subparsers.add_parser("translate", help="Translate text")
    translation.add_argument("text", nargs="?", help="Text to translate, or pipe it in")
    translation.add_argument("--target-language", default="English")
    translation.set_defaults(handler=command_translate)

    serve = subparsers.add_parser("serve", help="Start the API and browser interface")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(handler=command_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the console script."""
    configure_logging()
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except AppError as exc:
        print(f"\nError: {exc.message}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
