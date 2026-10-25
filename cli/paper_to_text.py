import argparse
from pathlib import Path

from soulxpodcast.document import CleaningOptions, extract_speech_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract speech-ready text from an academic PDF.")
    parser.add_argument("pdf", help="Path to a born-digital academic PDF")
    parser.add_argument("--output", "-o", help="Output .txt path; defaults beside the PDF")
    parser.add_argument("--grobid-url", default="http://localhost:8070")
    parser.add_argument("--no-title", action="store_true")
    parser.add_argument("--no-abstract", action="store_true")
    parser.add_argument("--no-headings", action="store_true")
    parser.add_argument("--keep-citations", action="store_true")
    args = parser.parse_args()

    pdf = Path(args.pdf)
    output = Path(args.output) if args.output else pdf.with_suffix(".speech.txt")
    options = CleaningOptions(
        include_title=not args.no_title,
        include_abstract=not args.no_abstract,
        include_headings=not args.no_headings,
        remove_inline_citations=not args.keep_citations,
    )
    result = extract_speech_text(pdf, grobid_url=args.grobid_url, options=options)
    output.write_text(result.text + "\n", encoding="utf-8")
    print(f"[INFO] Saved speech-ready text to: {output}")


if __name__ == "__main__":
    main()
