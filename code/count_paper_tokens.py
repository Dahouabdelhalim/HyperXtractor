import argparse
import csv
import json
from pathlib import Path

import tiktoken


def read_paper(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        source = file.read()

    if path.suffix.lower() == ".json":
        data = json.loads(source)
        sentences = data["sentences"]


        text = " ".join(
            " ".join(sentence)
            for sentence in sentences
        )

        paper_id = data.get("doc_key") or path.stem
        return paper_id, text, len(sentences)

    return path.stem, source, ""


def main():
    parser = argparse.ArgumentParser(
        description="Count tokens in Markdown and SciNLP JSON papers."
    )
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--output", type=Path, default=Path("paper_lengths.csv"))
    args = parser.parse_args()

    # Both Markdown and JSON files are supported.
    extensions = {".md", ".markdown", ".json", ".txt"}

    if args.input_path.is_file():
        candidates = [args.input_path]
    elif args.input_path.is_dir():
        candidates = args.input_path.iterdir()
    else:
        raise SystemExit(f"Input does not exist: {args.input_path}")

    files = sorted(
        path for path in candidates
        if path.is_file() and path.suffix.lower() in extensions
    )

    if not files:
        raise SystemExit("No Markdown or JSON files found.")

    if args.output.resolve() in {path.resolve() for path in files}:
        raise SystemExit("Output must not overwrite an input paper.")

    tokenizer = tiktoken.get_encoding("o200k_base")
    rows = []

    for path in files:
        paper_id, text, sentence_count = read_paper(path)

        # Tokenize the reconstructed document as one complete text.
        token_count = len(tokenizer.encode_ordinary(text))

        rows.append({
            "paper_id": paper_id,
            "num_sentences": sentence_count,
            "num_tokens": token_count,
            "num_words": len(text.split()),
            "num_characters": len(text),
        })

        print(f"{paper_id}: {token_count:,} tokens")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    total_tokens = sum(row["num_tokens"] for row in rows)

    print(f"\nNumber of papers: {len(rows)}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Average tokens per paper: {total_tokens / len(rows):,.2f}")
    print(f"CSV saved to: {args.output}")


if __name__ == "__main__":
    main()
