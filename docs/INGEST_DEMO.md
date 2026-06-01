# Ingest Demo

This demo shows how to load a source file, normalize it, split it into chunks, and clean the chunks using the project's `data_cleaner` utilities.

## Usage

Run the script with a source file path:

```bash
python scripts/ingest_demo.py path/to/file.txt
```

Supported file types:

- `.txt`
- `.csv`
- `.tsv`
- `.md`
- `.json`
- `.xls`, `.xlsx`, `.xlsm`, `.xlsb`

## Options

- `--chunk-size`: chunk size for text splitting (default `200`)
- `--chunk-overlap`: overlap size between chunks (default `50`)
- `--max-display`: maximum number of chunks printed in the demo (default `5`)

## What it does

1. Loads the source file.
2. Uses `extract_table_text` for structured sources and plain read for generic text.
3. Cleans text with `clean_text`.
4. Splits the cleaned text into chunks using `split_text_into_chunks`.
5. Runs `batch_clean` to produce cleaned chunk records.
6. Prints the first few cleaned chunks and metrics.

## Output

The demo prints:

- chunk id
- language
- token count
- cleaned chunk text preview
- data cleaner metrics

## Notes

- The script uses `utils.setup_logger` to initialize logging.
- If `pandas` is installed, Excel ingestion is supported.
