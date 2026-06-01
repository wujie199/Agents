"""Ingest demo: read a source file, clean text, split into chunks, and print cleaned records."""
import argparse
import json
import os

from utils import (
    setup_logger,
    logger,
    clean_text,
    extract_table_text,
    batch_clean,
    split_text_into_chunks,
    get_data_cleaner_metrics,
    reset_data_cleaner_metrics,
)

LOGGER = setup_logger('ingest_demo')


def load_source_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.csv', '.tsv', '.md', '.json', '.txt', '.xls', '.xlsx', '.xlsm', '.xlsb'):
        return extract_table_text(path)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception:
        LOGGER.exception('Failed to read file: %s', path)
        return ''


def demo_ingest(path: str, chunk_size: int, chunk_overlap: int, max_display: int):
    reset_data_cleaner_metrics()
    raw = load_source_text(path)
    if not raw:
        LOGGER.error('No text extracted from %s', path)
        return

    LOGGER.info('Loaded %d characters from %s', len(raw), path)
    cleaned = clean_text(raw)
    LOGGER.info('Cleaned text length: %d', len(cleaned))

    chunks = split_text_into_chunks(cleaned, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    LOGGER.info('Split into %d chunks (chunk_size=%d, chunk_overlap=%d)', len(chunks), chunk_size, chunk_overlap)

    records = [
        {'source': 'demo', 'source_id': path, 'raw_text': chunk}
        for chunk in chunks
    ]
    cleaned_records = list(batch_clean(records))

    for index, record in enumerate(cleaned_records[:max_display], start=1):
        print('\n--- chunk %d ---' % index)
        print('id:', record['id'])
        print('lang:', record['lang'])
        print('token_count:', record['token_count'])
        print(record['chunk_text'][:400])

    print('\nMetrics:')
    print(json.dumps(get_data_cleaner_metrics(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo ingestion pipeline for text and table files.')
    parser.add_argument('path', help='Path to the source file to ingest')
    parser.add_argument('--chunk-size', type=int, default=200, help='Chunk size for text splitting')
    parser.add_argument('--chunk-overlap', type=int, default=50, help='Chunk overlap for text splitting')
    parser.add_argument('--max-display', type=int, default=5, help='Maximum number of chunks to print')

    args = parser.parse_args()
    demo_ingest(args.path, args.chunk_size, args.chunk_overlap, args.max_display)
