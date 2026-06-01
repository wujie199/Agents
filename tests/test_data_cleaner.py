import pytest
from utils import (
    clean_text,
    postprocess_ocr,
    extract_table_text,
    normalize_metadata,
    dedupe_chunks,
    semantic_dedupe,
    batch_clean,
    ocr_confidence_score,
    get_data_cleaner_metrics,
    reset_data_cleaner_metrics,
)


def _get_first(records):
    items = list(batch_clean(records))
    return items[0] if items else None


def test_clean_text_basic():
    s = "\tHello\nWorld  "
    assert clean_text(s) == "Hello World"


def test_postprocess_ocr():
    s = "Hello\nWorï¿½ld"
    assert 'ï¿½' not in postprocess_ocr(s)


def test_extract_table_text():
    table = [[1, 2], [3, 4]]
    out = extract_table_text(table)
    assert '1' in out and '4' in out


def test_extract_table_text_csv():
    csv_text = 'a,b,c\n1,2,3\n4,5,6'
    out = extract_table_text(csv_text)
    assert 'a | b | c' in out
    assert '4 | 5 | 6' in out


def test_extract_table_text_html():
    html = '<table><tr><th>a</th><th>b</th></tr><tr><td>1</td><td>2</td></tr></table>'
    out = extract_table_text(html)
    assert 'a | b' in out
    assert '1 | 2' in out


def test_extract_table_text_markdown():
    md = '| a | b |\n|---|---|\n|1|2|\n|3|4|'
    out = extract_table_text(md)
    assert 'a | b' in out
    assert '3 | 4' in out


def test_extract_table_text_dict_of_lists():
    table = {'a': [1, 3], 'b': [2, 4]}
    out = extract_table_text(table)
    assert 'a | b' in out
    assert '1 | 2' in out


def test_extract_table_text_json():
    json_text = '[{"a": 1, "b": 2}, {"a": 3, "b": 4}]'
    out = extract_table_text(json_text)
    assert 'a | b' in out
    assert '3 | 4' in out


def test_extract_table_text_json_varying_keys():
    json_text = '[{"a": 1, "b": 2}, {"a": 3, "c": 5}]'
    out = extract_table_text(json_text)
    assert 'a | b | c' in out
    assert '3 |  | 5' in out


def test_extract_table_text_invalid_html_returns_text():
    out = extract_table_text('not really <table> data')
    assert 'not really' in out


def test_data_cleaner_metrics_and_error_handling():
    reset_data_cleaner_metrics()
    out = list(batch_clean([{'source': 'x', 'source_id': '1', 'raw_text': 'valid'}, 'bad record', {'raw_text': 'ok'}]))
    assert len(out) == 2
    metrics = get_data_cleaner_metrics()
    assert metrics.get('records_cleaned', 0) == 2
    assert metrics.get('record_errors', 0) == 1


def test_normalize_metadata():
    meta = {' author ': 'å¼ ä¸', 'ts': None}
    out = normalize_metadata(meta)
    assert 'author' in out and 'ts' not in out


def test_normalize_metadata_date_and_tags():
    meta = {
        'åå»ºæ¶é´': '2024/05/19',
        'updated': '2024-05-20',
        'tags': 'AI, RAG;æµè¯',
    }
    out = normalize_metadata(meta)
    assert 'created_at' in out
    assert out['created_at'].startswith('2024-05-19')
    assert 'updated_at' in out
    assert out['updated_at'].startswith('2024-05-20')
    assert out['tags'] == ['AI', 'RAG', 'æµè¯']


def test_dedupe_chunks():
    chunks = [
        {'chunk_text': 'a', 'meta': {}},
        {'chunk_text': 'a', 'meta': {}},
        {'chunk_text': 'b', 'meta': {}},
    ]
    res = dedupe_chunks(chunks)
    assert len(res) == 2


def test_batch_clean():
    records = [
        {'source': 'f', 'source_id': '1', 'raw_text': 'hi'},
    ]
    out = list(batch_clean(records))
    assert out and out[0]['chunk_text'] == 'hi'


def test_postprocess_ocr_layout():
    text = "è¿æ¯ OCR è¾åº\nçææ¬ï¼\néè¦åå¹¶æ¢è¡ã"
    out = postprocess_ocr(text, preserve_layout=True)
    assert 'è¿æ¯ OCR è¾åº' in out
    assert '\n' in out


def test_postprocess_ocr_table():
    text = "å1  å2\n1    2\n3    4"
    out = postprocess_ocr(text, preserve_tables=True)
    assert 'å1' in out
    assert '\n' in out


def test_ocr_confidence():
    clean = 'Hello world! This is a test.'
    noisy = 'Hâ¬llo w0rld 123 ### ???'
    assert ocr_confidence_score(clean) >= ocr_confidence_score(noisy)


def test_language_detection_and_tokenize():
    en = _get_first([{'source': 'x', 'source_id': '1', 'raw_text': 'Hello world!'}])
    zh = _get_first([{'source': 'x', 'source_id': '2', 'raw_text': 'ä½ å¥½ï¼ä¸ç'}])
    assert en is not None and en['lang'] in ('en', 'und')
    assert zh is not None and zh['lang'] in ('zh', 'und')
    assert en['token_count'] >= 1
    assert zh['token_count'] >= 1


def test_semantic_dedupe_minhash():
    chunks = [
        {'chunk_text': 'This is a test sentence.'},
        {'chunk_text': 'This is a test sentence!'},
        {'chunk_text': 'Completely different content.'},
    ]
    out = semantic_dedupe(chunks, threshold=0.8)
    assert len(out) == 2
    assert any('Completely different' in item['chunk_text'] for item in out)


def test_semantic_dedupe_embedding():
    def emb_fn(text):
        if 'different' in text.lower():
            return [0.0, 1.0]
        return [1.0, 0.0]

    chunks = [
        {'chunk_text': 'Hello A'},
        {'chunk_text': 'Hello B'},
        {'chunk_text': 'Different item'},
    ]
    out = semantic_dedupe(chunks, threshold=0.8, embedding_fn=emb_fn)
    assert len(out) == 2
    assert any('Different item' in item['chunk_text'] for item in out)
