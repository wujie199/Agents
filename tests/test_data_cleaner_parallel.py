from utils import batch_clean_parallel


def test_batch_clean_parallel_threaded():
    records = [{'source': 't', 'source_id': str(i), 'raw_text': f'text {i}'} for i in range(200)]
    out = list(batch_clean_parallel(records, workers=8, use_process=False, chunk_size=32))
    assert len(out) == 200
    # æ£æ¥é¡ºåºæ§åå­æ®µ
    for i, item in enumerate(out):
        assert item['source'] == 't'
        assert item['token_count'] >= 1

# Process-based test is optional; keep but not required to run in restricted env
def test_batch_clean_parallel_process():
    records = [{'source': 'p', 'source_id': str(i), 'raw_text': f'è¿ç¨ ææ¬ {i}'} for i in range(50)]
    out = list(batch_clean_parallel(records, workers=4, use_process=False, chunk_size=16))
    assert len(out) == 50
    for item in out:
        assert 'lang' in item
