"""索引 MD5 清单与 doc_id 生成。"""

from document.rag.application.indexing.index_manifest import (
    IndexManifest,
    doc_id_from_file_md5,
    file_md5_hex,
)
from document.rag.config import RagPipelineConfig, compute_index_config_hash


def test_file_md5_stable(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello", encoding="utf-8")
    assert file_md5_hex(f) == file_md5_hex(f)
    assert doc_id_from_file_md5(file_md5_hex(f)).startswith("doc_")


def test_manifest_skip_register(tmp_path):
    manifest_path = tmp_path / "indexed_by_md5.json"
    m = IndexManifest(manifest_path)
    md5 = "abc123" * 5 + "abcd"
    assert not m.is_indexed("t1", md5)
    m.register(
        "t1",
        md5,
        doc_id="doc_x",
        source_path="/x.pdf",
        model_version="v1",
        config_hash="hash_v1",
        chunk_count=3,
    )
    assert m.is_indexed("t1", md5)
    assert m.matches_index_config(
        "t1",
        md5,
        model_version="v1",
        config_hash="hash_v1",
    )
    assert not m.matches_index_config(
        "t1",
        md5,
        model_version="v2",
        config_hash="hash_v1",
    )
    assert not m.matches_index_config(
        "t1",
        md5,
        model_version="v1",
        config_hash="hash_v2",
    )
    entry = m.get_entry("t1", md5)
    assert entry["doc_id"] == "doc_x"
    assert entry["chunk_count"] == 3


def test_config_hash_changes_when_chunk_config_changes():
    base = RagPipelineConfig(model_version="v1", chunk_size=200)
    other = RagPipelineConfig(model_version="v1", chunk_size=400)
    assert compute_index_config_hash(base) != compute_index_config_hash(other)
