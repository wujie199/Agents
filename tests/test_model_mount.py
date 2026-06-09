"""外置模型盘挂载提醒。"""

import pytest

from document.model_mount import (
    is_external_volume_mounted,
    require_mounted_volume,
    unmounted_reminder,
    warn_if_unmounted,
)


def test_unmounted_volume_detected():
    assert is_external_volume_mounted("/Volumes/wj/model/ocr") is (
        __import__("pathlib").Path("/Volumes/wj").is_dir()
    )


def test_local_path_always_ok(tmp_path):
    assert is_external_volume_mounted(tmp_path / "models") is True


def test_unmounted_reminder_mentions_no_download():
    msg = unmounted_reminder(
        path="/Volumes/wj/model/ocr",
        purpose="OCR",
        env_hint="OCR_MODEL_ROOT",
    )
    assert "不会自动下载" in msg
    assert "/Volumes/wj" in msg
    assert "OCR_MODEL_ROOT" in msg


def test_require_mounted_raises_when_volume_missing(monkeypatch):
    monkeypatch.setattr(
        "document.model_mount.is_external_volume_mounted",
        lambda _p: False,
    )
    with pytest.raises(FileNotFoundError, match="外置模型盘未就绪"):
        require_mounted_volume(
            "/Volumes/wj/model/ocr",
            purpose="OCR",
            env_hint="OCR_MODEL_ROOT",
        )


def test_warn_if_unmounted_returns_false(monkeypatch):
    monkeypatch.setattr(
        "document.model_mount.is_external_volume_mounted",
        lambda _p: False,
    )
    assert (
        warn_if_unmounted(
            "/Volumes/wj/model/ocr",
            purpose="OCR",
        )
        is False
    )
