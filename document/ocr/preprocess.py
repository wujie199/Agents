# -*- coding: utf-8 -*-
"""页级预处理：文档朝向与曲面矫正（通用，与业务类型无关）。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from document.ocr.load_ocr import DOC_ORI_MODEL_NAME, UVDOC_MODEL_NAME, validate_model_dir


@dataclass(frozen=True)
class PreprocessPlan:
    """单次尝试的预处理计划。"""

    use_orientation: bool = False
    use_unwarp: bool = False


@dataclass
class PreprocessResult:
    image_path: Path
    orientation_angle: int | None = None
    orientation_applied: bool = False
    unwarp_applied: bool = False
    skipped: bool = True


def plan_for_attempt(
    attempt: int,
    *,
    preprocess_mode: str,
) -> PreprocessPlan:
    """按尝试轮次加重预处理（auto 模式，无 QC 提示时的默认阶梯）。"""
    mode = preprocess_mode.lower()
    if mode == "off":
        return PreprocessPlan()
    if mode == "on":
        return PreprocessPlan(use_orientation=True, use_unwarp=True)
    if attempt <= 0:
        return PreprocessPlan()
    if attempt == 1:
        return PreprocessPlan(use_orientation=True, use_unwarp=False)
    return PreprocessPlan(use_orientation=True, use_unwarp=True)


def resolve_preprocess_plan(
    attempt: int,
    *,
    preprocess_mode: str,
    suggest_heavier_preprocess: bool = False,
    suggest_table_e2e_only: bool = False,
) -> PreprocessPlan:
    """结合 QC 提示与轮次决定预处理（表格单失败时不盲目做朝向/矫正）。"""
    mode = preprocess_mode.lower()
    if mode == "off":
        return PreprocessPlan()
    if mode == "on":
        return PreprocessPlan(use_orientation=True, use_unwarp=True)

    if attempt <= 0:
        return PreprocessPlan()

    if suggest_table_e2e_only and not suggest_heavier_preprocess:
        return PreprocessPlan()

    if suggest_heavier_preprocess and mode == "off":
        return PreprocessPlan()

    if suggest_heavier_preprocess:
        if attempt == 1:
            return PreprocessPlan(use_orientation=True, use_unwarp=False)
        return PreprocessPlan(use_orientation=True, use_unwarp=True)

    return plan_for_attempt(attempt, preprocess_mode=preprocess_mode)


def _numpy_bgr_to_pil(arr: Any) -> Image.Image:
    return Image.fromarray(arr[:, :, ::-1])


class PagePreprocessor:
    """懒加载 DocPreprocessor，对单页图像做朝向/矫正。"""

    def __init__(
        self,
        *,
        model_root: Path,
        device: str,
        enable_mkldnn: bool,
    ) -> None:
        self.model_root = model_root
        self.device = device
        self.enable_mkldnn = enable_mkldnn
        self._pipes: dict[tuple[bool, bool], Any] = {}

    def _create_pipeline(self, plan: PreprocessPlan) -> Any:
        from paddleocr import DocPreprocessor

        key = (plan.use_orientation, plan.use_unwarp)
        if key in self._pipes:
            return self._pipes[key]

        root = self.model_root
        ori_dir = root / DOC_ORI_MODEL_NAME
        missing = validate_model_dir(ori_dir)
        if missing:
            raise FileNotFoundError(
                f"文档朝向模型不完整 ({ori_dir})，缺少: {', '.join(missing)}"
            )

        kw: dict[str, Any] = {"device": self.device}
        if self.enable_mkldnn and self.device.startswith("cpu"):
            kw["enable_mkldnn"] = True

        pipe_kw: dict[str, Any] = {
            "doc_orientation_classify_model_name": DOC_ORI_MODEL_NAME,
            "doc_orientation_classify_model_dir": str(ori_dir),
            "use_doc_orientation_classify": plan.use_orientation,
            "use_doc_unwarping": plan.use_unwarp,
            **kw,
        }
        if plan.use_unwarp:
            uw_dir = root / UVDOC_MODEL_NAME
            missing_uw = validate_model_dir(uw_dir)
            if missing_uw:
                raise FileNotFoundError(
                    f"曲面矫正模型不完整 ({uw_dir})，缺少: {', '.join(missing_uw)}"
                )
            pipe_kw["doc_unwarping_model_name"] = UVDOC_MODEL_NAME
            pipe_kw["doc_unwarping_model_dir"] = str(uw_dir)

        pipe = DocPreprocessor(**pipe_kw)
        self._pipes[key] = pipe
        return pipe

    def run(
        self,
        image_path: Path,
        scratch_dir: Path,
        plan: PreprocessPlan,
    ) -> PreprocessResult:
        if not plan.use_orientation and not plan.use_unwarp:
            return PreprocessResult(image_path=image_path, skipped=True)

        scratch_dir.mkdir(parents=True, exist_ok=True)
        out_path = scratch_dir / "preprocessed.png"
        pipe = self._create_pipeline(plan)
        outputs = list(pipe.predict(str(image_path)))
        if not outputs:
            return PreprocessResult(image_path=image_path, skipped=True)

        res = outputs[0]
        payload = res.json if hasattr(res, "json") else {}
        if isinstance(payload, dict) and "res" in payload:
            payload = payload["res"]
        angle_raw = payload.get("angle") if isinstance(payload, dict) else None
        angle: int | None = None
        if angle_raw is not None:
            try:
                angle = int(angle_raw)
            except (TypeError, ValueError):
                angle = None

        output_img = res["output_img"]
        _numpy_bgr_to_pil(output_img).save(out_path)

        return PreprocessResult(
            image_path=out_path,
            orientation_angle=angle,
            orientation_applied=plan.use_orientation,
            unwarp_applied=plan.use_unwarp,
            skipped=False,
        )
