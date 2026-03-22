"""
외부 추천 알고리즘 폴더(예: /data/recommendation_algorithm)를 수정하지 않고,
경로 + 진입점(모듈:함수)만 환경 변수로 지정해 로드합니다.

템플릿 쪽 코드는 절대 편집하지 마세요. 이 파일과 algorithm.py만 유지보수합니다.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def resolve_external_algorithm() -> Optional[Callable[..., Any]]:
    """
    RECOMMENDATION_ALGORITHM_PATH + RECOMMENDATION_ALGORITHM_ENTRYPOINT 가 모두 있으면
    import 가능한 callable을 반환. 설정이 없으면 None.
    """
    root = os.getenv("RECOMMENDATION_ALGORITHM_PATH", "").strip()
    entry = os.getenv("RECOMMENDATION_ALGORITHM_ENTRYPOINT", "").strip()
    if not root or not entry:
        return None
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"RECOMMENDATION_ALGORITHM_PATH 가 디렉터리가 아닙니다: {root!r}"
        )
    if root not in sys.path:
        sys.path.insert(0, root)

    mod_name, sep, func_name = entry.partition(":")
    mod_name, func_name = mod_name.strip(), func_name.strip()
    if not sep or not mod_name or not func_name:
        raise ValueError(
            "RECOMMENDATION_ALGORITHM_ENTRYPOINT 는 '모듈경로:함수이름' 형식이어야 합니다. "
            "예: main:run_recommendation"
        )
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, func_name, None)
    if fn is None:
        raise AttributeError(
            f"모듈 {mod_name!r} 에 {func_name!r} 이(가) 없습니다."
        )
    if not callable(fn):
        raise TypeError(f"{entry!r} 은 호출 가능한 객체가 아닙니다.")
    return fn


def _invoke_algorithm(
    fn: Callable[..., Any],
    *,
    user_info: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> Any:
    """템플릿 시그니처 차이를 흡수: 키워드 우선, 실패 시 위치 인자."""
    try:
        return fn(user_info=user_info, candidates=candidates)
    except TypeError:
        return fn(user_info, candidates)


def _use_pipeline_module() -> bool:
    """pipeline.run_full_pipeline + db.get_engine 경로 사용 여부."""
    path = os.getenv("RECOMMENDATION_ALGORITHM_PATH", "").strip()
    if not path:
        return False
    entry = os.getenv("RECOMMENDATION_ALGORITHM_ENTRYPOINT", "").strip().lower()
    # 비어 있으면 표준 파이프라인으로 간주
    if not entry:
        return True
    return entry == "pipeline:run_full_pipeline"


def try_run_external(
    *,
    user_info: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> Optional[Any]:
    """
    외부 알고리즘이 설정되어 있으면 실행하고 원시 결과를 반환.
    설정이 없으면 None (NotImplemented → LLM 폴백).

    - RECOMMENDATION_ALGORITHM_PATH 만 있거나 ENTRYPOINT 가 pipeline:run_full_pipeline 이면
      pipeline_adapter.run_full_pipeline_wrapped (run_full_pipeline + sync engine)
    - 그 외 ENTRYPOINT 는 모듈:함수 (user_info, candidates) 직접 호출
    """
    if _use_pipeline_module():
        from .pipeline_adapter import run_full_pipeline_wrapped

        try:
            return run_full_pipeline_wrapped(user_info=user_info, candidates=candidates)
        except Exception:
            logger.exception(
                "pipeline.run_full_pipeline 실패 (PATH=%s)",
                os.getenv("RECOMMENDATION_ALGORITHM_PATH"),
            )
            raise

    fn = resolve_external_algorithm()
    if fn is None:
        return None
    try:
        return _invoke_algorithm(fn, user_info=user_info, candidates=candidates)
    except Exception:
        logger.exception(
            "외부 추천 알고리즘 실행 실패 (RECOMMENDATION_ALGORITHM_PATH=%s, ENTRYPOINT=%s)",
            os.getenv("RECOMMENDATION_ALGORITHM_PATH"),
            os.getenv("RECOMMENDATION_ALGORITHM_ENTRYPOINT"),
        )
        raise
