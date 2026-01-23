from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Union

import yaml


def find_repo_root(start_path: Path = None) -> Path:
    """Find repository root by looking for .git directory or pyproject.toml."""
    if start_path is None:
        start_path = Path(__file__).resolve()

    current = start_path
    while current.parent != current:  # Stop at filesystem root
        if (current / ".git").is_dir() or (current / "pyproject.toml").exists():
            return current
        current = current.parent

    # Fallback to current working directory if repo root not found
    return Path.cwd()


def _to_posix(p: str | Path) -> str:
    return str(p).replace("\\", "/")


def _replace_base_dir(value: Any, base_dir_posix: str) -> Any:
    """Replace {base_dir} placeholder recursively."""
    if isinstance(value, str):
        return value.replace("{base_dir}", base_dir_posix)
    if isinstance(value, dict):
        return {k: _replace_base_dir(v, base_dir_posix) for k, v in value.items()}
    if isinstance(value, list):
        return [_replace_base_dir(v, base_dir_posix) for v in value]
    return value


def _resolve_split_structure_paths(cfg: dict[str, Any], config_path: Path) -> dict[str, Any]:
    """
    [분리 구조] data_drive + git_code 구조의 경로 자동 해석
    
    {git_code_dir} -> git_code 디렉토리 경로
    {data_drive_dir} -> data_drive 디렉토리 경로
    """
    # config.yaml이 있는 디렉토리 = git_code/configs
    git_code_dir = config_path.parent.parent.resolve()
    project_root = git_code_dir.parent
    data_drive_dir = project_root / "data_drive"
    
    def _replace_path_placeholders(value: Any) -> Any:
        if isinstance(value, str):
            value = value.replace("{git_code_dir}", str(git_code_dir))
            value = value.replace("{data_drive_dir}", str(data_drive_dir))
            return value
        if isinstance(value, dict):
            return {k: _replace_path_placeholders(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_replace_path_placeholders(v) for v in value]
        return value
    
    return _replace_path_placeholders(cfg)


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # [분리 구조] 경로 플레이스홀더 해석 ({git_code_dir}, {data_drive_dir})
    cfg = _resolve_split_structure_paths(cfg, path)

    paths = cfg.get("paths", {})
    base_dir = paths.get("base_dir")
    if not base_dir:
        raise KeyError("configs/config.yaml must define: paths.base_dir")

    # [경로 고정] base_dir 깨진 문자열 검증 및 제거
    base_dir_str = str(base_dir).strip()
    if "???" in base_dir_str or "??" in base_dir_str or not base_dir_str:
        expected_base_dir = os.getenv(
            "BASE_DIR", str(find_repo_root()))
        raise ValueError(
            f"[FATAL] base_dir contains corrupted characters: {base_dir_str}\n"
            f"Fix configs/config.yaml paths.base_dir to: {expected_base_dir}"
        )

    base_dir_posix = _to_posix(base_dir)
    cfg = _replace_base_dir(cfg, base_dir_posix)

    # Convert common path fields to Path objects (optional but useful)
    paths = cfg.get("paths", {})
    for k, v in list(paths.items()):
        if isinstance(v, str) and ("/" in v or "\\" in v):
            paths[k] = Path(v)
    cfg["paths"] = paths

    # [경로 고정] 런타임 강제 검증: base_dir이 정확한 경로인지 확인
    # [분리 구조] data_drive + git_code 구조 지원
    expected_base_dir = os.getenv("BASE_DIR", str(find_repo_root()))
    EXPECTED = Path(expected_base_dir).resolve()
    ACTUAL = Path(paths["base_dir"]).resolve()

    # [분리 구조] git_code에서 실행 중이면 base_dir이 git_code여야 함
    # 하지만 config.yaml에 절대 경로가 있으면 그대로 사용 (하위 호환성)
    if ACTUAL != EXPECTED:
        # 경고만 출력하고 계속 진행 (분리 구조에서는 정상)
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            f"[경로] base_dir mismatch (분리 구조에서는 정상일 수 있음).\n"
            f"  expected: {EXPECTED}\n"
            f"  actual  : {ACTUAL}\n"
            f"  config.yaml의 base_dir을 확인하세요."
        )

    # [경로 고정] 런타임 로그 출력
    import logging

    logger = logging.getLogger(__name__)
    logger.info(f"[RUNTIME] cwd={Path.cwd().resolve()}")
    logger.info(f"[RUNTIME] base_dir={ACTUAL}")

    return cfg


def get_path(cfg: dict[str, Any], key: str) -> Path:
    paths = cfg.get("paths", {})
    if key not in paths:
        raise KeyError(f"Missing paths.{key} in config")
    v = paths[key]
    return v if isinstance(v, Path) else Path(str(v))


# 기본값 정의
DEFAULT_CONFIG = {
    "params": {"start_date": "2016-01-01", "end_date": "2024-12-31"},
    "run": {
        "fail_on_validation_error": True,
        "save_formats": ["parquet", "csv"],
        "skip_if_exists": False,
        "timezone": "Asia/Seoul",
        "write_meta": True,
    },
    "l7": {
        "cost_bps": 10.0,
        "holding_days": 20,
        "top_k": 12,
        "rebalance_interval": 1,
        "target_volatility": 0.15,
    },
}


def load_yaml(
    path: Union[str, Path], defaults: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """
    YAML 파일을 로드하고 기본값을 병합합니다.

    Args:
        path: YAML 파일 경로
        defaults: 기본값 딕셔너리 (옵션)

    Returns:
        병합된 설정 딕셔너리

    Raises:
        FileNotFoundError: 파일이 존재하지 않을 때
        yaml.YAMLError: YAML 파싱 오류
        ValueError: 유효하지 않은 기본값 키
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"YAML 파싱 오류 ({path}): {e}")

    # 기본값 병합
    if defaults:
        config = merge_defaults(config, defaults)

    # 환경변수 치환
    config = substitute_env_vars(config)

    return config


def merge_defaults(config: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    """
    설정에 기본값을 재귀적으로 병합합니다.

    Args:
        config: 사용자 설정
        defaults: 기본값

    Returns:
        병합된 설정
    """
    merged = defaults.copy()

    def _merge_dict(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
        for key, value in source.items():
            if key in target:
                if isinstance(target[key], dict) and isinstance(value, dict):
                    target[key] = _merge_dict(target[key], value)
                else:
                    target[key] = value
            else:
                target[key] = value
        return target

    return _merge_dict(merged, config)


def substitute_env_vars(config: Any) -> Any:
    """
    설정에서 환경변수 플레이스홀더를 치환합니다.
    ${VAR_NAME} 또는 $VAR_NAME 형식을 지원합니다.

    Args:
        config: 설정 딕셔너리 (재귀적 처리)

    Returns:
        환경변수가 치환된 설정
    """
    if isinstance(config, str):
        import re

        # ${VAR} 또는 $VAR 패턴 찾기
        pattern = re.compile(r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)")

        def replace_var(match):
            var_name = match.group(1) or match.group(2)
            value = os.getenv(var_name)
            if value is None:
                raise ValueError(f"환경변수가 설정되지 않음: {var_name}")
            return value

        return pattern.sub(replace_var, config)

    elif isinstance(config, dict):
        return {k: substitute_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [substitute_env_vars(item) for item in config]
    else:
        return config


def load_yaml_with_defaults(path: Union[str, Path]) -> dict[str, Any]:
    """
    YAML 파일을 로드하고 글로벌 기본값을 병합합니다.

    Args:
        path: YAML 파일 경로

    Returns:
        기본값이 병합된 설정
    """
    return load_yaml(path, DEFAULT_CONFIG)
