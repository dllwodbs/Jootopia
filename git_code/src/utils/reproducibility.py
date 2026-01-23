"""
재현성을 위한 manifest.json 및 metrics.json 생성 유틸리티

이 모듈은 파이프라인 실행 시 재현에 필요한 메타데이터를 자동으로 생성합니다.
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.utils.config import load_config


def get_git_sha(repo_dir: Optional[Path] = None) -> Optional[str]:
    """Git commit SHA를 가져옵니다."""
    try:
        repo_dir = repo_dir or Path.cwd()
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def get_config_hash(config_path: str | Path) -> str:
    """설정 파일의 SHA256 해시를 계산합니다."""
    config_path = Path(config_path)
    if not config_path.exists():
        return "unknown"
    
    with config_path.open("rb") as f:
        content = f.read()
    return hashlib.sha256(content).hexdigest()[:16]


def create_run_id() -> str:
    """실행 ID를 생성합니다 (타임스탬프 기반)."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_manifest(
    *,
    run_id: str,
    config_path: str | Path,
    track: str = "unknown",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    universe: Optional[str] = None,
    cost_bps: Optional[float] = None,
    horizon_short: Optional[int] = None,
    horizon_long: Optional[int] = None,
    seed: Optional[int] = None,
    repo_dir: Optional[Path] = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    재현성을 위한 manifest.json을 생성합니다.
    
    Args:
        run_id: 실행 ID
        config_path: 설정 파일 경로
        track: Track 이름 (track_a, track_b 등)
        start_date: 시작 날짜
        end_date: 종료 날짜
        universe: 유니버스 (예: "KOSPI200")
        cost_bps: 거래 비용 (bps)
        horizon_short: 단기 호라이즌 (일)
        horizon_long: 장기 호라이즌 (일)
        seed: 랜덤 시드
        repo_dir: Git 저장소 디렉토리
        extra: 추가 메타데이터
    
    Returns:
        manifest 딕셔너리
    """
    config_path = Path(config_path)
    repo_dir = repo_dir or Path.cwd()
    
    # 설정 파일 해시 계산
    config_hash = get_config_hash(config_path)
    
    # Git SHA 가져오기
    git_sha = get_git_sha(repo_dir)
    
    # 설정 파일 요약 추출
    try:
        cfg = load_config(config_path)
        config_summary = {
            "l4": cfg.get("l4", {}),
            "l5": cfg.get("l5", {}),
            "l6r": cfg.get("l6r", {}),
            "l7": cfg.get("l7", {}),
            "l8_short": cfg.get("l8_short", {}),
            "l8_long": cfg.get("l8_long", {}),
        }
    except Exception:
        config_summary = {}
    
    manifest = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "track": track,
        "config": {
            "path": str(config_path),
            "hash": config_hash,
            "summary": config_summary,
        },
        "git": {
            "sha": git_sha,
            "repo_dir": str(repo_dir),
        },
        "parameters": {
            "start_date": start_date,
            "end_date": end_date,
            "universe": universe,
            "cost_bps": cost_bps,
            "horizon_short": horizon_short,
            "horizon_long": horizon_long,
            "seed": seed,
        },
    }
    
    if extra:
        manifest["extra"] = extra
    
    return manifest


def save_manifest(
    manifest: dict[str, Any],
    output_dir: Path,
    *,
    filename: str = "manifest.json",
) -> Path:
    """manifest.json을 저장합니다."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / filename
    
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    return manifest_path


def load_metrics_from_bt_metrics(bt_metrics_path: Path) -> Optional[dict[str, Any]]:
    """백테스트 메트릭 파일에서 지표를 추출합니다."""
    try:
        if bt_metrics_path.suffix == ".parquet":
            df = pd.read_parquet(bt_metrics_path)
        elif bt_metrics_path.suffix == ".csv":
            df = pd.read_csv(bt_metrics_path)
        else:
            return None
        
        # phase별로 메트릭 추출
        metrics = {}
        
        for phase in ["dev", "holdout"]:
            phase_df = df[df.get("phase", "") == phase] if "phase" in df.columns else df
            
            if phase_df.empty:
                continue
            
            # 첫 번째 행 사용 (일반적으로 phase별로 1행)
            row = phase_df.iloc[0]
            
            phase_metrics = {
                "net_sharpe": float(row.get("net_sharpe", 0)) if pd.notna(row.get("net_sharpe")) else None,
                "net_cagr": float(row.get("net_cagr", 0)) if pd.notna(row.get("net_cagr")) else None,
                "net_mdd": float(row.get("net_mdd", 0)) if pd.notna(row.get("net_mdd")) else None,
                "net_calmar_ratio": float(row.get("net_calmar_ratio", 0)) if pd.notna(row.get("net_calmar_ratio")) else None,
                "net_hit_ratio": float(row.get("net_hit_ratio", 0)) if pd.notna(row.get("net_hit_ratio")) else None,
                "net_total_return": float(row.get("net_total_return", 0)) if pd.notna(row.get("net_total_return")) else None,
            }
            
            # None 값 제거
            phase_metrics = {k: v for k, v in phase_metrics.items() if v is not None}
            
            if phase_metrics:
                metrics[phase] = phase_metrics
        
        return metrics if metrics else None
    
    except Exception:
        return None


def build_metrics(
    *,
    run_id: str,
    strategy: Optional[str] = None,
    bt_metrics_path: Optional[Path] = None,
    custom_metrics: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    metrics.json을 생성합니다.
    
    Args:
        run_id: 실행 ID
        strategy: 전략 이름 (예: "bt120_long")
        bt_metrics_path: 백테스트 메트릭 파일 경로
        custom_metrics: 사용자 정의 메트릭
    
    Returns:
        metrics 딕셔너리
    """
    metrics: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
    }
    
    if strategy:
        metrics["strategy"] = strategy
    
    # 백테스트 메트릭 로드
    if bt_metrics_path and bt_metrics_path.exists():
        bt_metrics = load_metrics_from_bt_metrics(bt_metrics_path)
        if bt_metrics:
            metrics["backtest"] = bt_metrics
    
    # 사용자 정의 메트릭 추가
    if custom_metrics:
        metrics["custom"] = custom_metrics
    
    return metrics


def save_metrics(
    metrics: dict[str, Any],
    output_dir: Path,
    *,
    filename: str = "metrics.json",
) -> Path:
    """metrics.json을 저장합니다."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / filename
    
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    return metrics_path


def save_run_artifacts(
    *,
    run_id: str,
    config_path: str | Path,
    track: str,
    output_base_dir: Path,
    bt_metrics_path: Optional[Path] = None,
    strategy: Optional[str] = None,
    **manifest_kwargs: Any,
) -> tuple[Path, Path]:
    """
    실행 아티팩트를 저장합니다 (manifest.json + metrics.json).
    
    Returns:
        (manifest_path, metrics_path) 튜플
    """
    # runs/<run_id>/ 디렉토리 생성
    run_dir = output_base_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # manifest.json 생성
    manifest = build_manifest(
        run_id=run_id,
        config_path=config_path,
        track=track,
        **manifest_kwargs,
    )
    manifest_path = save_manifest(manifest, run_dir)
    
    # metrics.json 생성
    metrics = build_metrics(
        run_id=run_id,
        strategy=strategy,
        bt_metrics_path=bt_metrics_path,
    )
    metrics_path = save_metrics(metrics, run_dir)
    
    return manifest_path, metrics_path
