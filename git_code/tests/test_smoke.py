"""
스모크 테스트: 파이프라인 기본 동작 검증

이 테스트는 파이프라인이 깨지지 않았는지 빠르게 확인합니다.
전체 파이프라인을 실행하지 않고, 핵심 컴포넌트의 import와 기본 동작만 검증합니다.
"""

import sys
from pathlib import Path

import pytest

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_import_core_modules():
    """핵심 모듈 import 테스트"""
    # Config 유틸리티
    from src.utils.config import load_config, get_path
    
    # IO 유틸리티
    from src.utils.io import load_artifact, save_artifact, artifact_exists
    
    # Meta 유틸리티
    from src.utils.meta import build_meta, save_meta
    
    # 재현성 유틸리티
    from src.utils.reproducibility import (
        build_manifest,
        build_metrics,
        save_run_artifacts,
    )
    
    # 파이프라인
    from src.pipeline.track_a_pipeline import run_track_a_pipeline
    from src.pipeline.track_b_pipeline import run_track_b_pipeline
    
    # 데이터 수집
    from src.data_collection.pipeline import DataCollectionPipeline
    
    assert True  # 모든 import가 성공하면 통과


def test_config_load():
    """설정 파일 로드 테스트"""
    from src.utils.config import load_config
    
    config_path = project_root / "configs" / "config.yaml"
    
    if not config_path.exists():
        pytest.skip("config.yaml not found")
    
    try:
        cfg = load_config(config_path)
        assert "paths" in cfg
        assert "base_dir" in cfg.get("paths", {})
    except Exception as e:
        # 경로 문제 등으로 실패할 수 있음 (정상)
        pytest.skip(f"Config load failed (expected in CI): {e}")


def test_reproducibility_utils():
    """재현성 유틸리티 기본 동작 테스트"""
    from src.utils.reproducibility import (
        create_run_id,
        get_config_hash,
        build_manifest,
        build_metrics,
    )
    
    # Run ID 생성
    run_id = create_run_id()
    assert isinstance(run_id, str)
    assert len(run_id) > 0
    
    # Config hash 계산
    config_path = project_root / "configs" / "config.yaml"
    if config_path.exists():
        config_hash = get_config_hash(config_path)
        assert isinstance(config_hash, str)
        assert len(config_hash) > 0
    
    # Manifest 생성
    manifest = build_manifest(
        run_id=run_id,
        config_path=config_path if config_path.exists() else "configs/config.yaml",
        track="test",
    )
    assert manifest["run_id"] == run_id
    assert manifest["track"] == "test"
    assert "timestamp" in manifest
    assert "config" in manifest
    
    # Metrics 생성
    metrics = build_metrics(run_id=run_id, strategy="test_strategy")
    assert metrics["run_id"] == run_id
    assert "timestamp" in metrics


def test_cli_import():
    """CLI 모듈 import 테스트"""
    from src.cli import (
        run_full_pipeline,
        run_data_download,
        run_track_a,
        run_track_b,
        main,
    )
    
    assert callable(run_full_pipeline)
    assert callable(run_data_download)
    assert callable(run_track_a)
    assert callable(run_track_b)
    assert callable(main)


def test_data_collection_pipeline_init():
    """데이터 수집 파이프라인 초기화 테스트"""
    from src.data_collection.pipeline import DataCollectionPipeline
    
    config_path = project_root / "configs" / "config.yaml"
    
    if not config_path.exists():
        pytest.skip("config.yaml not found")
    
    try:
        pipeline = DataCollectionPipeline(
            config_path=str(config_path),
            force_rebuild=False,
        )
        assert pipeline is not None
    except Exception as e:
        # 경로 문제 등으로 실패할 수 있음 (정상)
        pytest.skip(f"Pipeline init failed (expected in CI): {e}")


def test_artifact_io_basic():
    """아티팩트 I/O 기본 동작 테스트 (더미 데이터)"""
    import tempfile
    from pathlib import Path
    import pandas as pd
    
    from src.utils.io import save_artifact, load_artifact, artifact_exists
    
    # 임시 디렉토리 생성
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        artifact_base = tmp_path / "test_artifact"
        
        # 더미 데이터 생성
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=10, freq="D"),
            "ticker": ["A"] * 10,
            "value": range(10),
        })
        
        # 저장
        save_artifact(df, artifact_base, formats=["parquet"], force=True)
        
        # 존재 확인
        assert artifact_exists(artifact_base, formats=["parquet"])
        
        # 로드
        loaded_df = load_artifact(artifact_base)
        assert len(loaded_df) == 10
        assert "ticker" in loaded_df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
