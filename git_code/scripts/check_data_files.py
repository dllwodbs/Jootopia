"""
데이터 파일 검증 스크립트

외부 드라이브에서 다운로드한 데이터가 올바른지 확인합니다.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config, get_path
from src.utils.io import artifact_exists, load_artifact
import pandas as pd


def check_data_files():
    """필수 데이터 파일 확인"""
    cfg = load_config("configs/config.yaml")
    cal_data_dir = Path(get_path(cfg, "data_cal"))
    raw_data_dir = Path(get_path(cfg, "data_raw_data"))
    
    print("=" * 80)
    print("데이터 파일 검증")
    print("=" * 80)
    
    # Track A 필수 파일
    print("\n[Track A 필수 파일]")
    panel_path = cal_data_dir / "panel_merged_daily"
    dataset_path = cal_data_dir / "dataset_daily"
    
    track_a_ok = False
    if artifact_exists(panel_path):
        df = load_artifact(panel_path)
        print(f"  ✅ panel_merged_daily: {len(df):,}행, {len(df.columns)}컬럼")
        track_a_ok = True
    elif artifact_exists(dataset_path):
        df = load_artifact(dataset_path)
        print(f"  ✅ dataset_daily: {len(df):,}행, {len(df.columns)}컬럼")
        track_a_ok = True
    else:
        print("  ❌ panel_merged_daily 또는 dataset_daily 없음")
        print(f"     경로: {cal_data_dir}")
    
    # Track B 필수 파일
    print("\n[Track B 필수 파일]")
    
    required_files = {
        "dataset_daily": cal_data_dir / "dataset_daily",
        "cv_folds_short": cal_data_dir / "cv_folds_short",
        "cv_folds_long": cal_data_dir / "cv_folds_long",
        "universe_k200_membership_monthly": raw_data_dir / "universe_k200_membership_monthly",
    }
    
    track_b_ok = True
    for name, path in required_files.items():
        if artifact_exists(path):
            try:
                df = load_artifact(path)
                print(f"  ✅ {name}: {len(df):,}행")
            except Exception as e:
                print(f"  ⚠️ {name}: 파일은 있지만 로드 실패 - {e}")
                track_b_ok = False
        else:
            print(f"  ❌ {name} 없음: {path}")
            track_b_ok = False
    
    # Track A 산출물 (Track B 실행 전 필요)
    print("\n[Track A 산출물]")
    ranking_short_path = cal_data_dir / "ranking_short_daily"
    ranking_long_path = cal_data_dir / "ranking_long_daily"
    
    if artifact_exists(ranking_short_path) and artifact_exists(ranking_long_path):
        short_df = load_artifact(ranking_short_path)
        long_df = load_artifact(ranking_long_path)
        print(f"  ✅ ranking_short_daily: {len(short_df):,}행")
        print(f"  ✅ ranking_long_daily: {len(long_df):,}행")
        print("  → Track B 실행 가능")
    else:
        print("  ⚠️ ranking_*.parquet 없음 (Track A 먼저 실행 필요)")
        print(f"     경로: {cal_data_dir}")
    
    # 선택적 파일
    print("\n[선택적 파일]")
    optional_files = {
        "ohlcv_daily": raw_data_dir / "ohlcv_daily",
        "sector_map": raw_data_dir / "sector_map",
    }
    
    for name, path in optional_files.items():
        if artifact_exists(path):
            try:
                df = load_artifact(path)
                print(f"  ✅ {name}: {len(df):,}행 (선택적)")
            except Exception:
                print(f"  ✅ {name}: 존재 (선택적)")
        else:
            print(f"  ⚠️ {name}: 없음 (선택적, 없어도 실행 가능)")
    
    print("\n" + "=" * 80)
    if track_a_ok and track_b_ok:
        print("✅ 모든 필수 파일 확인 완료")
        print("\n다음 단계:")
        print("  1. Track A 실행: python -m src.pipeline.track_a_pipeline")
        print("  2. Track B 실행: python -m src.pipeline.track_b_pipeline bt120_long")
        return True
    else:
        print("❌ 일부 필수 파일이 없습니다.")
        if not track_a_ok:
            print("\n💡 Track A 실행을 위해 다음 파일이 필요합니다:")
            print(f"   - {cal_data_dir}/panel_merged_daily.parquet")
            print(f"   또는 {cal_data_dir}/dataset_daily.parquet")
        if not track_b_ok:
            print("\n💡 Track B 실행을 위해 다음 파일이 필요합니다:")
            for name, path in required_files.items():
                if not artifact_exists(path):
                    print(f"   - {path}")
        print("\n외부 드라이브에서 데이터를 다운로드하세요.")
        return False


if __name__ == "__main__":
    check_data_files()
