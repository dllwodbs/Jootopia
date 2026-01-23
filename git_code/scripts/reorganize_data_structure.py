"""
데이터 폴더 구조 재구성 스크립트

1. 필수/선택 데이터 구분
2. 원천 데이터 -> data/raw_data
3. 계산 데이터 -> data/cal_data
4. 설명 md 파일 생성
"""

import sys
from pathlib import Path
import shutil

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.config import load_config, get_path

cfg = load_config("configs/config.yaml")

# 경로 설정
base_data_dir = Path(get_path(cfg, "data_interim")).parent
raw_data_dir = base_data_dir / "raw_data"
cal_data_dir = base_data_dir / "cal_data"

# 기존 경로
interim_dir = Path(get_path(cfg, "data_interim"))
external_dir = Path(get_path(cfg, "data_ext"))

print("=" * 80)
print("데이터 폴더 구조 재구성")
print("=" * 80)

# 필수 원천 데이터 (다운로드 가능하지만 재산출을 위해 보관)
REQUIRED_RAW_DATA = {
    "universe_k200_membership_monthly": "L0 유니버스 (KOSPI200 멤버십)",
    "ohlcv_daily": "L1 OHLCV 데이터 (거래일별 가격/거래량)",
}

# 선택적 원천 데이터
OPTIONAL_RAW_DATA = {
    "fundamentals_annual": "L2 DART 재무 데이터 (선택, API 키 필요)",
    "pykrx_fundamentals_daily": "L1B pykrx 재무 데이터 (선택)",
}

# 외부 데이터 (선택)
EXTERNAL_DATA = {
    "esg_daily.parquet": "ESG 데이터 (선택)",
    "news_sentiment_daily.parquet": "뉴스 감성 데이터 (선택)",
    "sector_map.csv": "섹터 매핑 (선택)",
}

# 필수 계산 데이터
REQUIRED_CAL_DATA = {
    "panel_merged_daily": "L3 패널 병합 데이터",
    "dataset_daily": "L4 데이터셋 (타겟 포함)",
    "cv_folds_short": "L4 CV 분할 (단기)",
    "cv_folds_long": "L4 CV 분할 (장기)",
    "ranking_short_daily": "L8 단기 랭킹",
    "ranking_long_daily": "L8 장기 랭킹",
}

# 선택적 계산 데이터
OPTIONAL_CAL_DATA = {
    "rebalance_scores_from_ranking_interval_*": "L6R 리밸런싱 스코어",
    "bt_positions_*": "L7 백테스트 포지션",
    "bt_returns_*": "L7 백테스트 수익률",
    "bt_equity_curve_*": "L7 백테스트 자산 곡선",
    "bt_metrics_*": "L7 백테스트 성과 지표",
}

# 디렉토리 생성
raw_data_dir.mkdir(parents=True, exist_ok=True)
cal_data_dir.mkdir(parents=True, exist_ok=True)

print(f"\n[디렉토리 생성]")
print(f"  원천 데이터: {raw_data_dir}")
print(f"  계산 데이터: {cal_data_dir}")

# 원천 데이터 이동
print(f"\n[원천 데이터 이동]")
moved_raw = []

# 필수 원천 데이터
for artifact_name, description in REQUIRED_RAW_DATA.items():
    src = interim_dir / artifact_name
    if src.exists() or (interim_dir / f"{artifact_name}.parquet").exists():
        src_file = src if src.exists() else (interim_dir / f"{artifact_name}.parquet")
        dst = raw_data_dir / src_file.name
        if src_file.is_file():
            shutil.copy2(src_file, dst)
            moved_raw.append((artifact_name, "필수"))
            print(f"  ✓ {artifact_name} (필수)")
        elif src_file.is_dir():
            shutil.copytree(src_file, dst, dirs_exist_ok=True)
            moved_raw.append((artifact_name, "필수"))
            print(f"  ✓ {artifact_name} (필수, 폴더)")

# 선택적 원천 데이터
for artifact_name, description in OPTIONAL_RAW_DATA.items():
    src = interim_dir / artifact_name
    if src.exists() or (interim_dir / f"{artifact_name}.parquet").exists():
        src_file = src if src.exists() else (interim_dir / f"{artifact_name}.parquet")
        dst = raw_data_dir / src_file.name
        if src_file.is_file():
            shutil.copy2(src_file, dst)
            moved_raw.append((artifact_name, "선택"))
            print(f"  ✓ {artifact_name} (선택)")
        elif src_file.is_dir():
            shutil.copytree(src_file, dst, dirs_exist_ok=True)
            moved_raw.append((artifact_name, "선택"))
            print(f"  ✓ {artifact_name} (선택, 폴더)")

# 외부 데이터
if external_dir.exists():
    for file_name, description in EXTERNAL_DATA.items():
        src = external_dir / file_name
        if src.exists():
            dst = raw_data_dir / file_name
            shutil.copy2(src, dst)
            moved_raw.append((file_name, "선택(외부)"))
            print(f"  ✓ {file_name} (선택, 외부)")

# 계산 데이터 이동
print(f"\n[계산 데이터 이동]")
moved_cal = []

# 필수 계산 데이터
for artifact_name, description in REQUIRED_CAL_DATA.items():
    src = interim_dir / artifact_name
    if src.exists() or (interim_dir / f"{artifact_name}.parquet").exists():
        src_file = src if src.exists() else (interim_dir / f"{artifact_name}.parquet")
        dst = cal_data_dir / src_file.name
        if src_file.is_file():
            shutil.copy2(src_file, dst)
            moved_cal.append((artifact_name, "필수"))
            print(f"  ✓ {artifact_name} (필수)")
        elif src_file.is_dir():
            shutil.copytree(src_file, dst, dirs_exist_ok=True)
            moved_cal.append((artifact_name, "필수"))
            print(f"  ✓ {artifact_name} (필수, 폴더)")

# 선택적 계산 데이터 (패턴 매칭)
import glob
for pattern in OPTIONAL_CAL_DATA.keys():
    pattern_clean = pattern.replace("*", "")
    for src_file in interim_dir.glob(f"{pattern_clean}*"):
        if src_file.is_file():
            dst = cal_data_dir / src_file.name
            shutil.copy2(src_file, dst)
            moved_cal.append((src_file.name, "선택"))
            print(f"  ✓ {src_file.name} (선택)")

print("\n" + "=" * 80)
print(f"이동 완료: 원천 {len(moved_raw)}개, 계산 {len(moved_cal)}개")
print("=" * 80)
