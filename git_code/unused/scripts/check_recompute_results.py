"""재산출 결과 확인"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.config import load_config, get_path
from src.utils.io import artifact_exists, load_artifact

cfg = load_config("configs/config.yaml")
interim = Path(get_path(cfg, "data_interim"))

print("=" * 80)
print("계산 데이터 재산출 결과 확인")
print("=" * 80)

# L3
if artifact_exists(interim / "panel_merged_daily"):
    panel = load_artifact(interim / "panel_merged_daily")
    print(f"✓ L3 (패널 병합): {len(panel):,}행")
else:
    print("✗ L3: 없음")

# L4
if artifact_exists(interim / "dataset_daily"):
    dataset = load_artifact(interim / "dataset_daily")
    print(f"✓ L4 (데이터셋): {len(dataset):,}행")
    
    if artifact_exists(interim / "cv_folds_short"):
        cv_s = load_artifact(interim / "cv_folds_short")
        print(f"✓ L4 (CV 단기): {len(cv_s):,}개 folds")
    
    if artifact_exists(interim / "cv_folds_long"):
        cv_l = load_artifact(interim / "cv_folds_long")
        print(f"✓ L4 (CV 장기): {len(cv_l):,}개 folds")
else:
    print("✗ L4: 없음")

# L8
if artifact_exists(interim / "ranking_short_daily"):
    ranking_s = load_artifact(interim / "ranking_short_daily")
    print(f"✓ L8 (단기 랭킹): {len(ranking_s):,}행, {ranking_s['date'].nunique()}개 날짜")
else:
    print("✗ L8 (단기): 없음")

if artifact_exists(interim / "ranking_long_daily"):
    ranking_l = load_artifact(interim / "ranking_long_daily")
    print(f"✓ L8 (장기 랭킹): {len(ranking_l):,}행, {ranking_l['date'].nunique()}개 날짜")
else:
    print("✗ L8 (장기): 없음")

# L6R
import glob
l6r_files = list(interim.glob("rebalance_scores_from_ranking_interval_*.parquet"))
if l6r_files:
    for f in l6r_files:
        df = load_artifact(f)
        print(f"✓ L6R ({f.stem}): {len(df):,}행")
else:
    print("✗ L6R: 없음")

# L7
strategies = ["bt120_long", "bt120_ens", "bt20_short", "bt20_ens"]
for strategy in strategies:
    if artifact_exists(interim / f"bt_metrics_{strategy}"):
        metrics = load_artifact(interim / f"bt_metrics_{strategy}")
        print(f"✓ L7 ({strategy}): {len(metrics)}개 메트릭")
    else:
        print(f"✗ L7 ({strategy}): 없음")

print("=" * 80)
