"""영업일 기준 랭킹 산정 확인"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.config import load_config, get_path
from src.utils.io import load_artifact
import pandas as pd

cfg = load_config("configs/config.yaml")
interim = Path(get_path(cfg, "data_interim"))

print("=" * 80)
print("영업일 기준 랭킹 산정 확인")
print("=" * 80)

ranking_s = load_artifact(interim / "ranking_short_daily")
dates = pd.to_datetime(ranking_s['date'].unique())

print(f"\n랭킹 날짜 범위: {dates.min().date()} ~ {dates.max().date()}")
print(f"총 날짜 수: {len(dates):,}일")

# 요일별 분포
weekdays = pd.Series(dates).dt.weekday
weekday_names = ['월', '화', '수', '목', '금', '토', '일']
for i, name in enumerate(weekday_names):
    count = (weekdays == i).sum()
    pct = count / len(weekdays) * 100
    print(f"  {name}요일: {count:,}일 ({pct:.1f}%)")

print(f"\n월~금 비율: {(weekdays < 5).sum() / len(weekdays):.1%}")
print(f"주말 비율: {(weekdays >= 5).sum() / len(weekdays):.1%}")

# 연속된 날짜 확인
dates_sorted = pd.Series(dates).sort_values()
gaps = dates_sorted.diff().dt.days
print(f"\n날짜 간격 통계:")
print(f"  평균 간격: {gaps.mean():.2f}일")
print(f"  최소 간격: {gaps.min():.0f}일")
print(f"  최대 간격: {gaps.max():.0f}일")

# 주말이 포함되어 있는지 확인
has_weekend = (weekdays >= 5).any()
if has_weekend:
    print(f"\n⚠️ 주말 포함: {has_weekend}")
    weekend_dates = dates[weekdays >= 5]
    print(f"  주말 날짜 예시: {weekend_dates[:5]}")
else:
    print(f"\n✅ 주말 미포함: 영업일만 포함됨")

print("=" * 80)
