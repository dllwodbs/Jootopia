"""특정 날짜의 랭킹 1~20위 조회"""
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

# 날짜 파라미터
target_date = "2023-06-12"

print("=" * 80)
print(f"{target_date} 랭킹 1~20위")
print("=" * 80)

# 단기 랭킹 로드
ranking_short = load_artifact(interim / "ranking_short_daily")
ranking_short['date'] = pd.to_datetime(ranking_short['date'])

# 장기 랭킹 로드
ranking_long = load_artifact(interim / "ranking_long_daily")
ranking_long['date'] = pd.to_datetime(ranking_long['date'])

# 날짜 필터링
target_dt = pd.to_datetime(target_date)
short_filtered = ranking_short[ranking_short['date'] == target_dt].copy()
long_filtered = ranking_long[ranking_long['date'] == target_dt].copy()

if len(short_filtered) == 0:
    print(f"\n⚠️ {target_date} 날짜의 랭킹 데이터가 없습니다.")
    print(f"   사용 가능한 날짜 범위: {ranking_short['date'].min().date()} ~ {ranking_short['date'].max().date()}")
    sys.exit(1)

# 단기 랭킹 1~20위
print(f"\n[단기 랭킹] (Top 20)")
print("-" * 80)
short_top20 = short_filtered.nsmallest(20, 'rank_total')[['ticker', 'rank_total', 'score_total']].copy()
short_top20 = short_top20.sort_values('rank_total')
short_top20.index = range(1, len(short_top20) + 1)
print(short_top20.to_string())

# 장기 랭킹 1~20위
print(f"\n[장기 랭킹] (Top 20)")
print("-" * 80)
long_top20 = long_filtered.nsmallest(20, 'rank_total')[['ticker', 'rank_total', 'score_total']].copy()
long_top20 = long_top20.sort_values('rank_total')
long_top20.index = range(1, len(long_top20) + 1)
print(long_top20.to_string())

print("\n" + "=" * 80)
