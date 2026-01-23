"""계산 데이터 재산출 실행"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.pipeline.track_b_pipeline import run_track_b_pipeline

# Track B 실행
strategies = ["bt120_long", "bt120_ens", "bt20_short", "bt20_ens"]

for strategy in strategies:
    print(f"\n[{strategy}] 실행 중...")
    try:
        result = run_track_b_pipeline(strategy=strategy, force_rebuild=True)
        print(f"  ✓ 완료: {len(result['bt_metrics'])}개 메트릭")
    except Exception as e:
        print(f"  ✗ 실패: {e}")
