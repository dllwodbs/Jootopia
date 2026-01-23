"""
랭킹 산정 및 백테스트 실행 스크립트

외부 드라이브에서 데이터를 다운로드한 후 실행합니다.
사용자가 직접 랭킹과 백테스트 결과만 산출할 수 있습니다.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.track_a_pipeline import run_track_a_pipeline
from src.pipeline.track_b_pipeline import run_track_b_pipeline


def main():
    """랭킹 산정 및 백테스트 실행"""
    config_path = "configs/config.yaml"
    
    print("=" * 80)
    print("랭킹 산정 및 백테스트 실행")
    print("=" * 80)
    print("\n⚠️ 사전 확인:")
    print("   외부 드라이브에서 다음 데이터를 다운로드했는지 확인하세요:")
    print("   - data/cal_data/panel_merged_daily.parquet")
    print("   - data/cal_data/dataset_daily.parquet")
    print("   - data/cal_data/cv_folds_short.parquet")
    print("   - data/cal_data/cv_folds_long.parquet")
    print("   - data/raw_data/universe_k200_membership_monthly.parquet")
    print("\n데이터 검증: python scripts/check_data_files.py")
    print("=" * 80)
    
    # 1. Track A: 랭킹 산정
    print("\n[1/2] Track A: 랭킹 산정 실행 중...")
    try:
        result_a = run_track_a_pipeline(
            config_path=config_path,
            force_rebuild=False,  # 캐시 우선
        )
        print("✅ Track A 완료")
        print(f"   - ranking_short_daily: {len(result_a.get('ranking_short_daily', [])):,}행")
        print(f"   - ranking_long_daily: {len(result_a.get('ranking_long_daily', [])):,}행")
    except FileNotFoundError as e:
        print(f"❌ 오류: {e}")
        print("\n💡 해결 방법:")
        print("   1. 외부 드라이브에서 다음 파일을 다운로드하세요:")
        print("      - data/cal_data/panel_merged_daily.parquet")
        print("      - data/cal_data/dataset_daily.parquet (선택적)")
        print("   2. 데이터 검증: python scripts/check_data_files.py")
        return
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 2. Track B: 백테스트
    print("\n[2/2] Track B: 백테스트 실행 중...")
    strategies = ["bt120_long", "bt20_ens", "bt20_short", "bt120_ens"]
    
    success_count = 0
    for strategy in strategies:
        print(f"\n  → {strategy} 실행 중...")
        try:
            result_b = run_track_b_pipeline(
                config_path=config_path,
                strategy=strategy,
                force_rebuild=False,  # 캐시 우선
            )
            print(f"  ✅ {strategy} 완료")
            success_count += 1
        except FileNotFoundError as e:
            print(f"  ⚠️ {strategy} 실패: {e}")
            print("     Track A를 먼저 실행했는지 확인하세요.")
            continue
        except Exception as e:
            print(f"  ❌ {strategy} 오류: {e}")
            continue
    
    print("\n" + "=" * 80)
    if success_count > 0:
        print(f"✅ 실행 완료! ({success_count}/{len(strategies)} 전략 성공)")
    else:
        print("❌ 모든 전략 실행 실패")
    print("=" * 80)
    print("\n산출물 위치:")
    print("  - 랭킹: data/cal_data/ranking_*.parquet")
    print("  - 백테스트: data/cal_data/bt_metrics_*.parquet")
    print("  - 백테스트 상세: data/cal_data/bt_returns_*.parquet")
    print("\n성과 확인:")
    print("  python scripts/extract_latest_metrics.py")


if __name__ == "__main__":
    main()
