"""
계산 데이터 재산출 스크립트

원천 데이터는 제외하고 계산 데이터만 재산출합니다:
- L3: 패널 병합 (ohlcv_daily 기반)
- L4: CV 분할 및 타겟 생성
- L8: 랭킹 엔진 (Track A)
- L6R: 랭킹 스코어 변환
- L7: 백테스트 (Track B)
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_collection.pipeline import DataCollectionPipeline
from src.pipeline.track_a_pipeline import run_track_a_pipeline
from src.pipeline.track_b_pipeline import run_track_b_pipeline
from src.utils.config import load_config, get_path
from src.utils.io import artifact_exists, load_artifact
from src.utils.logging import get_logger, setup_logging
import shutil

logger = get_logger(__name__)


def main():
    """계산 데이터 재산출 메인 함수"""
    
    config_path = "configs/config.yaml"
    cfg = load_config(config_path)
    
    # 로깅 설정
    log_file = str(Path(get_path(cfg, "logs")) / "recompute_calculated_data.log")
    setup_logging(log_file=log_file)
    
    logger.info("=" * 80)
    logger.info("계산 데이터 재산출 시작 (원천 데이터 제외)")
    logger.info("=" * 80)
    
    # 계산 데이터는 data 폴더에 저장
    interim_dir = Path(get_path(cfg, "data_interim"))
    interim_dir.mkdir(parents=True, exist_ok=True)
    
    # data_backup의 외부 데이터를 data/external로 복사
    backup_dir = Path("data_backup")
    external_dir = Path(get_path(cfg, "data_ext"))
    external_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"[경로 설정]")
    logger.info(f"  계산 데이터 저장: {interim_dir}")
    logger.info(f"  외부 데이터 복사: {external_dir}")
    
    # data_backup의 외부 데이터 복사
    logger.info("[data_backup 외부 데이터 복사]")
    external_files = [
        "esg_daily.parquet",
        "news_sentiment_daily.parquet",
        "sector_map.csv",
    ]
    
    for file_name in external_files:
        src = backup_dir / file_name
        dst = external_dir / file_name
        if src.exists():
            if not dst.exists() or True:  # 항상 복사
                shutil.copy2(src, dst)
                logger.info(f"  ✓ {file_name} 복사 완료")
        else:
            logger.warning(f"  ⚠ {file_name}: data_backup에 없음")
    
    # 필수 원천 데이터 확인 (data/interim에서 또는 다운로드 필요)
    logger.info("[필수 원천 데이터 확인]")
    required_sources = {
        "universe_k200_membership_monthly": "L0 유니버스",
        "ohlcv_daily": "L1 OHLCV",
    }
    
    missing_sources = []
    
    for artifact_name, description in required_sources.items():
        artifact_path = interim_dir / artifact_name
        if artifact_exists(artifact_path):
            df = load_artifact(artifact_path)
            logger.info(f"  ✓ {description}: {len(df):,}행 (기존 데이터 사용)")
        else:
            logger.warning(f"  ⚠ {description}: 없음 (다운로드 필요)")
            missing_sources.append(description)
    
    # 원천 데이터가 없으면 다운로드 시도 (선택적)
    if missing_sources:
        logger.warning(f"원천 데이터가 없습니다: {', '.join(missing_sources)}")
        logger.info("원천 데이터 다운로드를 시도합니다 (인터넷 연결 필요)...")
        
        try:
            pipeline = DataCollectionPipeline(
                config_path=config_path,
                force_rebuild=False,  # 기존 데이터가 있으면 사용
            )
            
            if "universe_k200_membership_monthly" in missing_sources:
                logger.info("[L0] 유니버스 다운로드 중... (시간이 걸릴 수 있습니다)")
                try:
                    pipeline.run_l0()
                    logger.info("  ✓ L0 완료")
                except Exception as e:
                    logger.warning(f"  ⚠ L0 다운로드 실패: {e}")
                    logger.warning("  원천 데이터 없이는 계산 데이터 재산출 불가능")
                    return False
            
            if "ohlcv_daily" in missing_sources:
                logger.info("[L1] OHLCV 다운로드 중... (시간이 걸릴 수 있습니다)")
                try:
                    pipeline.run_l1()
                    logger.info("  ✓ L1 완료")
                except Exception as e:
                    logger.warning(f"  ⚠ L1 다운로드 실패: {e}")
                    logger.warning("  원천 데이터 없이는 계산 데이터 재산출 불가능")
                    return False
            
            # 다시 확인
            all_ok = True
            for artifact_name, description in required_sources.items():
                artifact_path = interim_dir / artifact_name
                if artifact_exists(artifact_path):
                    df = load_artifact(artifact_path)
                    logger.info(f"  ✓ {description}: {len(df):,}행")
                else:
                    logger.error(f"  ✗ {description}: 여전히 없음")
                    all_ok = False
            
            if not all_ok:
                logger.error("필수 원천 데이터가 없어 계산 데이터 재산출을 중단합니다")
                logger.error("원천 데이터를 먼저 준비하거나 다운로드하세요")
                return False
                    
        except Exception as e:
            logger.error(f"원천 데이터 다운로드 실패: {e}", exc_info=True)
            logger.error("인터넷 연결 및 pykrx 설치를 확인하세요")
            logger.error("또는 기존 원천 데이터를 data/interim 폴더에 준비하세요")
            return False
    
    # 선택적 원천 데이터 확인
    logger.info("[선택적 원천 데이터 확인]")
    optional_sources = {
        "fundamentals_annual": "L2 DART 재무",
        "pykrx_fundamentals_daily": "L1B pykrx 재무",
    }
    
    for artifact_name, description in optional_sources.items():
        artifact_path = interim_dir / artifact_name
        if artifact_exists(artifact_path):
            df = load_artifact(artifact_path)
            logger.info(f"  ✓ {description}: {len(df):,}행")
        else:
            logger.warning(f"  ⚠ {description}: 없음 (스킵 가능)")
    
    # L3: 패널 병합 재산출
    logger.info("=" * 80)
    logger.info("[L3] 패널 병합 재산출")
    logger.info("=" * 80)
    
    try:
        # 원천 데이터 직접 로드 (재다운로드 방지)
        logger.info("  → 원천 데이터 직접 로드")
        ohlcv_daily = load_artifact(interim_dir / "ohlcv_daily")
        universe_membership_monthly = load_artifact(interim_dir / "universe_k200_membership_monthly")
        
        logger.info(f"    - ohlcv_daily: {len(ohlcv_daily):,}행")
        logger.info(f"    - universe: {len(universe_membership_monthly):,}행")
        
        # 재무 데이터는 선택적
        fundamentals_annual = None
        fundamentals_path = interim_dir / "fundamentals_annual"
        if artifact_exists(fundamentals_path):
            fundamentals_annual = load_artifact(fundamentals_path)
            logger.info(f"    - fundamentals_annual: {len(fundamentals_annual):,}행")
        else:
            logger.info("    - fundamentals_annual: 없음 (스킵)")
        
        # L3 직접 호출
        from src.data_collection.collectors import collect_panel
        
        logger.info("  → L3 패널 병합 재산출 시작...")
        panel_result = collect_panel(
            ohlcv_daily=ohlcv_daily,
            fundamentals_annual=fundamentals_annual,
            universe_membership_monthly=universe_membership_monthly,
            fundamental_lag_days=cfg.get("l2", {}).get("fallback_lag_days", 90),
            filter_k200_members_only=cfg.get("l3", {}).get("filter_k200_members_only", True),
            config_path=config_path,
            save_to_cache=True,
            force_rebuild=True,  # L3 강제 재산출
        )
        logger.info(f"  ✓ 패널 병합 완료: {len(panel_result):,}행")
        
    except Exception as e:
        logger.error(f"L3 실행 실패: {e}", exc_info=True)
        return False
    
    # L4: CV 분할 재산출
    logger.info("=" * 80)
    logger.info("[L4] CV 분할 및 타겟 생성 재산출")
    logger.info("=" * 80)
    
    try:
        # L4 직접 호출
        from src.data_collection.collectors import collect_dataset
        
        logger.info("  → L4 CV 분할 재산출 시작...")
        dataset_result = collect_dataset(
            panel_merged_daily=panel_result,
            config_path=config_path,
            save_to_cache=True,
            force_rebuild=True,  # L4 강제 재산출
        )
        logger.info(f"  ✓ 데이터셋 생성 완료: {len(dataset_result['dataset_daily']):,}행")
        logger.info(f"  ✓ CV folds (단기): {len(dataset_result['cv_folds_short']):,}개")
        logger.info(f"  ✓ CV folds (장기): {len(dataset_result['cv_folds_long']):,}개")
        
    except Exception as e:
        logger.error(f"L4 실행 실패: {e}", exc_info=True)
        return False
    
    # Track A: L8 랭킹 엔진 재산출
    logger.info("=" * 80)
    logger.info("[Track A] L8 랭킹 엔진 재산출")
    logger.info("=" * 80)
    
    try:
        track_a_result = run_track_a_pipeline(
            config_path=config_path,
            force_rebuild=True,  # 강제 재산출
            run_ui_payload=False,
        )
        
        logger.info(f"  ✓ 단기 랭킹: {len(track_a_result['ranking_short_daily']):,}행")
        logger.info(f"  ✓ 장기 랭킹: {len(track_a_result['ranking_long_daily']):,}행")
        
    except Exception as e:
        logger.error(f"Track A 실행 실패: {e}", exc_info=True)
        return False
    
    # Track B: L6R, L7 백테스트 재산출
    logger.info("=" * 80)
    logger.info("[Track B] L6R, L7 백테스트 재산출")
    logger.info("=" * 80)
    
    strategies = ["bt120_long", "bt120_ens", "bt20_short", "bt20_ens"]
    
    for strategy in strategies:
        logger.info(f"[전략: {strategy}]")
        try:
            track_b_result = run_track_b_pipeline(
                config_path=config_path,
                strategy=strategy,
                force_rebuild=True,  # 강제 재산출
            )
            
            logger.info(f"  ✓ {strategy} 완료:")
            logger.info(f"    - 포지션: {len(track_b_result['bt_positions']):,}행")
            logger.info(f"    - 수익률: {len(track_b_result['bt_returns']):,}행")
            logger.info(f"    - 자산 곡선: {len(track_b_result['bt_equity_curve']):,}행")
            logger.info(f"    - 메트릭: {len(track_b_result['bt_metrics']):,}행")
            
        except Exception as e:
            logger.warning(f"  ⚠ {strategy} 실행 실패: {e}")
            continue
    
    logger.info("=" * 80)
    logger.info("✅ 계산 데이터 재산출 완료")
    logger.info("=" * 80)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
