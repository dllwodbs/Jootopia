"""
재현성 검증 스크립트

단계별로 데이터 재산출을 실행하여 재현성을 검증합니다.
"""
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import logging
from datetime import datetime

from src.data_collection.pipeline import run_data_collection_pipeline
from src.pipeline.track_a_pipeline import run_track_a_pipeline
from src.pipeline.track_b_pipeline import run_track_b_pipeline

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("reproducibility_test.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


def main():
    """재현성 검증 메인 함수"""
    logger.info("=" * 80)
    logger.info("재현성 검증 시작")
    logger.info(f"시작 시간: {datetime.now()}")
    logger.info("=" * 80)

    config_path = "configs/config.yaml"
    force_rebuild = True  # 재산출을 위해 캐시 무시

    try:
        # ============================================================
        # Step 1: L0~L4 데이터 수집 및 계산
        # ============================================================
        logger.info("\n" + "=" * 80)
        logger.info("Step 1: L0~L4 데이터 수집 및 계산")
        logger.info("=" * 80)

        data_result = run_data_collection_pipeline(
            config_path=config_path,
            stages=None,  # 전체 실행
            force_rebuild=force_rebuild,
        )

        logger.info(f"✅ L0~L4 완료")
        logger.info(f"  - 생성된 아티팩트: {list(data_result['artifacts'].keys())}")

        # ============================================================
        # Step 2: Track A (L8) 랭킹 생성
        # ============================================================
        logger.info("\n" + "=" * 80)
        logger.info("Step 2: Track A (L8) 랭킹 생성")
        logger.info("=" * 80)

        track_a_result = run_track_a_pipeline(
            config_path=config_path,
            force_rebuild=force_rebuild,
            run_ui_payload=False,
        )

        logger.info(f"✅ Track A 완료")
        logger.info(
            f"  - 단기 랭킹: {len(track_a_result['ranking_short_daily']):,}행"
        )
        logger.info(
            f"  - 장기 랭킹: {len(track_a_result['ranking_long_daily']):,}행"
        )

        # ============================================================
        # Step 3: Track B (L6R, L7) 백테스트
        # ============================================================
        logger.info("\n" + "=" * 80)
        logger.info("Step 3: Track B (L6R, L7) 백테스트")
        logger.info("=" * 80)

        # bt120_long 전략으로 테스트
        track_b_result = run_track_b_pipeline(
            config_path=config_path,
            strategy="bt120_long",
            force_rebuild=force_rebuild,
        )

        logger.info(f"✅ Track B 완료")
        logger.info(f"  - 전략: bt120_long")
        logger.info(f"  - 포지션: {len(track_b_result['bt_positions']):,}행")
        logger.info(f"  - 수익률: {len(track_b_result['bt_returns']):,}행")
        logger.info(f"  - 메트릭: {len(track_b_result['bt_metrics']):,}행")

        # ============================================================
        # 결과 요약
        # ============================================================
        logger.info("\n" + "=" * 80)
        logger.info("재현성 검증 완료")
        logger.info(f"종료 시간: {datetime.now()}")
        logger.info("=" * 80)

        logger.info("\n생성된 아티팩트:")
        for key, path in track_b_result.get("artifacts_path", {}).items():
            logger.info(f"  - {key}: {path}")

        return {
            "status": "success",
            "data_collection": data_result,
            "track_a": track_a_result,
            "track_b": track_b_result,
        }

    except Exception as e:
        logger.error(f"재현성 검증 실패: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
