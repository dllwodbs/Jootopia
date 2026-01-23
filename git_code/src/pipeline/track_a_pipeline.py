"""
Track A 전체 파이프라인 실행 모듈

Track A: 랭킹 엔진 (Ranking Engine)
- 목적: 피처들로 KOSPI200의 랭킹을 산정하여 이용자에게 제공
- L8: 랭킹 엔진 실행
- L11: UI Payload Builder
"""

from pathlib import Path

import pandas as pd

from src.utils.config import get_path, load_config
from src.utils.io import artifact_exists, load_artifact, save_artifact
from src.utils.logging import ExecutionSummary, get_logger, setup_logging

logger = get_logger(__name__)


def run_track_a_pipeline(
    config_path: str = "configs/config.yaml",
    force_rebuild: bool = False,
    run_ui_payload: bool = False,
) -> dict:
    """
    Track A 전체 파이프라인을 실행합니다.

    Track A는 랭킹 엔진으로, 피처 기반 랭킹을 생성하여 이용자에게 제공합니다.

    Args:
        config_path: 설정 파일 경로
        force_rebuild: True면 캐시 무시하고 재계산

    Returns:
        dict: 파이프라인 실행 결과
        {
            "ranking_daily": DataFrame,
            "ranking_short_daily": DataFrame,
            "ranking_long_daily": DataFrame,
            "ui_payload": dict,
            "artifacts_path": dict,
        }
    """
    # 로깅 설정 초기화
    log_file = str(Path(get_path(load_config(config_path), "logs")) / "track_a.log")
    setup_logging(log_file=log_file)

    # 실행 요약 초기화
    summary = ExecutionSummary("track_a", config_path)
    summary.add_parameter("force_rebuild", force_rebuild)
    summary.add_parameter("run_ui_payload", run_ui_payload)

    logger.info("=" * 80)
    logger.info("Track A: 랭킹 엔진 파이프라인 실행 시작")
    logger.info(f"설정 파일: {config_path}")
    logger.info(f"강제 재빌드: {force_rebuild}")
    logger.info(f"UI Payload 실행: {run_ui_payload}")
    logger.info("=" * 80)

    summary.start_step("pipeline_init")

    # 설정 로드
    cfg = load_config(config_path)
    cal_data_dir = Path(get_path(cfg, "data_cal"))
    cal_data_dir.mkdir(parents=True, exist_ok=True)

    summary.end_step("pipeline_init")

    artifacts = {}
    artifacts_path = {}

    # 공통 데이터 확인
    logger.info("[공통 데이터 확인]")
    summary.start_step("data_loading")

    # L3: 패널 데이터
    panel_path = cal_data_dir / "panel_merged_daily"
    # [개선안 41번] force_rebuild 의미 정리:
    # - 입력(공통 캐시)은 항상 로드 가능해야 한다. (L0~L4는 별도 파이프라인이 담당)
    # - force_rebuild는 Track A의 "출력(랭킹)"을 재생성할지 여부로만 사용한다.
    if artifact_exists(panel_path):
        artifacts["panel_merged_daily"] = load_artifact(panel_path)
        artifacts_path["panel"] = str(panel_path)
        panel_size = len(artifacts["panel_merged_daily"])
        logger.info(f"  ✓ 패널 데이터 로드: {panel_size:,}행")
        summary.add_metadata("input_panel_rows", panel_size)
        summary.add_output("panel_data", str(panel_path))
    else:
        logger.warning("  ✗ 패널 데이터가 없습니다. L0~L3까지 실행이 필요합니다.")
        logger.warning(
            "  python -m src.tools.run_two_track_and_export --force-shared  (또는 DataCollectionPipeline) 를 먼저 실행하세요."
        )
        raise FileNotFoundError("panel_merged_daily not found")

    # L4: CV 분할 (랭킹 엔진은 dataset_daily 사용 가능)
    dataset_path = cal_data_dir / "dataset_daily"
    if artifact_exists(dataset_path):
        artifacts["dataset_daily"] = load_artifact(dataset_path)
        artifacts_path["dataset"] = str(dataset_path)
        dataset_size = len(artifacts["dataset_daily"])
        logger.info(f"  ✓ 데이터셋 로드: {dataset_size:,}행")
        summary.add_metadata("input_dataset_rows", dataset_size)
    else:
        logger.info("  → dataset_daily가 없습니다. panel_merged_daily를 사용합니다.")
        artifacts["dataset_daily"] = artifacts["panel_merged_daily"]
        summary.add_metadata("input_dataset_rows", panel_size)

    summary.end_step("data_loading")

    # L8: 랭킹 엔진 실행
    logger.info("[L8] 랭킹 엔진 실행")
    summary.start_step("ranking_engine")

    from src.tracks.track_a.stages.ranking.l8_dual_horizon import (
        run_L8_long_rank_engine,
        run_L8_short_rank_engine,
    )

    ranking_short_path = cal_data_dir / "ranking_short_daily"
    ranking_long_path = cal_data_dir / "ranking_long_daily"
    ranking_integrated_path = cal_data_dir / "ranking_integrated_daily"

    if (
        artifact_exists(ranking_short_path)
        and artifact_exists(ranking_long_path)
        and not force_rebuild
    ):
        artifacts["ranking_short_daily"] = load_artifact(ranking_short_path)
        artifacts["ranking_long_daily"] = load_artifact(ranking_long_path)
        artifacts_path["ranking_short"] = str(ranking_short_path)
        artifacts_path["ranking_long"] = str(ranking_long_path)
        short_size = len(artifacts["ranking_short_daily"])
        long_size = len(artifacts["ranking_long_daily"])
        logger.info(f"  ✓ 캐시에서 로드: 단기 {short_size:,}행, 장기 {long_size:,}행")
        summary.add_metadata("ranking_short_rows", short_size)
        summary.add_metadata("ranking_long_rows", long_size)
        summary.add_output("ranking_short", str(ranking_short_path))
        summary.add_output("ranking_long", str(ranking_long_path))
    else:
        logger.info("  → 랭킹 엔진 재실행")
        summary.start_step("ranking_short_generation")
        # L8_short 실행
        logger.info("  → 단기 랭킹 생성 중...")
        outputs_short, warns_short = run_L8_short_rank_engine(
            cfg=cfg,
            artifacts=artifacts,
            force=force_rebuild,
        )
        artifacts["ranking_short_daily"] = outputs_short["ranking_short_daily"]
        summary.end_step("ranking_short_generation")

        summary.start_step("ranking_long_generation")
        # L8_long 실행
        logger.info("  → 장기 랭킹 생성 중...")
        outputs_long, warns_long = run_L8_long_rank_engine(
            cfg=cfg,
            artifacts=artifacts,
            force=force_rebuild,
        )
        artifacts["ranking_long_daily"] = outputs_long["ranking_long_daily"]
        summary.end_step("ranking_long_generation")

        save_artifact(artifacts["ranking_short_daily"], ranking_short_path, force=True)
        save_artifact(artifacts["ranking_long_daily"], ranking_long_path, force=True)
        artifacts_path["ranking_short"] = str(ranking_short_path)
        artifacts_path["ranking_long"] = str(ranking_long_path)
        short_size = len(artifacts["ranking_short_daily"])
        long_size = len(artifacts["ranking_long_daily"])
        logger.info(f"  ✓ 생성 완료: 단기 {short_size:,}행, 장기 {long_size:,}행")
        summary.add_metadata("ranking_short_rows", short_size)
        summary.add_metadata("ranking_long_rows", long_size)
        summary.add_output("ranking_short", str(ranking_short_path))
        summary.add_output("ranking_long", str(ranking_long_path))

    # 통합 랭킹 생성 (장기 0.8, 단기 0.2)
    logger.info("[통합 랭킹 생성]")
    summary.start_step("ranking_integrated_generation")
    
    if (
        artifact_exists(ranking_integrated_path)
        and not force_rebuild
    ):
        artifacts["ranking_integrated_daily"] = load_artifact(ranking_integrated_path)
        artifacts_path["ranking_integrated"] = str(ranking_integrated_path)
        integrated_size = len(artifacts["ranking_integrated_daily"])
        logger.info(f"  ✓ 캐시에서 로드: 통합 {integrated_size:,}행")
        summary.add_metadata("ranking_integrated_rows", integrated_size)
        summary.add_output("ranking_integrated", str(ranking_integrated_path))
    else:
        logger.info("  → 통합 랭킹 생성 (장기 0.8, 단기 0.2)")
        
        # 단기/장기 랭킹 병합
        ranking_short = artifacts["ranking_short_daily"].copy()
        ranking_long = artifacts["ranking_long_daily"].copy()
        
        # 날짜/티커 정규화
        ranking_short["date"] = pd.to_datetime(ranking_short["date"])
        ranking_long["date"] = pd.to_datetime(ranking_long["date"])
        ranking_short["ticker"] = ranking_short["ticker"].astype(str).str.zfill(6)
        ranking_long["ticker"] = ranking_long["ticker"].astype(str).str.zfill(6)
        
        # 병합
        merged = ranking_short[["date", "ticker", "score_total", "rank_total"]].merge(
            ranking_long[["date", "ticker", "score_total", "rank_total"]],
            on=["date", "ticker"],
            how="inner",
            suffixes=("_short", "_long"),
        )
        
        # 통합 스코어 계산 (단기 0.2, 장기 0.8)
        alpha_short = 0.2
        alpha_long = 0.8
        merged["score_total"] = (
            alpha_short * merged["score_total_short"] + 
            alpha_long * merged["score_total_long"]
        )
        
        # 통합 랭킹 계산 (날짜별로 그룹화하여 랭킹)
        merged["rank_total"] = merged.groupby("date")["score_total"].rank(
            method="min", ascending=False
        )
        
        # 원본 컬럼들 유지 (단기 랭킹 구조 기준)
        ranking_integrated = ranking_short[["date", "ticker"]].merge(
            merged[["date", "ticker", "score_total", "rank_total"]],
            on=["date", "ticker"],
            how="inner"
        )
        
        # 나머지 컬럼들 추가 (단기 랭킹에서)
        other_cols = [col for col in ranking_short.columns if col not in ["date", "ticker", "score_total", "rank_total"]]
        if other_cols:
            ranking_integrated = ranking_integrated.merge(
                ranking_short[["date", "ticker"] + other_cols],
                on=["date", "ticker"],
                how="left"
            )
        
        artifacts["ranking_integrated_daily"] = ranking_integrated
        save_artifact(ranking_integrated, ranking_integrated_path, force=True)
        artifacts_path["ranking_integrated"] = str(ranking_integrated_path)
        integrated_size = len(ranking_integrated)
        logger.info(f"  ✓ 통합 랭킹 생성 완료: {integrated_size:,}행 (단기 {alpha_short*100}%, 장기 {alpha_long*100}%)")
        summary.add_metadata("ranking_integrated_rows", integrated_size)
        summary.add_output("ranking_integrated", str(ranking_integrated_path))
    
    summary.end_step("ranking_integrated_generation")

    summary.end_step("ranking_engine")

    # L11: UI Payload Builder (선택적)
    # [개선안 41번] L11은 외부 API(지수/벤치마크 등) 의존이 있어 기본 OFF로 둔다.
    if run_ui_payload:
        logger.info("[L11] UI Payload Builder 실행 (선택적)")
        summary.start_step("ui_payload_builder")
        try:
            from src.tracks.track_a.stages.ranking.ui_payload_builder import (
                run_L11_ui_payload,
            )

            ohlcv_path = interim_dir / "ohlcv_daily"
            if artifact_exists(ohlcv_path):
                ohlcv_daily = load_artifact(ohlcv_path)
                # ranking_daily는 단기/장기 중 하나를 선택하거나 통합해야 함
                # 일단 단기 랭킹을 사용 (필요시 통합 랭킹 생성 가능)
                ranking_daily = artifacts["ranking_short_daily"].copy()

                outputs, warns = run_L11_ui_payload(
                    cfg=cfg,
                    artifacts={
                        "ranking_daily": ranking_daily,
                        "ohlcv_daily": ohlcv_daily,
                    },
                    force=force_rebuild,
                )
                artifacts["ui_payload"] = outputs
                logger.info("  ✓ UI Payload 생성 완료")
                summary.add_metadata("ui_payload_generated", True)
                summary.add_output("ui_payload", "generated")
            else:
                logger.warning("  ⚠ ohlcv_daily가 없어 UI Payload를 건너뜁니다.")
                artifacts["ui_payload"] = None
                summary.add_metadata("ui_payload_skipped", "ohlcv_missing")
        except Exception as e:
            logger.warning(f"  ⚠ UI Payload Builder 실행 실패: {e}")
            artifacts["ui_payload"] = None
            summary.add_metadata("ui_payload_error", str(e))
        summary.end_step("ui_payload_builder")
    else:
        artifacts["ui_payload"] = None
        summary.add_metadata("ui_payload_skipped", "disabled")

    # 실행 요약 저장
    summary_path = summary.save_summary(cal_data_dir)
    logger.info(f"실행 요약 저장: {summary_path}")

    logger.info("=" * 80)
    logger.info("✅ Track A: 랭킹 엔진 파이프라인 실행 완료")
    logger.info("=" * 80)

    return {
        "ranking_short_daily": artifacts["ranking_short_daily"],
        "ranking_long_daily": artifacts["ranking_long_daily"],
        "ranking_integrated_daily": artifacts.get("ranking_integrated_daily"),
        "ui_payload": artifacts.get("ui_payload"),
        "artifacts_path": artifacts_path,
        "run_summary": summary_path,
    }


if __name__ == "__main__":
    # 로깅은 함수 내부에서 설정됨
    try:
        result = run_track_a_pipeline()
        print(
            f"\n✅ 완료: 단기 랭킹 {len(result['ranking_short_daily']):,}행, 장기 랭킹 {len(result['ranking_long_daily']):,}행"
        )
        print(f"실행 요약: {result.get('run_summary', 'N/A')}")
    except Exception as e:
        logger.error(f"Track A 파이프라인 실행 실패: {e}", exc_info=True)
        raise
