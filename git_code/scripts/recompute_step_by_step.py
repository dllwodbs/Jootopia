"""
계산 데이터 단계별 재산출 스크립트

각 단계를 개별적으로 실행하여 오류를 확인할 수 있습니다.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.config import load_config, get_path
from src.utils.io import artifact_exists, load_artifact, save_artifact
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def step_l3():
    """L3: 패널 병합 재산출"""
    config_path = "configs/config.yaml"
    cfg = load_config(config_path)
    
    log_file = str(Path(get_path(cfg, "logs")) / "recompute_l3.log")
    setup_logging(log_file=log_file)
    
    interim_dir = Path(get_path(cfg, "data_interim"))
    
    logger.info("=" * 80)
    logger.info("[L3] 패널 병합 재산출")
    logger.info("=" * 80)
    
    # 원천 데이터 로드
    ohlcv_daily = load_artifact(interim_dir / "ohlcv_daily")
    universe_membership_monthly = load_artifact(interim_dir / "universe_k200_membership_monthly")
    
    logger.info(f"  ohlcv_daily: {len(ohlcv_daily):,}행")
    logger.info(f"  universe: {len(universe_membership_monthly):,}행")
    
    fundamentals_annual = None
    if artifact_exists(interim_dir / "fundamentals_annual"):
        fundamentals_annual = load_artifact(interim_dir / "fundamentals_annual")
        logger.info(f"  fundamentals_annual: {len(fundamentals_annual):,}행")
    
    # L3 실행
    from src.data_collection.collectors import collect_panel
    
    panel_result = collect_panel(
        ohlcv_daily=ohlcv_daily,
        fundamentals_annual=fundamentals_annual,
        universe_membership_monthly=universe_membership_monthly,
        fundamental_lag_days=90,
        filter_k200_members_only=True,
        config_path=config_path,
        save_to_cache=True,
        force_rebuild=True,
    )
    
    logger.info(f"  ✓ 완료: {len(panel_result):,}행")
    return panel_result


def step_l4(panel_result):
    """L4: CV 분할 재산출"""
    config_path = "configs/config.yaml"
    cfg = load_config(config_path)
    
    log_file = str(Path(get_path(cfg, "logs")) / "recompute_l4.log")
    setup_logging(log_file=log_file)
    
    logger.info("=" * 80)
    logger.info("[L4] CV 분할 재산출")
    logger.info("=" * 80)
    
    from src.data_collection.collectors import collect_dataset
    
    dataset_result = collect_dataset(
        panel_merged_daily=panel_result,
        config_path=config_path,
        save_to_cache=True,
        force_rebuild=True,
    )
    
    logger.info(f"  ✓ 완료: dataset {len(dataset_result['dataset_daily']):,}행")
    return dataset_result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["l3", "l4", "l8", "l6r", "l7"], help="실행할 단계")
    args = parser.parse_args()
    
    if args.step == "l3":
        step_l3()
    elif args.step == "l4":
        panel = load_artifact(Path(get_path(load_config("configs/config.yaml"), "data_interim")) / "panel_merged_daily")
        step_l4(panel)
    else:
        print("사용법: python scripts/recompute_step_by_step.py --step l3|l4|l8|l6r|l7")
