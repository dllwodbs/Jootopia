"""
Command Line Interface for Quantitative Trading Portfolio

This module provides a unified CLI for running the entire quantitative trading pipeline,
including data collection, feature engineering, modeling, ranking, portfolio construction,
and backtesting.

Usage:
    python -m src.cli run --config configs/config.yaml
    python -m src.cli data-download --config configs/config.yaml
    python -m src.cli track-a --config configs/config.yaml
    python -m src.cli track-b --strategy bt120_long --config configs/config.yaml
"""

import sys
from pathlib import Path
import argparse
import logging

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

logger = get_logger(__name__)


def run_full_pipeline(config_path: str = "configs/config.yaml"):
    """전체 파이프라인 실행 (원천 데이터 제외)"""

    # 로깅 설정
    log_file = str(Path("logs") / "full_pipeline.log")
    setup_logging(log_file=log_file)

    logger.info("=" * 80)
    logger.info("전체 파이프라인 실행 시작")
    logger.info("=" * 80)

    try:
        # Track A: 랭킹 엔진
        logger.info("Track A (랭킹 엔진) 실행 중...")
        run_track_a_pipeline(config_path)

        # Track B: 백테스트 (bt20_short로 빠르게)
        logger.info("Track B (백테스트) 실행 중...")
        run_track_b_pipeline(config_path, "bt20_short")

        logger.info("파이프라인 실행 완료!")

    except Exception as e:
        logger.error(f"파이프라인 실행 실패: {e}")
        raise


def run_data_download(config_path: str = "configs/config.yaml"):
    """데이터 다운로드 실행"""

    # 로깅 설정 (config 로드 전에)
    log_file = str(Path("logs") / "data_download.log")
    setup_logging(log_file=log_file)

    logger.info("데이터 다운로드 시작...")

    try:
        pipeline = DataCollectionPipeline(config_path)
        pipeline.run_all()

        logger.info("데이터 다운로드 완료!")

    except Exception as e:
        logger.error(f"데이터 다운로드 실패: {e}")
        raise


def run_track_a(config_path: str = "configs/config.yaml"):
    """Track A (랭킹 엔진) 실행"""

    # 로깅 설정 (config 로드 전에)
    log_file = str(Path("logs") / "track_a.log")
    setup_logging(log_file=log_file)

    logger.info("Track A 실행 시작...")

    try:
        run_track_a_pipeline(config_path)
        logger.info("Track A 실행 완료!")

    except Exception as e:
        logger.error(f"Track A 실행 실패: {e}")
        raise


def run_track_b(config_path: str = "configs/config.yaml", strategy: str = "bt120_long"):
    """Track B (백테스트) 실행"""

    # 로깅 설정 (config 로드 전에)
    log_file = str(Path("logs") / f"track_b_{strategy}.log")
    setup_logging(log_file=log_file)

    logger.info(f"Track B ({strategy}) 실행 시작...")

    try:
        run_track_b_pipeline(config_path, strategy)
        logger.info(f"Track B ({strategy}) 실행 완료!")

    except Exception as e:
        logger.error(f"Track B 실행 실패: {e}")
        raise


def main():
    """메인 CLI 함수"""

    parser = argparse.ArgumentParser(description="Quantitative Trading Portfolio CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run command
    run_parser = subparsers.add_parser("run", help="Run full pipeline")
    run_parser.add_argument("--config", default="configs/config.yaml", help="Config file path")

    # data-download command
    data_parser = subparsers.add_parser("data-download", help="Download data")
    data_parser.add_argument("--config", default="configs/config.yaml", help="Config file path")

    # track-a command
    track_a_parser = subparsers.add_parser("track-a", help="Run Track A (ranking)")
    track_a_parser.add_argument("--config", default="configs/config.yaml", help="Config file path")

    # track-b command
    track_b_parser = subparsers.add_parser("track-b", help="Run Track B (backtest)")
    track_b_parser.add_argument("--config", default="configs/config.yaml", help="Config file path")
    track_b_parser.add_argument("--strategy", default="bt120_long", help="Strategy name")

    args = parser.parse_args()

    if args.command == "run":
        run_full_pipeline(args.config)
    elif args.command == "data-download":
        run_data_download(args.config)
    elif args.command == "track-a":
        run_track_a(args.config)
    elif args.command == "track-b":
        run_track_b(args.config, args.strategy)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()