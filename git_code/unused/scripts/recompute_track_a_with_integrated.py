"""
Track A 성과 지표 재산출 스크립트

재구성된 data 폴더(raw_data, cal_data)를 사용하여:
1. 단기 랭킹 재산출
2. 장기 랭킹 재산출
3. 통합 랭킹 생성 (단기 0.2, 장기 0.8)
4. 성과 지표 계산
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.config import load_config, get_path
from src.utils.io import artifact_exists, load_artifact, save_artifact
from src.utils.logging import get_logger, setup_logging
import pandas as pd
import numpy as np

logger = get_logger(__name__)


def load_data_from_reorganized_structure(config_path: str):
    """재구성된 data 폴더에서 데이터 로드"""
    cfg = load_config(config_path)
    base_data_dir = Path(get_path(cfg, "data_interim")).parent
    
    # 재구성된 폴더 경로
    raw_data_dir = base_data_dir / "raw_data"
    cal_data_dir = base_data_dir / "cal_data"
    
    logger.info("=" * 80)
    logger.info("재구성된 data 폴더에서 데이터 로드")
    logger.info("=" * 80)
    logger.info(f"  원천 데이터: {raw_data_dir}")
    logger.info(f"  계산 데이터: {cal_data_dir}")
    
    artifacts = {}
    
    # 원천 데이터 로드
    logger.info("\n[원천 데이터 로드]")
    
    # universe
    universe_path = raw_data_dir / "universe_k200_membership_monthly"
    if artifact_exists(universe_path):
        artifacts["universe_k200_membership_monthly"] = load_artifact(universe_path)
        logger.info(f"  ✓ 유니버스: {len(artifacts['universe_k200_membership_monthly']):,}행")
    else:
        raise FileNotFoundError(f"유니버스 데이터 없음: {universe_path}")
    
    # ohlcv
    ohlcv_path = raw_data_dir / "ohlcv_daily"
    if artifact_exists(ohlcv_path):
        artifacts["ohlcv_daily"] = load_artifact(ohlcv_path)
        logger.info(f"  ✓ OHLCV: {len(artifacts['ohlcv_daily']):,}행")
    else:
        raise FileNotFoundError(f"OHLCV 데이터 없음: {ohlcv_path}")
    
    # 계산 데이터 로드
    logger.info("\n[계산 데이터 로드]")
    
    # panel_merged_daily
    panel_path = cal_data_dir / "panel_merged_daily"
    if artifact_exists(panel_path):
        artifacts["panel_merged_daily"] = load_artifact(panel_path)
        logger.info(f"  ✓ 패널: {len(artifacts['panel_merged_daily']):,}행")
    else:
        raise FileNotFoundError(f"패널 데이터 없음: {panel_path}")
    
    # dataset_daily
    dataset_path = cal_data_dir / "dataset_daily"
    if artifact_exists(dataset_path):
        artifacts["dataset_daily"] = load_artifact(dataset_path)
        logger.info(f"  ✓ 데이터셋: {len(artifacts['dataset_daily']):,}행")
    else:
        logger.warning("  ⚠ 데이터셋 없음, 패널 사용")
        artifacts["dataset_daily"] = artifacts["panel_merged_daily"]
    
    return artifacts, raw_data_dir, cal_data_dir


def create_integrated_ranking(
    ranking_short: pd.DataFrame,
    ranking_long: pd.DataFrame,
    alpha_short: float = 0.2,
    alpha_long: float = 0.8,
) -> pd.DataFrame:
    """
    통합 랭킹 생성 (단기 0.2, 장기 0.8)
    
    Args:
        ranking_short: 단기 랭킹
        ranking_long: 장기 랭킹
        alpha_short: 단기 가중치 (기본 0.2)
        alpha_long: 장기 가중치 (기본 0.8)
    
    Returns:
        통합 랭킹 DataFrame
    """
    logger.info(f"\n[통합 랭킹 생성]")
    logger.info(f"  단기 가중치: {alpha_short}")
    logger.info(f"  장기 가중치: {alpha_long}")
    
    # 날짜/티커 정규화
    ranking_short = ranking_short.copy()
    ranking_long = ranking_long.copy()
    
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
    
    # 통합 스코어 계산
    # score_total을 정규화하여 결합
    # 단기와 장기의 스코어 범위가 다를 수 있으므로 정규화 필요
    
    # 방법 1: 스코어 직접 결합 (정규화된 스코어라고 가정)
    merged["score_total_integrated"] = (
        alpha_short * merged["score_total_short"] + 
        alpha_long * merged["score_total_long"]
    )
    
    # 통합 랭킹 계산 (날짜별로 그룹화하여 랭킹)
    merged["rank_total_integrated"] = merged.groupby("date")["score_total_integrated"].rank(
        method="min", ascending=False
    )
    
    # 결과 정리
    result = merged[[
        "date", "ticker", 
        "score_total_integrated", "rank_total_integrated",
        "score_total_short", "rank_total_short",
        "score_total_long", "rank_total_long"
    ]].copy()
    
    result = result.rename(columns={
        "score_total_integrated": "score_total",
        "rank_total_integrated": "rank_total",
    })
    
    logger.info(f"  ✓ 통합 랭킹 생성 완료: {len(result):,}행")
    logger.info(f"  ✓ 날짜 범위: {result['date'].min().date()} ~ {result['date'].max().date()}")
    logger.info(f"  ✓ 날짜 수: {result['date'].nunique():,}개")
    
    return result


def calculate_performance_metrics(
    ranking: pd.DataFrame,
    dataset_daily: pd.DataFrame,
    horizon_days: int = 20,
    name: str = "랭킹",
) -> dict:
    """
    랭킹 성과 지표 계산
    
    Args:
        ranking: 랭킹 데이터
        dataset_daily: 데이터셋 (타겟 변수 포함)
        horizon_days: 수익률 계산 기간
        name: 랭킹 이름
    
    Returns:
        성과 지표 딕셔너리
    """
    logger.info(f"\n[{name} 성과 지표 계산]")
    
    # 날짜/티커 정규화
    ranking = ranking.copy()
    dataset_daily = dataset_daily.copy()
    
    ranking["date"] = pd.to_datetime(ranking["date"])
    dataset_daily["date"] = pd.to_datetime(dataset_daily["date"])
    ranking["ticker"] = ranking["ticker"].astype(str).str.zfill(6)
    dataset_daily["ticker"] = dataset_daily["ticker"].astype(str).str.zfill(6)
    
    # 타겟 변수 컬럼 찾기
    ret_col = f"ret_fwd_{horizon_days}d"
    if ret_col not in dataset_daily.columns:
        logger.warning(f"  ⚠ {ret_col} 컬럼 없음, 성과 지표 계산 스킵")
        return {}
    
    # 랭킹과 수익률 병합
    merged = ranking[["date", "ticker", "rank_total"]].merge(
        dataset_daily[["date", "ticker", ret_col]],
        on=["date", "ticker"],
        how="inner",
    )
    
    # 수익률이 있는 데이터만
    merged = merged[merged[ret_col].notna()].copy()
    
    if len(merged) == 0:
        logger.warning(f"  ⚠ 병합된 데이터 없음")
        return {}
    
    # IC 계산 (Information Coefficient)
    # 상위 랭킹과 수익률의 상관관계
    ic_by_date = merged.groupby("date").apply(
        lambda g: g["rank_total"].corr(g[ret_col])
    )
    ic_mean = ic_by_date.mean()
    ic_std = ic_by_date.std()
    icir = ic_mean / ic_std if ic_std > 0 else 0
    
    # Rank IC 계산
    rank_ic_by_date = merged.groupby("date").apply(
        lambda g: g["rank_total"].corr(g[ret_col], method="spearman")
    )
    rank_ic_mean = rank_ic_by_date.mean()
    rank_ic_std = rank_ic_by_date.std()
    rank_icir = rank_ic_mean / rank_ic_std if rank_ic_std > 0 else 0
    
    # Top 20 수익률
    top20_returns = []
    for date, group in merged.groupby("date"):
        top20 = group.nsmallest(20, "rank_total")
        if len(top20) > 0:
            avg_return = top20[ret_col].mean()
            top20_returns.append(avg_return)
    
    top20_mean_return = np.mean(top20_returns) if top20_returns else 0
    
    metrics = {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "icir": icir,
        "rank_ic_mean": rank_ic_mean,
        "rank_ic_std": rank_ic_std,
        "rank_icir": rank_icir,
        "top20_mean_return": top20_mean_return,
        "n_dates": len(ic_by_date),
        "n_observations": len(merged),
    }
    
    logger.info(f"  ✓ IC: {ic_mean:.4f} (IR: {icir:.4f})")
    logger.info(f"  ✓ Rank IC: {rank_ic_mean:.4f} (IR: {rank_icir:.4f})")
    logger.info(f"  ✓ Top 20 평균 수익률: {top20_mean_return*100:.2f}%")
    
    return metrics


def main():
    """Track A 성과 지표 재산출 메인 함수"""
    config_path = "configs/config.yaml"
    cfg = load_config(config_path)
    
    # 로깅 설정
    log_file = str(Path(get_path(cfg, "logs")) / "track_a_recompute.log")
    setup_logging(log_file=log_file)
    
    logger.info("=" * 80)
    logger.info("Track A 성과 지표 재산출 시작")
    logger.info("=" * 80)
    
    # 재구성된 data 폴더에서 데이터 로드
    artifacts, raw_data_dir, cal_data_dir = load_data_from_reorganized_structure(config_path)
    
    # Track A 파이프라인이 interim_dir에서 데이터를 찾으므로 임시로 복사
    interim_dir = Path(get_path(cfg, "data_interim"))
    interim_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n[임시 데이터 복사]")
    logger.info(f"  cal_data → interim_dir (Track A 파이프라인 호환)")
    
    # 필수 데이터 복사
    import shutil
    required_files = ["panel_merged_daily", "dataset_daily"]
    for file_name in required_files:
        # parquet 파일 우선 확인
        src_parquet = cal_data_dir / f"{file_name}.parquet"
        src_dir = cal_data_dir / file_name
        
        dst = interim_dir / file_name
        
        copied = False
        if src_parquet.exists():
            shutil.copy2(src_parquet, interim_dir / f"{file_name}.parquet")
            logger.info(f"  ✓ {file_name}.parquet 복사 완료")
            copied = True
        elif src_dir.exists() and src_dir.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src_dir, dst)
            logger.info(f"  ✓ {file_name} (폴더) 복사 완료")
            copied = True
        
        if not copied:
            logger.warning(f"  ⚠ {file_name} 복사 실패: 파일 없음")
    
    # Track A 파이프라인 실행
    logger.info("\n" + "=" * 80)
    logger.info("[Track A] 랭킹 엔진 실행")
    logger.info("=" * 80)
    
    from src.pipeline.track_a_pipeline import run_track_a_pipeline
    
    track_a_result = run_track_a_pipeline(
        config_path=config_path,
        force_rebuild=True,
        run_ui_payload=False,
    )
    
    ranking_short = track_a_result["ranking_short_daily"]
    ranking_long = track_a_result["ranking_long_daily"]
    
    logger.info(f"  ✓ 단기 랭킹: {len(ranking_short):,}행")
    logger.info(f"  ✓ 장기 랭킹: {len(ranking_long):,}행")
    
    # 통합 랭킹 생성 (단기 0.2, 장기 0.8)
    logger.info("\n" + "=" * 80)
    logger.info("[통합 랭킹] 생성")
    logger.info("=" * 80)
    
    ranking_integrated = create_integrated_ranking(
        ranking_short=ranking_short,
        ranking_long=ranking_long,
        alpha_short=0.2,
        alpha_long=0.8,
    )
    
    # 통합 랭킹 저장 (cal_data와 interim_dir 모두)
    integrated_path_cal = cal_data_dir / "ranking_integrated_daily"
    integrated_path_interim = interim_dir / "ranking_integrated_daily"
    save_artifact(ranking_integrated, integrated_path_cal, force=True)
    save_artifact(ranking_integrated, integrated_path_interim, force=True)
    logger.info(f"  ✓ 통합 랭킹 저장: {integrated_path_cal}")
    logger.info(f"  ✓ 통합 랭킹 저장: {integrated_path_interim}")
    
    # 성과 지표 계산
    logger.info("\n" + "=" * 80)
    logger.info("[성과 지표 계산]")
    logger.info("=" * 80)
    
    dataset_daily = artifacts["dataset_daily"]
    
    # 단기 성과 지표
    metrics_short = calculate_performance_metrics(
        ranking=ranking_short,
        dataset_daily=dataset_daily,
        horizon_days=20,
        name="단기",
    )
    
    # 장기 성과 지표
    metrics_long = calculate_performance_metrics(
        ranking=ranking_long,
        dataset_daily=dataset_daily,
        horizon_days=120,
        name="장기",
    )
    
    # 통합 성과 지표
    metrics_integrated = calculate_performance_metrics(
        ranking=ranking_integrated,
        dataset_daily=dataset_daily,
        horizon_days=20,  # 통합은 단기 기준
        name="통합",
    )
    
    # 결과 요약
    logger.info("\n" + "=" * 80)
    logger.info("[성과 지표 요약]")
    logger.info("=" * 80)
    
    print("\n단기 랭킹:")
    print(f"  IC: {metrics_short.get('ic_mean', 0):.4f} (IR: {metrics_short.get('icir', 0):.4f})")
    print(f"  Rank IC: {metrics_short.get('rank_ic_mean', 0):.4f} (IR: {metrics_short.get('rank_icir', 0):.4f})")
    print(f"  Top 20 평균 수익률: {metrics_short.get('top20_mean_return', 0)*100:.2f}%")
    
    print("\n장기 랭킹:")
    print(f"  IC: {metrics_long.get('ic_mean', 0):.4f} (IR: {metrics_long.get('icir', 0):.4f})")
    print(f"  Rank IC: {metrics_long.get('rank_ic_mean', 0):.4f} (IR: {metrics_long.get('rank_icir', 0):.4f})")
    print(f"  Top 20 평균 수익률: {metrics_long.get('top20_mean_return', 0)*100:.2f}%")
    
    print("\n통합 랭킹 (단기 0.2, 장기 0.8):")
    print(f"  IC: {metrics_integrated.get('ic_mean', 0):.4f} (IR: {metrics_integrated.get('icir', 0):.4f})")
    print(f"  Rank IC: {metrics_integrated.get('rank_ic_mean', 0):.4f} (IR: {metrics_integrated.get('rank_icir', 0):.4f})")
    print(f"  Top 20 평균 수익률: {metrics_integrated.get('top20_mean_return', 0)*100:.2f}%")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ Track A 성과 지표 재산출 완료")
    logger.info("=" * 80)
    
    return {
        "ranking_short": ranking_short,
        "ranking_long": ranking_long,
        "ranking_integrated": ranking_integrated,
        "metrics_short": metrics_short,
        "metrics_long": metrics_long,
        "metrics_integrated": metrics_integrated,
    }


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
