"""
최신 성과 지표 추출 스크립트

Track A, B의 최신 성과 지표를 추출하여 README_PORTFOLIO.md 업데이트용 데이터를 생성합니다.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.utils.config import load_config, get_path
from src.utils.io import load_artifact, artifact_exists


def extract_backtest_metrics():
    """백테스트 성과 지표 추출"""
    cfg = load_config("configs/config.yaml")
    
    # data_cal 경로 확인
    cal_data_dir = Path(get_path(cfg, "data_cal"))
    
    strategies = ["bt120_long", "bt120_ens", "bt20_short", "bt20_ens"]
    results = {}
    
    for strategy in strategies:
        metrics_path = cal_data_dir / f"bt_metrics_{strategy}"
        
        if not artifact_exists(metrics_path):
            print(f"⚠️ {strategy}: 메트릭 파일 없음")
            continue
        
        try:
            df = load_artifact(metrics_path)
            
            # phase별로 분리
            if "phase" not in df.columns:
                print(f"⚠️ {strategy}: phase 컬럼 없음")
                continue
            
            dev_df = df[df["phase"] == "dev"]
            holdout_df = df[df["phase"] == "holdout"]
            
            results[strategy] = {
                "dev": dev_df.iloc[0].to_dict() if not dev_df.empty else None,
                "holdout": holdout_df.iloc[0].to_dict() if not holdout_df.empty else None,
            }
            
        except Exception as e:
            print(f"❌ {strategy}: 오류 - {e}")
            continue
    
    return results


def format_metric(value, fmt=".4f"):
    """메트릭 값 포맷팅"""
    if pd.isna(value) or value is None:
        return "N/A"
    try:
        if isinstance(value, (int, float)):
            return f"{value:{fmt}}"
        return str(value)
    except Exception:
        return "N/A"


def format_percentage(value):
    """퍼센트 포맷팅"""
    if pd.isna(value) or value is None:
        return "N/A"
    try:
        return f"{value*100:.2f}%"
    except Exception:
        return "N/A"


def generate_metrics_table(results, phase="holdout"):
    """성과 테이블 생성"""
    strategies = ["bt120_long", "bt20_ens", "bt20_short", "bt120_ens"]
    
    rows = []
    for strategy in strategies:
        if strategy not in results:
            continue
        
        data = results[strategy].get(phase)
        if not data:
            continue
        
        row = {
            "전략": strategy,
            "Sharpe": format_metric(data.get("net_sharpe"), ".4f"),
            "CAGR": format_percentage(data.get("net_cagr")),
            "MDD": format_percentage(data.get("net_mdd")),
            "Calmar": format_metric(data.get("net_calmar_ratio"), ".4f"),
            "Hit Ratio": format_percentage(data.get("net_hit_ratio")),
            "리밸런싱 수": format_metric(data.get("n_rebalances"), ".0f"),
        }
        rows.append(row)
    
    return rows


def main():
    """메인 함수"""
    print("=" * 80)
    print("최신 성과 지표 추출")
    print("=" * 80)
    
    # 백테스트 성과 추출
    results = extract_backtest_metrics()
    
    if not results:
        print("❌ 성과 데이터를 찾을 수 없습니다.")
        return
    
    # Holdout 구간 성과 테이블
    print("\n📊 Holdout 구간 성과 (2024-01-01 ~ 2024-12-31)")
    print("-" * 80)
    holdout_rows = generate_metrics_table(results, phase="holdout")
    
    for row in holdout_rows:
        print(f"| {row['전략']:15} | {row['Sharpe']:8} | {row['CAGR']:8} | {row['MDD']:8} | {row['Calmar']:8} | {row['Hit Ratio']:8} | {row['리밸런싱 수']:8} |")
    
    # Dev 구간 성과 테이블
    print("\n📊 Dev 구간 성과 (2016-01-01 ~ 2023-12-31)")
    print("-" * 80)
    dev_rows = generate_metrics_table(results, phase="dev")
    
    for row in dev_rows:
        print(f"| {row['전략']:15} | {row['Sharpe']:8} | {row['CAGR']:8} | {row['MDD']:8} | {row['Calmar']:8} | {row['Hit Ratio']:8} | {row['리밸런싱 수']:8} |")
    
    # Markdown 테이블 형식으로 출력
    print("\n" + "=" * 80)
    print("Markdown 테이블 (Holdout)")
    print("=" * 80)
    print("| 전략 | Sharpe | CAGR | MDD | Calmar | Hit Ratio | 리밸런싱 수 |")
    print("|------|--------|------|-----|--------|-----------|------------|")
    for row in holdout_rows:
        print(f"| {row['전략']} | {row['Sharpe']} | {row['CAGR']} | {row['MDD']} | {row['Calmar']} | {row['Hit Ratio']} | {row['리밸런싱 수']} |")
    
    print("\n" + "=" * 80)
    print("Markdown 테이블 (Dev)")
    print("=" * 80)
    print("| 전략 | Sharpe | CAGR | MDD | Calmar | Hit Ratio | 리밸런싱 수 |")
    print("|------|--------|------|-----|--------|-----------|------------|")
    for row in dev_rows:
        print(f"| {row['전략']} | {row['Sharpe']} | {row['CAGR']} | {row['MDD']} | {row['Calmar']} | {row['Hit Ratio']} | {row['리밸런싱 수']} |")
    
    # 상세 데이터 저장
    output_file = project_root / "docs" / "latest_metrics.json"
    import json
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n✅ 상세 데이터 저장: {output_file}")


if __name__ == "__main__":
    main()
