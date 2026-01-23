"""Track B 성과 지표 확인"""
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

print("=" * 80)
print("Track B 성과 지표")
print("=" * 80)

# 전략 목록
strategies = ["bt120_long", "bt120_ens", "bt20_short", "bt20_ens"]

for strategy in strategies:
    metrics_path = interim / f"bt_metrics_{strategy}"
    
    # parquet 파일 우선, 없으면 csv
    if not metrics_path.exists():
        metrics_path = interim / f"bt_metrics_{strategy}.parquet"
    
    if not metrics_path.exists():
        print(f"\n⚠️ {strategy}: 메트릭 파일 없음")
        continue
    
    try:
        metrics = load_artifact(metrics_path)
        
        print(f"\n[{strategy}]")
        print("-" * 80)
        
        # DataFrame인 경우
        if isinstance(metrics, pd.DataFrame):
            if len(metrics) > 0:
                # 주요 성과 지표 컬럼 선택
                key_cols = ['phase', 'net_total_return', 'net_cagr', 'net_sharpe', 
                           'net_mdd', 'net_hit_ratio', 'net_profit_factor', 
                           'gross_total_return', 'gross_cagr', 'gross_sharpe',
                           'ic', 'rank_ic', 'icir', 'rank_icir',
                           'n_rebalances', 'avg_n_tickers', 'date_start', 'date_end']
                available_cols = [c for c in key_cols if c in metrics.columns]
                
                if available_cols:
                    display_df = metrics[available_cols].copy()
                    # 숫자 포맷팅
                    for col in display_df.select_dtypes(include=['float64']).columns:
                        if col in ['net_total_return', 'gross_total_return']:
                            display_df[col] = display_df[col].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")
                        elif col in ['net_cagr', 'gross_cagr']:
                            display_df[col] = display_df[col].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")
                        elif col in ['net_sharpe', 'gross_sharpe', 'icir', 'rank_icir']:
                            display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
                        elif col in ['net_mdd', 'gross_mdd']:
                            display_df[col] = display_df[col].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")
                        elif col in ['net_hit_ratio', 'gross_hit_ratio']:
                            display_df[col] = display_df[col].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")
                        elif col in ['net_profit_factor', 'gross_profit_factor']:
                            display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
                        elif col in ['ic', 'rank_ic']:
                            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
                        elif col in ['n_rebalances', 'avg_n_tickers']:
                            display_df[col] = display_df[col].apply(lambda x: f"{int(x)}" if pd.notna(x) else "N/A")
                    
                    pd.set_option('display.max_columns', None)
                    pd.set_option('display.width', None)
                    pd.set_option('display.max_colwidth', 20)
                    print(display_df.to_string(index=False))
                else:
                    # 모든 컬럼 표시
                    pd.set_option('display.max_columns', None)
                    pd.set_option('display.width', None)
                    print(metrics.to_string(index=False))
            else:
                print("  메트릭 데이터 없음")
        # Dict인 경우
        elif isinstance(metrics, dict):
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:,.4f}")
                else:
                    print(f"  {key}: {value}")
        # 기타
        else:
            print(f"  {metrics}")
            
    except Exception as e:
        print(f"\n⚠️ {strategy}: 로드 실패 - {e}")

print("\n" + "=" * 80)

# 추가: 자산 곡선 요약 정보
print("\n[자산 곡선 요약]")
print("-" * 80)

for strategy in strategies:
    equity_path = interim / f"bt_equity_curve_{strategy}"
    
    # parquet 우선
    if not equity_path.exists():
        equity_path = interim / f"bt_equity_curve_{strategy}.parquet"
    
    if equity_path.exists():
        try:
            equity = load_artifact(equity_path)
            if isinstance(equity, pd.DataFrame) and len(equity) > 0:
                if 'date' in equity.columns and 'equity' in equity.columns:
                    equity['date'] = pd.to_datetime(equity['date'])
                    equity = equity.sort_values('date')
                    
                    initial = equity['equity'].iloc[0]
                    final = equity['equity'].iloc[-1]
                    total_return = (final / initial - 1) * 100
                    
                    # 최대 자산, 최소 자산
                    max_equity = equity['equity'].max()
                    min_equity = equity['equity'].min()
                    max_dd = (min_equity / max_equity - 1) * 100 if max_equity > 0 else 0
                    
                    print(f"{strategy}:")
                    print(f"  시작 자산: {initial:,.0f}")
                    print(f"  최종 자산: {final:,.0f}")
                    print(f"  최대 자산: {max_equity:,.0f}")
                    print(f"  최소 자산: {min_equity:,.0f}")
                    print(f"  총 수익률: {total_return:.2f}%")
                    print(f"  최대 낙폭: {max_dd:.2f}%")
                    print()
        except Exception as e:
            print(f"  {strategy}: {e}")

print("=" * 80)
