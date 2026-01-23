# Strategy Card

## 1. Objective

**Strategy Name**: KOSPI200 Dual-Horizon Ranking System

**Investment Objective**:
- Generate quantitative stock rankings for KOSPI200 universe
- Provide alpha signals through multi-horizon ranking approach
- Maintain market-neutral exposure with controlled risk

**Target Audience**: Quantitative portfolio managers, systematic traders

## 2. Universe & Data

### Universe Definition
- **Index**: KOSPI200
- **Constituents**: ~200 stocks (varies by membership)
- **Rebalancing**: Monthly (index rebalancing schedule)
- **Coverage**: All KOSPI200 members during membership period

### Data Sources
- **Primary**: OHLCV (daily price/volume)
- **Secondary**: Fundamentals (annual financials)
- **Optional**: ESG scores, news sentiment
- **Period**: 2016-01-01 ~ 2024-12-31

### Data Quality Controls
- **Missing Data**: Forward/backward fill for technical indicators
- **Outliers**: 99th percentile winsorizing
- **Survival Bias**: Only includes currently listed companies
- **Look-ahead Bias**: Prevented through proper temporal splits

## 3. Signal Generation

### Ranking Methodology

#### Short-term Ranking (Momentum Focus)
- **Horizon**: 20 trading days
- **Features**: Price momentum, volatility, volume ratios
- **Weighting**: Equal-weighted feature combination
- **Normalization**: Percentile ranking

#### Long-term Ranking (Value Focus)
- **Horizon**: 120 trading days
- **Features**: Fundamental ratios, trend indicators
- **Weighting**: IC-optimized feature weights
- **Normalization**: Percentile ranking

#### Integrated Ranking
- **Combination**: Short-term (20%) + Long-term (80%)
- **Purpose**: Balanced alpha capture
- **Rebalancing**: Daily ranking updates

### Feature Engineering

#### Technical Features
- Price momentum (5d, 20d, 60d)
- Volatility measures (20d, 60d)
- Volume ratios and turnover
- Moving averages and trend indicators

#### Fundamental Features (Optional)
- Valuation ratios (PER, PBR, PCR)
- Profitability metrics (ROE, ROA)
- Growth indicators
- Size factors (market cap, float)

## 4. Portfolio Construction

### Portfolio Rules

#### Long/Short Construction
- **Top K**: Top 15 stocks for long positions
- **Bottom K**: Bottom 15 stocks for short positions (optional)
- **Weighting**: Equal-weighted within long/short buckets
- **Market Neutral**: Dollar-neutral exposure

#### Rebalancing
- **Frequency**: Daily (ranking updates)
- **Buffer Zone**: 20% buffer to reduce turnover
- **Minimum Holding**: 1 trading day
- **Transaction Costs**: Applied at rebalancing

### Risk Management

#### Position Limits
- **Stock Level**: Max 5% per stock
- **Sector Level**: Max 20% per sector
- **Factor Exposure**: Controlled through ranking constraints

#### Risk Metrics
- **Target Volatility**: 15% annualized
- **Max Drawdown**: 20% limit
- **Value-at-Risk**: 5% 1-day VaR

## 5. Costs & Execution Assumptions

### Transaction Costs
- **Commission**: 0.015% per trade (round-trip)
- **Market Impact**: 0.05% for large trades
- **Spread**: 0.02% bid-ask spread
- **Total Cost**: ~0.085% per round-trip trade

### Execution Assumptions
- **Liquidity**: KOSPI200 stocks assumed liquid
- **Market Hours**: Regular trading session only
- **Order Type**: Market orders (VWAP execution)
- **Settlement**: T+2 settlement cycle

### Cost Impact Analysis
- **Turnover Sensitivity**: High turnover increases costs
- **Holding Period**: Longer holding reduces cost drag
- **Buffer Zone**: Reduces unnecessary trading

## 6. Evaluation Framework

### Performance Metrics

#### Risk-Adjusted Returns
- **Sharpe Ratio**: > 0.5 target
- **Sortino Ratio**: Downside risk focus
- **Calmar Ratio**: Drawdown-adjusted return

#### Attribution
- **Total Return**: Cumulative portfolio return
- **Annual Return**: Time-weighted annual return
- **Benchmark**: KOSPI200 index

#### Risk Metrics
- **Volatility**: Realized volatility
- **Max Drawdown**: Peak-to-trough decline
- **Value-at-Risk**: Portfolio risk measure

### Validation Methodology

#### Walk-Forward Validation
- **Training Window**: 3-5 years expanding window
- **Test Window**: 20 trading days
- **Holdout**: 2024 data for final validation

#### Cross-Validation
- **Folds**: 5-fold time-series split
- **Gap**: 20-day embargo between folds
- **Refit**: Monthly model updates

### Benchmark Comparison
- **Primary**: KOSPI200 total return
- **Secondary**: Equal-weighted KOSPI200
- **Risk Parity**: Volatility-targeted benchmark

## 7. Known Failure Modes & Limitations

### Data-Related Issues
- **Missing Data**: Technical indicators may be unreliable with gaps
- **Corporate Actions**: Stock splits/mergers affect continuity
- **Market Microstructure**: Intraday data not available

### Model-Related Issues
- **Overfitting**: Complex features may overfit to training data
- **Regime Dependency**: Performance varies by market conditions
- **Feature Stability**: Feature importance may change over time

### Execution-Related Issues
- **Slippage**: Real execution may differ from model assumptions
- **Liquidity**: Low liquidity stocks may have higher costs
- **Market Impact**: Large orders may move prices

### Market Condition Dependency
- **Bull Markets**: Momentum strategies perform well
- **Bear Markets**: Value strategies may outperform
- **High Volatility**: Risk management becomes critical
- **Low Volatility**: Alpha generation becomes challenging

## 8. Implementation Details

### Code Structure
- **Track A**: Ranking engine (`src/tracks/track_a/`)
- **Track B**: Backtest engine (`src/tracks/track_b/`)
- **Shared**: Common utilities (`src/utils/`)

### Configuration
- **Main Config**: `configs/config.yaml`
- **Feature Config**: `configs/feature_groups/`
- **Experiment Config**: `configs/experiments/`

### Execution Modes
- **Development**: Single strategy backtest
- **Production**: Multi-strategy portfolio
- **Research**: Parameter sensitivity analysis

## 9. Performance Expectations

### Expected Returns
- **Target Alpha**: 3-5% annual outperformance
- **Sharpe Ratio**: 0.5-0.8
- **Max Drawdown**: 15-20%

### Risk Parameters
- **Volatility Target**: 12-15% annualized
- **VaR (95%)**: 2-3% daily
- **Stress Loss**: 10% maximum loss scenario

### Capacity Considerations
- **AUM Limit**: ₩500B based on liquidity
- **Position Size**: Max 5% per stock
- **Daily Turnover**: 10-20% of portfolio

## 10. Monitoring & Maintenance

### Daily Monitoring
- **Performance**: Daily P&L, risk metrics
- **Data Quality**: Missing data alerts
- **Execution**: Fill rates, slippage monitoring

### Monthly Reviews
- **Attribution**: Factor contribution analysis
- **Risk**: Stress testing results
- **Costs**: Transaction cost analysis

### Quarterly Updates
- **Rebalancing**: Universe and parameter updates
- **Model Refresh**: Feature importance review
- **Benchmark**: Performance comparison updates

---

**Strategy Version**: v2.0
**Last Updated**: 2026-01-22
**Review Date**: Quarterly
**Owner**: Quant Trading Team