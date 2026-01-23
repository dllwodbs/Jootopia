# Data Card

## Overview

This document describes the data used in the KOSPI200 quantitative trading system, including sources, collection methods, quality metrics, and known limitations.

## Dataset Summary

| Dataset | Source | Frequency | Period | Universe | Status |
|---------|--------|-----------|--------|----------|--------|
| OHLCV | KRX via pykrx | Daily | 2016-01-04 ~ 2024-12-30 | KOSPI200 | ✅ Available |
| Universe | KRX via pykrx | Monthly | 2016-01 ~ 2024-12 | KOSPI200 | ✅ Available |
| Fundamentals | DART via OpenDartReader | Annual | 2016 ~ 2023 | KOSPI200 | ⚠️ Optional |
| ESG | External | Annual | 2020 ~ 2023 | KOSPI200 | ⚠️ Optional |
| News Sentiment | External | Daily | 2020 ~ 2023 | KOSPI200 | ⚠️ Optional |

## Detailed Dataset Descriptions

### 1. OHLCV Data (Primary)

#### Description
Daily Open, High, Low, Close, Volume data for KOSPI200 stocks

#### Source
- **Provider**: Korea Exchange (KRX)
- **API**: pykrx library (`stock.get_market_ohlcv_by_date`)
- **Access**: Public API (no authentication required)

#### Collection Method
```python
import pykrx.stock as stock
df = stock.get_market_ohlcv_by_date("20160101", "20241231", "005930")
```

#### Data Quality
- **Completeness**: 99.8% (trading days only)
- **Accuracy**: High (official exchange data)
- **Timeliness**: Real-time available
- **Coverage**: All KOSPI200 stocks during membership period

#### Known Issues
- **Missing Data**: No data for non-trading days (weekends, holidays)
- **Adjustments**: Price adjustments not applied (raw prices)
- **Corporate Actions**: Stock splits/mergers may affect continuity

### 2. Universe Data

#### Description
KOSPI200 index membership information

#### Source
- **Provider**: Korea Exchange (KRX)
- **API**: pykrx library (`stock.get_index_portfolio_deposit_file`)
- **Access**: Public API

#### Collection Method
Monthly collection of index constituents with membership dates

#### Data Quality
- **Completeness**: 100%
- **Accuracy**: Official index composition
- **Updates**: Monthly rebalancing

#### Known Issues
- **Timing**: Membership changes effective at month-end
- **Historical**: May not reflect intra-month changes

### 3. Fundamental Data (Optional)

#### Description
Annual financial statements from DART system

#### Source
- **Provider**: Financial Supervisory Service (FSS) via DART
- **API**: OpenDartReader library
- **Access**: Requires API key

#### Collection Method
```python
import OpenDartReader
dart = OpenDartReader("your_api_key")
df = dart.finstate("005930", 2023)
```

#### Data Quality
- **Completeness**: 95% (some companies missing data)
- **Accuracy**: Regulatory filings (high reliability)
- **Timeliness**: Annual reports (4-6 month delay)

#### Known Issues
- **API Limits**: Rate limiting applies
- **Coverage**: Not all companies file complete data
- **Timing**: Fiscal year vs calendar year differences

### 4. ESG Data (Optional)

#### Description
Environmental, Social, Governance scores

#### Source
- **Provider**: External data vendor
- **Format**: CSV files
- **Access**: Local files

#### Data Quality
- **Completeness**: 85%
- **Accuracy**: Vendor methodology
- **Updates**: Annual

#### Known Issues
- **Coverage**: Limited to larger companies
- **Methodology**: Black box scoring
- **Timeliness**: Annual updates

### 5. News Sentiment Data (Optional)

#### Description
Daily news sentiment scores for stocks

#### Source
- **Provider**: External NLP processing
- **Format**: Parquet files
- **Access**: Local files

#### Data Quality
- **Completeness**: 78%
- **Accuracy**: NLP model dependent
- **Updates**: Daily

#### Known Issues
- **Noise**: Financial news can be volatile
- **Coverage**: Limited to actively covered stocks
- **Language**: Korean text processing

## Data Processing Pipeline

### Stage L0: Universe Construction
- Input: KRX index data
- Output: Monthly membership lists
- Quality Checks: Continuity validation

### Stage L1: OHLCV Collection
- Input: Universe + date range
- Output: Daily price/volume data
- Quality Checks: Missing data imputation

### Stage L2: Fundamental Data (Optional)
- Input: Universe + fiscal years
- Output: Annual financial metrics
- Quality Checks: Data completeness validation

### Stage L3: Feature Engineering
- Input: OHLCV + fundamentals + external data
- Output: Technical + fundamental features
- Quality Checks: Feature distribution analysis

### Stage L4: Dataset Preparation
- Input: Features + universe
- Output: ML-ready dataset with targets
- Quality Checks: Target distribution, missing values

## Data Quality Metrics

### Completeness
- OHLCV: 99.8% (trading days only)
- Universe: 100%
- Fundamentals: 95%
- ESG: 85%
- News: 78%

### Temporal Coverage
- Training: 2016-01-01 ~ 2022-12-31
- Validation: 2023-01-01 ~ 2023-12-31
- Holdout: 2024-01-01 ~ 2024-12-31

### Universe Coverage
- KOSPI200 stocks during membership period
- ~200 stocks at any given time
- Market cap weighted representation

## Data Bias and Limitations

### Selection Bias
- **Survivorship Bias**: Only includes currently listed companies
- **Size Bias**: KOSPI200 favors large-cap stocks
- **Liquidity Bias**: More liquid stocks have better data quality

### Temporal Bias
- **Holiday Effects**: No trading data on holidays
- **Market Hours**: Only regular trading session data
- **Announcement Timing**: Financial data lags market events

### Data Quality Issues
- **Missing Values**: Varies by dataset (0.2% ~ 22%)
- **Outliers**: Extreme price movements, corporate actions
- **Data Errors**: API failures, parsing errors

## Data Versioning

### Version Control
- Raw data: External storage with hash verification
- Processed data: Git-tracked artifacts with manifest files
- Configuration: YAML files with parameter versioning

### Reproducibility
- Random seeds: Fixed for reproducible results
- API calls: Cached responses
- External data: Versioned downloads

## Usage Recommendations

### For Development
1. Use cached data for faster iteration
2. Validate data quality before model training
3. Monitor for data drift in production

### For Production
1. Implement data quality monitoring
2. Set up automated data validation pipelines
3. Maintain backup data sources

### For Research
1. Document data assumptions clearly
2. Report data quality metrics
3. Consider robustness to data changes

## Data Ethics and Compliance

### Privacy Considerations
- No personal identifiable information used
- Aggregated market data only

### Regulatory Compliance
- Uses publicly available market data
- No insider information
- Educational/research purpose only

### Fair Use
- Respects API rate limits
- Caches data to minimize external calls
- Provides attribution to data sources