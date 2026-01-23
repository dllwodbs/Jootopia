# Model Card

## Overview

This system uses a **rule-based quantitative ranking approach** rather than machine learning models. The "model" consists of predefined feature engineering rules and linear weighting schemes for stock ranking.

## Model Details

### Model Type
- **Type**: Rule-based scoring system
- **Architecture**: Linear combination of normalized features
- **Training**: No training required (rule-based)
- **Parameters**: Predefined feature weights and normalization rules

### Input Features

#### Short-term Features (Momentum Focus)
| Feature | Description | Weight | Normalization |
|---------|-------------|--------|---------------|
| price_momentum_20d | 20-day price momentum | 0.021 | Percentile |
| price_momentum_60d | 60-day price momentum | - | Percentile |
| volatility_20d | 20-day volatility | - | Percentile |
| volume_ratio | Trading volume ratio | - | Percentile |
| momentum_reversal | Momentum reversal signal | - | Percentile |

#### Long-term Features (Value Focus)
| Feature | Description | Weight | Normalization |
|---------|-------------|--------|---------------|
| roe | Return on Equity | 0.062 | Percentile |
| per | Price-to-Earnings Ratio | - | Percentile |
| pbr | Price-to-Book Ratio | - | Percentile |
| market_cap | Market Capitalization | - | Percentile |
| trend_strength | Long-term trend strength | - | Percentile |

### Scoring Algorithm

#### Individual Feature Scoring
```python
# Percentile normalization
feature_score = percentile_rank(feature_value, universe)
```

#### Composite Scoring
```python
# Linear combination
total_score = Σ(feature_weight[i] × feature_score[i])
```

#### Final Ranking
```python
# Rank by total score (higher is better)
final_rank = rank(total_score, descending=True)
```

### Performance Characteristics

#### Training Data Performance
- **Period**: 2016-01-01 ~ 2022-12-31
- **Universe**: KOSPI200 constituents
- **IC (Information Coefficient)**: 0.0120 (short), 0.0160 (long)
- **Rank IC**: 0.0374 (short), 0.0443 (long)

#### Validation Data Performance
- **Period**: 2023-01-01 ~ 2023-12-31
- **IC**: 0.0120 (short), 0.0160 (long)
- **Rank IC**: 0.0374 (short), 0.0443 (long)

#### Holdout Data Performance
- **Period**: 2024-01-01 ~ 2024-12-31
- **IC**: 0.0105 (integrated)
- **Rank IC**: 0.0407 (integrated)

## Intended Use

### Primary Use Case
- **Quantitative Stock Ranking**: Generate daily rankings for KOSPI200 stocks
- **Portfolio Construction**: Support systematic portfolio management
- **Research**: Academic and industry research on quantitative strategies

### Target Users
- Quantitative portfolio managers
- Systematic traders
- Academic researchers
- Financial analysts

### Usage Context
- **Frequency**: Daily ranking updates
- **Horizon**: Short-term (20 days) and long-term (120 days)
- **Market**: Korean equity market (KOSPI200)

## Limitations

### Data Limitations
- **Historical Data Only**: Trained on past market conditions
- **Survival Bias**: Only includes currently listed companies
- **Market Regime Dependency**: Performance varies by market conditions
- **Liquidity Assumptions**: Assumes KOSPI200 liquidity levels

### Model Limitations
- **Linear Assumptions**: Assumes linear relationships between features and returns
- **Static Weights**: Feature weights are fixed, not adaptive
- **No Machine Learning**: Cannot learn complex non-linear patterns
- **Feature Engineering Dependent**: Performance depends on feature quality

### Performance Limitations
- **IC Range**: 0.01-0.04 (modest predictive power)
- **Regime Sensitivity**: Performance varies across market conditions
- **Transaction Costs**: Model assumes idealized execution
- **Implementation Gap**: Real-world execution may differ from backtest

## Ethical Considerations

### Fairness
- **Equal Treatment**: All stocks in universe treated equally
- **No Bias**: No intentional bias toward specific sectors or companies
- **Transparent Rules**: All ranking rules are explicit and auditable

### Impact
- **Market Impact**: Large implementation may affect market prices
- **Liquidity**: May stress liquidity in smaller stocks
- **Information Asymmetry**: Provides systematic signals that may be replicated

### Responsible Use
- **Educational Purpose**: Primarily for research and education
- **Risk Disclosure**: Users should understand quantitative risks
- **Professional Advice**: Not a substitute for professional investment advice

## Maintenance

### Monitoring
- **Performance Tracking**: Regular IC and ranking quality monitoring
- **Data Quality**: Feature calculation and data completeness checks
- **Market Regime**: Performance across different market conditions

### Updates
- **Feature Review**: Annual review of feature importance
- **Weight Optimization**: Periodic weight recalibration
- **Universe Changes**: Adaptation to index changes

### Validation
- **Walk-Forward**: Regular out-of-sample validation
- **Stress Testing**: Performance under extreme market conditions
- **Robustness Checks**: Sensitivity to parameter changes

## Technical Details

### Implementation
- **Language**: Python 3.11+
- **Libraries**: pandas, numpy, pykrx
- **Execution**: CLI-based with configuration files

### Configuration
- **Feature Definitions**: `configs/feature_groups/`
- **Weights**: `configs/feature_weights/`
- **Parameters**: `configs/config.yaml`

### Reproducibility
- **Version Control**: All code and configurations versioned
- **Random Seeds**: Fixed for reproducible results
- **Environment**: Requirements.txt for dependency management

## Alternative Approaches

### Potential Improvements
1. **Machine Learning Integration**: Replace rule-based with ML models
2. **Adaptive Weights**: Dynamic feature weighting
3. **Ensemble Methods**: Multiple ranking models combination
4. **Deep Learning**: Neural networks for feature extraction

### Comparison with ML Approaches
| Aspect | Rule-based (Current) | ML-based (Potential) |
|--------|---------------------|---------------------|
| Interpretability | High | Medium |
| Development Time | Low | High |
| Computational Cost | Low | High |
| Adaptability | Low | High |
| Performance Ceiling | Medium | High |

## Conclusion

This rule-based ranking system provides a transparent, interpretable approach to quantitative stock ranking with modest but consistent predictive power. While limited by its linear assumptions and static weights, it offers a solid foundation for systematic investing with clear advantages in terms of simplicity and auditability.

The model's primary value lies in its transparency and consistency rather than superior predictive power, making it suitable for applications where interpretability and reliability are prioritized over maximum performance.