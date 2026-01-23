# Architecture Documentation

## Overview

This project implements a **Two-Track Quantitative Trading System** for KOSPI200 universe:

- **Track A**: Ranking Engine - Generates stock rankings using feature-based scoring
- **Track B**: Investment Model - Executes backtests using ranking-based portfolio construction

## System Architecture

### High-Level Pipeline

```
Raw Data → Data Collection → Feature Engineering → Ranking → Portfolio → Backtest → Evaluation
    ↓           ↓                    ↓             ↓          ↓          ↓           ↓
External   L0-L4 Pipeline      L3 Features   L8 Engine  L6R/L7   L7 Backtest  Reports
```

### Detailed Stage Breakdown

#### Data Pipeline (L0-L4)
- **L0**: Universe Construction (KOSPI200 membership)
- **L1**: OHLCV Data Collection (Daily price/volume data)
- **L2**: Fundamental Data Collection (DART financials - optional)
- **L3**: Feature Engineering (Technical indicators, fundamental ratios)
- **L4**: Dataset Preparation (CV splits, target generation)

#### Track A: Ranking Engine (L8)
- **Input**: Feature-engineered dataset
- **Process**: Multi-horizon ranking (short-term + long-term)
- **Output**: Daily rankings for each stock

#### Track B: Investment Model (L6R-L7)
- **L6R**: Ranking Score Transformation (Convert rankings to portfolio signals)
- **L7**: Backtesting (Portfolio construction, trade execution, performance calculation)

## Module Structure

```
src/
├── data_collection/   # Data collection pipeline (L0-L4)
│   ├── __init__.py
│   └── pipeline.py    # DataCollectionPipeline class
├── stages/            # Individual pipeline stages
│   ├── data/          # L0-L4 data stages
│   │   ├── l0_universe.py
│   │   ├── l1_ohlcv.py
│   │   └── ...
│   └── modeling/      # L5-L7 modeling stages
│       ├── l5_train_models.py
│       └── ...
├── tracks/            # Two-track architecture
│   ├── track_a/       # Ranking engine (L8, L11)
│   │   └── stages/
│   ├── track_b/       # Investment model (L6R, L7)
│   │   └── stages/
│   └── shared/        # Shared stages (L0-L4)
│       └── stages/
├── pipeline/           # Pipeline orchestration
│   ├── track_a_pipeline.py  # Track A execution
│   └── track_b_pipeline.py  # Track B execution
├── features/          # Feature engineering utilities
├── ranking/           # Ranking algorithms
├── portfolio/         # Portfolio construction
├── evaluation/        # Performance evaluation
├── utils/             # Shared utilities
│   ├── config.py      # Configuration management
│   ├── io.py          # Data I/O
│   ├── meta.py        # Metadata management
│   ├── logging.py     # Logging utilities
│   └── reproducibility.py  # Reproducibility tools
├── tools/             # Utility scripts
└── cli.py             # Command-line interface
```

## Data Flow

### Input Data
- **Raw Data**: OHLCV, fundamentals, ESG data
- **Configuration**: Strategy parameters, universe, rebalancing rules
- **External**: Benchmark data, transaction costs assumptions

### Output Data
- **Rankings**: Daily stock rankings (short/long term)
- **Portfolios**: Rebalanced portfolio positions
- **Performance**: Backtest results, risk metrics, attribution

## Key Design Decisions

### 1. Two-Track Architecture
- **Rationale**: Separates signal generation (Track A) from portfolio management (Track B)
- **Benefits**: Modularity, easier testing, flexible strategy combinations

### 2. Walk-Forward Validation
- **Rationale**: Simulates real-world deployment with expanding window validation
- **Benefits**: Reduces overfitting, provides realistic performance estimates

### 3. Modular Pipeline Design
- **Rationale**: Each stage is independent and testable
- **Benefits**: Easier debugging, partial re-runs, pipeline composition

### 4. Configuration-Driven Approach
- **Rationale**: All parameters in YAML files for reproducibility
- **Benefits**: Easy experimentation, audit trail, version control

## Dependencies

### Core Dependencies
- `pandas`: Data manipulation
- `numpy`: Numerical computations
- `pykrx`: KRX data access
- `OpenDartReader`: DART financial data (optional)

### Development Dependencies
- `pytest`: Testing
- `black`: Code formatting
- `flake8`: Linting

## Configuration Management

### Config Files
- `configs/config.yaml`: Main configuration
- `configs/experiments/`: Experiment-specific overrides

### Environment Variables
- `DART_API_KEY`: For fundamental data access
- `DATA_PATHS`: Custom data directory paths

## Performance Considerations

### Memory Optimization
- Chunked data processing for large datasets
- Efficient pandas operations
- Garbage collection for large intermediate results

### Speed Optimization
- Vectorized operations where possible
- Parallel processing for independent calculations
- Caching of expensive computations

## Testing Strategy

### Unit Tests
- Individual functions and classes
- Edge cases and error conditions
- Mock external dependencies

### Integration Tests
- End-to-end pipeline execution
- Data validation checks
- Performance regression tests

## Deployment Considerations

### Local Development
- `python -m src.cli run --config configs/config.yaml`
- Individual stage execution
- Debug mode with detailed logging

### Production Deployment
- Containerized execution (Docker)
- Scheduled runs (cron/Airflow)
- Result storage and alerting

## Future Extensions

### Planned Features
- Machine learning model integration
- Alternative data sources
- Multi-asset support
- Real-time execution

### Potential Improvements
- GPU acceleration for large-scale backtests
- Distributed computing for massive datasets
- Advanced risk management
- Portfolio optimization algorithms