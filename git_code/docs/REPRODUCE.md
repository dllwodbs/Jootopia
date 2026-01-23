# Reproducibility Guide

## Overview

This guide provides step-by-step instructions to reproduce the results of the KOSPI200 quantitative trading system.

## Quick Start (3 Commands)

```bash
# 1. Install dependencies
pip install -e .

# 2. Run Track A (Ranking)
python -m src.cli track-a --config configs/config.yaml

# 3. Run Track B (Backtest)
python -m src.cli track-b --strategy bt120_long --config configs/config.yaml
```

## Prerequisites

### System Requirements
- **Python**: 3.11 or higher
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 5GB free space
- **Internet**: Required for initial data download

### Dependencies

#### Core Dependencies
```bash
pip install pandas numpy pykrx requests
```

#### Optional Dependencies (for full functionality)
```bash
pip install OpenDartReader  # For fundamental data
pip install plotly  # For advanced plotting
```

#### Development Dependencies
```bash
pip install pytest black flake8  # Testing and code quality
```

## Environment Setup

### 1. Clone Repository
```bash
git clone <repository-url>
cd quant-trading-portfolio
```

### 2. Install Package
```bash
pip install -e .
```

### 3. Configure Environment
```bash
# Copy environment template
cp env_example.txt .env

# Edit .env file with your API keys
# DART_API_KEY=your_dart_api_key_here
```

### 4. Verify Installation
```bash
python -c "import src; print('✅ Installation successful')"
```

## Data Preparation

### Raw Data Download
```bash
# Download OHLCV data
python -c "
from src.data_collection.pipeline import DataCollectionPipeline
pipeline = DataCollectionPipeline()
pipeline.run_l0()  # Universe
pipeline.run_l1()  # OHLCV
"

# Optional: Download fundamentals (requires API key)
python -c "
pipeline = DataCollectionPipeline()
pipeline.run_l2()  # DART fundamentals
"
```

### Data Structure
After download, your `data/` folder should look like:
```
data/
├── raw_data/
│   ├── universe_k200_membership_monthly.parquet
│   ├── ohlcv_daily.parquet
│   └── fundamentals_annual.parquet (optional)
└── cal_data/
    ├── panel_merged_daily.parquet
    ├── dataset_daily.parquet
    └── ...
```

## Execution Modes

### Mode 1: Full Pipeline (Recommended)

```bash
# Run complete pipeline
python scripts/recompute_calculated_data.py
```

This will execute:
- L3: Feature engineering
- L4: Dataset preparation
- L8: Ranking generation
- L6R-L7: Backtesting

### Mode 2: Individual Components

#### Track A Only (Ranking)
```bash
# Generate rankings
python -m src.pipeline.track_a_pipeline --force-rebuild
```

#### Track B Only (Backtest)
```bash
# Run backtest for specific strategy
python -m src.pipeline.track_b_pipeline --strategy bt120_long --force-rebuild
```

#### CLI Interface
```bash
# Using CLI (when implemented)
python -m src.cli run --config configs/config.yaml
python -m src.cli track-a --config configs/config.yaml
python -m src.cli track-b --strategy bt120_long --config configs/config.yaml
```

### Mode 3: Development/Debug

#### Data Validation
```bash
# Check data integrity
python -c "
from src.utils.io import load_artifact
from pathlib import Path

# Load and validate data
universe = load_artifact(Path('data/raw_data/universe_k200_membership_monthly.parquet'))
ohlcv = load_artifact(Path('data/raw_data/ohlcv_daily.parquet'))
print(f'Universe: {len(universe)} records')
print(f'OHLCV: {len(ohlcv)} records')
"
```

#### Partial Execution
```bash
# Run only L3 and L4
python -c "
from src.data_collection.pipeline import DataCollectionPipeline
pipeline = DataCollectionPipeline(force_rebuild=True)
# Add your partial execution logic
"
```

## Expected Outputs

### Track A Results
- `data/cal_data/ranking_short_daily.parquet`: Short-term rankings
- `data/cal_data/ranking_long_daily.parquet`: Long-term rankings
- `data/cal_data/ranking_integrated_daily.parquet`: Integrated rankings

### Track B Results
- `data/cal_data/bt_positions_*.parquet`: Portfolio positions
- `data/cal_data/bt_returns_*.parquet`: Portfolio returns
- `data/cal_data/bt_equity_curve_*.parquet`: Equity curves
- `data/cal_data/bt_metrics_*.parquet`: Performance metrics

### Reports
- `TRACK_A_PERFORMANCE_REPORT.md`: Ranking performance analysis
- `RECOMPUTE_FINAL_REPORT.md`: Full pipeline results

## Configuration

### Main Configuration
Edit `configs/config.yaml` for:
- Universe settings
- Feature parameters
- Rebalancing rules
- Cost assumptions

### Experiment Configuration
Create experiment-specific configs in `configs/experiments/`:
```yaml
# configs/experiments/custom_experiment.yaml
inherit_from: ../config.yaml
l8_short:
  feature_weights_config: configs/feature_weights_custom.yaml
```

## Troubleshooting

### Common Issues

#### 1. Missing Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 2. Data Download Failures
```bash
# Check internet connection
ping www.google.com

# Retry download
python -c "from src.data_collection.pipeline import DataCollectionPipeline; DataCollectionPipeline().run_l0()"
```

#### 3. Memory Issues
```bash
# For large datasets, increase memory
# Or process in chunks
export PYTHONPATH=/your/project/path:$PYTHONPATH
python scripts/recompute_calculated_data.py
```

#### 4. API Key Issues
```bash
# Check .env file
cat .env

# Test API key
python -c "import OpenDartReader; dart = OpenDartReader('your_key'); print('API key valid')"
```

### Debug Mode

#### Verbose Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with debug
from src.pipeline.track_a_pipeline import run_track_a_pipeline
result = run_track_a_pipeline(force_rebuild=True)
```

#### Data Inspection
```python
# Check data quality
from src.utils.io import load_artifact
import pandas as pd

ranking = load_artifact('data/cal_data/ranking_short_daily.parquet')
print(ranking.head())
print(ranking.describe())
```

## Validation

### Performance Validation
```python
# Check results match expected
from src.utils.io import load_artifact

# Load metrics
metrics = load_artifact('data/cal_data/bt_metrics_bt120_long.parquet')
print("Performance metrics:")
print(metrics)
```

### Data Integrity
```python
# Validate data completeness
import pandas as pd

ranking = pd.read_parquet('data/cal_data/ranking_short_daily.parquet')
print(f"Date range: {ranking['date'].min()} to {ranking['date'].max()}")
print(f"Unique dates: {ranking['date'].nunique()}")
print(f"Unique tickers: {ranking['ticker'].nunique()}")
```

## Performance Benchmarks

### Expected Execution Times
- Data download: 5-10 minutes
- Feature engineering: 2-5 minutes
- Ranking generation: 1-2 minutes
- Backtesting: 3-5 minutes
- **Total**: 15-30 minutes

### System Requirements
- **CPU**: Quad-core recommended
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB for data and artifacts

## Version Control

### Git Workflow
```bash
# Clone repository
git clone <repo-url>
cd quant-trading-portfolio

# Create feature branch
git checkout -b feature/new-experiment
# Make changes...

# Commit changes
git add .
git commit -m "Add new experiment configuration"

# Push and create PR
git push origin feature/new-experiment
```

### Configuration Versioning
- All config files are version controlled
- Experiment configs inherit from base config
- Results include config hash for traceability

## Contributing

### Code Style
```bash
# Format code
black src/ tests/

# Check style
flake8 src/ tests/

# Run tests
pytest tests/
```

### Documentation
- Update docstrings for new functions
- Add type hints
- Update this reproducibility guide

## Support

### Documentation Links
- [ARCHITECTURE.md](ARCHITECTURE.md): System design
- [STRATEGY_CARD.md](STRATEGY_CARD.md): Strategy details
- [DATA_CARD.md](DATA_CARD.md): Data specifications
- [MODEL_CARD.md](MODEL_CARD.md): Model limitations

### Issue Reporting
- Check existing issues first
- Provide reproducible example
- Include system information
- Attach relevant logs