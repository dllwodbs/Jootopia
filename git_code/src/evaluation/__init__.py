"""
Evaluation module for quantitative trading strategies.

This module provides comprehensive evaluation tools for:
- Performance metrics calculation
- Risk analysis
- Attribution analysis
- Benchmark comparisons
- Statistical significance testing
"""

from .metrics import calculate_performance_metrics
from .risk import calculate_risk_metrics
from .attribution import calculate_attribution

__all__ = [
    'calculate_performance_metrics',
    'calculate_risk_metrics',
    'calculate_attribution'
]