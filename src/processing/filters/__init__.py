# src/filters/__init__.py
from .base_filter import BaseFilter
from .moving_average import MovingAverageFilter

__all__ = ['BaseFilter', 'MovingAverageFilter']