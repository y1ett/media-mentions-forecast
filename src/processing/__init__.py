"""Обработка данных: тексты -> staging -> витрина mart.topic_daily -> контроль качества.
"""
from .mart import build_all
from .quality import run_dq
from .text import top_terms

__all__ = ["build_all", "run_dq", "top_terms"]
