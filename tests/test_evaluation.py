"""
Entry point: run all evaluations + generate all reports.

Usage:
    python -m tests.test_evaluation

This file orchestrates the 3 modules:
    - fake_data.py:      fake PipelineState + Gold standards
    - run_evaluation.py: 7 evaluation metrics (benchmark + no-gold)
    - run_reports.py:    reports (text, JSON, tables, plots)
"""
from tests.run_reports import generate_all_reports

if __name__ == "__main__":
    generate_all_reports()
