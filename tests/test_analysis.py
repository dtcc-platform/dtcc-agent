"""Tests for the analysis module."""

import numpy as np
import pytest

from dtcc_agent.analysis import summarize_field, compare_fields


class TestSummarizeField:
    def test_basic_stats(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = summarize_field(values, "test")
        assert result["field_name"] == "test"
        assert result["min"] == 1.0
        assert result["max"] == 5.0
        assert result["mean"] == 3.0
        assert result["median"] == 3.0
        assert result["num_values"] == 5

    def test_nan_filtered(self):
        values = np.array([1.0, np.nan, 3.0, np.inf, 5.0])
        result = summarize_field(values, "test")
        assert result["num_values"] == 3  # only 1, 3, 5

    def test_all_nan(self):
        values = np.array([np.nan, np.nan])
        result = summarize_field(values, "test")
        assert "error" in result

    def test_2d_flattened(self):
        values = np.array([[1, 2], [3, 4]])
        result = summarize_field(values, "test")
        assert result["num_values"] == 4
        assert result["mean"] == 2.5


class TestCompareFields:
    def test_basic_comparison(self):
        a = np.array([10.0, 20.0, 30.0])
        b = np.array([15.0, 25.0, 35.0])
        result = compare_fields(a, b, "baseline", "heatwave", "temperature")

        assert "baseline" in result
        assert "heatwave" in result
        assert "difference" in result
        assert result["difference"]["mean"] == pytest.approx(5.0)

    def test_size_mismatch(self):
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 2.0, 3.0])
        result = compare_fields(a, b)
        assert "error" in result

    def test_identical_fields(self):
        a = np.array([1.0, 2.0, 3.0])
        result = compare_fields(a, a, "A", "B", "field")
        assert result["difference"]["mean"] == 0.0
        assert result["difference"]["max"] == 0.0
