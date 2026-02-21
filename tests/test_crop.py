"""Tests for spatial cropping of cached objects."""

import numpy as np
import pytest

from dtcc_agent.crop import crop_to_bounds


def test_crop_pointcloud_filters_points():
    """PointCloud-like object: only points within bounds are kept."""
    class FakePC:
        def __init__(self):
            self.points = np.array([
                [100.0, 200.0, 5.0],
                [150.0, 250.0, 6.0],
                [300.0, 400.0, 7.0],  # outside
            ])
            self.classification = np.array([1, 2, 3])

    pc = FakePC()
    cropped = crop_to_bounds(pc, [90, 190, 200, 300])

    assert len(cropped.points) == 2
    assert len(cropped.classification) == 2
    assert cropped.points[0][0] == 100.0


def test_crop_returns_original_if_unknown_type():
    """Unknown types are returned as-is (no cropping)."""
    obj = {"data": 123}
    result = crop_to_bounds(obj, [0, 0, 100, 100])
    assert result is obj
