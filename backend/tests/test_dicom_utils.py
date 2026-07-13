"""
test_dicom_utils.py
-------------------
Unit tests for the DICOM utility helpers.
These tests run without any DICOM files by testing the pure numeric functions.
"""

import numpy as np
import pytest
from app.utils.dicom_utils import (
    apply_window,
    normalize_generic,
    get_window_preset,
    WINDOW_PRESETS,
)


class TestApplyWindow:
    def test_values_clipped_to_range(self):
        arr = np.array([-1000, -600, 0, 400, 3000], dtype=np.float32)
        result = apply_window(arr, window_center=-600, window_width=1500)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_output_dtype_is_uint8(self):
        arr = np.zeros((10, 10), dtype=np.float32)
        result = apply_window(arr, window_center=40, window_width=400)
        assert result.dtype == np.uint8

    def test_uniform_array_maps_to_boundary(self):
        # An array all at the lower window edge → all 0
        arr = np.full((5, 5), -40.0)   # center=40, width=80 → lower = 0
        result = apply_window(arr, window_center=40, window_width=80)
        assert np.all(result == 0)

    def test_upper_window_edge_maps_to_255(self):
        # An array all at the upper window edge → all 255
        arr = np.full((5, 5), 80.0)   # center=40, width=80 → upper = 80
        result = apply_window(arr, window_center=40, window_width=80)
        assert np.all(result == 255)

    def test_output_shape_preserved(self):
        arr = np.random.rand(64, 64).astype(np.float32) * 1000
        result = apply_window(arr, window_center=40, window_width=400)
        assert result.shape == (64, 64)


class TestNormalizeGeneric:
    def test_output_dtype_is_uint8(self):
        arr = np.array([0, 50, 100, 200], dtype=np.float32)
        result = normalize_generic(arr)
        assert result.dtype == np.uint8

    def test_output_range_0_to_255(self):
        arr = np.array([10, 50, 100, 500], dtype=np.float32)
        result = normalize_generic(arr)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_uniform_array_returns_original_value(self):
        arr = np.full((4, 4), 42.0)
        result = normalize_generic(arr)
        # When max == min there's nothing to scale — the value is simply
        # cast to uint8 (no division path is taken).
        assert np.all(result == 42)

    def test_min_maps_to_0_max_maps_to_255(self):
        arr = np.array([0.0, 255.0])
        result = normalize_generic(arr)
        assert result[0] == 0
        assert result[1] == 255


class TestGetWindowPreset:
    def test_returns_brain_preset_for_brain_description(self):
        preset = get_window_preset("Brain MRI", "", "MR")
        assert preset == WINDOW_PRESETS["brain"]

    def test_returns_lung_preset_for_chest_body_part(self):
        preset = get_window_preset("", "CHEST", "CT")
        assert preset == WINDOW_PRESETS["chest"]

    def test_returns_none_for_plain_xray_modality(self):
        preset = get_window_preset("", "", "CR")
        assert preset is None

    def test_returns_none_for_dx_modality(self):
        preset = get_window_preset("", "", "DX")
        assert preset is None

    def test_returns_default_for_unknown_ct(self):
        preset = get_window_preset("Unknown Study", "Unknown", "CT")
        assert preset == WINDOW_PRESETS["default"]

    def test_case_insensitive_matching(self):
        preset = get_window_preset("BRAIN SCAN", "", "MR")
        assert preset == WINDOW_PRESETS["brain"]

    @pytest.mark.parametrize("keyword", list(WINDOW_PRESETS.keys()))
    def test_all_presets_reachable(self, keyword):
        if keyword == "default":
            return  # default is the fallback, not matched by keyword
        preset = get_window_preset(keyword, "", "CT")
        assert preset == WINDOW_PRESETS[keyword]
