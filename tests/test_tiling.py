"""Unit tests for high-resolution tiled image stitching (v1.5.0).

Tests cover:
  - tile grid computation (_compute_tile_grid)
  - blend mask generation (_make_blend_mask)
  - tile stitching with overlap blending (stitch_tiles)
  - position label generation (_tile_position_label)
  - image param get/set API (get_image_params, update_image_params)
"""

import numpy as np
import pytest
from PIL import Image

from researcher.image import (
    _compute_tile_grid,
    _make_blend_mask,
    _tile_position_label,
    stitch_tiles,
    get_image_params,
    update_image_params,
    DEFAULT_IMAGE_PARAMS,
)


# ---------------------------------------------------------------------------
# _compute_tile_grid
# ---------------------------------------------------------------------------

class TestComputeTileGrid:
    def test_single_tile_when_target_equals_tile(self):
        positions = _compute_tile_grid(512, 512, 512, 512, overlap=128)
        assert positions == [(0, 0)]

    def test_2x2_grid(self):
        # 2 tiles wide, 2 tiles tall with 128px overlap
        # stride = 512 - 128 = 384; need target > 512
        positions = _compute_tile_grid(896, 896, 512, 512, overlap=128)
        assert len(positions) == 4
        # All positions within bounds
        for x, y in positions:
            assert x >= 0 and x + 512 <= 896
            assert y >= 0 and y + 512 <= 896

    def test_positions_cover_target(self):
        """Every pixel in the target area should be covered by at least one tile."""
        target_w, target_h = 2048, 2048
        tile_w, tile_h = 512, 512
        overlap = 128
        positions = _compute_tile_grid(target_w, target_h, tile_w, tile_h, overlap)

        coverage = np.zeros((target_h, target_w), dtype=bool)
        for x, y in positions:
            coverage[y : y + tile_h, x : x + tile_w] = True
        assert coverage.all(), "Some pixels are not covered by any tile"

    def test_tiles_do_not_exceed_target(self):
        target_w, target_h = 2048, 1024
        tile_w, tile_h = 512, 512
        overlap = 128
        positions = _compute_tile_grid(target_w, target_h, tile_w, tile_h, overlap)
        for x, y in positions:
            assert x + tile_w <= target_w, f"Tile at ({x},{y}) exceeds target width"
            assert y + tile_h <= target_h, f"Tile at ({x},{y}) exceeds target height"

    def test_no_overlap(self):
        positions = _compute_tile_grid(1024, 512, 512, 512, overlap=0)
        assert len(positions) == 2
        assert (0, 0) in positions
        assert (512, 0) in positions

    def test_asymmetric_target(self):
        """Panoramic target: wider than tall."""
        positions = _compute_tile_grid(2048, 512, 512, 512, overlap=64)
        # Should have multiple columns but only one row
        ys = {y for _, y in positions}
        assert len(ys) == 1


# ---------------------------------------------------------------------------
# _make_blend_mask
# ---------------------------------------------------------------------------

class TestMakeBlendMask:
    def test_shape(self):
        mask = _make_blend_mask(512, 512, overlap=128)
        assert mask.shape == (512, 512)

    def test_center_is_one(self):
        mask = _make_blend_mask(512, 512, overlap=64)
        # Center region should be 1.0
        assert mask[256, 256] == pytest.approx(1.0)

    def test_corners_near_zero_when_all_feathered(self):
        mask = _make_blend_mask(512, 512, overlap=128)
        # Corner (0,0) should be near 0 when all edges are feathered
        assert mask[0, 0] < 0.01

    def test_corner_full_when_border(self):
        """Border tile: top-left corner, only feather right and bottom."""
        mask = _make_blend_mask(
            512, 512, overlap=128,
            feather_left=False, feather_top=False,
            feather_right=True, feather_bottom=True,
        )
        # Top-left corner should be 1.0 (no feathering on those edges)
        assert mask[0, 0] == pytest.approx(1.0)

    def test_edge_ramp(self):
        mask = _make_blend_mask(512, 512, overlap=128)
        # Left edge midpoint should be ~0.5
        assert 0.3 < mask[256, 64] < 0.7

    def test_no_overlap_all_ones(self):
        mask = _make_blend_mask(512, 512, overlap=0)
        assert np.allclose(mask, 1.0)

    def test_values_in_range(self):
        mask = _make_blend_mask(256, 256, overlap=64)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_no_feather_all_ones(self):
        """No feathering on any edge → all ones."""
        mask = _make_blend_mask(
            256, 256, overlap=64,
            feather_left=False, feather_right=False,
            feather_top=False, feather_bottom=False,
        )
        assert np.allclose(mask, 1.0)


# ---------------------------------------------------------------------------
# stitch_tiles
# ---------------------------------------------------------------------------

class TestStitchTiles:
    def test_single_tile_passthrough(self):
        """A single tile should produce an identical image."""
        tile = Image.new("RGB", (512, 512), color=(128, 64, 32))
        result = stitch_tiles([tile], [(0, 0)], 512, 512, overlap=0)
        assert result.size == (512, 512)
        arr = np.asarray(result)
        assert np.allclose(arr, [128, 64, 32], atol=1)

    def test_two_tiles_horizontal(self):
        """Two horizontally overlapping solid-color tiles should blend."""
        tile_w, tile_h = 256, 256
        overlap = 64
        target_w = tile_w * 2 - overlap  # 448
        target_h = tile_h

        tile1 = Image.new("RGB", (tile_w, tile_h), color=(255, 0, 0))
        tile2 = Image.new("RGB", (tile_w, tile_h), color=(0, 0, 255))

        positions = _compute_tile_grid(target_w, target_h, tile_w, tile_h, overlap)
        result = stitch_tiles([tile1, tile2], positions, target_w, target_h, overlap)

        assert result.size == (target_w, target_h)
        arr = np.asarray(result)

        # Left edge should be red
        assert arr[128, 0, 0] > 200
        # Right edge should be blue
        assert arr[128, target_w - 1, 2] > 200
        # Center overlap zone should be a blend (purple-ish)
        mid_x = target_w // 2
        assert arr[128, mid_x, 0] > 50  # some red
        assert arr[128, mid_x, 2] > 50  # some blue

    def test_output_size(self):
        """Stitched image should have the exact target dimensions."""
        target_w, target_h = 1024, 768
        tile_w, tile_h = 256, 256
        overlap = 64
        positions = _compute_tile_grid(target_w, target_h, tile_w, tile_h, overlap)
        tiles = [Image.new("RGB", (tile_w, tile_h), color=(100, 100, 100))
                 for _ in positions]
        result = stitch_tiles(tiles, positions, target_w, target_h, overlap)
        assert result.size == (target_w, target_h)

    def test_uniform_tiles_produce_uniform_image(self):
        """All-same-color tiles should stitch into a uniform image."""
        target_w, target_h = 1024, 1024
        tile_w, tile_h = 512, 512
        overlap = 128
        color = (42, 84, 168)
        positions = _compute_tile_grid(target_w, target_h, tile_w, tile_h, overlap)
        tiles = [Image.new("RGB", (tile_w, tile_h), color=color) for _ in positions]
        result = stitch_tiles(tiles, positions, target_w, target_h, overlap)
        arr = np.asarray(result)
        # Should be uniform (within rounding tolerance)
        assert np.allclose(arr, color, atol=2)

    def test_mismatched_lengths_raises(self):
        tile = Image.new("RGB", (64, 64))
        with pytest.raises(ValueError, match="same length"):
            stitch_tiles([tile], [(0, 0), (10, 10)], 128, 64, overlap=0)

    def test_4k_grid_dimensions(self):
        """Verify grid for 4K target produces correct tile count."""
        target_w, target_h = 3840, 2160
        tile_w, tile_h = 512, 512
        overlap = 128
        positions = _compute_tile_grid(target_w, target_h, tile_w, tile_h, overlap)
        # Sanity check: should be a reasonable number of tiles
        assert len(positions) > 9  # at least 3x3
        assert len(positions) < 100  # not too many


# ---------------------------------------------------------------------------
# _tile_position_label
# ---------------------------------------------------------------------------

class TestTilePositionLabel:
    def test_single_tile(self):
        label = _tile_position_label(0, 0, cols=1, rows=1, stride_x=384, stride_y=384)
        assert label == ""

    def test_top_left(self):
        label = _tile_position_label(0, 0, cols=3, rows=3, stride_x=384, stride_y=384)
        assert label == "top-left"

    def test_bottom_right(self):
        label = _tile_position_label(768, 768, cols=3, rows=3, stride_x=384, stride_y=384)
        assert label == "bottom-right"

    def test_center(self):
        label = _tile_position_label(384, 384, cols=3, rows=3, stride_x=384, stride_y=384)
        assert label == "middle-center"

    def test_single_row(self):
        label = _tile_position_label(384, 0, cols=3, rows=1, stride_x=384, stride_y=384)
        assert label == "center"


# ---------------------------------------------------------------------------
# Image param get/set
# ---------------------------------------------------------------------------

class TestImageParams:
    def setup_method(self):
        """Reset to defaults before each test."""
        update_image_params(dict(DEFAULT_IMAGE_PARAMS))

    def test_get_defaults(self):
        params = get_image_params()
        assert params["hires_enabled"] is False
        assert params["hires_width"] == 2048
        assert params["hires_height"] == 2048
        assert params["tile_overlap"] == 128
        assert params["hires_strength"] == 0.45

    def test_update_hires_enabled(self):
        result = update_image_params({"hires_enabled": True})
        assert result["hires_enabled"] is True

    def test_update_dimensions(self):
        result = update_image_params({"hires_width": 3840, "hires_height": 2160})
        assert result["hires_width"] == 3840
        assert result["hires_height"] == 2160

    def test_overlap_clamped(self):
        result = update_image_params({"tile_overlap": 999})
        assert result["tile_overlap"] == 256  # clamped to max

        result = update_image_params({"tile_overlap": -10})
        assert result["tile_overlap"] == 0  # clamped to min

    def test_width_minimum(self):
        result = update_image_params({"hires_width": 100})
        assert result["hires_width"] == 256  # clamped to min

    def test_unknown_keys_ignored(self):
        result = update_image_params({"nonexistent_key": 42})
        assert "nonexistent_key" not in result

    def test_roundtrip(self):
        update_image_params({"hires_enabled": True, "hires_width": 4096})
        params = get_image_params()
        assert params["hires_enabled"] is True
        assert params["hires_width"] == 4096

    def test_update_strength(self):
        result = update_image_params({"hires_strength": 0.5})
        assert result["hires_strength"] == 0.5

    def test_strength_clamped(self):
        result = update_image_params({"hires_strength": 2.0})
        assert result["hires_strength"] == 1.0  # clamped to max

        result = update_image_params({"hires_strength": 0.0})
        assert result["hires_strength"] == 0.05  # clamped to min
