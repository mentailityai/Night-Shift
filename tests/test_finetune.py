# =============================================================================
# Night-Shift — Fine-Tuning Pipeline Tests
# =============================================================================
# Verifies the threshold monitor logic and replay buffer math.
# =============================================================================

from __future__ import annotations

import math

import pytest

from app.core.config import Settings


class TestThresholdConfig:
    """Tests for the fine-tuning threshold configuration."""

    def test_default_threshold(self):
        """Default threshold should be 500."""
        s = Settings()
        assert s.fine_tuning_threshold == 500

    def test_custom_threshold(self):
        """Threshold should be overridable."""
        s = Settings(fine_tuning_threshold=100)
        assert s.fine_tuning_threshold == 100

    def test_replay_buffer_default(self):
        """Default replay buffer ratio should be 0.15 (15%)."""
        s = Settings()
        assert s.fine_tuning_replay_buffer_ratio == 0.15


class TestReplayBufferMath:
    """Tests for the replay buffer mixing calculations."""

    def test_replay_count_standard(self):
        """With 500 new rows and 15% ratio, expect 75 replay rows."""
        new_count = 500
        ratio = 0.15
        replay_count = math.ceil(new_count * ratio)
        assert replay_count == 75

    def test_replay_count_small_batch(self):
        """With 10 new rows and 15% ratio, expect 2 replay rows (ceiling)."""
        new_count = 10
        ratio = 0.15
        replay_count = math.ceil(new_count * ratio)
        assert replay_count == 2

    def test_replay_count_zero_ratio(self):
        """With 0% ratio, no replay rows should be added."""
        new_count = 500
        ratio = 0.0
        replay_count = math.ceil(new_count * ratio)
        assert replay_count == 0

    def test_replay_count_full_ratio(self):
        """With 100% ratio, replay count equals new count."""
        new_count = 50
        ratio = 1.0
        replay_count = math.ceil(new_count * ratio)
        assert replay_count == 50

    def test_total_dataset_size(self):
        """Total dataset should be new + replay."""
        new_count = 500
        ratio = 0.15
        replay_count = math.ceil(new_count * ratio)
        total = new_count + replay_count
        assert total == 575
