"""
tests/unit/test_config.py
--------------------------
Unit tests for MentorConfig — covers defaults, validation errors, and
the summary() method.
"""

from __future__ import annotations

import pytest

from mentorml.config import MentorConfig
from mentorml.core.exceptions import ConfigurationError


class TestMentorConfigDefaults:
    """Verify that default values are sensible and present."""

    def test_default_construction_succeeds(self) -> None:
        config = MentorConfig()
        assert config is not None

    def test_default_missing_threshold(self) -> None:
        assert MentorConfig().missing_threshold == 0.4

    def test_default_correlation_threshold(self) -> None:
        assert MentorConfig().correlation_threshold == 0.95

    def test_default_cv_folds(self) -> None:
        assert MentorConfig().cv_folds == 5

    def test_default_random_state(self) -> None:
        assert MentorConfig().random_state == 42

    def test_default_task_type_is_none(self) -> None:
        assert MentorConfig().task_type is None

    def test_default_target_column_is_none(self) -> None:
        assert MentorConfig().target_column is None

    def test_default_verbose_is_true(self) -> None:
        assert MentorConfig().verbose is True


class TestMentorConfigValidation:
    """Verify that invalid configurations raise ConfigurationError."""

    def test_missing_threshold_below_zero(self) -> None:
        with pytest.raises(ConfigurationError, match="missing_threshold"):
            MentorConfig(missing_threshold=-0.1)

    def test_missing_threshold_above_one(self) -> None:
        with pytest.raises(ConfigurationError, match="missing_threshold"):
            MentorConfig(missing_threshold=1.1)

    def test_missing_threshold_boundary_zero(self) -> None:
        """0.0 is a valid boundary value."""
        config = MentorConfig(missing_threshold=0.0)
        assert config.missing_threshold == 0.0

    def test_missing_threshold_boundary_one(self) -> None:
        """1.0 is a valid boundary value."""
        config = MentorConfig(missing_threshold=1.0)
        assert config.missing_threshold == 1.0

    def test_correlation_threshold_zero(self) -> None:
        """0.0 is excluded from the valid range."""
        with pytest.raises(ConfigurationError, match="correlation_threshold"):
            MentorConfig(correlation_threshold=0.0)

    def test_test_size_zero(self) -> None:
        with pytest.raises(ConfigurationError, match="test_size"):
            MentorConfig(test_size=0.0)

    def test_test_size_one(self) -> None:
        with pytest.raises(ConfigurationError, match="test_size"):
            MentorConfig(test_size=1.0)

    def test_cv_folds_one(self) -> None:
        with pytest.raises(ConfigurationError, match="cv_folds"):
            MentorConfig(cv_folds=1)

    def test_cv_folds_two_is_valid(self) -> None:
        config = MentorConfig(cv_folds=2)
        assert config.cv_folds == 2

    def test_invalid_task_type(self) -> None:
        with pytest.raises(ConfigurationError, match="task_type"):
            MentorConfig(task_type="clustering")

    def test_valid_task_types(self) -> None:
        MentorConfig(task_type="classification")
        MentorConfig(task_type="regression")

    def test_cardinality_threshold_zero(self) -> None:
        with pytest.raises(ConfigurationError, match="cardinality_threshold"):
            MentorConfig(cardinality_threshold=0)


class TestMentorConfigSummary:
    """Tests for the summary() method."""

    def test_summary_contains_class_name(self) -> None:
        s = MentorConfig().summary()
        assert "MentorConfig" in s

    def test_summary_contains_key_fields(self) -> None:
        config = MentorConfig(target_column="y", cv_folds=10)
        s = config.summary()
        assert "target_column" in s
        assert "cv_folds" in s
        assert "10" in s
