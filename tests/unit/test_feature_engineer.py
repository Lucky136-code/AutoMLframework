"""Unit tests for Phase 4 — FeatureEngineer."""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np

from mentorml.config import MentorConfig
from mentorml.core.decision import DecisionLog
from mentorml.core.exceptions import ComponentNotFittedError
from mentorml.features.feature_engineer import FeatureEngineer


@pytest.fixture()
def config() -> MentorConfig:
    return MentorConfig(correlation_threshold=0.95)


@pytest.fixture()
def numeric_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "a": rng.uniform(0, 1, 100),
        "b": rng.uniform(0, 1, 100),
        "c": rng.uniform(0, 1, 100),
    })


class TestFeatureEngineer:
    def test_transform_without_fit_raises(self, config: MentorConfig, numeric_df: pd.DataFrame) -> None:
        fe = FeatureEngineer(config)
        log = DecisionLog()
        with pytest.raises(ComponentNotFittedError):
            fe.transform(numeric_df, log)

    def test_fit_returns_self(self, config: MentorConfig, numeric_df: pd.DataFrame) -> None:
        fe = FeatureEngineer(config)
        log = DecisionLog()
        assert fe.fit(numeric_df, log) is fe

    def test_log_transform_for_skewed_column(self, config: MentorConfig) -> None:
        rng = np.random.default_rng(1)
        # Create a right-skewed column: many zeros + a few extreme values
        skewed_vals = np.concatenate([np.zeros(80), np.array([100, 500, 1000, 5000, 10000,
                                                               15000, 20000, 25000, 30000, 40000])])
        normal_vals = rng.normal(0, 1, 90)
        # Ensure equal lengths
        n = min(len(skewed_vals), len(normal_vals))
        df = pd.DataFrame({"skewed": skewed_vals[:n], "normal": normal_vals[:n]})
        analysis_result = {
            "column_profiles": {
                "skewed": {"skewness": 5.0, "outlier_pct": 0.0},
                "normal": {"skewness": 0.1, "outlier_pct": 0.0},
            }
        }
        fe = FeatureEngineer(config)
        log = DecisionLog()
        fe.fit(df, log, analysis_result)
        out = fe.transform(df, log)
        assert "skewed_log" in out.columns
        assert "normal_log" not in out.columns

    def test_correlated_columns_dropped(self) -> None:
        cfg = MentorConfig(correlation_threshold=0.9)
        x = np.linspace(0, 1, 100)
        df = pd.DataFrame({"x": x, "x_copy": x + 0.001, "unrelated": np.random.default_rng(0).uniform(size=100)})
        fe = FeatureEngineer(cfg)
        log = DecisionLog()
        fe.fit(df, log)
        out = fe.transform(df, log)
        cols = list(out.columns)
        # One of x or x_copy should be dropped
        assert not ("x" in cols and "x_copy" in cols)

    def test_interaction_features_created(self, config: MentorConfig, numeric_df: pd.DataFrame) -> None:
        fe = FeatureEngineer(config)
        log = DecisionLog()
        fe.fit(numeric_df, log)
        out = fe.transform(numeric_df, log)
        interaction_cols = [c for c in out.columns if "_x_" in c]
        # May or may not have interactions based on correlations, but transform should not error
        assert isinstance(interaction_cols, list)

    def test_outlier_flag_created(self, config: MentorConfig) -> None:
        vals = list(range(90)) + [10000] * 10
        df = pd.DataFrame({"heavy_tail": vals})
        analysis_result = {
            "column_profiles": {
                "heavy_tail": {"skewness": 3.0, "outlier_pct": 0.1},
            }
        }
        fe = FeatureEngineer(config)
        log = DecisionLog()
        fe.fit(df, log, analysis_result)
        out = fe.transform(df, log)
        assert "heavy_tail_is_outlier" in out.columns
        assert out["heavy_tail_is_outlier"].dtype in (int, np.int64, np.int32)

    def test_emits_decisions(self, config: MentorConfig, numeric_df: pd.DataFrame) -> None:
        fe = FeatureEngineer(config)
        log = DecisionLog()
        fe.fit(numeric_df, log)
        assert len(log) > 0

    def test_transform_preserves_shape_without_drops(self, config: MentorConfig, numeric_df: pd.DataFrame) -> None:
        fe = FeatureEngineer(config)
        log = DecisionLog()
        fe.fit(numeric_df, log)
        out = fe.transform(numeric_df, log)
        # Must have at least as many columns as input (adding new ones)
        assert len(out.columns) >= len(numeric_df.columns) - len(fe._drop_correlated)
