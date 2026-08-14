"""Unit tests for Phase 3 — ExplainablePreprocessor."""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np

from mentorml.config import MentorConfig
from mentorml.core.decision import DecisionLog, Severity
from mentorml.core.exceptions import ComponentNotFittedError
from mentorml.preprocessing.preprocessor import ExplainablePreprocessor


@pytest.fixture()
def config() -> MentorConfig:
    return MentorConfig(cardinality_threshold=5)


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "age": rng.integers(18, 80, 50).astype(float),
        "income": rng.uniform(20_000, 100_000, 50),
        "city": rng.choice(["NY", "LA", "SF"], 50),
        "score": rng.uniform(0, 1, 50),
    })
    df.loc[0, "age"] = np.nan
    df.loc[1, "city"] = np.nan
    return df


class TestExplainablePreprocessor:
    def test_transform_without_fit_raises(self, config: MentorConfig, sample_df: pd.DataFrame) -> None:
        prep = ExplainablePreprocessor(config)
        log = DecisionLog()
        with pytest.raises(ComponentNotFittedError):
            prep.transform(sample_df, log)

    def test_fit_returns_self(self, config: MentorConfig, sample_df: pd.DataFrame) -> None:
        prep = ExplainablePreprocessor(config)
        log = DecisionLog()
        result = prep.fit(sample_df, log)
        assert result is prep

    def test_fit_transform_no_missing_after(self, config: MentorConfig, sample_df: pd.DataFrame) -> None:
        prep = ExplainablePreprocessor(config)
        log = DecisionLog()
        prep.fit(sample_df, log)
        out = prep.transform(sample_df, log)
        assert out.isnull().sum().sum() == 0

    def test_categorical_ohe_creates_dummies(self, config: MentorConfig, sample_df: pd.DataFrame) -> None:
        prep = ExplainablePreprocessor(config)
        log = DecisionLog()
        prep.fit(sample_df, log)
        out = prep.transform(sample_df, log)
        # city should be OHE'd (3 unique values <= cardinality_threshold=5)
        ohe_cols = [c for c in out.columns if c.startswith("city__")]
        assert len(ohe_cols) == 3

    def test_numeric_columns_scaled(self, config: MentorConfig, sample_df: pd.DataFrame) -> None:
        prep = ExplainablePreprocessor(config)
        log = DecisionLog()
        prep.fit(sample_df, log)
        out = prep.transform(sample_df, log)
        # After StandardScaler, mean should be near 0
        numeric_out = out.select_dtypes(include="number")
        assert numeric_out.shape[1] > 0

    def test_constant_column_dropped(self, config: MentorConfig) -> None:
        df = pd.DataFrame({
            "x": [1.0] * 50,  # constant
            "y": list(range(50)),
        })
        analysis_result = {
            "column_profiles": {
                "x": {"is_constant": True, "is_id_like": False, "dtype_category": "numeric"},
                "y": {"is_constant": False, "is_id_like": False, "dtype_category": "numeric"},
            }
        }
        prep = ExplainablePreprocessor(config)
        log = DecisionLog()
        prep.fit(df, log, analysis_result=analysis_result)
        out = prep.transform(df, log)
        assert "x" not in out.columns

    def test_id_like_column_dropped(self, config: MentorConfig) -> None:
        df = pd.DataFrame({
            "user_id": list(range(50)),
            "value": list(range(50)),
        })
        analysis_result = {
            "column_profiles": {
                "user_id": {"is_constant": False, "is_id_like": True, "dtype_category": "numeric"},
                "value": {"is_constant": False, "is_id_like": False, "dtype_category": "numeric"},
            }
        }
        prep = ExplainablePreprocessor(config)
        log = DecisionLog()
        prep.fit(df, log, analysis_result=analysis_result)
        out = prep.transform(df, log)
        assert "user_id" not in out.columns

    def test_fit_emits_decisions(self, config: MentorConfig, sample_df: pd.DataFrame) -> None:
        prep = ExplainablePreprocessor(config)
        log = DecisionLog()
        prep.fit(sample_df, log)
        assert len(log) > 0
        actions = {r.action for r in log}
        assert "fit_start" in actions
        assert "fit_complete" in actions

    def test_high_cardinality_uses_label_encoding(self) -> None:
        config = MentorConfig(cardinality_threshold=3)
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "country": rng.choice([str(i) for i in range(20)], 100),
            "value": rng.uniform(0, 1, 100),
        })
        prep = ExplainablePreprocessor(config)
        log = DecisionLog()
        prep.fit(df, log)
        out = prep.transform(df, log)
        assert "country" in out.columns
        assert out["country"].dtype != object
