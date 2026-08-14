"""Unit tests for Phase 5 — ModelSelector."""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_diabetes

from mentorml.config import MentorConfig
from mentorml.core.decision import DecisionLog
from mentorml.modeling.model_selector import ModelSelector


@pytest.fixture()
def config() -> MentorConfig:
    return MentorConfig(cv_folds=3, n_jobs=1)


@pytest.fixture()
def classification_data() -> tuple[pd.DataFrame, pd.Series]:
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series((data.target == 0).astype(int), name="target")  # binary
    return X, y


@pytest.fixture()
def regression_data() -> tuple[pd.DataFrame, pd.Series]:
    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y


class TestModelSelector:
    def test_select_returns_dict(
        self,
        config: MentorConfig,
        classification_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = classification_data
        selector = ModelSelector(config)
        log = DecisionLog()
        result = selector.select(X, y, log, task_type="classification")
        assert isinstance(result, dict)

    def test_result_has_required_keys(
        self,
        config: MentorConfig,
        classification_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = classification_data
        selector = ModelSelector(config)
        log = DecisionLog()
        result = selector.select(X, y, log, task_type="classification")
        assert "best_model" in result
        assert "best_model_name" in result
        assert "cv_scores" in result
        assert "scoring_metric" in result
        assert "task_type" in result

    def test_best_model_is_fitted(
        self,
        config: MentorConfig,
        classification_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = classification_data
        selector = ModelSelector(config)
        log = DecisionLog()
        result = selector.select(X, y, log, task_type="classification")
        model = result["best_model"]
        predictions = model.predict(X)
        assert len(predictions) == len(X)

    def test_classification_uses_roc_auc(
        self,
        config: MentorConfig,
        classification_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = classification_data
        selector = ModelSelector(config)
        log = DecisionLog()
        result = selector.select(X, y, log, task_type="classification")
        assert result["scoring_metric"] == "roc_auc"

    def test_regression_uses_rmse(
        self,
        config: MentorConfig,
        regression_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = regression_data
        selector = ModelSelector(config)
        log = DecisionLog()
        result = selector.select(X, y, log, task_type="regression")
        assert result["scoring_metric"] == "neg_root_mean_squared_error"

    def test_all_candidates_scored(
        self,
        config: MentorConfig,
        classification_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = classification_data
        selector = ModelSelector(config)
        log = DecisionLog()
        result = selector.select(X, y, log, task_type="classification")
        assert len(result["cv_scores"]) >= 2

    def test_emits_decision_records(
        self,
        config: MentorConfig,
        classification_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = classification_data
        selector = ModelSelector(config)
        log = DecisionLog()
        selector.select(X, y, log, task_type="classification")
        actions = {r.action for r in log}
        assert "model_selection_start" in actions
        assert any("best_model_selected" in a for a in actions)

    def test_best_model_name_is_string(
        self,
        config: MentorConfig,
        classification_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = classification_data
        selector = ModelSelector(config)
        log = DecisionLog()
        result = selector.select(X, y, log, task_type="classification")
        assert isinstance(result["best_model_name"], str)
        assert len(result["best_model_name"]) > 0
