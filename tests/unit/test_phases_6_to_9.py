"""Unit tests for Phases 6–9 and the Pipeline."""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

from mentorml.config import MentorConfig
from mentorml.core.decision import DecisionLog, Severity
from mentorml.explainability.shap_explainer import SHAPExplainer
from mentorml.export.model_exporter import ModelExporter
from mentorml.narrative.narrator import BusinessNarrator
from mentorml.preprocessing.preprocessor import ExplainablePreprocessor
from mentorml.reporting.html_report import HTMLReportGenerator


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> MentorConfig:
    return MentorConfig(cv_folds=2, n_jobs=1)


@pytest.fixture()
def classification_xy() -> tuple[pd.DataFrame, pd.Series]:
    data = load_breast_cancer()
    X = pd.DataFrame(data.data[:80], columns=data.feature_names)
    y = pd.Series(data.target[:80], name="target")
    return X, y


@pytest.fixture()
def fitted_model(classification_xy: tuple[pd.DataFrame, pd.Series]) -> RandomForestClassifier:
    X, y = classification_xy
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)
    return model


# ---------------------------------------------------------------------------
# Phase 6 — SHAPExplainer
# ---------------------------------------------------------------------------


class TestSHAPExplainer:
    def test_explain_returns_dict(
        self,
        config: MentorConfig,
        fitted_model: RandomForestClassifier,
        classification_xy: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = classification_xy
        explainer = SHAPExplainer(config)
        log = DecisionLog()
        result = explainer.explain(fitted_model, X, log, y=y)
        assert isinstance(result, dict)

    def test_feature_importances_sorted(
        self,
        config: MentorConfig,
        fitted_model: RandomForestClassifier,
        classification_xy: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = classification_xy
        explainer = SHAPExplainer(config)
        log = DecisionLog()
        result = explainer.explain(fitted_model, X, log, y=y)
        importances = result["feature_importances"]
        values = [i["importance"] for i in importances]
        assert values == sorted(values, reverse=True)

    def test_all_features_covered(
        self,
        config: MentorConfig,
        fitted_model: RandomForestClassifier,
        classification_xy: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = classification_xy
        explainer = SHAPExplainer(config)
        log = DecisionLog()
        result = explainer.explain(fitted_model, X, log, y=y)
        features_in_result = {i["feature"] for i in result["feature_importances"]}
        assert features_in_result == set(X.columns)

    def test_top_features_is_list(
        self,
        config: MentorConfig,
        fitted_model: RandomForestClassifier,
        classification_xy: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = classification_xy
        explainer = SHAPExplainer(config)
        log = DecisionLog()
        result = explainer.explain(fitted_model, X, log, y=y)
        assert isinstance(result["top_features"], list)
        assert len(result["top_features"]) <= 5

    def test_emits_decision_records(
        self,
        config: MentorConfig,
        fitted_model: RandomForestClassifier,
        classification_xy: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = classification_xy
        explainer = SHAPExplainer(config)
        log = DecisionLog()
        explainer.explain(fitted_model, X, log, y=y)
        assert len(log) >= 2


# ---------------------------------------------------------------------------
# Phase 7 — BusinessNarrator
# ---------------------------------------------------------------------------


class TestBusinessNarrator:
    def test_narrate_returns_string(self) -> None:
        narrator = BusinessNarrator(project_name="Test Project")
        log = DecisionLog()
        narrative = narrator.narrate(log)
        assert isinstance(narrative, str)
        assert len(narrative) > 0

    def test_narrative_contains_project_name(self) -> None:
        narrator = BusinessNarrator(project_name="My Churn Model")
        log = DecisionLog()
        narrative = narrator.narrate(log)
        assert "My Churn Model" in narrative

    def test_narrative_has_sections(self) -> None:
        narrator = BusinessNarrator()
        log = DecisionLog()
        narrative = narrator.narrate(log)
        assert "Executive Summary" in narrative
        assert "Decision Log" in narrative

    def test_narrative_with_quality_report(self) -> None:
        narrator = BusinessNarrator()
        log = DecisionLog()
        quality_report = {
            "overall_severity": Severity.WARNING,
            "issues": [
                {
                    "column": "age",
                    "issue_type": "missing_high",
                    "severity": Severity.WARNING,
                    "detail": "20% missing",
                    "metric": 0.2,
                    "recommendation": "Impute with median",
                }
            ],
            "n_issues_by_severity": {Severity.WARNING: 1, Severity.CRITICAL: 0, Severity.INFO: 0},
            "columns_to_drop": [],
            "columns_to_impute": ["age"],
        }
        narrative = narrator.narrate(log, quality_report=quality_report)
        assert "age" in narrative

    def test_narrative_with_model_selection(self) -> None:
        narrator = BusinessNarrator()
        log = DecisionLog()
        model_result = {
            "best_model_name": "RandomForestClassifier",
            "task_type": "classification",
            "scoring_metric": "roc_auc",
            "cv_scores": {"RandomForestClassifier": 0.95, "LogisticRegression": 0.90},
        }
        narrative = narrator.narrate(log, model_selection_result=model_result)
        assert "RandomForestClassifier" in narrative
        assert "0.9500" in narrative

    def test_narrative_is_valid_markdown(self) -> None:
        narrator = BusinessNarrator()
        log = DecisionLog()
        narrative = narrator.narrate(log)
        assert "##" in narrative or "#" in narrative


# ---------------------------------------------------------------------------
# Phase 8 — HTMLReportGenerator
# ---------------------------------------------------------------------------


class TestHTMLReportGenerator:
    def test_generate_creates_file(self, config: MentorConfig) -> None:
        gen = HTMLReportGenerator(project_name="Test")
        log = DecisionLog()
        with tempfile.TemporaryDirectory() as tmp:
            path = gen.generate(log, output_dir=tmp)
            assert os.path.exists(path)
            assert path.endswith(".html")

    def test_generated_file_is_non_empty(self, config: MentorConfig) -> None:
        gen = HTMLReportGenerator(project_name="Test")
        log = DecisionLog()
        with tempfile.TemporaryDirectory() as tmp:
            path = gen.generate(log, output_dir=tmp)
            assert os.path.getsize(path) > 100

    def test_html_contains_project_name(self) -> None:
        gen = HTMLReportGenerator(project_name="My ML Project")
        log = DecisionLog()
        with tempfile.TemporaryDirectory() as tmp:
            path = gen.generate(log, output_dir=tmp)
            content = open(path, encoding="utf-8").read()
            assert "My ML Project" in content

    def test_html_with_quality_report(self) -> None:
        gen = HTMLReportGenerator(project_name="Test")
        log = DecisionLog()
        quality_report = {
            "overall_severity": "ok",
            "issues": [],
            "n_issues_by_severity": {},
            "columns_to_drop": [],
            "columns_to_impute": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = gen.generate(log, output_dir=tmp, quality_report=quality_report)
            content = open(path, encoding="utf-8").read()
            assert "html" in content.lower()


# ---------------------------------------------------------------------------
# Phase 9 — ModelExporter
# ---------------------------------------------------------------------------


class TestModelExporter:
    def test_export_creates_directory(
        self,
        fitted_model: RandomForestClassifier,
        classification_xy: tuple[pd.DataFrame, pd.Series],
        config: MentorConfig,
    ) -> None:
        X, y = classification_xy
        prep = ExplainablePreprocessor(config)
        log = DecisionLog()
        prep.fit(X, log)
        exporter = ModelExporter(project_name="test_project")
        with tempfile.TemporaryDirectory() as tmp:
            path = exporter.export(
                model=fitted_model,
                preprocessor=prep,
                feature_names=list(X.columns),
                log=log,
                output_dir=tmp,
            )
            assert os.path.isdir(path)

    def test_export_contains_required_files(
        self,
        fitted_model: RandomForestClassifier,
        classification_xy: tuple[pd.DataFrame, pd.Series],
        config: MentorConfig,
    ) -> None:
        X, y = classification_xy
        prep = ExplainablePreprocessor(config)
        log = DecisionLog()
        prep.fit(X, log)
        exporter = ModelExporter(project_name="test_project")
        with tempfile.TemporaryDirectory() as tmp:
            path = exporter.export(
                model=fitted_model,
                preprocessor=prep,
                feature_names=list(X.columns),
                log=log,
                output_dir=tmp,
            )
            files = os.listdir(path)
            assert "model.joblib" in files
            assert "preprocessor.joblib" in files
            assert "feature_names.json" in files
            assert "decisions.json" in files
            assert "predict.py" in files
            assert "manifest.json" in files

    def test_feature_names_json_correct(
        self,
        fitted_model: RandomForestClassifier,
        classification_xy: tuple[pd.DataFrame, pd.Series],
        config: MentorConfig,
    ) -> None:
        X, y = classification_xy
        prep = ExplainablePreprocessor(config)
        log = DecisionLog()
        prep.fit(X, log)
        exporter = ModelExporter(project_name="test_project")
        with tempfile.TemporaryDirectory() as tmp:
            path = exporter.export(
                model=fitted_model,
                preprocessor=prep,
                feature_names=list(X.columns),
                log=log,
                output_dir=tmp,
            )
            with open(os.path.join(path, "feature_names.json")) as f:
                names = json.load(f)
            assert names == list(X.columns)

    def test_decisions_json_is_valid(
        self,
        fitted_model: RandomForestClassifier,
        classification_xy: tuple[pd.DataFrame, pd.Series],
        config: MentorConfig,
    ) -> None:
        X, y = classification_xy
        prep = ExplainablePreprocessor(config)
        log = DecisionLog()
        prep.fit(X, log)
        exporter = ModelExporter(project_name="test_project")
        with tempfile.TemporaryDirectory() as tmp:
            path = exporter.export(
                model=fitted_model,
                preprocessor=prep,
                feature_names=list(X.columns),
                log=log,
                output_dir=tmp,
            )
            with open(os.path.join(path, "decisions.json")) as f:
                decisions = json.load(f)
            assert isinstance(decisions, list)

    def test_emits_export_records(
        self,
        fitted_model: RandomForestClassifier,
        classification_xy: tuple[pd.DataFrame, pd.Series],
        config: MentorConfig,
    ) -> None:
        X, y = classification_xy
        prep = ExplainablePreprocessor(config)
        log = DecisionLog()
        prep.fit(X, log)
        initial_len = len(log)
        exporter = ModelExporter(project_name="test_project")
        with tempfile.TemporaryDirectory() as tmp:
            exporter.export(
                model=fitted_model,
                preprocessor=prep,
                feature_names=list(X.columns),
                log=log,
                output_dir=tmp,
            )
        assert len(log) > initial_len
