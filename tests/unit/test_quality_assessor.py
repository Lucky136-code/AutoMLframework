"""Unit tests for Phase 2 — DataQualityAssessor."""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np

from mentorml.config import MentorConfig
from mentorml.core.decision import DecisionLog, Severity
from mentorml.core.exceptions import DataValidationError
from mentorml.quality.quality_assessor import (
    DataQualityAssessor,
    DuplicateInspector,
    MissingValueInspector,
    OutlierInspector,
    QualityIssue,
)


@pytest.fixture()
def config() -> MentorConfig:
    return MentorConfig(missing_threshold=0.4)


@pytest.fixture()
def clean_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "age": rng.integers(18, 80, 100).astype(float),
        "income": rng.uniform(20_000, 100_000, 100),
        "city": rng.choice(["A", "B", "C"], 100),
    })


# ---------------------------------------------------------------------------
# QualityIssue
# ---------------------------------------------------------------------------

class TestQualityIssue:
    def test_to_dict_has_required_keys(self) -> None:
        issue = QualityIssue(
            column="age",
            issue_type="missing_high",
            severity=Severity.WARNING,
            detail="30% missing",
            metric=0.3,
            recommendation="Impute with median",
        )
        d = issue.to_dict()
        assert set(d.keys()) == {
            "column", "issue_type", "severity", "detail", "metric", "recommendation"
        }

    def test_none_metric_stays_none(self) -> None:
        issue = QualityIssue(
            column=None,
            issue_type="duplicate_rows",
            severity=Severity.WARNING,
            detail="5 dupes",
        )
        assert issue.to_dict()["metric"] is None


# ---------------------------------------------------------------------------
# MissingValueInspector
# ---------------------------------------------------------------------------

class TestMissingValueInspector:
    def test_no_issues_when_clean(self, clean_df: pd.DataFrame, config: MentorConfig) -> None:
        log = DecisionLog()
        issues = MissingValueInspector.inspect(clean_df, config, log)
        assert issues == []
        assert len(log) == 0

    def test_critical_when_above_threshold(self, config: MentorConfig) -> None:
        df = pd.DataFrame({"a": [np.nan] * 60 + [1.0] * 40})
        log = DecisionLog()
        issues = MissingValueInspector.inspect(df, config, log)
        assert len(issues) == 1
        assert issues[0].severity == Severity.CRITICAL
        assert issues[0].issue_type == "missing_critical"

    def test_warning_between_5_and_threshold(self, config: MentorConfig) -> None:
        df = pd.DataFrame({"a": [np.nan] * 20 + [1.0] * 80})
        log = DecisionLog()
        issues = MissingValueInspector.inspect(df, config, log)
        assert len(issues) == 1
        assert issues[0].severity == Severity.WARNING

    def test_info_below_5_percent(self, config: MentorConfig) -> None:
        df = pd.DataFrame({"a": [np.nan] * 3 + [1.0] * 97})
        log = DecisionLog()
        issues = MissingValueInspector.inspect(df, config, log)
        assert len(issues) == 1
        assert issues[0].severity == Severity.INFO

    def test_emits_decision_records(self, config: MentorConfig) -> None:
        df = pd.DataFrame({"a": [np.nan] * 20 + [1.0] * 80})
        log = DecisionLog()
        MissingValueInspector.inspect(df, config, log)
        assert len(log) == 1


# ---------------------------------------------------------------------------
# OutlierInspector
# ---------------------------------------------------------------------------

class TestOutlierInspector:
    def test_no_outliers_in_normal_data(self) -> None:
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"x": rng.normal(0, 1, 200)})
        log = DecisionLog()
        issues = OutlierInspector.inspect(df, log)
        # Normal distribution should have < 10% outliers by IQR
        assert len(issues) == 0

    def test_outliers_detected_in_heavy_tail(self) -> None:
        vals = list(range(90)) + [10000, 20000, 30000, -5000, -6000,
                                  50000, 100000, 200000, 300000, 400000,
                                  500000]
        df = pd.DataFrame({"x": vals})
        log = DecisionLog()
        issues = OutlierInspector.inspect(df, log)
        assert len(issues) == 1
        assert issues[0].issue_type == "outliers_high"

    def test_zero_iqr_skipped(self) -> None:
        df = pd.DataFrame({"x": [5.0] * 100})
        log = DecisionLog()
        issues = OutlierInspector.inspect(df, log)
        assert len(issues) == 0

    def test_non_numeric_columns_skipped(self) -> None:
        df = pd.DataFrame({"cat": ["a", "b", "c"] * 33 + ["d"]})
        log = DecisionLog()
        issues = OutlierInspector.inspect(df, log)
        assert issues == []


# ---------------------------------------------------------------------------
# DuplicateInspector
# ---------------------------------------------------------------------------

class TestDuplicateInspector:
    def test_no_duplicates(self, clean_df: pd.DataFrame) -> None:
        log = DecisionLog()
        issues = DuplicateInspector.inspect(clean_df, log)
        assert issues == []

    def test_duplicates_detected(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 1, 2, 3]})
        log = DecisionLog()
        issues = DuplicateInspector.inspect(df, log)
        assert len(issues) == 1
        assert issues[0].issue_type == "duplicate_rows"
        assert issues[0].metric == pytest.approx(2 / 5)

    def test_critical_when_high_duplicate_rate(self) -> None:
        df = pd.DataFrame({"a": [1] * 60 + [2] * 40})
        log = DecisionLog()
        issues = DuplicateInspector.inspect(df, log)
        assert issues[0].severity == Severity.CRITICAL


# ---------------------------------------------------------------------------
# DataQualityAssessor
# ---------------------------------------------------------------------------

class TestDataQualityAssessor:
    def test_empty_df_raises(self, config: MentorConfig) -> None:
        assessor = DataQualityAssessor(config)
        log = DecisionLog()
        with pytest.raises(DataValidationError):
            assessor.analyze(pd.DataFrame(), log)

    def test_clean_data_returns_ok(self, clean_df: pd.DataFrame, config: MentorConfig) -> None:
        assessor = DataQualityAssessor(config)
        log = DecisionLog()
        report = assessor.analyze(clean_df, log)
        assert report["overall_severity"] == "ok"
        assert report["issues"] == []

    def test_report_has_required_keys(self, clean_df: pd.DataFrame, config: MentorConfig) -> None:
        assessor = DataQualityAssessor(config)
        log = DecisionLog()
        report = assessor.analyze(clean_df, log)
        required = {"n_rows", "n_cols", "issues", "n_issues_by_severity",
                    "overall_severity", "columns_to_drop", "columns_to_impute"}
        assert required.issubset(report.keys())

    def test_columns_to_drop_identified(self, config: MentorConfig) -> None:
        df = pd.DataFrame({
            "good": [1.0] * 100,
            "bad": [np.nan] * 60 + [1.0] * 40,
        })
        assessor = DataQualityAssessor(config)
        log = DecisionLog()
        report = assessor.analyze(df, log)
        assert "bad" in report["columns_to_drop"]

    def test_columns_to_impute_identified(self, config: MentorConfig) -> None:
        df = pd.DataFrame({
            "x": [np.nan] * 10 + [1.0] * 90,
        })
        assessor = DataQualityAssessor(config)
        log = DecisionLog()
        report = assessor.analyze(df, log)
        assert "x" in report["columns_to_impute"]

    def test_emits_start_and_complete_records(self, clean_df: pd.DataFrame, config: MentorConfig) -> None:
        assessor = DataQualityAssessor(config)
        log = DecisionLog()
        assessor.analyze(clean_df, log)
        actions = {r.action for r in log}
        assert "quality_assessment_start" in actions
        assert "quality_assessment_complete" in actions

    def test_n_rows_n_cols_correct(self, clean_df: pd.DataFrame, config: MentorConfig) -> None:
        assessor = DataQualityAssessor(config)
        log = DecisionLog()
        report = assessor.analyze(clean_df, log)
        assert report["n_rows"] == len(clean_df)
        assert report["n_cols"] == len(clean_df.columns)
