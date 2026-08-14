"""
Integration test — full MentorPipeline end-to-end.

Uses sklearn's breast_cancer dataset (569 samples, 30 features, binary classification).
"""

from __future__ import annotations

import os
import tempfile

import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes

from mentorml.config import MentorConfig
from mentorml.pipeline import MentorPipeline, PipelineResult


@pytest.fixture()
def cancer_df() -> pd.DataFrame:
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df


@pytest.fixture()
def diabetes_df() -> pd.DataFrame:
    data = load_diabetes()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df


@pytest.mark.integration
class TestMentorPipelineIntegration:
    def test_pipeline_runs_classification(self, cancer_df: pd.DataFrame) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = MentorConfig(
                target_column="target",
                task_type="classification",
                cv_folds=2,
                n_jobs=1,
                report_output_dir=tmp,
                export_output_dir=tmp,
            )
            pipeline = MentorPipeline(
                config,
                project_name="Cancer Test",
                generate_report=True,
                export_artefacts=True,
            )
            result = pipeline.fit(cancer_df)
            assert isinstance(result, PipelineResult)

    def test_pipeline_result_has_model(self, cancer_df: pd.DataFrame) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = MentorConfig(
                target_column="target",
                task_type="classification",
                cv_folds=2,
                n_jobs=1,
                report_output_dir=tmp,
                export_output_dir=tmp,
            )
            pipeline = MentorPipeline(config, project_name="Cancer Test")
            result = pipeline.fit(cancer_df)
            assert result.model is not None
            assert isinstance(result.model_name, str)

    def test_pipeline_narrative_is_string(self, cancer_df: pd.DataFrame) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = MentorConfig(
                target_column="target",
                task_type="classification",
                cv_folds=2,
                n_jobs=1,
                report_output_dir=tmp,
                export_output_dir=tmp,
            )
            pipeline = MentorPipeline(config, project_name="Cancer Test")
            result = pipeline.fit(cancer_df)
            assert isinstance(result.narrative, str)
            assert len(result.narrative) > 100

    def test_pipeline_report_file_exists(self, cancer_df: pd.DataFrame) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = MentorConfig(
                target_column="target",
                task_type="classification",
                cv_folds=2,
                n_jobs=1,
                report_output_dir=tmp,
                export_output_dir=tmp,
            )
            pipeline = MentorPipeline(
                config, project_name="Cancer Test", generate_report=True
            )
            result = pipeline.fit(cancer_df)
            if result.report_path:
                assert os.path.exists(result.report_path)
                assert result.report_path.endswith(".html")

    def test_pipeline_export_dir_has_files(self, cancer_df: pd.DataFrame) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = MentorConfig(
                target_column="target",
                task_type="classification",
                cv_folds=2,
                n_jobs=1,
                report_output_dir=tmp,
                export_output_dir=tmp,
            )
            pipeline = MentorPipeline(
                config, project_name="Cancer Test", export_artefacts=True
            )
            result = pipeline.fit(cancer_df)
            if result.export_dir:
                assert os.path.isdir(result.export_dir)
                files = os.listdir(result.export_dir)
                assert "model.joblib" in files

    def test_pipeline_log_has_decisions(self, cancer_df: pd.DataFrame) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = MentorConfig(
                target_column="target",
                task_type="classification",
                cv_folds=2,
                n_jobs=1,
                report_output_dir=tmp,
                export_output_dir=tmp,
            )
            pipeline = MentorPipeline(config, project_name="Cancer Test")
            result = pipeline.fit(cancer_df)
            assert len(result.log) > 10

    def test_pipeline_summary_is_string(self, cancer_df: pd.DataFrame) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = MentorConfig(
                target_column="target",
                task_type="classification",
                cv_folds=2,
                n_jobs=1,
                report_output_dir=tmp,
                export_output_dir=tmp,
            )
            pipeline = MentorPipeline(config, project_name="Cancer Test")
            result = pipeline.fit(cancer_df)
            summary = result.summary()
            assert "MentorPipeline" in summary
            assert result.model_name in summary

    def test_pipeline_regression(self, diabetes_df: pd.DataFrame) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = MentorConfig(
                target_column="target",
                task_type="regression",
                cv_folds=2,
                n_jobs=1,
                report_output_dir=tmp,
                export_output_dir=tmp,
            )
            pipeline = MentorPipeline(
                config,
                project_name="Diabetes Regression",
                generate_report=True,
                export_artefacts=True,
            )
            result = pipeline.fit(diabetes_df)
            assert result.task_type == "regression"
            assert result.model is not None
