"""MLflow experiment tracker for LLM pipelines."""
from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


@dataclass
class LLMExperimentConfig:
    experiment_name: str
    model_name: str
    model_version: str
    prompt_template: str
    temperature: float = 0.1
    max_tokens: int = 1024
    tags: dict[str, str] = field(default_factory=dict)


class LLMOpsTracker:
    """MLflow-based experiment tracker for LLM pipeline runs."""

    def __init__(self, tracking_uri: str, registry_uri: str | None = None) -> None:
        mlflow.set_tracking_uri(tracking_uri)
        if registry_uri:
            mlflow.set_registry_uri(registry_uri)
        self.client = MlflowClient()

    @contextmanager
    def start_run(self, config: LLMExperimentConfig):
        """Context manager for a tracked LLM experiment run."""
        mlflow.set_experiment(config.experiment_name)

        with mlflow.start_run(tags={"model": config.model_name, **config.tags}) as run:
            # Log configuration
            mlflow.log_params({
                "model_name": config.model_name,
                "model_version": config.model_version,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
            })
            mlflow.log_text(config.prompt_template, "prompt_template.txt")

            logger.info("Started MLflow run: %s", run.info.run_id)
            yield run

    def log_inference(
        self,
        run_id: str,
        question: str,
        answer: str,
        latency_ms: float,
        cost_usd: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log a single LLM inference result."""
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics({
                "latency_ms": latency_ms,
                "cost_usd": cost_usd,
            })
            mlflow.log_text(
                json.dumps({"q": question, "a": answer, "meta": metadata or {}}, indent=2),
                f"inferences/{int(time.time())}.json",
            )

    def log_eval_results(
        self,
        run_id: str,
        faithfulness: float,
        answer_relevance: float,
        context_recall: float,
        hallucination_rate: float,
    ) -> None:
        """Log RAGAS evaluation metrics to MLflow."""
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics({
                "eval/faithfulness": faithfulness,
                "eval/answer_relevance": answer_relevance,
                "eval/context_recall": context_recall,
                "eval/hallucination_rate": hallucination_rate,
                "eval/composite_score": (
                    faithfulness * 0.4
                    + answer_relevance * 0.3
                    + context_recall * 0.3
                    - hallucination_rate * 0.5
                ),
            })

    def register_model(
        self,
        run_id: str,
        model_name: str,
        promote_to: str = "Staging",
    ) -> str:
        """Register model in MLflow registry and promote to stage."""
        model_uri = f"runs:/{run_id}/model"
        mv = mlflow.register_model(model_uri, model_name)
        self.client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage=promote_to,
        )
        logger.info("Registered %s v%s to %s", model_name, mv.version, promote_to)
        return mv.version

    def get_best_run(
        self,
        experiment_name: str,
        metric: str = "eval/composite_score",
        ascending: bool = False,
    ) -> dict:
        """Retrieve the best run by metric from an experiment."""
        experiment = self.client.get_experiment_by_name(experiment_name)
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
            max_results=1,
        )
        if not runs:
            return {}
        best = runs[0]
        return {
            "run_id": best.info.run_id,
            "metrics": best.data.metrics,
            "params": best.data.params,
        }
