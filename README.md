# ⚙️ LLMOps Platform: MLflow + Vertex AI

> Production **LLMOps** framework for tracking, versioning, evaluating, and A/B testing LLM pipelines using MLflow + Vertex AI Model Registry — with automated deployment gates.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![MLflow](https://img.shields.io/badge/MLflow-2.10-orange) ![Vertex AI](https://img.shields.io/badge/Vertex%20AI-Model%20Registry-green) ![GCP](https://img.shields.io/badge/GCP-Cloud%20Run-red)

## 🎯 Problem Statement

LLM teams deploy models without tracking prompt versions, evaluating quality regressions, or comparing model variants systematically. This platform brings **engineering rigor to LLM development** — treating prompts, models, and evaluation metrics as first-class versioned artifacts.

## 🏗️ Architecture

```
Experiment Definition
    │
    ▼
[MLflow Tracking Server] ──► Log: prompts, params, metrics
    │
    ▼
[Evaluation Runner] ──► RAGAS / custom evals on golden dataset
    │
    ▼
[A/B Traffic Router] ──► Split traffic: ModelA vs ModelB
    │
    ▼
[Vertex AI Model Registry] ──► Promote winning model to production
    │
    ▼
[Cloud Run Deployment] ──► Auto-deploy with rollback guard
```

## ✨ Key Features

- **Prompt versioning** — track every prompt template change as MLflow artifact
- **Golden dataset evaluation** — automated regression testing on 500+ curated Q&A pairs
- **A/B testing framework** — statistical significance testing for model comparisons
- **Vertex AI integration** — seamless model registration and deployment promotion
- **Deployment gates** — block deployments if eval score drops >5%
- **Streamlit dashboard** — visual comparison of experiment runs
- **Cost tracking** — monitor LLM API costs per experiment

## 📁 Repository Structure

```
src/
├── tracking/
│   ├── mlflow_tracker.py       # MLflow experiment logging
│   ├── prompt_registry.py      # Versioned prompt management
│   └── artifact_store.py      # GCS-backed artifact storage
├── evaluation/
│   ├── evaluator.py            # Multi-metric LLM evaluation
│   ├── golden_dataset.py       # Golden Q&A dataset management
│   └── regression_checker.py  # Automated regression detection
├── ab_testing/
│   ├── traffic_splitter.py     # Request routing for A/B tests
│   ├── statistical_test.py     # Mann-Whitney U / t-test analysis
│   └── winner_selector.py     # Auto-promote winning variant
├── registry/
│   └── vertex_registry.py     # Vertex AI Model Registry integration
└── dashboard/
    └── app.py                 # Streamlit experiment comparison UI
```

## 🚀 Quick Start

```bash
pip install -r requirements.txt

# Start MLflow tracking server
mlflow server --backend-store-uri gs://your-bucket/mlflow \
              --default-artifact-root gs://your-bucket/artifacts \
              --host 0.0.0.0 --port 5000

# Run an experiment
python src/tracking/mlflow_tracker.py \
  --experiment_name "gemini-1.5-pro-v2" \
  --model gemini-1.5-pro \
  --prompt_template templates/rag_v2.txt

# Run evaluation suite
python src/evaluation/evaluator.py \
  --run_id <mlflow_run_id> \
  --golden_dataset data/golden_qa.jsonl

# Launch A/B test
python src/ab_testing/traffic_splitter.py \
  --model_a gemini-1.5-pro \
  --model_b gemini-1.5-flash \
  --traffic_split 50:50 \
  --duration_hours 24
```

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Faithfulness | Answer grounded in context |
| Answer Relevance | Answer addresses the question |
| Context Recall | Retrieved context covers answer |
| Latency P95 | 95th percentile response time |
| Cost per Query | LLM API cost per request |
| Hallucination Rate | % responses with unsupported claims |

## 📄 License

MIT License
