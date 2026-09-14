# AIPyCraft Kubernetes Experiments

This repository contains the Kubernetes task-authoring pipeline, the AIPyCraft execution pipeline, benchmark tasks and prompts, and experiment utilities.

## Setup

Use Python 3.10+ from the repository root. Docker Desktop must be running for Kubernetes execution.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-aipycraft-k8s.txt
```

For paid runs, create an ignored `.env` file containing:

```text
OPENROUTER_API_KEY=your_key_here
```

## Task authoring

Plans or generates benchmark task descriptions with the configured writer and review roles.

```powershell
python -m task_authoring.cli plan
python -m task_authoring.cli --difficulty-plan easy:5,medium:5,hard:5,very_hard:5 run --confirm-paid-calls I_ACCEPT_PAID_OPENROUTER_CALLS
```

Generated tasks and their executable index are written under `benchmark/pilot_runs/RUN_ID/`.

## AIPyCraft pipeline

Generates Kubernetes YAML, validates it, deploys it to an isolated Kind cluster, and evaluates the result.

```powershell
python -m aipycraft_k8s.cli plan
python -m aipycraft_k8s.cli prepare
python -m aipycraft_k8s.cli doctor
python -m aipycraft_k8s.cli smoke
python -m aipycraft_k8s.cli run --task all --confirm-paid-calls I_ACCEPT_PAID_OPENROUTER_CALLS
```

To execute tasks from a task-authoring run directly:

```powershell
python -m aipycraft_k8s.cli run --task all --task-index benchmark/pilot_runs/RUN_ID/task-index.json --confirm-paid-calls I_ACCEPT_PAID_OPENROUTER_CALLS
```

## Paired studies

Generates immutable candidates and runs resumable treatment comparisons.

```powershell
python -m aipycraft_k8s.study paired --name example --generator-config benchmark/aipycraft_config.json --treatment-config benchmark/aipycraft_config.json --task easy-001 --replicates 1 --confirm-paid-calls I_ACCEPT_PAID_OPENROUTER_CALLS
python -m aipycraft_k8s.study status --manifest benchmark/study_runs/example/manifest.json
```

## Results and evaluators

Export study rows, build the results ledger, or evaluate an already deployed candidate.

```powershell
python -m aipycraft_k8s.study_analysis --manifest benchmark/study_runs/example/manifest.json --output benchmark/study_results/example
python -m aipycraft_k8s.results_ledger
python -m benchmark.private_tests.run --task easy-001 --context CONTEXT --candidate candidate.yaml
```
