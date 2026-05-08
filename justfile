# AoE2 LLM Arena — Monorepo Commands

# Run the gameplay agent
agent *ARGS:
    python -m gameplay_agent {{ARGS}}

# Run the detection server
server *ARGS:
    python -m server {{ARGS}}

# Editable install — agent core only.
install:
    pip install -e .

# Editable install with dev tooling (ruff, pytest, pyright, pre-commit).
install-dev:
    pip install -e ".[dev]"

# Editable install + the detection server extras (fastapi, uvicorn).
install-server:
    pip install -e ".[server]"

# Everything for full local work: agent + server + dev tooling.
install-all:
    pip install -e ".[dev,server]"

# Lint with ruff (report only)
lint:
    ruff check .

# Auto-fix lint issues that ruff considers safe
lint-fix:
    ruff check . --fix

# Format with ruff
format:
    ruff format .

# Static type-check with pyright (AI agent code only — see [tool.pyright])
typecheck:
    pyright

# Run tests
test *ARGS:
    pytest {{ARGS}}

# Test coverage report (term-missing format)
coverage:
    pytest --cov --cov-report=term-missing

# Full local quality gate: lint + typecheck + tests
check: lint typecheck test

# Export model to ONNX
export-onnx model="detection/inference/models/aoe2_yolo_v5.pt":
    python -m detection.training.export_onnx --model {{model}}

# Export model to CoreML
export-coreml model="detection/inference/models/aoe2_yolo_v5.pt":
    python -m detection.training.export_coreml --model {{model}}

# Run YOLO training
train model="yolo11n.pt" *ARGS:
    python -m detection.training.train_yolo --model {{model}} {{ARGS}}

# Sync classes.yaml to server bundle
sync-classes:
    cp detection/training/config/classes.yaml server/classes.yaml

# Run scenario evaluations against the executor (~$0.50 for 10 scenarios)
eval *ARGS:
    python -m evaluation.runner {{ARGS}}

# Run all 10 scenarios
eval-all:
    python -m evaluation.runner --all
