# AoE2 LLM Arena — Monorepo Commands

# Run the gameplay agent
agent *ARGS:
    python -m gameplay_agent {{ARGS}}

# Run the detection server
server *ARGS:
    python -m server {{ARGS}}

# Install agent dependencies
install-agent:
    pip install -r gameplay_agent/requirements.txt

# Install server dependencies
install-server:
    pip install -r server/requirements.txt

# Install dev tooling (ruff, pytest)
install-dev:
    pip install -r requirements-dev.txt

# Lint with ruff (report only)
lint:
    ruff check .

# Auto-fix lint issues that ruff considers safe
lint-fix:
    ruff check . --fix

# Format with ruff
format:
    ruff format .

# Run tests
test *ARGS:
    pytest {{ARGS}}

# Full local quality gate: lint + tests
check: lint test

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
