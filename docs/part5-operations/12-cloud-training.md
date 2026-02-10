# Chapter 12: Cloud Training

YOLO training runs on Lambda Labs GPU instances. A single A100 completes 150 epochs in ~60 minutes for ~$1.30.

## 12.1 Why Cloud Training

| Platform | GPU | Time (150 epochs) | Cost |
|----------|-----|-------------------|------|
| M2 Pro (local) | Integrated | ~6.5 days | $0 (electricity) |
| Lambda Labs A100 | A100 40GB SXM4 | ~50-60 min | ~$1.30 |
| Lambda Labs H100 | H100 80GB | ~30 min | ~$1.65 |

Local training on Apple Silicon is impractical for iteration. Cloud GPUs provide 100-200x speedup at negligible cost.

## 12.2 Instance Setup

### SSH Access

```bash
ssh -i ~/Downloads/lambda-aoe2-training.pem ubuntu@<instance-ip>
```

The SSH key `lambda-aoe2-training.pem` is stored locally.

### Base Image

Lambda Stack 22.04 comes with PyTorch and CUDA pre-installed. Two setup steps required:

**1. Fix numpy compatibility:**

```bash
pip install 'numpy<2'
```

numpy 2.x breaks PyTorch's C extensions. This must be done before any training.

**2. Install ultralytics:**

```bash
pip install ultralytics
```

## 12.3 Dataset Upload

Package and upload the hybrid dataset:

```bash
# Local: compress dataset
cd agent/detection
tar -czf training_data_v2.tar.gz training_data_v2/

# Upload to Lambda
scp -i ~/Downloads/lambda-aoe2-training.pem \
    training_data_v2.tar.gz \
    ubuntu@<instance-ip>:~/

# Remote: extract
ssh ... 'cd ~ && tar -xzf training_data_v2.tar.gz'
```

### dataset.yaml Path Fix

The `dataset.yaml` path must be **absolute** on the Lambda instance:

```yaml
# WRONG (fails on Lambda):
path: .

# CORRECT:
path: /home/ubuntu/training_data_v2
train: train/images
val: val/images
```

This is because ultralytics resolves relative paths from its own installation directory, not the current working directory.

## 12.4 Training Execution

Training script template at `tmp/train_v2_lambda.sh`:

```bash
cd /home/ubuntu
python -c "
from ultralytics import YOLO
model = YOLO('yolo11n.pt')
model.train(
    data='/home/ubuntu/training_data_v2/dataset.yaml',
    epochs=150,
    batch=16,
    imgsz=640,
    patience=20,
    name='aoe2_v2',
    # Isometric-tuned augmentation
    flipud=0.0,
    fliplr=0.5,
    degrees=10,
    scale=0.5,
    mosaic=1.0,
    mixup=0.1,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
)
"
```

Training progress is logged to the console with per-epoch metrics: box loss, cls loss, mAP50, mAP50-95.

## 12.5 Model Download

After training completes:

```bash
# Download best weights
scp -i ~/Downloads/lambda-aoe2-training.pem \
    ubuntu@<instance-ip>:~/runs/detect/aoe2_v2/weights/best.pt \
    agent/detection/inference/models/aoe2_yolo_v2.pt

# Optional: export and download ONNX
ssh ... 'python -c "from ultralytics import YOLO; YOLO(\"runs/detect/aoe2_v2/weights/best.pt\").export(format=\"onnx\")"'
scp -i ~/Downloads/lambda-aoe2-training.pem \
    ubuntu@<instance-ip>:~/runs/detect/aoe2_v2/weights/best.onnx \
    agent/detection/inference/models/aoe2_yolo_v2.onnx
```

## 12.6 Cost Management

Lambda Labs charges by the hour. A typical training session:

| Phase | Duration | Cost |
|-------|----------|------|
| Instance startup | 2-3 min | ~$0.05 |
| Dataset upload | 1-2 min | -- |
| Training (150 epochs) | 50-60 min | ~$1.15 |
| Model download + ONNX export | 2-3 min | ~$0.05 |
| **Total** | **~65 min** | **~$1.30** |

Terminate the instance immediately after downloading weights to avoid idle charges.

---

## Summary

- Lambda Labs A100: ~60 min, ~$1.30 per training run
- Setup: `pip install 'numpy<2' ultralytics`
- Dataset.yaml must use absolute paths on Lambda
- Download best.pt and optionally export to ONNX
- Terminate instance immediately after download

## Related Topics

- [Chapter 8: Training Pipeline](../part3-entity-detection/08-training-pipeline.md) -- dataset preparation and hyperparameters
- [Chapter 7: Detector Architecture](../part3-entity-detection/07-detector-architecture.md) -- how the model is loaded at runtime
