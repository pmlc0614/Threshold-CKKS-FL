# Centralized CIFAR-100 ResNet-32 baseline

Run the centralized plaintext baseline from the repository root:

```bash
python -m plaintext_fl.centralized \
  --dataset cifar100 \
  --model resnet32 \
  --epochs 200 \
  --batch-size 128 \
  --eval-batch-size 512 \
  --optimizer sgd \
  --learning-rate 0.1 \
  --minimum-learning-rate 0.0 \
  --schedule cosine \
  --momentum 0.9 \
  --weight-decay 5e-4 \
  --device cuda \
  --num-workers 2 \
  --seed 42 \
  --deterministic \
  --output-dir runs/centralized-cifar100-resnet32
```

Omit `--deterministic` for faster cuDNN autotuning when exact rerun
reproducibility is not required. Add `--no-download` after CIFAR-100 has
already been downloaded under `--data-dir` (default: `./data`).

The projection-shortcut model has 472,756 trainable parameters, 1,232
BatchNorm running-mean values, and 1,232 BatchNorm running-variance values.
Detailed epoch metrics are written to `metrics.csv`; configuration, model
statistics, checkpoints, and the final summary are stored in the selected
output directory.
