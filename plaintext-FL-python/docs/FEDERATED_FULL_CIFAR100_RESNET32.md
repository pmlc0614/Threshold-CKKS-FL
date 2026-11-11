# Plaintext FULL CIFAR-100/ResNet-32

`plaintext_fl.federated_full` runs genuine dense FedAvg. It sends each full
FP32 floating-point model-state delta in fixed layout order. It does not split
the model into chunks and therefore does not charge or create chunk indices,
labels, padding, residuals, or cohort metadata.

The following command matches the eight-client non-IID partition and
quantized client weights used by the protocol experiments:

```bash
python -m plaintext_fl.federated_full \
  --dataset cifar100 \
  --model resnet32 \
  --rounds 200 \
  --clients 8 \
  --local-epochs 1 \
  --batch-size 128 \
  --eval-batch-size 512 \
  --split noniid \
  --alpha 1.0 \
  --partition-file partitions/cifar100-n8-dirichlet-alpha1-seed42.json \
  --partition-seed 42 \
  --weight-mode quantized \
  --weight-denominator 128 \
  --optimizer sgd \
  --learning-rate 0.1 \
  --schedule cosine \
  --momentum 0.9 \
  --weight-decay 0.0005 \
  --device cuda \
  --output-dir runs/cifar100-resnet32-full
```

To reproduce the FULL plaintext comparison with protocol-domain Gaussian
noise, add the fixed design bounds and exponents:

```bash
  --u-max 0.5507141365 \
  --u-max-aux 73.02073288 \
  --noise-p 12 \
  --aux-noise-p 18
```

This applies `sigma = U_max * 2^-p` independently to every aggregated
trainable-parameter coordinate. It applies
`sigma_aux = U_max_aux * 2^-p_aux` using independent random streams for the
aggregated BatchNorm `running_mean` and `running_var` coordinates. Omitting an
exponent disables its corresponding noise stream.

Per-round and cumulative communication in `summary.json` uses exact unpadded
FP32 payload sizes. Uplink is the sum over all clients; logical downlink is one
model copy; unicast downlink is one copy per client. The console reports
all-client uplink and all-unicast downlink. `online_runtime_parallel` is the
slowest complete client phase plus server aggregation/update time; evaluation
and network time are excluded.
