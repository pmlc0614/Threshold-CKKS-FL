# Plaintext sparse CIFAR-100/ResNet-32

The plaintext sparse runner supports CIFAR-100 with ResNet-32 while retaining
the same chunk top-k, error-feedback, quantized-weight, and minimum-cohort
paths used by the CIFAR-10 experiment.

Run a bound-profiling experiment without injected noise:

```bash
python -m plaintext_fl.federated_sparse \
  --dataset cifar100 \
  --model resnet32 \
  --rounds 200 \
  --clients 8 \
  --local-epochs 1 \
  --batch-size 128 \
  --eval-batch-size 512 \
  --split iid \
  --weight-mode quantized \
  --weight-denominator 128 \
  --rho 0.3 \
  --chunk-size 1024 \
  --minimum-cohort-size 3 \
  --residual-beta 1.0 \
  --optimizer sgd \
  --learning-rate 0.1 \
  --schedule cosine \
  --momentum 0.9 \
  --weight-decay 0.0005 \
  --device cuda \
  --num-workers 0 \
  --deterministic \
  --output-dir runs/cifar100-resnet32-sparse-profile
```

The final `observed_bounds.json` records the observed aggregate parameter and
auxiliary maxima and their 1.5x recommendations. Supplying bounds alone does
not add noise. A later same-seed noise run is enabled by supplying both the
bounds and precision exponents:

```bash
  --u-max <fixed-parameter-bound> \
  --u-max-aux <fixed-auxiliary-bound> \
  --noise-p 16 \
  --aux-noise-p 16
```

These options inject update-domain Gaussian noise with standard deviations
`U_max * 2^-p` and `U_max_aux * 2^-p_aux`. They do not emulate coefficient-
domain FHE smudging.

Each round prints and records three distinct sparsity ratios. `rho_requested`
is the command-line value, `rho_selected` is the actual per-client selection
ratio after converting rho to an integer chunk count, and `rho_eff_round` is
the accepted client/chunk contribution count divided by `clients *
total_chunk_count`. The minimum-cohort filter affects only the last value.
`rho_eff_running` is its mean over completed rounds, while final `summary.json`
stores that experiment-wide mean as `rho_eff`. `cohort_retention_ratio` is the
fraction of selected client/chunk contributions that survived the filter.

Plaintext communication accounting transmits only valid FP32 model and
auxiliary coordinates; zero padding used by the internal chunk tensors is not
charged. Sparse runs use the v28 support-bitmap codec and require
`0 < rho < 1`.
Configurations whose rounded chunk count selects every model chunk are also
rejected; use `python -m plaintext_fl.federated_full` for fixed-order FULL
training. The `communication_profile` and
`plaintext_payload_padding_included` fields record the accounting mode.

The supported dataset/model pairs are deliberately restricted to
`cifar10/resnet20` and `cifar100/resnet32` so an accidental class-count or
layout mismatch fails before loading data.

## Sparse runtime profiling

Each `client_metrics.csv` row separates local SGD, model-delta generation,
sparsification preparation, chunk-score calculation, top-k selection,
selection validation, residual generation, and selection-bitmap processing.
The per-round CSV/JSONL files contain both the all-client sequential sums and
per-phase maximum-client estimates, plus server-side sparse aggregation,
admission-bitmap, and rejected-residual restoration timings.

The final `summary.json` collects the same values under
`plaintext_sparse_runtime_breakdown`. The `sparsification_total` value is the
measured wall time for the entire error-feedback compression call; its
component phases are diagnostic subdivisions and must not be added to it a
second time. Per-phase client maxima are also diagnostic and are not an
additive critical-path estimate because different clients may be the slowest
in different phases.

The CPU implementation batches cohort counting, weighted sparse aggregation,
compression-residual subtraction, and rejected-chunk restoration with PyTorch
index operations. Bitmap decoding uses the same canonical MSB0 wire format
with a byte lookup instead of scanning every possible bit position. These are
implementation-only fast paths: chunk selection, float64 aggregation,
minimum-cohort admission, residual semantics, and transmitted bytes are
unchanged and are covered by reference-equivalence tests.

## Reusing an exact non-IID assignment

Use `--partition-file` to share exactly the same client sample indices across
plaintext and encrypted experiments. On the first run the file is created;
later runs validate and load it instead of sampling a new partition:

```bash
--split noniid \
--alpha 1.0 \
--partition-seed 42 \
--partition-file partitions/cifar10-n8-dirichlet-alpha1-seed42.json
```

`--partition-seed` is needed only to control creation. Once the file exists it
may be omitted, allowing `--seed` to change model initialization, loader order,
noise, and cryptographic randomness without changing client membership. If
`--partition-seed` is supplied while loading, it must match the seed recorded
in the file. The file also records exact indices, a target-label digest, and an
assignment checksum; dataset, client-count, split, or alpha mismatches fail
before training. Each run's `partition.json` records whether the assignment was
`created`, `loaded`, or generated without a persistent file.
