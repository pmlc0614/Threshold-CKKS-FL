# Federated Learning in Python

This directory contains the Python training code for *Communication-Efficient Threshold-CKKS Aggregation for Secure Cross-Silo Federated Learning*. It provides plaintext baselines and encrypted training with the [OpenFHE C++ backend](../ciphertext-FL-cpp/).

| Training mode | Command | C++ backend |
| --- | --- | --- |
| Plaintext Full | `federated-full` | No |
| Plaintext Sparse | `python -m plaintext_fl.federated_sparse` | No |
| Encrypted Full | `encrypted-full` | Yes |
| Encrypted Sparse | `encrypted-sparse` | Yes |

See the [top-level README](../README.md) for the protocol overview, figures, and experimental results.

## Installation

The commands below use Bash. Start in the root of `Threshold-CKKS-FL` with a Python 3.10 or later environment activated.

### 1. Install the Python package

```bash
cd plaintext-FL-python
python -m pip install -e .
```

This installs the training commands and Python dependencies, including PyTorch 2.5.1 and torchvision 0.20.1. For plaintext training, continue to the [CIFAR-10 example](#cifar-10-example).

### 2. Build the C++ backend for encrypted training

Encrypted training also requires CMake 3.16.3 or later, a C++17 compiler, and Boost headers. Run the following commands from `plaintext-FL-python/`:

```bash
python -m pip install 'pybind11==2.13.6'

cmake -S ../ciphertext-FL-cpp -B ../ciphertext-FL-cpp/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_FEDERATED_PYTHON_BINDINGS=ON \
  -DBUILD_UNITTESTS=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_BENCHMARKS=OFF \
  -DPython3_EXECUTABLE="$(command -v python)" \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"

cmake --build ../ciphertext-FL-cpp/build \
  --target ciphertext_fl_cpp -j 4

export PYTHONPATH="$(realpath ../ciphertext-FL-cpp/build/python)${PYTHONPATH:+:$PYTHONPATH}"
```

Check which native module Python loads:

```bash
python -c "import ciphertext_fl_cpp; print(ciphertext_fl_cpp.__file__)"
```

The path should point into `ciphertext-FL-cpp/build/python/`. In a new shell, activate the same Python environment and set `PYTHONPATH` again.

This implementation uses protocol and wire version **v17** and native binding ABI **22**. When updating the code, reinstall the Python package and rebuild the binding from the matching C++ checkout. The `-j 4` option controls compilation jobs. The `--crypto-threads` training option controls runtime parallelism.

## Supported profiles

Encrypted training uses the JSON profiles in [`plaintext_fl/profiles/`](plaintext_fl/profiles/). Each profile specifies the dataset, model, aggregation mode, and cryptographic parameters for a run.

| Dataset / model | Mode | Profile | N | k | B | ρ |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 / ResNet-20 | Full | [`resnet20-cifar10-full-shared-v17.json`](plaintext_fl/profiles/resnet20-cifar10-full-shared-v17.json) | 8192 | 1 | 8192 | 1 |
| CIFAR-10 / ResNet-20 | Sparse | [`resnet20-cifar10-sp-k64-rho020-shared-v17.json`](plaintext_fl/profiles/resnet20-cifar10-sp-k64-rho020-shared-v17.json) | 8192 | 64 | 128 | 0.2 |
| CIFAR-100 / ResNet-32 | Full | [`resnet32-cifar100-full-shared-v17.json`](plaintext_fl/profiles/resnet32-cifar100-full-shared-v17.json) | 8192 | 1 | 8192 | 1 |
| CIFAR-100 / ResNet-32 | Sparse | [`resnet32-cifar100-sp-shared-v17.json`](plaintext_fl/profiles/resnet32-cifar100-sp-shared-v17.json) | 8192 | 4 | 2048 | 0.2 |

`N` is the ring dimension. In Sparse, `B` is the chunk size, `k = N/B` is the maximum number of chunks packed into one RLWE ciphertext, and `ρ` is the target fraction of chunks selected by each client. Full processes dense RLWE blocks directly.

Use a Full profile with `encrypted-full` and a Sparse profile with `encrypted-sparse`. Encrypted training reads the chunk size and transmission ratio from the profile. If supplied on the command line, these values must match the profile. The dataset partition and number of training rounds are selected separately.

## CIFAR-10 example

The examples below train ResNet-20 with eight clients for 200 rounds. All four runs use the same non-IID Dirichlet partition (`α = 1.0`) and training seed (`42`). Run the shared settings first, then choose the training command you want.

### Shared settings

Run this block once in the same Bash session, from `plaintext-FL-python/`. The example uses a CUDA-enabled PyTorch environment.

```bash
mkdir -p runs
export OMP_NUM_THREADS=1
export OMP_DYNAMIC=FALSE

TRAIN_ARGS=(
  --dataset cifar10
  --model resnet20
  --data-dir ./data
  --partition-file partitions/cifar10-n8-dirichlet-alpha1-seed42.json
  --partition-seed 42
  --split noniid
  --alpha 1.0
  --clients 8
  --rounds 200
  --local-epochs 1
  --batch-size 128
  --eval-batch-size 512
  --optimizer sgd
  --learning-rate 0.1
  --minimum-learning-rate 0.0
  --schedule cosine
  --momentum 0.9
  --weight-decay 0.0005
  --seed 42
  --device cuda
  --num-workers 0
)
```

### Plaintext Full

```bash
federated-full "${TRAIN_ARGS[@]}" \
  --weight-mode quantized \
  --weight-denominator 128 \
  --output-dir ./runs/plaintext-full-cifar10
```

### Plaintext Sparse

```bash
python -m plaintext_fl.federated_sparse "${TRAIN_ARGS[@]}" \
  --weight-mode quantized \
  --weight-denominator 128 \
  --rho 0.2 \
  --chunk-size 128 \
  --minimum-cohort-size 3 \
  --residual-beta 1.0 \
  --output-dir ./runs/plaintext-sparse-cifar10
```

These plaintext examples use quantized client weights with denominator 128 to match the encrypted weighting convention. The Sparse example also matches the packaged CIFAR-10 encrypted Sparse profile in chunk size, transmission ratio, and minimum cohort size. For an element-wise plaintext baseline, set `--chunk-size 1`.

### Encrypted Full

```bash
encrypted-full "${TRAIN_ARGS[@]}" \
  --crypto-parameter-profile plaintext_fl/profiles/resnet20-cifar10-full-shared-v17.json \
  --crypto-threads 64 \
  --collusion-bound 1 \
  --comparison-mode none \
  --output-dir ./runs/encrypted-full-cifar10
```

### Encrypted Sparse

```bash
encrypted-sparse "${TRAIN_ARGS[@]}" \
  --crypto-parameter-profile plaintext_fl/profiles/resnet20-cifar10-sp-k64-rho020-shared-v17.json \
  --crypto-threads 64 \
  --collusion-bound 1 \
  --comparison-mode none \
  --residual-beta 1.0 \
  --output-dir ./runs/encrypted-sparse-cifar10
```

`--comparison-mode none` runs encrypted training without computing a separate plaintext reference. To inspect all available options, append `--help` to any of the four training commands.

### Running CIFAR-100

In the shared settings, change the dataset to `cifar100`, the model to `resnet32`, and the partition file to `partitions/cifar100-n8-dirichlet-alpha1-seed42.json`. Execute that block again before launching training, and use a CIFAR-100 output directory.

For encrypted training, select the corresponding CIFAR-100 profile from the [profile table](#supported-profiles). The packaged CIFAR-100 Sparse profile uses **B = 2048** and **ρ = 0.2**. To compare it with plaintext Sparse at the same chunk size, set `--chunk-size 2048` in the plaintext command.

## Runtime and communication

Training runs in a single process that executes clients sequentially. Cryptographic operations can use multiple threads within one client or on the server.

| Metric | Interpretation |
| --- | --- |
| Online runtime | Calculated under a parallel-client execution model using the longest client time in each phase plus server processing time. Setup, evaluation, and network latency are excluded. |
| `round_time` | Wall-clock time observed for a round in the single-host simulation. |
| Online communication | Logical bit-packed protocol payload exchanged during training. The accounting excludes transport framing and acknowledgements. |
| Setup communication | One-time traffic for protocol setup, reported separately from online communication. |

The detailed communication fields are `v17_itemization` and `v17_itemized_communication`. In Sparse, module-to-ring repacking counts as server processing.

## Dropout experiment

Dropout is disabled by default. To run the post-upload dropout experiment, add the following options to an encrypted training command and use a separate output directory:

```bash
--experimental-dynamic-cohort \
--dropout-round-count 20 \
--dropout-seed 42 \
--dropout-target-cap 8
```

With eight clients and 200 rounds, this selects 20 rounds in which one client drops out after all clients have uploaded their updates and before the first collective-decryption request. The server reuses the uploaded ciphertexts and retains the original weights of the surviving clients. Keys are cached by survivor set. In Sparse, a returning client resets its residual state.

The implementation simulates client and server roles and permits one attempt per logical round. The paper reports this dropout experiment separately from its fixed-cohort security analysis.

## Tests

From `plaintext-FL-python/`, run the Python test suite:

```bash
python -m unittest discover -s tests
```

To build and run the native functional tests:

```bash
cmake --build ../ciphertext-FL-cpp/build --target federated1 -j 4
../ciphertext-FL-cpp/build/myexp/federated1 all
```

To run the optional full-dimension integration tests with synthetic updates:

```bash
FL_V17_NATIVE_TEST=1 python -m unittest tests.test_shared_v17_native
```

The functional tests check implementation behavior, including encoding and decryption. For a new experiment, check that model and auxiliary updates satisfy the bounds in the selected profile. The training code applies no clipping, and passing the tests does not certify the security of a parameter choice.