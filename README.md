# FullSD / SparseSD v17

Reference implementation for the v17 protocol documents, with Python training
and a native OpenFHE engine. Both repositories must be updated together.
Protocol and wire identifiers are **v17**; native binding ABI is **22**.
Old v12 profiles, binary headers and binding modules are rejected. Existing
experiment records remain historical; do not relabel them as v17 results.

## Installation

Use Python 3.10 or later, CMake, a C++ compiler and Boost headers. In an activated
Python environment, with the repositories in sibling directories:

```bash
cd ~/federated-workspace/plaintext-FL-python
python -m pip install -e .
python -m pip install 'pybind11==2.13.6'
cmake -S ../ciphertext-FL-cpp -B ../ciphertext-FL-cpp/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_FEDERATED_PYTHON_BINDINGS=ON \
  -DBUILD_UNITTESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF \
  -DPython3_EXECUTABLE="$(command -v python)" \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build ../ciphertext-FL-cpp/build --target ciphertext_fl_cpp -j 4
export PYTHONPATH="$(realpath ../ciphertext-FL-cpp/build/python)${PYTHONPATH:+:$PYTHONPATH}"
```

Reinstall the editable Python package after upgrading: this creates the new
`encrypted-full` command. Rebuild C++ to obtain ABI 22; changing PYTHONPATH alone
does not rebuild an old module. The `-j` value controls build jobs, not runtime
crypto threads.

## Supported profiles

Only the following v17 Python profiles are packaged. All names below are under
`plaintext_fl/profiles/`.

| Dataset / model | Mode | Profile | N | k | B | rho |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 / ResNet20 | Full | `resnet20-cifar10-full-shared-v17.json` | 8192 | 1 | 8192 | 1 |
| CIFAR-10 / ResNet20 | Sparse | `resnet20-cifar10-sp-k64-rho020-shared-v17.json` | 8192 | 64 | 128 | 0.2 |
| CIFAR-100 / ResNet32 | Full | `resnet32-cifar100-full-shared-v17.json` | 8192 | 1 | 8192 | 1 |
| CIFAR-100 / ResNet32 | Sparse | `resnet32-cifar100-sp-shared-v17.json` | 8192 | 4 | 2048 | 0.2 |

The default for each mode is its CIFAR-10 profile. Explicit profile paths are
recommended for reproducibility. A Full command rejects a Sparse profile and
vice versa, before loading datasets or setting up cryptography. Full uses direct
RLWE blocks and does not perform sparse selection or R2M/M2R. Its block length
is not a sparsification parameter.

## CIFAR-10 example

```bash
mkdir -p runs partitions
set -o pipefail
OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE \
  encrypted-full \
  --dataset cifar10 --model resnet20 --data-dir ./data \
  --crypto-parameter-profile plaintext_fl/profiles/resnet20-cifar10-full-shared-v17.json \
  --partition-file partitions/cifar10-n8-dirichlet-alpha1-seed42.json \
  --split noniid --alpha 1.0 --partition-seed 42 \
  --clients 8 --rounds 200 --local-epochs 1 \
  --batch-size 128 --eval-batch-size 512 \
  --optimizer sgd --learning-rate 0.1 --schedule cosine --minimum-learning-rate 0 \
  --momentum 0.9 --weight-decay 0.0005 \
  --device cuda --num-workers 0 --crypto-threads 64 --collusion-bound 1 \
  --seed 42 --comparison-mode none \
  --output-dir ./runs/full-cifar10-v17-abi22 \
  2>&1 | tee ./runs/full-cifar10-v17-abi22.log
```

For Sparse, use `encrypted-sparse`, its Sparse profile above, a separate output
path/log, and `--residual-beta 1.0`. For CIFAR-100, select `cifar100`, `resnet32`,
the matching profile, and a CIFAR-100 partition path. Preserve the same partition
file when comparing Full and Sparse on the same dataset. The profile does not
set the number of rounds or dataset partition. Sample counts determine integer
client weights with fixed denominator 128.

## Runtime and measurement scope

Clients are simulated sequentially in one process. Crypto threads parallelize
work within one client, or within the server; they do not mix client batches.
Sparse M2R schedules `(group, component)` jobs in the existing pool and applies
ModDown separately for each source before reduction. All M2R phases count as
server work. The independent-client latency estimate adds phase-wise client
maxima and server work; evaluation and network latency are excluded.
`round_time` is actual single-host wall time and is reported separately.

`--comparison-mode none` does not compute a plaintext answer. Explicit comparison
modes and test-only phase diagnostics retain correctness oracles. Report fields
use `v17_itemization` and `v17_itemized_communication`. Communication is logical
bit-packed cryptographic payload, not measured socket traffic; control-plane
framing and ACK bytes are excluded. Setup traffic is reported separately.

No clipping is added. Check the selected model/auxiliary bounds for the intended
experiment. A profile is not a runtime STATIC-PASS certificate. Old candidate
screening and empirical acceptance tools have been removed; remaining numerical
checks test codec/decryption correctness and do not certify security parameters.

## Dropout subset

Dropout is off by default. The supported optional experiment excludes one client
after all uploads and before the first decryption request. Use
`--experimental-dynamic-cohort --dropout-round-count 20 --dropout-seed 42
--dropout-target-cap 8` for 20 affected rounds with eight clients. Target keys
are cached by exact survivor set, and weights are not renormalized.

This implementation fixes the attempt cap at one: no same-logical-round retry.
The broader v17 extension permits bounded pre-request retries, but they are not
implemented here. Post-request retries remain forbidden. Offline Sparse clients
reset residuals on rejoin. ACKs are simulated; this is not an authenticated
network deployment or a complete implementation of all dropout extension cases.

## Tests

```bash
cmake --build ../ciphertext-FL-cpp/build --target federated1 -j 4
../ciphertext-FL-cpp/build/myexp/federated1 all
python -m unittest discover -s tests
```

Native tests are an explicit build target, not part of the default training
build. Small test fixtures and plaintext oracles are not public experiment
profiles. `FL_V17_NATIVE_TEST=1 python -m unittest tests.test_shared_v17_native`
opts into expensive full-dimension profile integration tests. They use synthetic
updates, not a 200-round accuracy experiment. Plaintext baseline runners remain
available separately (`federated-full` and `python -m plaintext_fl.federated_sparse`).
