# Threshold-CKKS Aggregation in C++

This directory contains the OpenFHE-based C++ backend for *Communication-Efficient Threshold-CKKS Aggregation for Secure Cross-Silo Federated Learning*. The [Python training code](../plaintext-FL-python/) handles datasets, local training, and evaluation. It calls the C++ backend to encrypt client updates, aggregate them on the server, and collectively decrypt the aggregate.

| Goal | CMake target | Build output |
| --- | --- | --- |
| Run encrypted training from Python | `ciphertext_fl_cpp` | Python module in `build/python/` |
| Run the C++ functional tests | `federated1` | Executable at `build/myexp/federated1` |

If you have already built the backend using the [Python installation instructions](../plaintext-FL-python/README.md#installation), the same build can be used here. From `ciphertext-FL-cpp/`, continue to [Verify the Python binding](#verify-the-python-binding) or [Tests](#tests).

## Implementation

The Full and Sparse profiles use the same Python binding. The training command and its parameter profile select the aggregation path.

| Component | Role | Source |
| --- | --- | --- |
| Full aggregation | Aggregates dense model updates in packed ring learning with errors (RLWE) ciphertexts. | [`protocol-full-model-path.cpp`](myexp/federated1/protocol-full-model-path.cpp) |
| Sparse aggregation | Aligns chunks selected by different clients and aggregates contributions at matching model positions. | [`protocol-sparse-path.cpp`](myexp/federated1/protocol-sparse-path.cpp) |
| Ring and module conversion | Converts packed RLWE ciphertexts into module learning with errors (MLWE) ciphertexts for chunk-wise aggregation, then repacks the aggregated chunks into RLWE ciphertexts. | [`protocol-ring-conversion.cpp`](myexp/federated1/protocol-ring-conversion.cpp) |
| Collective decryption | Combines clients' partial-decryption responses to recover the aggregate. | [`protocol-v17-decryption.cpp`](myexp/federated1/protocol-v17-decryption.cpp) |
| Python binding | Exposes the native engine and its version information to the Python training code. | [`python-bindings.cpp`](myexp/federated1/python-bindings.cpp) |

See the [top-level README](../README.md) for the protocol diagram and experimental results.

## Installation

The commands below use Bash. They assume that `ciphertext-FL-cpp/` and `plaintext-FL-python/` are sibling directories under `Threshold-CKKS-FL/`.

Build requirements:

- Python 3.10 or later in an activated environment, with Python development headers
- CMake 3.16.3 or later
- A C++17 compiler with OpenMP support
- Boost headers

### Build the Python binding

From the root of `Threshold-CKKS-FL`, run:

```bash
cd ciphertext-FL-cpp
python -m pip install 'pybind11==2.13.6'

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_FEDERATED_PYTHON_BINDINGS=ON \
  -DBUILD_UNITTESTS=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_BENCHMARKS=OFF \
  -DPython3_EXECUTABLE="$(command -v python)" \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"

cmake --build build --target ciphertext_fl_cpp -j 4
```

The `-j 4` option sets the number of parallel compilation jobs. For encrypted training, use the Python option `--crypto-threads` to configure parallelism in the native engine.

### Verify the Python binding

From `ciphertext-FL-cpp/`, add the build output to `PYTHONPATH` and import the module:

```bash
export PYTHONPATH="$(realpath build/python)${PYTHONPATH:+:$PYTHONPATH}"

python -c 'import ciphertext_fl_cpp as m; print(m.__file__); print(m.PROTOCOL_VERSION, m.BINDING_ABI_VERSION)'
```

The first line should show the path to the compiled module under `ciphertext-FL-cpp/build/python/`. The second line should show:

```text
v17 22
```

`v17` is the protocol version, and `22` is the native binding ABI version expected by the Python training code. When opening a new shell, activate the same Python environment and set `PYTHONPATH` again.

## Running encrypted training

Start training from [`plaintext-FL-python/`](../plaintext-FL-python/) after installing the Python package and verifying the native binding. The [CIFAR-10 example](../plaintext-FL-python/README.md#cifar-10-example) provides the training commands.

| Mode | Python command | Profile to select |
| --- | --- | --- |
| Full | `encrypted-full` | Full profile for the chosen dataset and model |
| Sparse | `encrypted-sparse` | Sparse profile for the chosen dataset and model |

For `--crypto-parameter-profile`, use a JSON file from [`plaintext-FL-python/plaintext_fl/profiles/`](../plaintext-FL-python/plaintext_fl/profiles/). The [supported profiles table](../plaintext-FL-python/README.md#supported-profiles) lists the dataset, model, chunk size, and transmission ratio for each file. The C++ directory [`myexp/federated1/profiles/`](myexp/federated1/profiles/) contains the native parameter records.

## Tests

From `ciphertext-FL-cpp/`, build the native test executable and run all functional suites:

```bash
cmake --build build --target federated1 -j 4
./build/myexp/federated1 all
```

To list the available suites:

```bash
./build/myexp/federated1 --help
```

To run individual suites for ring conversion or threshold CKKS:

```bash
./build/myexp/federated1 ring
./build/myexp/federated1 threshold
```

The full suite also covers coefficient encoding, protocol setup, Full and Sparse aggregation, and the native engine interface used by the Python training code.

## Dropout support

The optional dropout experiment excludes one client after all clients have uploaded their ciphertexts and before the first collective-decryption request. The server reuses the uploaded ciphertexts to aggregate over the surviving clients. The implementation simulates client and server roles in one process and uses one attempt per logical round.

See the [Python README](../plaintext-FL-python/README.md) for the dropout options and training commands.

## OpenFHE and acknowledgments

This implementation builds on [OpenFHE](https://github.com/openfheorg/openfhe-development). We thank the OpenFHE developers and contributors for making the library available. The threshold-CKKS federated aggregation code for this project is in [`myexp/federated1/`](myexp/federated1/).

- [OpenFHE documentation](https://openfhe-development.readthedocs.io/en/latest/)
- [OpenFHE: Open-Source Fully Homomorphic Encryption Library](https://eprint.iacr.org/2022/915)
- [OpenFHE project acknowledgments](https://github.com/openfheorg/openfhe-development#acknowledgments)
- [OpenFHE BSD 2-Clause License](LICENSE)