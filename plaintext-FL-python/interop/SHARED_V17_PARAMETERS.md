> Current release: v17, binding ABI 22. See the [execution guide](https://github.com/pmlc0614/plaintext-FL-python/blob/codex/README.md).

# Shared v17 representative execution profiles

The two `resnet32-cifar100-{full,sp}-shared-v17.json` files select the shared
v17 document parameters. Protocol/wire version is **v17**, ABI **22**.
Only v17 profiles are supported; see the README for CIFAR-10 and CIFAR-100 mode-specific defaults.

| Setting | Model | Auxiliary |
| --- | ---: | ---: |
| Exact scale | 2^86 | 2^80 |
| c1 / ca / c2 / cd | 20 / 57 / 16 / 13 | 27 / 61 / 23 / 19 |
| Smudging sigma | 2^78 | 2^72 |
| Delivery scale | 2774 | 3589 |

Shared N=8192; q bits=(48,48); P bits=(60,); sigma_e=3.2.
Full rank=1, selection=1; Sparse rank=4, B=2048, rho=0.2.
Expected Q primes: 281474976694273, 140737488486401;
P: 1152921504606830593. The adapter passes FirstModSize=48 and
ScalingModSize=47 to OpenFHE 1.4.0 (FIXEDMANUAL/BV, depth=1).
The full reproduction settings and reference candidate IDs are in
[SHARED_V17_PARAMETERS.json](SHARED_V17_PARAMETERS.json).

## Selecting a profile

Python: pass `--crypto-parameter-profile
plaintext_fl/profiles/resnet32-cifar100-sp-shared-v17.json` to the existing
encrypted runner (use `full` in the filename for FullSD).
The model is ResNet32/CIFAR100; choose matching existing runner/model options.
C++ candidate files are under `myexp/federated1/profiles/` with the same names.
Large scales use decimal strings to preserve every bit in the numeric records and Python loader.
The C++ JSON files are numeric records; the old candidate loader and measurement CLI have been removed.

## Scope and experiment inputs

The reference certificate assumes n=8, T=200, c=1 and integer weights
(13,15,17,16,14,19,18,16), denominator 128. C++ profiles store these inputs;
Python numeric profiles do not set the dataset partition or training duration.
The training runner derives weights from registered sample counts: configure
those inputs to match before interpreting a run using the reference certificate.
Do not infer that selecting a numeric profile enforces the complete experiment.
Model dimension is 472756; auxiliary dimension is 2464 (1232+1232).
Input bounds are 1.472850219 and 73.02073288; Sparse applies the model bound
to update+residual. No automatic clipping has been added.

S1-S9 reference STATIC-PASS is for the mathematical reference sampler/codec,
not a production native certificate. Native sampler/codec conformance, dataset
training and dropout-extension coverage are separate checks. See DROPOUT_V17.md
for the supported pre-request dropout subset; same-round retries are disabled.

## Integration check

From the Python repository with the built ABI 22 extension on PYTHONPATH:

```sh
PYTHONPATH=../cpp-repo/build/python FL_V17_NATIVE_TEST=1 \
  python3 -m unittest tests.test_shared_v17_native -v
```

This opt-in test loads the actual packaged profiles, uses N=8192 and the full
model/aux dimensions, registers the specified eight weights and executes one
compressed encrypted round per mode. Synthetic updates are checked against a
separate weighted plaintext sum; Sparse checks admitted support and inactive
zeros. Both generated prime arrays must match the witness. Nonzero document
noise is used. Finite-sample RMSE tolerances are integration checks, not a
security certificate or a 200-round/dataset learning experiment.
