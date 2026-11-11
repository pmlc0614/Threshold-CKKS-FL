> Current release: v17, binding ABI 22. See the [execution guide](https://github.com/pmlc0614/plaintext-FL-python/blob/codex/README.md).

# FL protocol v17 interoperability contract

This directory defines the current contract shared by the Python orchestrator
and the C++ native engine.

The Python process is the top-level FL orchestrator.  The C++ component is a
stateful cryptographic protocol engine.  pybind11 is the runtime transport;
the binary framing below is for golden tests, replay, and debugging.

## Version and primitive types

| Field | Contract |
| --- | --- |
| Protocol | `v17` |
| Schema | `1.0` |
| Binding ABI | `22` |
| Replay/layout byte order | little endian; crypto addresses/public setup use BE64 |
| Model/auxiliary wire values | IEEE-754 `float32` |
| IDs, dimensions, offsets, lengths | unsigned 64-bit |
| Weight denominator | `128` |
| Hash | SHA-256 over canonical bytes |

Native C++ structs, `size_t`, pointer values, and compiler-dependent padding
must never be serialized across the boundary.

## Three independent layouts

The ResNet-20 state is represented by three layouts and three hashes:

1. `ParameterLayout`: trainable parameters only.
2. `RunningMeanLayout`: BatchNorm `running_mean` entries only.
3. `RunningVarianceLayout`: BatchNorm `running_var` entries only.

`num_batches_tracked` and other non-floating buffers are not transmitted.
Model parameters and auxiliary values are never concatenated into one
plaintext.  Each entry contains UTF-8 name, shape, offset, element count, and
wire dtype.  Entries are contiguous, unique, and in the deterministic PyTorch
ordering chosen by the Python model adapter.

Canonical layout bytes are:

```text
"FLV17LAY"                    8 bytes
schema_major                  uint16
schema_minor                  uint16
layout_kind                   uint8 (1 parameter, 2 mean, 3 variance)
model_name_length             uint32
model_name                    UTF-8 bytes
entry_count                   uint64
for each entry:
  name_length                 uint32
  name                        UTF-8 bytes
  dtype                       uint8 (1 float32)
  rank                        uint32
  shape[rank]                 uint64 each
  offset                      uint64
  numel                       uint64
```

The layout identifier is lowercase hexadecimal SHA-256 of these bytes.

## Registered integer weights

For registered client `i`, let `n_i` be its positive local training-example
count.  Compute the exact rational quota `128*n_i/sum(n)`; take every floor,
then give remaining units to larger rational remainders.  Equal remainders are
resolved by ascending numeric client ID.  The result remains aligned with the
input client order.

The setup is valid only if every result satisfies `alpha_i >= 1` and the sum is
exactly 128.  A setup that would produce zero is rejected.  Weights are fixed
for the setup and are not renormalized for survivors, sparse cohorts, labels,
blocks, or dropout.

## Sparse two-phase boundary

`update_bound` and `aux_update_bound` bound the admitted integer-weighted
aggregate after division by the fixed weight denominator. They are not
per-client clipping or admission bounds. Individual float32 payloads must be
finite; the native coefficient codec checks them against the actual scaled
centered-Q representation. A privileged single-machine observer may validate
the plaintext aggregate after cohort admission, but is not part of the
label-only server interface.

1. Python trains locally and sends selected actual chunk IDs only to the local
   C++ client role.
2. The C++ client maps actual IDs to shuffled labels.  Only labels enter the
   server-public admission state.
3. The C++ server checks the entire registered cohort and key-switch-key
   readiness, then performs minimum-contributor admission per label.
4. Each local C++ client maps accepted labels back to actual IDs and returns
   those IDs to its Python trainer.
5. Python submits values only for the accepted actual chunks.  Running-mean and
   running-variance deltas use their independent direct-RLWE streams.

Actual chunk IDs and the secret permutation are forbidden in server-public
messages and state.

## Round state machine

```text
CREATED -> SELECTION_COLLECTING -> ADMISSION_FINALIZED
        -> VALUES_COLLECTING -> ENCRYPTED -> DECRYPTING -> COMMITTED
```

Each nonterminal phase may transition to `ABORTED`.  `COMMITTED` and `ABORTED`
are terminal.  Python model, optimizer-visible model state, auxiliary state,
and error-feedback residual stay pending until `COMMITTED`.

The default requires the registered cohort. Optional pre-request dropout freezes
a survivor set before the first decryption request; see [DROPOUT_V17.md](DROPOUT_V17.md).
The implementation permits one attempt per logical round and no retry.

## Binary replay header

The fixed 184-byte header is:

```text
magic                         8 bytes: "FLV17BIN"
schema_major                  uint16
schema_minor                  uint16
message_type                  uint16
flags                         uint16
setup_id                      16 bytes
round_id                      uint64
client_id                     uint64 (UINT64_MAX means server-global)
parameter_layout_sha256       32 bytes
running_mean_layout_sha256    32 bytes
running_variance_layout_sha256 32 bytes
payload_length                uint64
payload_sha256                32 bytes
```

Message types are setup manifest, client selection, admission result, accepted
chunk values, auxiliary delta, and round result.  Unknown schema versions,
message types, flags, lengths, hashes, or layout identifiers fail closed.

## Golden vectors

`golden-v1.json` is byte-identical in both repositories.  Python and C++ tests
must reproduce every expected alpha, layout hash, and binary-header byte.  A
change to a vector requires a schema or ABI review; silently updating only one
repository is forbidden.

## Current key domains and compatibility

Full and Sparse independently sample local encryption secrets x_i and collective
secrets s_i. Sparse also samples independent M2R target secrets t_i. Client
key switching maps x_i to s_S; Sparse M2R targets t_S. Auxiliary streams use
their own protocol domains.

ABI 22 requires rebuilding the extension together with the Python package.
Earlier ABI modules and v12 wire/layout identifiers are rejected. See
[SD_V17_WIRE_FORMAT.md](SD_V17_WIRE_FORMAT.md) for exact byte encodings.
