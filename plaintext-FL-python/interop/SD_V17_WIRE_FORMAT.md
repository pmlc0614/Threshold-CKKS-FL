> Current release: v17, binding ABI 22. See the [execution guide](https://github.com/pmlc0614/plaintext-FL-python/blob/codex/README.md).

# SD v17 byte contract and setup boundary (ABI 22)

The source TeX SHA-256 values are recorded in `SD_V17_BYTE_VECTORS.json`.
This contract separates document-defined cryptographic addresses from local
implementation framing. It does not define authenticated network transport or
serialize a complete distributed ciphertext transcript.

## Document-defined addresses

`|` means concatenation. Integers below are unsigned BE64. Seeds and setup IDs
are 16 raw bytes; tags are ASCII without terminators. SHAKE128 words are read
as BE64 and high-tail rejection is used before reduction modulo the requested
range; reducing every word modulo the range would introduce bias.

| Purpose | Bytes |
| --- | --- |
| Uniform polynomial | `UPOLY | seed | tag | BE64(N) | BE64(prime)` |
| Public key tag | `PK-P1 | setup_id` |
| M2R tag | `M2R-A | setup_id | BE64(j) | BE64(lambda)` |
| Local model/aux A tag | `ENC-FULL-A`, `ENC-SP-A`, or `ENC-AUX-A`, then `setup_id | BE64(rid) | BE64(crypto_client_id) | BE64(block)` |
| DD offset tag | `C2-OFFSET | setup_id | BE64(rid) | path | BE64(instance)` |
| DD dither tag | `C2-DITH | setup_id | BE64(rid) | path | BE64(instance) | BE64(crypto_client_id)` |
| Index PRF input | `IDX-PERM | setup_id | BE64(rid) | BE64(J)` |
| Index shuffle stream | `IDX-FY | kappa | BE64(J)` |

Paths are `FULL-MODEL`, `SP-MODEL`, and `AUX`. The index PRF is HMAC-SHA256
under the client-side Kidx, truncated to its first 16 bytes. Fisher-Yates runs
backwards with unbiased range sampling; J=1 produces identity. Rank-one Sparse
execution skips M2R key switching. A synthetic M2R tag vector is not evidence
that rank-one execution invokes M2R. DD model and auxiliary paths use the same
late root after both aggregate-a stages; fixed-cohort/no-retry rules are unchanged.

Sparse support is exactly ceil(J/8) bytes, bit `7 - (label mod 8)` in byte
`floor(label/8)`. Unused low bits must be zero. Decoding returns ascending labels;
upload support is the exact intersection. Full uses support codec NONE.
Sparse stream coefficients retain positions whose index modulo rank is below
the active-slot count, preserving padding-slot correspondence.

## Implementation-defined containers

Coefficient codes are packed MSB-first, with zero low padding bits in the final
byte. The decoder requires the exact length for the supplied count and width.
The v17 text specifies the logical codes; this byte container is a repository
convention. Existing `FLV17LAY` layout hashes and 184-byte replay headers retain
their little-endian format from `PROTOCOL_CONTRACT_V17.md`; that byte order does
not apply to cryptographic addresses or to the setup envelope below.

The new public setup envelope is:

```text
ASCII("SDV17PUB")                   8 bytes
BE64(1)                            envelope schema (not binding ABI)
setup_id                           16 bytes
public CRS seed                    16 bytes
BE64(descriptor length)
canonical parameter descriptor     descriptor length bytes
BE64(roster count)
for each registered position:
    BE64(position)                 cryptographic client ID
    BE64(external client ID)        application alias
parameter layout SHA-256           32 bytes
running-mean layout SHA-256        32 bytes
running-variance layout SHA-256    32 bytes
```

Descriptor and roster must be nonempty; aliases must be unique. Positions must
be exactly 0..n-1 in registration order. Truncation, trailing bytes, another
schema, and noncanonical positions are rejected. For aliases `(30,10,20,40)`,
crypto IDs are `(0,1,2,3)` and crypto ID zero denotes alias 30. Do not sort aliases
or confuse numerical minimum alias with the protocol's minimum crypto ID.
This does not change the existing registered-weight quantization/tie-break rule.

The envelope contains no Kidx, private x/s/t keys, or server-only alpha weights.
It is an equality binding checked at the Python/native boundary, not a signature
or an authenticated transcript. The adapter reconstructs the descriptor and
complete envelope independently and rejects a mismatch before exposing setup.
`NativeCryptoEngine.public_setup_binding` exposes immutable bytes after success.
The native setup result also exposes `crs_seed` and `setup_descriptor`.

## Canonical parameter descriptor field order

Strings are BE64 byte length followed by ASCII. Reals are finite IEEE-754
binary64 in big endian, after conversion from the native configuration type.
Integers are BE64 except the single-byte support codec and the two scales.
Each scale is a length-prefixed canonical decimal ASCII string in [1, 2^256). Arrays carry a BE64
count followed by their entries. This descriptor grammar is an implementation
choice; the TeX does not prescribe this complete grammar.

1. Strings `FEDERATED-SETUP-DESCRIPTOR-2`, `v17`; mode (Sparse=0, Full=1);
   one-byte support codec (Sparse=1, Full=0).
2. Client count, model dimension, ring dimension; actual Q-prime array,
   actual P-prime array (not merely requested bit sizes).
3. Hybrid large digits, maximum QP bits, MLWE rank, weight denominator.
4. Four real setup noise sigmas: public-key, M2R, client key-switch, encryption;
   string `exact-rational-truncated-dg-v3`.
5. Auxiliary dimension (BE64), model scaling factor (decimal string),
   auxiliary scaling factor (decimal string),
   minimum cohort, collusion bound, error-feedback flag (0/1), model precision,
   auxiliary precision.
6. Eight compression widths: model c1, ca, c2, cd; auxiliary c1, ca, c2, cd.
7. Seven reals: selection ratio, model bound, auxiliary bound, model smudging
   sigma, auxiliary smudging sigma, model delivery scale, auxiliary delivery scale.

## Validation and limits

Both repositories carry identical fixed JSON vectors with synthetic seeds and
keys. The Python stdlib reference and native production primitives are checked
against them: non-byte-aligned packing, high 64-bit addresses, high-tail rejection,
rank one, empty payloads, all model/aux domains, HMAC/Fisher-Yates and support slots.
Native contract tests also check the fixed public envelope hex. Python integration
tests check actual setup in Full, Sparse rank 4 and Sparse rank 1, and reject
modified descriptors, CRS seeds, envelopes, bitmap lengths/padding and code padding.

ABI 21 and earlier extensions are rejected: rebuild the native executable and extension for
ABI 22 together with the Python package. These tests are byte-boundary conformance
checks, not S1-S9 security/precision certification or real dataset training.
A full distributed wire format, ciphertext transport and authenticated channels
remain outside this local pybind implementation.
