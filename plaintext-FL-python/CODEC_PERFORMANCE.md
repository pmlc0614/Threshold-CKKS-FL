# Optional compression timing

Update and rebuild `ciphertext-FL-cpp`'s `ciphertext_fl_cpp` target with the
Python interpreter used for training. Before the normal training command:

```bash
export FL_CODEC_TIMING=1
```

The existing round JSON's `native_diagnostics.codec_work_timings` now preserves
the native breakdown. Each category contains `work_seconds` and `calls` for
native finalization: public dither/offset generation, Gaussian generation,
compression format/CRT conversion, quantization, pack/unpack, code summation
and reconstruction. `code_sum` includes nested `unpack` time. Public polynomial
and Gaussian categories include their existing format conversions.

These are accumulated work times across threads, not elapsed parallel round
time. Do not sum overlapping categories or replace `encrypted_path_parallel`
with their sum. Run one active engine per process for attribution. Setup and
operations outside finalization are excluded. `unset FL_CODEC_TIMING` disables
the diagnostics for performance comparisons. No raw audit files are enabled.

Use binding ABI 22 with the current Python package. Functional codec checks
are retained in the native regression suites.
