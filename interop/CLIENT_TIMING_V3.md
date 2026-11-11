> Current release: v17, binding ABI 22. See the [execution guide](https://github.com/pmlc0614/plaintext-FL-python/blob/codex/README.md).

# Client timing v3: separate executor contention from batch service

Runtime/wire v17 and binding ABI 22 are the current versions. Rebuild the native module
and update the Python runner together. Native diagnostics identify these
semantics with `crypto_timing_model: client-batch-service-v3`; old modules
remain labelled `legacy-wait-inclusive` by the Python adapter.

Model and auxiliary paths have separate caller flows for the shared late-seed
barrier, but share one bounded ciphertext executor. Previously its batch timer
started before acquiring the executor mutex. A path waiting for another path's
batch therefore reported that wait as its own operation time. Phase maxima
could include the same physical interval more than once.

Each stage now reports:

- `wall_seconds`: batch service elapsed time after acquiring the executor;
  includes dispatch, task execution and completion notification, not pure CPU time.
- `executor_wait_seconds`: elapsed time before acquiring the executor.
- `elapsed_including_executor_wait_seconds`: sum of the preceding two.
- `work_seconds`: sum of runner elapsed times, not client latency.

The service timer drives per-client encryption/key-switch/partial-decryption
vectors and their phase maxima. Both model and auxiliary service times are
summed for the same client before taking a maximum. Client batches are never
merged across client IDs. One-worker executors now also serialize independent
batches; recursive work stays inline. The shared pool's configured budget is
unchanged. Server batches retain their own metrics.

The measured whole native/round elapsed time is unchanged in definition.
The native elapsed calculation also fixes a double count: uploads prepared
inside `FinalizeRound` are already in its elapsed time. Only upload time
cached before entering that call is added separately. This corrects the
server/coordination residual as well as the parallel estimate.
Native server/coordination residual remains whole elapsed minus the sum of
client service times (clamped at zero), not a sum of caller wait intervals.
Do not subtract executor wait from whole elapsed: it overlaps another batch's
service. The Python estimate remains the sum of protocol phase maxima and
server/coordination time, with the existing whole-path cap. It is an estimate
for independent clients with a per-client worker budget, not a distributed run.
It excludes network latency. Schema v3 distinguishes it from old results.

Console output reports model and auxiliary maximum partial-decryption waits
separately. Per-client/per-stage wait values remain in each round's native
diagnostics. Maxima of service and wait need not belong to the same client,
so adding the two printed maxima does not reconstruct any one client's time.

This is a measurement correction, not a claim of faster cryptographic work.
No sampler, security parameter, ciphertext, cohort or retry rule changes.
Regression coverage forces contention at worker counts 1 and 4, checks batch
non-overlap, checks separate service/wait, resets, and recursive executor use.

Validation: crypto-execution and protocol-external-engine C++ suites passed;
43 Python timing/communication/native/training tests passed. Actual CIFAR-100
subset runs completed two FullSD and two SparseSD rounds using v17 profiles,
four crypto workers and the local OpenMP-disabled build. Checkpoints and Sparse
residuals reloaded successfully. All four native elapsed values were bounded by
the enclosing finalize call, and all combined client partial times equalled
model plus auxiliary service for that same client. In Sparse round 2, model
partial service max was 0.644 s, auxiliary 0.130 s, and auxiliary executor wait
max 4.761 s. This is evidence of timing attribution, not a server speedup claim.

After pulling both codex branches, rebuild `ciphertext_fl_cpp`, then start a
new run/output directory. Keep the same thread settings for a server comparison.
Existing running processes and saved old logs are not retroactively corrected.
