> Current release: v17, binding ABI 22. See the [execution guide](https://github.com/pmlc0614/plaintext-FL-python/blob/codex/README.md).

# v17 post-upload dropout experiment (ABI 22)

FullSD and SparseSD accept a scheduled single-client dropout after all model
and auxiliary uploads and before the first decryption request. Clients are
processed sequentially on the simulation host; crypto threads parallelize
work inside one client. Timing model `client-batch-service-v3` is unchanged.

Add these flags to the existing 200-round command (use a new output directory):

```bash
--experimental-dynamic-cohort \
--dropout-round-count 20 \
--dropout-seed 42 \
--dropout-target-cap 8 \
--collusion-bound 1 \
--comparison-mode none
```

This example selects 20 rounds and one offline client in each selected round.
The schedule is saved to `dropout_schedule.json`. With eight registered clients
and single-client dropout, eight distinct proper targets suffice. The general
target cap defaults to 128; it is an operating limit, not a security certificate.
No dropout occurs without the schedule flag. Rebuild the native binding; ABI 21
and earlier are rejected by the updated Python adapter.

## State and cryptography

- Setup encryption/collective secret keys remain separate and unchanged.
  New target public keys and `x_i -> s_S` key-switch keys use the existing
  survivor-key path. Sparse M2R targets sum cached original shares.
- Cache scope is setup and exact target set. A distinct proper target is counted
  before key generation starts, including failed construction. The original
  cohort does not consume the cap. Attempt cap is one. Both caps are bound into
  the experimental setup descriptor; exhaustion closes the current attempt.
- Reaggregation uses original uploads. Sparse admission/support is recomputed
  for survivors, while original upload slot/label mappings are retained.
  Weights keep the setup denominator (128), without survivor renormalization.
- A finalization notice contains exact setup ID, logical round, attempt ID,
  revision, survivor IDs, and Sparse support bitmap and packing groups. Every
  final survivor acknowledges this context before aggregate-a compression/DD.
  Shrinking the target invalidates prior acknowledgements. Missing acknowledgements
  abort; stale contexts and non-survivor acknowledgements are rejected. The
  native state machine freezes this context through decryption and commit.
- An empty Sparse model support still runs the auxiliary path and commits.
- The offline client misses the successful commit. Before its next selection,
  its Sparse residual is reset to zero; local training starts from the current
  committed global model including auxiliary state. Other clients retain normal
  error feedback. This experiment models offline exclusion, not an online
  excluded client that receives Commit and preserves its old residual.
- Post-request dropout injection is disabled in the production CLI. The native
  failure-injection test remains available to verify abort/no update and rejection
  of another attempt for the same logical round. A later logical round is allowed
  in experimental mode. The Sparse CLI's late-abort handler saves unchanged
  model/residual state to `no_update_checkpoint.pt` and advances; other unexpected
  errors stop execution. The checkpoint is not a serialized crypto-session resume.

## Measurement scope

Uploads count all clients; aggregate-a and partial-decryption traffic count only
final survivors. Fresh-target key traffic is charged once, cache hits cost no
new key upload. Result delivery retains the existing all-registered-client
accounting convention, including eventual synchronization of offline clients.
Control-plane envelopes, acknowledgements, and extra checkpoint transport are
excluded from the existing cryptographic-payload communication metric.

This is an in-process experiment, with simulated acknowledgements, not a
distributed authenticated transport. It does not implement arbitrary upload
failures, ACK timeout/re-shrink networking, or post-DD recovery. The fixed-core
STATIC-PASS result and security proof do not certify the entire dropout view.
