# Lovecraft recurrent text model — delivery incomplete

This checkout contains documentation only. The described Python program,
dependency manifest, initialized weights, corpus and verification reports have
not been delivered to this repository. It cannot currently generate text or run
the documented self-test.

The [original delivery description](docs/ORIGINAL_DELIVERY_DESCRIPTION.md)
is preserved for recovery. Its numerical and verification claims remain
unverified until the corresponding files can be recovered and checked.

## Recovery status

Checked on 2026-11-13 against main at
`a06875b7d9d6f6c9d3ef7ec0e31d66170c9e7919`:

- The tree contains the original README and an Actions Dependabot configuration.
- Neither commit in the available history contains the implementation or model.
- Main is the only branch; there are no releases or Actions artifacts in the
  two available runs.
- A filename search of accessible non-credential local artifact locations found
  no Lovecraft implementation or weight archive.

Recovery therefore requires the original source/model package or another
authoritative copy. Creating a different model would be new work and would not
validate the original delivery claims.

## Restore and verify

Recover these files from the original package before publishing runnable
instructions:

- `lovecraft_rnn.py`, `requirements.txt`, `lovecraft_weights.npz`;
- `lovecraft_corpus.json`, `corpus_manifest.json`, `coverage_audit.json`;
- `model_info.json`, `generation_samples.json`, `validation_report.json`,
  `runtime_metrics.json`, `SHA256SUMS.txt`.

Then verify the package checksums and corpus/model identities, install the
recorded dependencies in an isolated environment, run the documented self-test,
inspect the actual model dimensions, and repeat a seeded generation from a clean
checkout. Record commands, versions and results. Preserve any failed checks as
open delivery work.

The unused GitHub Actions Dependabot entry has been removed because this
repository has no workflows or action manifest to update. Add dependency updates
alongside the actual recovered dependency manifests and smoke workflow.
