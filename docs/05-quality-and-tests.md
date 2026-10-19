# Quality and tests

## Current controls
- Upload size validation limits images to 10 MB.
- HTTP Basic authentication protects the UI and try-on endpoint through a Modal secret.
- Per-IP limit is 5 requests per hour.
- Sample command-line flow and image guidance are documented.

## Missing evidence to complete
The repository does not currently contain an automated test suite or recorded latency benchmark. Before presenting production-readiness, add:
1. API tests for invalid content type, missing image fields, over-size upload, invalid credentials, and rate-limit response.
2. Deterministic unit tests for validation and rate-limit behaviour.
3. A rights-cleared evaluation set with qualitative output review criteria.
4. Measured p50/p95 warm and cold latency, GPU execution/cost per request, failure rate, and output-quality notes.
5. A screenshot or short GIF using only images you own or are licensed to use.

## Acceptance test checklist
- Valid authorised request produces a result.
- Invalid/missing/oversize input fails safely.
- Incorrect credentials are rejected.
- Sixth request in one hour receives rate-limit response.
- No secret appears in logs, source, or client-side JavaScript.
