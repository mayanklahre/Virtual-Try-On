# Virtual Try-On

**Solo project | GPU-powered virtual try-on prototype using CatVTON and Modal**

## Problem
A virtual try-on workflow needs to accept a person image and garment image, produce a generated preview, and avoid uncontrolled GPU spend or unauthorised use.

## Who it is for
An authorised developer or prototype user evaluating an image-based try-on workflow. It is not a claim of a production ecommerce service.

## Goal and success criteria
Provide an authenticated, documented API/UI path with input validation and abuse controls. Before production use, success must be measured through p50/p95 latency, cold-start behaviour, GPU cost per request, failure rate, and output-quality review. Those performance numbers have not yet been measured.

## Scope
**In:** CatVTON inference on Modal serverless A10G GPU, ONNX Runtime body parsing, 10 MB upload validation, HTTP Basic authentication, 5-requests-per-hour per-IP rate limit, model-weight setup, and usage guidance.

**Out:** production SLA, guaranteed fit/quality, user-image retention policy, commerce checkout, a public unauthenticated service, and measured production performance.

## Key decisions
- Modal serverless GPU provides a practical prototype environment.
- CatVTON is the open-source try-on model foundation.
- ONNX Runtime body parsing produces modular masks.
- The IP-Adapter stage was removed after testing suggested comparable quality with a simpler/lower-latency path; no before/after latency figure is claimed.
- Modal secrets, input validation, and rate limiting limit obvious abuse/cost risks.

See the [decision log](docs/03-decision-log.md).

## Outcome
The repository contains a Modal-based try-on path, persistent weights, a protected API/UI, image validation, and a 5-requests-per-hour client IP limit. No measured latency, GPU-cost, quality, or production-usage metric is claimed.

## Risks and mitigations
Primary risks are abuse/GPU cost, slow cold starts, variable generative output, credentials, and user-image handling. Controls include authentication, request limits, upload limits, secrets, and explicit documentation. See [risks](docs/04-risks.md).

## How to run
Requirements: Python 3.10+, a Modal account, authenticated Modal CLI, and A10G workspace access.

    python3 -m pip install modal
    modal setup
    modal secret create vto-web-auth VTO_USERNAME='your-login-name' VTO_PASSWORD='a-long-unique-password'
    python3 -m modal run app.py::download_weights
    python3 -m modal serve app.py

For a deployed Modal endpoint:

    python3 -m modal deploy app.py

Use sample images in data for a command-line check:

    python3 -m modal run app.py

## Next steps
- Add validation/rate-limit/authentication tests.
- Record warm and cold p50/p95 latency, cost per request, failure rate, and an output-quality evaluation.
- Add a rights-cleared screenshot or GIF demo.
- Define identity, retention, deletion, monitoring, and incident controls before real-user image handling.

## Documentation
- [Brief](docs/01-brief.md)
- [Requirements](docs/02-requirements.md)
- [Decision log](docs/03-decision-log.md)
- [Risks](docs/04-risks.md)
- [Quality and tests](docs/05-quality-and-tests.md)
- [Retrospective](docs/06-retrospective.md)
