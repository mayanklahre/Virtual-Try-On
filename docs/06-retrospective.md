# Retrospective

**Retrospective documentation, written on 19 October 2026.** This is a later summary of the solo repository, not backdated delivery evidence.

## What worked
- Modal offered a practical route to GPU-backed prototype execution.
- Persistent model storage and a small web/API surface kept the setup understandable.
- Validation, authentication, and rate limiting acknowledged real cost and abuse risks early.

## What was difficult
- Image quality depends heavily on framing, lighting, garment source, and model behaviour.
- Serverless GPU cold starts and cost trade-offs cannot be judged responsibly without measurements.
- Removing IP-Adapter simplified the path, but comparable quality needs a reproducible evaluation set.

## Improvements
- Add automated tests and a rights-cleared demo.
- Measure p50/p95 warm and cold latency and cost per completed request.
- Add stronger identity, observability, image-retention, and security controls before real user images.
- Maintain a curated test set and quality-review rubric for model changes.
