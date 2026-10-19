# Risks and mitigations

| Risk | Impact | Mitigation | Residual limit |
| --- | --- | --- | --- |
| Uncontrolled GPU spend or abuse | High | HTTP Basic authentication, 10 MB upload limit, and 5 requests per client IP per hour. | IP rate limiting is not a complete anti-abuse system. |
| Slow cold starts or high latency | Medium | Measure warm/cold latency and monitor request timing before defining a service level. | No numeric latency/SLA is currently recorded. |
| Poor output quality | Medium | Give clear input-image guidance and evaluate representative, rights-cleared test images. | Generative output may still have artifacts or poor garment fidelity. |
| Sensitive image exposure | High | Use secret-backed authentication; define retention/storage/deletion controls before real-user use. | Prototype documentation does not establish a full privacy program. |
| Credential leakage | High | Store credentials in Modal secrets, never browser JavaScript or source code. | Rotation and access governance are still needed for production. |
| Dependency/model availability | Medium | Use persistent Modal volume and documented setup; test deployment after dependency changes. | Model/provider changes may still require maintenance. |
