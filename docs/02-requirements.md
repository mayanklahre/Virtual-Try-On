# Requirements and acceptance criteria

| ID | User story | Acceptance criteria |
| --- | --- | --- |
| RQ-01 | As an authorised user, I can submit a person and garment image for a try-on result. | The API accepts multipart image fields, validates input, and returns an image result or controlled error. |
| RQ-02 | As an operator, I can protect access to the endpoint. | HTTP Basic authentication is required for the UI and try-on endpoint using a Modal secret, not source-code credentials. |
| RQ-03 | As an operator, I can limit abuse and GPU spend. | A client IP is limited to 5 try-on requests per hour and receives a clear rate-limit response. |
| RQ-04 | As a user, I receive usable input guidance. | Documentation specifies supported image size and recommended front-facing subject/clean garment images. |
| RQ-05 | As a maintainer, I can install weights and operate the service. | Setup, Modal authentication, secret configuration, persistent volume, development and deploy commands are documented. |
| RQ-06 | As a maintainer, I can assess performance before a production decision. | Latency, cold-start behaviour, output quality, request failure rate, and GPU cost are measured and recorded against a test set. This is pending. |
