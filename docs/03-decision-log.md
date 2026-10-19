# Decision log

| ID | Decision | Options considered | Choice and rationale | Consequence |
| --- | --- | --- | --- | --- |
| D-01 | Compute environment | Local GPU; managed hosted endpoint; serverless GPU | **Modal serverless GPU** for on-demand A10G execution and persistent model storage without owning local GPU infrastructure. | Cold starts and provider cost require measurement/monitoring. |
| D-02 | Try-on model | Custom training; alternative open source models; CatVTON | **CatVTON** as an available open-source virtual try-on foundation appropriate for the prototype scope. | Output quality depends strongly on input images and model limits. |
| D-03 | Human parsing | No masks; separate parsing service; embedded monolith | **Modular ONNX Runtime body parsing** to generate clothing, face, and background masks behind one endpoint. | More model/dependency complexity. |
| D-04 | IP-Adapter stage | Keep it; remove it | **Removed IP-Adapter** after testing indicated comparable output quality with lower expected inference complexity/latency. | This is a qualitative testing statement; before/after latency must be measured before making numeric claims. |
| D-05 | Abuse controls | Public unrestricted endpoint; validation only; validation plus auth/rate limit | **Upload validation, HTTP Basic auth, and 5 requests/hour per IP** to reduce cost and misuse risk. | Basic auth and simple IP limits are not a full production identity/security model. |
