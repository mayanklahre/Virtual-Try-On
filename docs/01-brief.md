# Project brief

**Project:** Virtual Try-On API Pipeline  
**Delivery:** Solo project using CatVTON on Modal serverless GPU infrastructure.

## Problem and users
Virtual try-on needs an API that accepts a person image and garment image, generates a result, and controls GPU cost and misuse. The intended user is a developer or authorised user testing an image-based try-on workflow.

## Goal and success criteria
Provide a secure, documented image-to-image try-on path with sensible input validation and abuse controls. Success criteria for a future production evaluation are measured latency, defined GPU-cost cap, output-quality review, and verified abuse protection. No latency or cost numbers are currently claimed.

## Scope
**In:** Modal-based inference, CatVTON weights, upload validation, HTTP Basic authentication, per-IP rate limit, image guidance, and local/deployment instructions.

**Out:** production SLA, guaranteed garment fit/accuracy, storage of user images, a full ecommerce frontend, and measured production performance.
