# Testing strategy

Lab 3 applies the ML testing pyramid: schema tests protect input boundaries, API integration tests verify the deployed contract, and behavioural tests ensure IDs are normalized before inference. The deterministic fallback model keeps CI reproducible; a trained SVD pickle can replace it without changing the API.

The CI workflow runs Ruff and the full suite with a hard 80% coverage gate. The container-validation workflow builds the production image and calls its health endpoint before a deployment is considered valid.
