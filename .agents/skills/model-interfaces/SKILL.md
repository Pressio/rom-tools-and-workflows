---
name: model-interfaces
description: Use when implementing or adapting a model for ROM Tools workflows, including QoIs and execution semantics.
---
# Model interfaces
Inspect the workflow's actual model protocol and existing model examples before implementing methods. Check `romtools/workflows/`, `examples/models/`, and `tests/romtools/workflows/`. Distinguish run-directory preparation, model execution, and QoI extraction; check return values and blocking/nonblocking behavior against current code. Do not assume an inverse driver accepts a sampling dispatcher argument.
