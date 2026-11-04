# Steering

[api.pyi](api.pyi) defines the shared interface for `caa/`, `icv/`, `sae/`, and
`reft_r1/`.

- `load_intervention(intervention_dir, model=model)` loads a saved intervention and
  checks its base-model revision and intervention locations against the model.
- `apply(model, intervention, strength=alpha)` returns a context manager that
  applies steering and restores model state on every exit, including exceptions.

Strength is method-specific. Equal values do not imply equal effects. The
unsteered baseline bypasses steering with `contextlib.nullcontext()`.

Train ReFT-r1 with `uv run python -m steering.reft_r1.main`, which takes about
6 minutes on a laptop. Its strength is a multiple of `max_act`.

Follow the [proposal](../CS3268_Project_Proposal_Group_6.md), use
[shared examples](../shared_examples/README.md), and save outputs under
[steering_vectors/](../steering_vectors/README.md). Load pretrained SAEs from
the external model cache.
