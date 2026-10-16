# Steering

[api.pyi](api.pyi) defines the shared inference contract. The four method folders
re-export its signatures; each future `api.py` must supply its own implementation.
These stubs do not execute steering.

- `load_intervention(checkpoint_dir, model=model)` loads a saved intervention and
  checks its base-model revision and intervention locations against the model.
- `apply(model, intervention, strength=alpha)` returns a context manager that
  applies steering and restores model state on every exit, including exceptions.

The intervention representation is method-specific, so the contract uses `Any`.
Model and tokenizer types remain `Any` until the shared loader is implemented;
the annotations do not establish compatibility.

Construction remains method-specific: averaging for CAA, PCA for ICV, feature
selection for SAE, and training for ReFT-r1. See the
[proposal](../CS3268_Project_Proposal_Group_6.md) for the experiment design and
[checkpoints](../checkpoints/README.md) for saved-intervention metadata.

The unsteered baseline uses `contextlib.nullcontext()` in the shared evaluation
path. Strength is method-specific; equal numbers do not imply equal effects.
