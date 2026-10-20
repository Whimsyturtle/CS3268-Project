# Steering

[api.pyi](api.pyi) defines the inference contract for all four methods.
Implementations belong in `caa/`, `icv/`, `sae/`, and `reft_r1/` and must follow
these signatures. The shared stub does not execute steering.

- `load_intervention(intervention_dir, model=model)` loads a saved intervention and
  checks its base-model revision and intervention locations against the model.
- `apply(model, intervention, strength=alpha)` returns a context manager that
  applies steering and restores model state on every exit, including exceptions.

The intervention representation is method-specific, so the contract uses `Any`.
Model and tokenizer types remain `Any` until the shared loader is implemented;
the annotations do not establish compatibility.

Construction remains method-specific: averaging for CAA, PCA for ICV, feature
selection for SAE, and training for ReFT-r1. Follow the
[proposal](../CS3268_Project_Proposal_Group_6.md) for the experiment design.

Keep source code, construction examples, and custom split definitions directly
under each method folder. Use the same reviewed source examples and splits
across methods; configurations can reference one file in a method folder rather
than maintain separate copies. Construction data must satisfy the proposal's
data separation requirement.

Put generated interventions in `steering/<method>/saved/`, along with construction
settings, source example IDs, exact model revision, and intervention locations.
SAE also records its compatible pretrained SAE checkpoint and selected features.
These folders are empty and ignored by Git except for `.gitkeep` placeholders.

The unsteered baseline uses `contextlib.nullcontext()` in the shared evaluation
path. Strength is method-specific; equal numbers do not imply equal effects.
