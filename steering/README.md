# Steering

[api.pyi](api.pyi) defines the shared interface for `caa/`, `icv/`, `sae/`, and
`reft_r1/`.

- `load_intervention(intervention_dir, model=model)` loads a saved intervention and
  checks its base-model revision and intervention locations against the model.
- `apply(model, intervention, strength=alpha)` returns a context manager that
  applies steering and restores model state on every exit, including exceptions.

Strength is method-specific. Equal values do not imply equal effects. The
unsteered baseline bypasses steering with `contextlib.nullcontext()`.

Follow the [proposal](../CS3268_Project_Proposal_Group_6.md), use
[shared examples](../shared_examples/README.md), and save outputs under
[steering_vectors/](../steering_vectors/README.md). Load pretrained SAEs from
the external model cache.

See [SAE steering](sae/README.md) for pretrained feature selection and application checks.

## Construction and evaluation

Supplied construction responses are examples used to extract or train an
intervention. Supplying a response does not establish that the unsteered model
would generate it. Construction AUROC, activation differences, and likelihood
of supplied responses do not by themselves establish reduced bias in generated
answers; they can also reflect wording, response style, or length.

Use the [proposal's separate held-out evaluations](../CS3268_Project_Proposal_Group_6.md#32-dataset-and-metrics)
to assess bias reduction and general capability on newly generated answers,
without supplying reference continuations. Report construction/training checks
separately from benchmark results, and state this distinction as a methodological
limitation in the report. See also
[AxBench's separate concept-detection and model-steering tasks](https://arxiv.org/html/2501.17148v3#S3).
