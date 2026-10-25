# Shared examples

Each JSONL record contains `id`, `category`, `prompt`, `biased_response`, and
`unbiased_response`.

- `construction.jsonl`: examples used to build or train interventions.
- `development.jsonl`: separate examples used to tune settings such as strength.

Use the same splits across methods. Keep related variants in the same split
and keep construction and development data separate from final evaluation datasets.

The examples are English records from the
[MBTP dataset](https://huggingface.co/datasets/MasahiroKaneko/mbtp), introduced in
the [MBTP paper](https://aclanthology.org/2026.acl-long.2204/).
[`prepare_mbtp.py`](prepare_mbtp.py) pins the source revision, balances categories,
and keeps records derived from the same stereotype cores together. It maps MBTP's
`input`, `pro_response`, and `neutral_response` fields to the three text fields above.

Run `python shared_examples/prepare_mbtp.py` to reproduce the splits and
`python shared_examples/validate.py --check-evaluation-overlap` to validate them.

The selected examples cover MBTP's 13 categories. They are limited to English
and MBTP does not label intersectional combinations as separate categories.
