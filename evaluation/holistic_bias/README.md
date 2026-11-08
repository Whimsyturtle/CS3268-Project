# HolisticBias

This implementation evaluates demographic differences in generated conversational
styles using [Smith et al. (2022), Appendix
A.6–A.7](https://arxiv.org/abs/2205.09209).

## Protocol

`load_examples` keeps one released sentence per `(axis, descriptor, template)` cell.
It prefers the noun `person`, then `woman` or `man`, then another noun, and finally a
noun-less phrase. Noun-only controls are excluded, and `(axis, descriptor)` identifies
a descriptor. The configured dataset revision produces 15,973 prompts across 623
descriptors, 13 axes, and 26 templates.

The configured seed controls the released stylistic variations: expanding `I'm`,
removing descriptor hyphens, lowercasing the noun phrase, and removing the final
period. The shared runner generates one greedy continuation per prompt.

Before classification, descriptor mentions are replaced with `left-handed` and the
BlenderBot safety marker is removed. The released ParlAI current-utterance classifier
must provide all 217 styles and the labels in the six reported clusters.

The scorer reports raw, unscaled values:

- `full_gen_bias`: sum of the variances of descriptor-mean probabilities over every
  style, averaged over templates.
- `partial_gen_bias_<cluster>`: the same per-style variances, summed over styles in
  one cluster and averaged over templates.
- `summed_cluster_gen_bias_<cluster>`: style probabilities are first summed within
  the cluster, then their descriptor variance is averaged over templates.

## Classifier environment

Create a separate Python 3.10 environment for ParlAI 1.7.2:

```bash
python3.10 -m venv .venv-holistic-bias
source .venv-holistic-bias/bin/activate
python -m pip install --upgrade pip
python -m pip install -r evaluation/holistic_bias/classifier-requirements.txt
export HOLISTIC_BIAS_CLASSIFIER_PYTHON="$PWD/.venv-holistic-bias/bin/python"
```

Then download `zoo:style_gen/curr_only_classifier/model` once to the configured
`~/.cache/parlai` directory. ParlAI's own downloader restarts the 4.9 GB archive
from zero after any stall, so fetch it with `curl` (rerun to resume), then let
ParlAI extract it and mark it complete. Extraction needs about 11 GB free and
leaves 5.4 GB.

```bash
MODEL_DIR="$HOME/.cache/parlai/models/style_gen/curr_only_classifier"
mkdir -p "$MODEL_DIR"
curl -L -C - -o "$MODEL_DIR/v1.1.tar.gz" \
  https://dl.fbaipublicfiles.com/parlai/_models/style_gen/curr_only_classifier/v1.1.tar.gz
"$HOLISTIC_BIAS_CLASSIFIER_PYTHON" -c \
  "from parlai.zoo.style_gen.curr_only_classifier import download; download('$HOME/.cache/parlai')"
```

Runs stop before generation if this download is incomplete.

Each run additionally saves:

- `classifier_inputs.jsonl`: censored responses passed to the classifier.
- `style_probabilities.jsonl`: ordered probability vectors keyed by `example_id`.
- `classifier_manifest.json`: style order, software versions, input hash, and SHA-256
  hashes of the resolved model, dictionary, options, and class-list files.
