"""Shared evaluation code: the BBQ split and the benchmark runner used by every steering method.

Run the unsteered baseline with `uv run python evaluation.py --output-dir results/baseline`.
To evaluate a steering method, attach its hooks to the model, then call `run_benchmarks`.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import lm_eval
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "google/gemma-3-1b-it"
SEED = 0
BBQ_DATASET = "oskarvanderwal/bbq"  # The copy lm-eval's `bbq` task reads; split indices refer to its order.
BBQ_BUILD_FRACTION = 0.2
# Greedy decoding for HolisticBias and AlpacaEval; lm-eval sets its own for MMLU-Pro (also 2048 tokens).
HOLISTIC_BIAS_MAX_NEW_TOKENS = 512
ALPACA_EVAL_MAX_NEW_TOKENS = 2048  # 512 cut off AlpacaEval answers mid-sentence in a test run.


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return model.to(device).eval(), tokenizer


def load_bbq():
    return load_dataset(BBQ_DATASET, "All", split="test")


def bbq_split(categories, question_indices):
    """Split row indices into (build, eval), giving BBQ_BUILD_FRACTION of each category's question templates to build.

    Whole templates are split because rows from one template share contexts and names.
    """
    rows_by_template = defaultdict(list)
    for row, template in enumerate(zip(categories, question_indices)):
        rows_by_template[template].append(row)
    rng = random.Random(SEED)
    build, evaluation = [], []
    for category in sorted(set(categories)):
        templates = sorted(template for template in rows_by_template if template[0] == category)
        rng.shuffle(templates)
        build_size = round(len(templates) * BBQ_BUILD_FRACTION)
        for template in templates[:build_size]:
            build += rows_by_template[template]
        for template in templates[build_size:]:
            evaluation += rows_by_template[template]
    # lm-eval requires sorted indices when selecting samples.
    return sorted(build), sorted(evaluation)


def bbq_build_set():
    """BBQ rows reserved for constructing steering vectors. Never evaluate on these."""
    bbq = load_bbq()
    build, _ = bbq_split(bbq["category"], bbq["question_index"])
    return bbq.select(build)


def holistic_bias_prompts():
    """One random sentence per (template, descriptor) pair, so bias can be compared across descriptors per template.

    Only "<descriptor> person" and fixed phrases (like "a wheelchair user") are kept, so differences between
    descriptors are not mixed with noun gender or phrasing.
    """
    sentences = load_dataset("fairnlp/holistic-bias", "sentences", split="test")
    sentences = sentences.filter(lambda row: row["noun"] == "person" or row["noun_phrase_type"] == "fixed_phrase")
    pairs = sentences.select_columns(["template", "descriptor"]).to_pandas()
    chosen = pairs.groupby(["template", "descriptor"]).sample(n=1, random_state=SEED).index
    return list(sentences.select(sorted(chosen)))


def alpaca_eval_prompts():
    path = hf_hub_download("tatsu-lab/alpaca_eval", "alpaca_eval.json", repo_type="dataset")
    return json.loads(Path(path).read_text())


@torch.no_grad()
def generate(model, tokenizer, prompts, batch_size, max_new_tokens):
    """Greedy chat responses to single-turn user prompts."""
    tokenizer.padding_side = "left"
    responses = []
    for start in range(0, len(prompts), batch_size):
        chats = [[{"role": "user", "content": prompt}] for prompt in prompts[start : start + batch_size]]
        inputs = tokenizer.apply_chat_template(
            chats, add_generation_prompt=True, padding=True, return_tensors="pt", return_dict=True
        ).to(model.device)
        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
        new_tokens = outputs[:, inputs["input_ids"].shape[1] :]
        responses += tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    return responses


def run_benchmarks(model, tokenizer, output_dir, method_name, batch_size=16, limit=None):
    """Score BBQ and MMLU-Pro, and save HolisticBias and AlpacaEval responses for later scoring.

    `limit` caps examples per benchmark (per subject for MMLU-Pro); use it only for test runs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)

    bbq = load_bbq()
    _, bbq_eval = bbq_split(bbq["category"], bbq["question_index"])
    bbq_eval = bbq_eval[:limit]
    bbq = lm_eval.simple_evaluate(model=lm, tasks=["bbq"], samples={"bbq": bbq_eval}, log_samples=False)
    (output_dir / "bbq.json").write_text(json.dumps(bbq["results"], indent=2))

    mmlu_pro = lm_eval.simple_evaluate(model=lm, tasks=["mmlu_pro"], limit=limit, log_samples=False)
    (output_dir / "mmlu_pro.json").write_text(json.dumps(mmlu_pro["results"], indent=2))

    holistic_bias = holistic_bias_prompts()[:limit]
    responses = generate(model, tokenizer, [row["text"] for row in holistic_bias], batch_size, HOLISTIC_BIAS_MAX_NEW_TOKENS)
    with open(output_dir / "holistic_bias_responses.jsonl", "w") as file:
        for row, response in zip(holistic_bias, responses):
            file.write(json.dumps({**row, "response": response}) + "\n")

    # Saved in the format the alpaca_eval package reads.
    alpaca_eval = alpaca_eval_prompts()[:limit]
    responses = generate(model, tokenizer, [row["instruction"] for row in alpaca_eval], batch_size, ALPACA_EVAL_MAX_NEW_TOKENS)
    outputs = [{**row, "output": response, "generator": method_name} for row, response in zip(alpaca_eval, responses)]
    (output_dir / "alpaca_eval_outputs.json").write_text(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the unsteered baseline.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, help="Examples per benchmark, per subject for MMLU-Pro (test runs only).")
    args = parser.parse_args()
    model, tokenizer = load_model()
    run_benchmarks(model, tokenizer, args.output_dir, "unsteered_baseline", args.batch_size, args.limit)
