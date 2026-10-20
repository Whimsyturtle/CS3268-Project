"""HolisticBias: demographic bias in open-ended text generation (Smith et al., 2022)."""

from datasets import load_dataset

from evaluation.common import outputs_by_id

GROUP_COLUMNS = ["axis", "bucket", "template", "descriptor"]


def load_examples(config):
    """One random sentence per (axis, bucket, template, descriptor), so descriptors can be compared per template.

    Only "<descriptor> person" and fixed phrases (like "a wheelchair user") are kept, so differences between
    descriptors are not mixed with noun gender or phrasing.
    """
    rows = load_dataset(config["dataset"], "sentences", split=config["split"], revision=config["revision"])
    groups = rows.select_columns(GROUP_COLUMNS + ["noun", "noun_phrase_type"]).to_pandas()
    groups = groups[(groups["noun"] == "person") | (groups["noun_phrase_type"] == "fixed_phrase")]
    # dropna=False keeps the no-descriptor control sentences, which have no axis.
    chosen = groups.groupby(GROUP_COLUMNS, dropna=False).sample(n=1, random_state=config["seed"]).index
    return [{"example_id": str(index), "data": rows[int(index)]} for index in sorted(chosen)]


def format_prompt(example, config):
    return example["data"]["text"]


def score(examples, predictions, config):
    outputs_by_id(examples, predictions)
    raise NotImplementedError("HolisticBias scoring needs the style classifier of Smith et al. (2020a); not implemented yet.")
