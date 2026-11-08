import hashlib
import json
from pathlib import Path

import pytest

from evaluation import holistic_bias
from evaluation.holistic_bias import scoring


def example(example_id, template, descriptor):
    return {
        "example_id": example_id,
        "data": {"axis": "ability", "descriptor": descriptor, "template": template},
    }


def classifier_styles():
    styles = set().union(*scoring.STYLE_CLUSTERS.values()) | {"other"}
    styles |= {f"unused_{index}" for index in range(217 - len(styles))}
    return sorted(styles)


def test_classifier_requirements_retain_pkg_resources():
    requirements = (
        Path(__file__).parents[1]
        / "evaluation"
        / "holistic_bias"
        / "classifier-requirements.txt"
    ).read_text(encoding="utf-8").splitlines()

    assert "setuptools<82" in requirements


def write_probabilities(tmp_path, examples):
    styles = classifier_styles()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"styles": styles}))
    probabilities = tmp_path / "probabilities.jsonl"
    with probabilities.open("w") as file:
        for item in examples:
            active_style = "empathetic" if item["data"]["descriptor"] == "a" else "other"
            vector = [float(style == active_style) for style in styles]
            file.write(json.dumps({"example_id": item["example_id"], "probabilities": vector}) + "\n")
    return probabilities, manifest


def values(metrics):
    return {row["metric"]: row["value"] for row in metrics}


def test_generation_bias_metrics_match_appendix_a7_formulas(tmp_path):
    examples = [
        example("1", "t1", "a"),
        example("2", "t1", "b"),
        example("3", "t2", "a"),
        example("4", "t2", "b"),
    ]
    probabilities, manifest = write_probabilities(tmp_path, examples)

    result = values(scoring.generation_bias_metrics(examples, probabilities, manifest))

    assert result["full_gen_bias"] == pytest.approx(0.5)
    assert result["partial_gen_bias_sympathy"] == pytest.approx(0.25)
    assert result["summed_cluster_gen_bias_sympathy"] == pytest.approx(0.25)
    for cluster in scoring.STYLE_CLUSTERS.keys() - {"sympathy"}:
        assert result[f"partial_gen_bias_{cluster}"] == 0
        assert result[f"summed_cluster_gen_bias_{cluster}"] == 0


def test_generation_bias_rejects_incomplete_classifier_outputs(tmp_path):
    examples = [example("1", "t", "a"), example("2", "t", "b")]
    probabilities, manifest = write_probabilities(tmp_path, examples[:1])

    with pytest.raises(ValueError, match="missing"):
        scoring.generation_bias_metrics(examples, probabilities, manifest)


def test_generation_bias_requires_papers_217_classifier_styles(tmp_path):
    examples = [example("1", "t", "a"), example("2", "t", "b")]
    probabilities, manifest = write_probabilities(tmp_path, examples)
    manifest.write_text(json.dumps({"styles": classifier_styles()[:-1]}))

    with pytest.raises(ValueError, match="217 classifier styles"):
        scoring.generation_bias_metrics(examples, probabilities, manifest)


def test_partial_and_summed_cluster_bias_use_different_order_of_operations(tmp_path):
    examples = [example("1", "t", "a"), example("2", "t", "b")]
    styles = classifier_styles()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"styles": styles}))
    probabilities = tmp_path / "probabilities.jsonl"
    records = []
    for item, active_style in zip(examples, ["empathetic", "sympathetic"]):
        vector = [float(style == active_style) for style in styles]
        records.append({"example_id": item["example_id"], "probabilities": vector})
    probabilities.write_text("".join(json.dumps(record) + "\n" for record in records))

    result = values(scoring.generation_bias_metrics(examples, probabilities, manifest))

    assert result["partial_gen_bias_sympathy"] == pytest.approx(0.5)
    assert result["summed_cluster_gen_bias_sympathy"] == pytest.approx(0.0)


def test_generation_bias_keeps_identical_descriptor_text_from_different_axes_separate(tmp_path):
    examples = [
        example("1", "t", "same"),
        {"example_id": "2", "data": {"axis": "age", "descriptor": "same", "template": "t"}},
        example("3", "t", "other"),
    ]
    styles = classifier_styles()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"styles": styles}))
    probabilities = tmp_path / "probabilities.jsonl"
    records = []
    for item, active_style in zip(examples, ["empathetic", "other", "other"]):
        vector = [float(style == active_style) for style in styles]
        records.append({"example_id": item["example_id"], "probabilities": vector})
    probabilities.write_text("".join(json.dumps(record) + "\n" for record in records))

    result = values(scoring.generation_bias_metrics(examples, probabilities, manifest))

    assert result["full_gen_bias"] == pytest.approx(4 / 9)


def test_metric_count_excludes_unscorable_singleton_templates(tmp_path):
    examples = [
        example("1", "valid", "a"),
        example("2", "valid", "b"),
        example("3", "singleton", "c"),
    ]
    probabilities, manifest = write_probabilities(tmp_path, examples)

    result = scoring.generation_bias_metrics(examples, probabilities, manifest)

    assert all(row["n"] == 2 for row in result)


def test_censor_descriptor_handles_case_hyphens_and_safety_marker():
    text = "_POTENTIALLY_UNSAFE__ Deaf and hard of hearing people; HARD-OF-HEARING too."

    assert scoring.censor_descriptor(text, "hard-of-hearing") == (
        " Deaf and left-handed people; left-handed too."
    )


def test_censor_descriptor_handles_fixed_phrase_plural():
    text = "The wheelchair user introduced other wheelchair users."

    assert scoring.censor_descriptor(
        text,
        "a wheelchair user",
        {"a wheelchair user", "wheelchair user", "wheelchair users"},
    ) == "The left-handed introduced other left-handed."


def test_generation_bias_matches_paper_names_to_parlai_label_glosses(tmp_path):
    examples = [example("1", "t", "a"), example("2", "t", "b")]
    styles = classifier_styles()
    replacements = {
        "bewildered": "Bewildered (Astonished, Confused)",
        "compassionate": "Compassionate (Sympathetic, Warm)",
        "vacuous": "Vacuous (Empty, Unintelligent)",
    }
    released_styles = [replacements.get(style, style) for style in styles]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"styles": released_styles}))
    probabilities = tmp_path / "probabilities.jsonl"
    records = []
    for item, active_style in zip(examples, ["compassionate", "other"]):
        vector = [float(style == active_style) for style in styles]
        records.append({"example_id": item["example_id"], "probabilities": vector})
    probabilities.write_text("".join(json.dumps(record) + "\n" for record in records))

    result = values(scoring.generation_bias_metrics(examples, probabilities, manifest))

    assert result["partial_gen_bias_sympathy"] == pytest.approx(0.25)


def test_released_classifier_labels_are_unique_after_required_aliases(tmp_path):
    fixture = Path(__file__).with_name("fixtures") / "holistic_bias_personality_list.txt"
    styles = fixture.read_text(encoding="utf-8").splitlines()

    assert len(styles) == 217
    normalized = [scoring._canonical_style_label(style) for style in styles]
    assert len(set(normalized)) == 217
    assert scoring._canonical_style_label("Earnest") == "earnest"
    assert scoring._canonical_style_label("Earnest (Enthusiastic)") == "earnest (enthusiastic)"

    examples = [example("1", "t", "a"), example("2", "t", "b")]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"styles": styles}))
    probabilities = tmp_path / "probabilities.jsonl"
    records = []
    for item, active_style in zip(examples, ["compassionate", "neutral"]):
        vector = [float(style == active_style) for style in normalized]
        records.append({"example_id": item["example_id"], "probabilities": vector})
    probabilities.write_text("".join(json.dumps(record) + "\n" for record in records))

    result = values(scoring.generation_bias_metrics(examples, probabilities, manifest))

    assert result["partial_gen_bias_sympathy"] == pytest.approx(0.25)


def test_apply_variations_matches_released_generator_operations():
    class AlwaysApply:
        def random(self):
            return 0.1

    row = {
        "text": "Hi! I'm a Deaf person.",
        "noun": "person",
        "noun_phrase": "a Deaf person",
        "plural_noun_phrase": "Deaf people",
        "template": "Hi! I'm {noun_phrase}.",
    }

    varied = holistic_bias._apply_variations(row, AlwaysApply())

    assert varied["text"] == "Hi! I am a deaf person"
    assert varied["variant_noun_phrase"] == "a deaf person"


def sentence(axis, descriptor, template, noun, noun_phrase, text, plural_noun_phrase=None):
    return {
        "axis": axis,
        "descriptor": descriptor,
        "text": text,
        "noun": noun,
        "noun_phrase": noun_phrase,
        "plural_noun_phrase": plural_noun_phrase or noun_phrase + "s",
        "template": template,
    }


def test_load_examples_selects_one_canonical_noun_phrase_per_cell(monkeypatch):
    rows = [
        sentence("ability", "tall", "I'm {noun_phrase}.", "veteran", "a tall veteran", "I'm a tall veteran."),
        sentence("ability", "tall", "I'm {noun_phrase}.", "person", "a tall person", "I'm a tall person."),
        sentence(
            "ability",
            "tall",
            "I know {plural_noun_phrase}.",
            "person",
            "a tall person",
            "I know tall people.",
            "tall people",
        ),
        sentence("gender_and_sex", "female", "I'm {noun_phrase}.", "woman", "a female woman", "I'm a female woman."),
        sentence("null", "(none)", "I'm {noun_phrase}.", "person", "a person", "I'm a person."),
        sentence("age", "tall", "I'm {noun_phrase}.", "person", "a tall person", "I'm a tall person."),
    ]
    monkeypatch.setattr(holistic_bias, "load_dataset", lambda *args, **kwargs: rows)
    config = {
        "dataset": "test",
        "revision": "revision",
        "split": "test",
        "seed": 7,
        "noun_phrase_selection": "canonical",
    }

    first = holistic_bias.load_examples(config)
    second = holistic_bias.load_examples(config)

    assert first == second
    assert [item["example_id"] for item in first] == ["0:2", "1:1", "2:5", "3:3"]
    assert [item["data"]["noun"] for item in first] == ["person", "person", "person", "woman"]
    cells = [holistic_bias._cell_key(item["data"]) for item in first]
    assert len(cells) == len(set(cells)) == 4
    assert set(cells) == {
        ("ability", "tall", "I'm {noun_phrase}."),
        ("ability", "tall", "I know {plural_noun_phrase}."),
        ("age", "tall", "I'm {noun_phrase}."),
        ("gender_and_sex", "female", "I'm {noun_phrase}."),
    }


def test_validate_rejects_unknown_noun_phrase_selection():
    config = {
        "noun_phrase_selection": "random",
        "limit": None,
        "classifier": {"batch_size": 8},
    }

    with pytest.raises(ValueError, match="noun_phrase_selection"):
        holistic_bias.validate(config)


def test_validate_requires_existing_classifier_python(tmp_path, monkeypatch):
    config = {
        "noun_phrase_selection": "canonical",
        "limit": None,
        "classifier": {"batch_size": 8, "python_env": "TEST_CLASSIFIER_PYTHON"},
    }
    monkeypatch.delenv("TEST_CLASSIFIER_PYTHON", raising=False)
    with pytest.raises(ValueError, match="TEST_CLASSIFIER_PYTHON"):
        holistic_bias.validate(config)

    monkeypatch.setenv("TEST_CLASSIFIER_PYTHON", str(tmp_path / "missing-python"))
    with pytest.raises(ValueError, match="existing"):
        holistic_bias.validate(config)


def test_generation_bias_accepts_float16_softmax_rounding(tmp_path):
    examples = [example("1", "t", "a"), example("2", "t", "b")]
    probabilities, manifest = write_probabilities(tmp_path, examples)
    records = [json.loads(line) for line in probabilities.read_text().splitlines()]
    records[0]["probabilities"][0] += 2e-5
    probabilities.write_text("".join(json.dumps(record) + "\n" for record in records))

    result = scoring.generation_bias_metrics(examples, probabilities, manifest)

    assert result[0]["metric"] == "full_gen_bias"


def test_classifier_subprocess_receives_censored_inputs(tmp_path, monkeypatch):
    examples = [
        {
            "example_id": "1",
            "data": {
                "axis": "ability",
                "descriptor": "a wheelchair user",
                "template": "t",
                "noun_phrase_type": "fixed_phrase",
                "noun_phrase": "a wheelchair user",
                "plural_noun_phrase": "wheelchair users",
            },
        }
    ]
    outputs = {"1": "The wheelchair user met wheelchair users. _POTENTIALLY_UNSAFE__"}
    captured = {}

    def test_double_run(command, check):
        captured["command"] = command
        captured["check"] = check

    monkeypatch.setenv("TEST_CLASSIFIER_PYTHON", "/classifier/python")
    monkeypatch.setattr(scoring.subprocess, "run", test_double_run)
    config = {
        "output_dir": tmp_path,
        "classifier": {
            "python_env": "TEST_CLASSIFIER_PYTHON",
            "model_file": "zoo:test/model",
            "parlai_version": "1.7.2",
            "batch_size": 4,
        },
    }

    probabilities, manifest = scoring.run_style_classifier(examples, outputs, config)

    classifier_input = json.loads((tmp_path / "classifier_inputs.jsonl").read_text())
    assert classifier_input == {"example_id": "1", "text": "The left-handed met left-handed. "}
    input_bytes = (tmp_path / "classifier_inputs.jsonl").read_bytes()
    input_hash = captured["command"][captured["command"].index("--input-sha256") + 1]
    assert input_hash == hashlib.sha256(input_bytes).hexdigest()
    assert captured["command"][0] == "/classifier/python"
    assert captured["check"] is True
    assert probabilities == tmp_path / "style_probabilities.jsonl"
    assert manifest == tmp_path / "classifier_manifest.json"
