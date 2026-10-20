from collections import defaultdict

from evaluation import BBQ_BUILD_FRACTION, bbq_split

CATEGORIES = ["Age"] * 100 + ["Religion"] * 50 + ["SES"] * 30
QUESTION_INDICES = [i % 10 for i in range(100)] + [i % 5 for i in range(50)] + [i % 3 for i in range(30)]


def test_bbq_split_is_disjoint_complete_and_sorted():
    build, evaluation = bbq_split(CATEGORIES, QUESTION_INDICES)

    assert not set(build) & set(evaluation)
    assert sorted(build + evaluation) == list(range(len(CATEGORIES)))
    assert build == sorted(build) and evaluation == sorted(evaluation)


def test_bbq_split_keeps_templates_whole_and_stratified():
    build, _ = bbq_split(CATEGORIES, QUESTION_INDICES)
    build_rows = set(build)
    sides_by_template = defaultdict(set)
    for row, template in enumerate(zip(CATEGORIES, QUESTION_INDICES)):
        sides_by_template[template].add(row in build_rows)

    assert all(len(sides) == 1 for sides in sides_by_template.values())
    for category in set(CATEGORIES):
        templates = [template for template in sides_by_template if template[0] == category]
        build_templates = [template for template in templates if True in sides_by_template[template]]
        assert len(build_templates) == round(len(templates) * BBQ_BUILD_FRACTION)


def test_bbq_split_is_reproducible():
    assert bbq_split(CATEGORIES, QUESTION_INDICES) == bbq_split(CATEGORIES, QUESTION_INDICES)
