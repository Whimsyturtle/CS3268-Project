import json
from pathlib import Path

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_sae_strength_1_config_differs_from_main_only_in_sae_strength():
    main = json.loads((CONFIGS / "main.json").read_text())
    appendix = json.loads((CONFIGS / "sae_strength_1.json").read_text())
    assert appendix["methods"]["sae"]["strength"] == 1.0
    main["methods"]["sae"]["strength"] = 1.0
    assert appendix == main
