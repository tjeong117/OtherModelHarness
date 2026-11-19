import json
from pathlib import Path

import pytest


def test_suppression_distinguishes_inactive_from_unobserved(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    from summarize_sae_guided import measured_suppression

    fitted = dict(type="tool", tool="fit_sae", arguments={"prompts": ["train"], "validation_prompts": ["validation"]},
                  result={"dictionary_id": "dictionary", "training": {"prompts": 1}, "top_k": 2})
    baseline = [{"margin": 1.0, "observations": [{"position": 2, "top": top}]}
                for top in ([{"index": 3, "value": 2.0}, {"index": 2, "value": 0.5}],
                            [{"index": 3, "value": 2.0}])]
    inspected = dict(type="tool", tool="inspect_target", call=2,
                     arguments={"prompts": ["validation", "fresh"], "intervention": {"scale": 0, "feature": 1},
                                "token_ids": [4, 7]},
                     result={"probe": {"kind": "sae", "layer": 1, "dictionary_id": "dictionary"},
                             "baseline_outputs": baseline, "outputs": [{"margin": 1.}, {"margin": 2.}],
                             "margin_changes": [0., 1.]})
    log = tmp_path / "trajectory.jsonl"
    log.write_text("\n".join(map(json.dumps, (fitted, inspected))))
    first, second = measured_suppression(tmp_path)
    assert first["fit_prompt_split"] == "validation" and second["fit_prompt_split"] == "new_public_prompt"
    assert first["baseline_feature_activations"][0]["activation"] == 0.0
    assert second["baseline_feature_activations"][0]["activation"] is None
    inspected["result"]["margin_changes"][0] = .5
    log.write_text("\n".join(map(json.dumps, (fitted, inspected))))
    with pytest.raises(ValueError, match="margin shift"):
        measured_suppression(tmp_path)
