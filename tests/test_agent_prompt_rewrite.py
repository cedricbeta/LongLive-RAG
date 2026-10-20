"""Caption rewrite for the absent-entity regenerate action: LLM first, validated, template fallback."""
import json

from utils.agent_prompt_rewrite import (
    REWRITE_SCHEMA,
    AbsentEntityPromptRewriter,
    template_regenerate_caption,
    validate_rewrite,
)

CAPTION = "Noah works at the desk. Iris enters and takes a seat to Noah's left."
RELATIONS = [{"subject": "Iris", "relation": "left", "anchor": "Noah"}]


class Client:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.calls = []

    def call(self, **kwargs):
        self.calls.append(kwargs)
        return {"parsed": next(self.answers), "call_id": f"call{len(self.calls)}"}


def _rewrite(client, attempt=1, previous=()):
    return AbsentEntityPromptRewriter(client).rewrite(
        caption=CAPTION, global_caption="scene", absent=[("Iris", "woman with red hair")],
        present=[("Noah", "tall man in green shirt")], relations=RELATIONS, shot_index=1,
        attempt=attempt, previous=list(previous),
    )


def test_llm_rewrite_is_text_only_and_kept_when_valid():
    client = Client([{"caption": "Iris, a woman with red hair, walks in and sits to Noah's left while Noah works at the desk."}])
    result = _rewrite(client)
    assert result["source"] == "llm"
    assert result["prompt"].startswith("Iris, a woman with red hair")
    assert result["call_id"] == "call1"
    assert result["rejected"] == []
    call = client.calls[0]
    assert call["image_paths"] == []
    assert call["schema"] == REWRITE_SCHEMA
    request = json.loads(call["prompt"])
    assert request["absent_entities"] == [{"name": "Iris", "description": "woman with red hair"}]
    assert request["spatial_relations"] == RELATIONS
    assert request["variant"] == 1


def test_invalid_rewrite_falls_back_to_template_and_records_why():
    client = Client([{"caption": "Noah keeps working at the desk alone."}])
    result = _rewrite(client)
    assert result["source"] == "template"
    assert result["rejected"][0]["problems"] == ["absent entity not named: Iris"]
    assert result["prompt"].startswith("Iris, woman with red hair, is clearly visible in the frame, to Noah's left.")
    assert result["prompt"].endswith(CAPTION)


def test_dropping_a_present_entity_or_running_long_is_rejected():
    assert validate_rewrite("Iris sits down and reads.", original=CAPTION, absent_names=["Iris"], present_names=["Noah"]) == [
        "present entity dropped: Noah"
    ]
    long = "Iris and Noah " + "talk " * 95
    assert "longer than 90 words" in validate_rewrite(long, original=CAPTION, absent_names=["Iris"], present_names=["Noah"])
    assert validate_rewrite("", original=CAPTION, absent_names=["Iris"], present_names=["Noah"]) == ["empty"]
    # a name that only appears inside another word does not count
    assert validate_rewrite("Irises bloom while Noah works.", original=CAPTION, absent_names=["Iris"], present_names=["Noah"])[0].startswith("absent entity not named")


def test_previous_rewrites_and_variant_reach_the_request():
    client = Client([{"caption": "Iris, a woman with red hair, joins Noah at the desk and sits on his left."}])
    result = _rewrite(client, attempt=2, previous=["Iris walks in beside Noah."])
    request = json.loads(client.calls[0]["prompt"])
    assert request["variant"] == 2
    assert request["previous_rewrites"] == ["Iris walks in beside Noah."]
    assert result["source"] == "llm"


def test_template_without_relation_or_client():
    text = template_regenerate_caption("Noah reads.", [("Iris", "woman with red hair")])
    assert text == "Iris, woman with red hair, is clearly visible in the frame. Noah reads."
    result = AbsentEntityPromptRewriter(None).rewrite(
        caption="Noah reads.", global_caption="", absent=[("Iris", "woman with red hair")], present=[("Noah", "a man")],
        relations=[], shot_index=0, attempt=1, previous=[],
    )
    assert result["source"] == "template" and result["call_id"] == ""
