"""Caption rewrite for the absent-entity regenerate action.

A masked repaint cannot grow a person: the 2026-09-04 placement test put a
person-sized box next to the anchor and renoised it at sigma 1.0 four times
without producing anyone (docs/narrastream-constraint-smoke-20260903.zh.md).
The regenerate action instead resamples the whole shot from the parent's KV
state with a rewritten caption.  The caption is rewritten by the judge's own
language model, not by a rule: the absent entities become the subject of the
first sentence with their scripted description, the present characters, the
setting and any spatial relation are kept, and the answer is strict JSON.

The only rule-shaped part is validation.  A rewrite that fails to name every
absent entity, drops a present one or runs too long is rejected and the
deterministic template takes its place, so the action never depends on a
lucky generation and every rewrite is recorded with its source.
"""
from __future__ import annotations

import json
import re
from typing import Any, Iterable, Mapping, Sequence

REWRITE_VERSION = "absent_entity_rewrite_v1"
MAX_WORDS = 90

REWRITE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"caption": {"type": "string", "maxLength": 700}},
    "required": ["caption"],
}


def _names_in(text: str, names: Iterable[str]) -> set[str]:
    found = set()
    for name in names:
        if re.search(rf"\b{re.escape(name)}\b", text, flags=re.I):
            found.add(name)
    return found


def template_regenerate_caption(
    caption: str,
    absent: Sequence[tuple[str, str]],
    relations: Sequence[Mapping[str, str]] = (),
) -> str:
    """Deterministic fallback: put the absent entities in front of the caption."""
    lead = []
    for name, description in absent:
        where = ""
        for relation in relations:
            if relation.get("subject") == name and relation.get("anchor"):
                side = relation.get("relation", "beside")
                where = (
                    f", {side} {relation['anchor']}" if side in ("beside", "behind", "in front of")
                    else f", to {relation['anchor']}'s {side}"
                )
                break
        lead.append(f"{name}, {description}, is clearly visible in the frame{where}.")
    return " ".join(lead + [caption.strip()]).strip()


def validate_rewrite(
    candidate: str,
    *,
    original: str,
    absent_names: Sequence[str],
    present_names: Sequence[str],
) -> list[str]:
    """Return the reasons a rewrite is unusable; empty means acceptable."""
    text = " ".join(str(candidate).split())
    problems = []
    if not text:
        return ["empty"]
    missing = set(absent_names) - _names_in(text, absent_names)
    if missing:
        problems.append("absent entity not named: " + ", ".join(sorted(missing)))
    kept = _names_in(original, present_names)
    dropped = kept - _names_in(text, present_names)
    if dropped:
        problems.append("present entity dropped: " + ", ".join(sorted(dropped)))
    if len(text.split()) > MAX_WORDS:
        problems.append(f"longer than {MAX_WORDS} words")
    return problems


class AbsentEntityPromptRewriter:
    """LLM caption rewrite with validation and a recorded template fallback."""

    def __init__(self, client: Any, *, max_words: int = MAX_WORDS) -> None:
        self.client = client
        self.max_words = int(max_words)

    def rewrite(
        self,
        *,
        caption: str,
        global_caption: str,
        absent: Sequence[tuple[str, str]],
        present: Sequence[tuple[str, str]],
        relations: Sequence[Mapping[str, str]],
        shot_index: int,
        attempt: int,
        previous: Sequence[str],
        inputs: Iterable[Any] = (),
    ) -> dict[str, Any]:
        absent_names = [name for name, _ in absent]
        present_names = [name for name, _ in present]
        rejected: list[dict[str, Any]] = []
        request = {
            "task_version": REWRITE_VERSION,
            "shot_caption": caption,
            "scene_caption": global_caption,
            "absent_entities": [{"name": n, "description": d} for n, d in absent],
            "present_entities": [{"name": n, "description": d} for n, d in present],
            "spatial_relations": [
                {"subject": r.get("subject"), "relation": r.get("relation"), "anchor": r.get("anchor")}
                for r in relations
            ],
            "variant": int(attempt),
            "previous_rewrites": list(previous),
            "task": (
                "The video model rendered this shot without the absent entities on screen. Rewrite the "
                "shot caption so that each absent entity is unmistakably visible from the first frame: "
                "name it in the first sentence as the grammatical subject together with its description, "
                "say where it stands relative to a present character when a spatial relation is given, and "
                "keep every present character, the setting and the actions. Do not add new characters and "
                "do not remove anyone. One paragraph of plain English, at most "
                f"{self.max_words} words, mentioning every listed name exactly as written. Use a different "
                "wording from previous_rewrites. Strict JSON only."
            ),
        }
        raw = None
        call_id = ""
        if self.client is not None:
            response = self.client.call(
                prompt=json.dumps(request, sort_keys=True, ensure_ascii=False),
                image_paths=[],
                schema=REWRITE_SCHEMA,
                call_kind=REWRITE_VERSION,
                shot_index=int(shot_index),
                temporal_inputs=inputs,
            )
            call_id = str(response.get("call_id", ""))
            raw = " ".join(str(response["parsed"].get("caption", "")).split())
            problems = validate_rewrite(
                raw, original=caption, absent_names=absent_names, present_names=present_names
            )
            if not problems:
                return {"prompt": raw, "source": "llm", "call_id": call_id, "rejected": rejected}
            rejected.append({"caption": raw, "problems": problems})
        prompt = template_regenerate_caption(caption, absent, relations)
        return {"prompt": prompt, "source": "template", "call_id": call_id, "rejected": rejected}
