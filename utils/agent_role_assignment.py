"""Exclusive index protocol, same-shot continuity and candidate exhaustion.

This calibration candidate never redraws a box or overrides a definite VLM
assignment/absence. It is not enabled by the live demo until calibration passes.
"""
import json

from utils.agent_entity_judge import VLMLocalizer, box_iou, validate_role_assignments


def resolve_role_assignments(entities, boxes, raw, previous_boxes, *, min_iou=0.65, margin=0.20):
    rows = validate_role_assignments(entities, boxes, raw)
    invalid_reasons = {"invalid or missing role assignment", "conflicting candidate claims"}
    eligible = {name for name, row in rows.items()
                if row["status"] == "uncertain" and row.get("reason") not in invalid_reasons}
    owned = {row["subject_index"] for row in rows.values() if row["status"] == "assigned"}
    # Require an unambiguous spatial winner over ALL candidates, including
    # claimed ones. Never attach a role to its second-best remaining person.
    proposals = {}
    for name in eligible:
        if name not in previous_boxes or not boxes:
            continue
        overlaps = sorted(((box_iou(previous_boxes[name], box), i) for i, box in enumerate(boxes)), reverse=True)
        best, index = overlaps[0]
        runner_up = overlaps[1][0] if len(overlaps) > 1 else 0.0
        if index not in owned and best >= min_iou and best - runner_up >= margin:
            proposals.setdefault(index, []).append(name)
    for index, names in proposals.items():
        if len(names) == 1:
            name = names[0]
            rows[name] = dict(status="assigned", subject_index=index,
                             reason="unique high-overlap same-shot continuation", resolution="temporal_candidate_match")
            owned.add(index)
    # Relative to the enumerated proposals, there is no remaining instance for
    # another role when every candidate already has a different valid owner.
    # An enumeration miss remains an upstream error, not something this infers.
    if boxes and len(owned) == len(boxes):
        for name in eligible:
            if rows[name]["status"] == "uncertain":
                rows[name] = dict(status="absent", subject_index=-1,
                                 reason="all enumerated candidates assigned to other roles", resolution="candidate_exhaustion")
    return validate_role_assignments(entities, boxes, rows)


class _IndexProtocolClient:
    """Adapt only the joint-role RPC; the underlying client owns all caching.

The old independent status/index fields admitted assigned + -1. One integer
with two reserved negative values makes that contradiction unrepresentable.
"""
    def __init__(self, client):
        self.client = client

    def __getattr__(self, name):
        return getattr(self.client, name)

    def call(self, **kwargs):
        if kwargs.get("call_kind") != "entity_assign":
            return self.client.call(**kwargs)
        prompt = json.loads(kwargs["prompt"])
        count = len(prompt["candidate_boxes_pixels"])
        schema = {
            "type": "object", "additionalProperties": False,
            "properties": {role["name"]: {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "reason": {"type": "string", "maxLength": 80},
                    "subject_index": {"type": "integer", "minimum": -2, "maximum": count - 1},
                },
                "required": ["reason", "subject_index"],
            } for role in prompt["roles"]},
            "required": [role["name"] for role in prompt["roles"]],
        }
        prompt["task_version"] = "exclusive_index_v3"
        prompt["task"] = (
            "Assign ALL scripted roles jointly to the subject IDs PRINTED on the first image. Candidate order "
            "is arbitrary, not left-to-right. Use activity, positions, earlier same-shot boxes, references and "
            "the other roles together to decide which visible subject is PLAYING each role. Hair, clothing, "
            "face, age and gender can be wrong: appearance drift is not absence, and is judged separately. "
            "A sole visible actor doing the sole role's task still plays that role despite wrong appearance. "
            "Each nonnegative index belongs to at most one role. Never redraw a box or assign an established "
            "actor to a missing third role. If the visible actors are accounted for by other roles and no "
            "additional actor exists for this role, return -1, regardless of the script saying it is present. "
            "Output subject_index >= 0 to select a printed candidate, -1 for no visible instance, and -2 only "
            "when available evidence cannot resolve role ownership. There is no separate status field. "
            "Previous boxes are spatial continuity hints, not rules across camera cuts. Ambiguous scripted "
            "presence does not force absence. Give a short positional/activity reason, not an appearance grade."
        )
        response = self.client.call(**{**kwargs, "prompt": json.dumps(prompt, sort_keys=True, ensure_ascii=False),
                                      "schema": schema, "call_kind": "entity_assign_index_v3"})
        rows = {}
        for name, row in response["parsed"].items():
            index = row.get("subject_index")
            status = ("assigned" if index >= 0 else "absent" if index == -1 else "uncertain") if type(index) is int else "uncertain"
            # Public Localizer rows keep the legacy -1 sentinel for both
            # absence and uncertainty; the protocol's -2 survives in the raw
            # cached response, not as an invalid public candidate index.
            public_index = -1 if type(index) is int and index in (-2, -1) else index
            rows[name] = {**row, "subject_index": public_index, "status": status}
        return {**response, "parsed": rows}


class ContinuityRoleLocalizer(VLMLocalizer):
    def __init__(self, client):
        super().__init__(_IndexProtocolClient(client))

    def assign_subjects(self, entities, reference_crops, frame, boxes, *, caption, previous_boxes, shot_index, inputs):
        raw = super().assign_subjects(entities, reference_crops, frame, boxes, caption=caption,
                                      previous_boxes=previous_boxes, shot_index=shot_index, inputs=inputs)
        return resolve_role_assignments(entities, boxes, raw, previous_boxes)
