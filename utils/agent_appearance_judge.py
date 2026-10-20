"""Experimental observation-first comparer; not enabled by the live demo.

Observe a crop without seeing any requested appearance, then compare with that
record visible alongside the original evidence. This is a calibration candidate,
not a claim that self-consistency or text descriptions are ground truth.
"""
from __future__ import annotations

import json

from utils.agent_entity_judge import COMPARE_SCHEMA, VLMComparer

OBSERVATION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        name: {"type": "string", "maxLength": 160}
        for name in ("subject_type", "hair_or_surface", "clothing_or_material", "face_or_shape", "visibility")
    },
    "required": ["subject_type", "hair_or_surface", "clothing_or_material", "face_or_shape", "visibility"],
}


class ObservedAppearanceComparer(VLMComparer):
    def observe(self, crop, *, shot_index, inputs):
        return self.client.call(
            prompt=(
                "Describe only the central subject ACTUALLY VISIBLE in this crop. No target description is supplied. "
                "Record subject_type (person/animal/object), hair_or_surface (actual hair colour, length, texture or "
                "surface pattern), clothing_or_material (actual garments and colours or materials), face_or_shape "
                "(visible identifying features), visibility (what is occluded, blurred or outside the crop). "
                "Do not invent hidden details, names, actions, story roles or intended appearance. For inapplicable "
                "fields write not applicable; for unobservable details say unknown. Strict JSON."
            ),
            image_paths=[crop], schema=OBSERVATION_SCHEMA, call_kind="entity_blind_appearance",
            shot_index=shot_index, temporal_inputs=inputs,
        )["parsed"]

    def _compare_crop(self, entity, reference_crop, crop, *, shot_index, inputs):
        current = self.observe(crop, shot_index=shot_index, inputs=inputs)
        reference = self.observe(reference_crop, shot_index=shot_index, inputs=inputs) if reference_crop else None
        prompt = {
            "task_version": "observation_first_v1",
            "script_description": entity.description,
            "basis": "reference continuity" if reference_crop else "authored visible attributes",
            "current_blind_observation": current,
            "reference_blind_observation": reference,
            "images": "reference then current crop" if reference_crop else "current crop only",
            "task": (
                "Judge appearance, not role ownership. The observations were made without seeing the script. "
                "Check them against the images; never replace an actual observed colour or hairstyle with the "
                "desired description. With a reference, compare the current subject's face/identity and visible "
                "appearance to the reference, not to an idealised script version. Without a reference, check "
                "each visible attribute explicitly required by the script. One clearly contradicted visible "
                "attribute means matches_reference=false even if all other attributes match. "
                "Do not penalise pose, action, camera, lighting, unobservable coat length or unspecified details. "
                "Broad descriptions such as a person, a man or a woman impose no hair, age, build or garment "
                "requirements. Give differences as a short factual description of any supported mismatch; empty "
                "when none is supported. target_phrase is diagnostic only and must not mention wrong features. "
                "Return strict JSON."
            ),
        }
        result = self.client.call(
            prompt=json.dumps(prompt, sort_keys=True, ensure_ascii=False),
            image_paths=[p for p in (reference_crop, crop) if p], schema=COMPARE_SCHEMA,
            call_kind="entity_observed_compare_crop", shot_index=shot_index, temporal_inputs=inputs,
        )["parsed"]
        return {**result, "blind_observation": current, "reference_observation": reference}
