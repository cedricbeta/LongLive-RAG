"""Development candidate: reject only grounded, visible appearance conflicts."""
import json
from concurrent.futures import ThreadPoolExecutor

from utils.agent_appearance_judge import ObservedAppearanceComparer

EVIDENCE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {name: {
        "type": "array", "maxItems": 4, "items": {"type": "string", "maxLength": 110},
    } for name in ("compatible_visible_requirements", "unobservable_or_unspecified", "supported_contradictions")},
    "required": ["compatible_visible_requirements", "unobservable_or_unspecified", "supported_contradictions"],
}


class EvidenceAppearanceComparer(ObservedAppearanceComparer):
    def compare(self, entity, reference_crop, crops, *, shot_index, inputs):
        # Separate crop requests can batch on the server. The fingerprinted
        # client serializes identical reference observations, not unrelated work.
        with ThreadPoolExecutor(max_workers=min(3, max(1, len(crops)))) as pool:
            votes = list(pool.map(lambda crop: self._compare_crop(entity, reference_crop, crop,
                                                                 shot_index=shot_index, inputs=inputs), crops))
        return self._aggregate_votes(votes, crops)

    def _compare_crop(self, entity, reference_crop, crop, *, shot_index, inputs):
        current = self.observe(crop, shot_index=shot_index, inputs=inputs)
        reference = self.observe(reference_crop, shot_index=shot_index, inputs=inputs) if reference_crop else None
        prompt = {
            "task_version": "visible_evidence_v2",
            "script_description": entity.description,
            "basis": "reference continuity" if reference_crop else "authored visible attributes",
            "current_blind_observation": current, "reference_blind_observation": reference,
            "images": "reference then current crop" if reference_crop else "current crop only",
            "task": (
                "Audit visible appearance evidence, NOT role ownership. With a reference, compare face identity "
                "and visible appearance against that reference. Without one, compare only the attributes "
                "explicitly required by the script. The blind descriptions are observations, not ground truth: "
                "verify them against the image. Return three short evidence lists. supported_contradictions must "
                "contain only unmistakable visually supported conflicts, stating expected versus actually visible. "
                "Unspecified features cannot contradict a requirement. Occluded or out-of-crop details belong "
                "in unobservable_or_unspecified, NEVER in contradictions. In particular a partial view of outerwear "
                "cannot establish its total length or whether a similar lapelled garment is a jacket or longer coat; "
                "do not claim a length/type conflict unless its distinguishing structure or hem is visible. "
                "A broad colour word in an observation does not exclude its narrower shades. Allow normal "
                "photographic shade/lighting variation and semantically overlapping descriptions; do not turn "
                "uncertain fine colour distinctions into contradictions. Clearly different colour families, "
                "hairstyles, faces or garment patterns remain contradictions when required and visible. "
                "Broad a person/man/woman descriptions impose no additional hair, age, build or clothes criteria. "
                "compatible_visible_requirements should list supported visible agreements, not copied ideals. "
                "Ignore actions, pose, camera and background. Strict JSON only."
            ),
        }
        evidence = self.client.call(
            prompt=json.dumps(prompt, sort_keys=True, ensure_ascii=False),
            image_paths=[p for p in (reference_crop, crop) if p], schema=EVIDENCE_SCHEMA,
            call_kind="entity_visible_evidence_v2", shot_index=shot_index, temporal_inputs=inputs,
        )["parsed"]
        contradictions = [s for s in evidence["supported_contradictions"] if s.strip()]
        compatible = [s for s in evidence["compatible_visible_requirements"] if s.strip()]
        return {
            "matches_reference": bool(compatible) and not contradictions,
            "differences": "; ".join(contradictions), "target_phrase": "",
            "appearance_evidence": evidence,
            "blind_observation": current, "reference_observation": reference,
        }
