"""Entity-level drift judge for the edit_shot repair loop.

Replaces the single free-form comparative call (``ComparativeDriftVerifier``)
with three narrow questions whose answers form a data structure; the verdict
and the repaint region are derived from that structure instead of from a
judge-written phrase.  Nothing here is specific to people: an entity is
whatever the script names (a man, a fish, a pocket watch).

    enumerate for each chunk-first frame: "box every main subject" -- boxes
              are the only allowed localisation candidates
    assign    jointly match scripted roles to candidate indices, allowing
              changed appearance, with one role per candidate and abstention
    compare   for each located entity: "does it still match its reference?"
              (any model-authored target phrase is diagnostic only)

SAM2 turns the per-chunk boxes into shot-level tracks (forward + reverse
propagation from every prompted frame, so a chunk the locator missed is filled
from its neighbours).  Repaint mask = drifted-entity tracks ∪ low-confidence
first-appearance tracks ∪ unassigned-subject tracks; repaint prompt = authored
caption + the frozen ``EntitySpec`` description of every affected entity.
Free-form comparer text never enters the repair prompt.  An absent entity with
nothing unassigned yields no track to repaint.  With ``place_absent`` enabled
the judge then synthesises a *placement* region instead: a person-sized box next
to an anchor named by a spatial relation in the caption ("Clara stands to Nora's
left"), or the entity's box from its last accepted shot.  Masked renoise inside
that box is the same text-channel repair; only "where to paint" changes.  With
no relation and no history the old "nothing to mask" data state remains.
Spatial relations between located entities are also assessed geometrically
(left/right by box centre, beside by gap and vertical overlap) and exposed as
``relation`` constraints; they do not gate acceptance yet.

Verdict metrics carry ``repair_score`` (strictly matching expected entities
minus unassigned subjects, over expected entities).  Failed edit branches are
not delivered; the session returns to the nearest stable parent.

Heavy dependencies (SAM2, PIL, cv2) are imported lazily; the decision logic is
pure and unit-testable with stub localiser/comparer objects.
"""

from __future__ import annotations

import copy
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import torch

from utils.agent_constraints import attach_constraint_assessments
from utils.agent_edit import FRAMES_PER_CHUNK, LATENT_GRID, read_video_frames, save_mask_payload
from utils.agent_runtime import Action, NormativeView, ShotObservation, Verdict, Verifier, file_sha256, fingerprint
from utils.agent_vlm import FingerprintedVLMClient

Box = tuple[int, int, int, int]

ENUMERATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "subjects": {
            "type": "array",
            "minItems": 0,
            "maxItems": 6,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "label": {"type": "string", "maxLength": 40},
                    "bbox_2d": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "integer"}},
                },
                "required": ["label", "bbox_2d"],
            },
        }
    },
    "required": ["subjects"],
}

COMPARE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "matches_reference": {"type": "boolean"},
        "differences": {"type": "string", "maxLength": 200},
        "target_phrase": {"type": "string", "maxLength": 120},
    },
    "required": ["matches_reference", "differences", "target_phrase"],
}


@dataclass(frozen=True)
class EntitySpec:
    """One scripted entity: name, script description, shots in which it is expected."""

    name: str
    description: str
    shots: tuple[int, ...] | None = None  # None = expected in every shot
    presence: Mapping[int, str] = field(default_factory=dict)

    def presence_in(self, shot_index: int) -> str:
        k = int(shot_index)
        return self.presence.get(k, "required" if self.shots is None or k in self.shots else "not_required")

    def expected_in(self, shot_index: int) -> bool:
        return self.presence_in(shot_index) == "required"

    def __post_init__(self) -> None:
        if any(int(k) < 0 or v not in ("required", "ambiguous") for k, v in self.presence.items()):
            raise ValueError("presence must map non-negative shot indices to required or ambiguous")
        object.__setattr__(self, "presence", {int(k): v for k, v in self.presence.items()})

    @classmethod
    def from_value(cls, value: Mapping[str, Any]) -> "EntitySpec":
        shots = value.get("shots")
        return cls(
            name=str(value["name"]),
            description=str(value.get("description", value["name"])),
            shots=None if shots is None else tuple(int(s) for s in shots),
            presence={int(k): str(v) for k, v in value.get("presence", {}).items()},
        )


def load_entity_specs(path: str | Path, scene: str) -> list[EntitySpec]:
    """Read ``{scene: [{name, description, shots}]}`` and return the scene's specs."""
    table = json.loads(Path(path).read_text(encoding="utf-8"))
    if scene not in table:
        raise KeyError(f"no entity specs for scene {scene!r} in {path}")
    return [EntitySpec.from_value(item) for item in table[scene]]


_PRONOUNS = ("him", "her", "them")


def parse_spatial_relations(caption: str, entity_names: Sequence[str]) -> list[dict[str, str]]:
    """Extract ``subject <relation> anchor`` triples between named entities.

    Relations: ``left`` / ``right`` (screen convention: "to Nora's left" is
    screen-left of Nora, which is how the benchmark captions read; the geometric
    check uses the same convention so placement and verification agree),
    ``beside`` (beside / next to / alongside), ``behind`` and ``front``.  The
    subject is the entity mentioned last before the relation phrase in the same
    sentence; a pronoun anchor resolves to the other entity mentioned before it.
    """
    canonical = {str(n).lower(): str(n) for n in entity_names}
    if not canonical or not caption:
        return []
    names = sorted(canonical.values(), key=len, reverse=True)
    name_pat = "|".join(re.escape(n) for n in names)
    pron = "|".join(_PRONOUNS)
    patterns = [
        (re.compile(rf"\b(?:to|on|at)\s+({name_pat})'s\s+(left|right)\b", re.I), lambda m: (m.group(1), m.group(2).lower())),
        (re.compile(rf"\b(?:to|on|at)\s+the\s+(left|right)\s+of\s+({name_pat}|{pron})\b", re.I), lambda m: (m.group(2), m.group(1).lower())),
        (re.compile(rf"\b(?:beside|next to|alongside)\s+({name_pat}|{pron})\b", re.I), lambda m: (m.group(1), "beside")),
        (re.compile(rf"\bbehind\s+({name_pat}|{pron})\b", re.I), lambda m: (m.group(1), "behind")),
        (re.compile(rf"\bin front of\s+({name_pat}|{pron})\b", re.I), lambda m: (m.group(1), "front")),
    ]
    mention_pat = re.compile(rf"\b({name_pat})\b", re.I)
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for sentence in re.split(r"(?<=[.!?])\s+", caption):
        mentions = [(m.start(), canonical[m.group(1).lower()]) for m in mention_pat.finditer(sentence)]
        if not mentions:
            continue
        for pattern, extract in patterns:
            for match in pattern.finditer(sentence):
                raw_anchor, relation = extract(match)
                before = [name for pos, name in mentions if pos < match.start()]
                if not before:
                    continue
                subject = before[-1]
                if raw_anchor.lower() in _PRONOUNS:
                    others = [name for name in reversed(before) if name != subject]
                    if not others:
                        continue
                    anchor = others[0]
                else:
                    anchor = canonical[raw_anchor.lower()]
                if subject == anchor:
                    continue
                key = (subject, relation, anchor)
                if key in seen:
                    continue
                seen.add(key)
                out.append({"subject": subject, "relation": relation, "anchor": anchor, "text": match.group(0)})
    return out


def spatial_relation_holds(subject: Box, anchor: Box, relation: str) -> bool | None:
    """Geometric check of a parsed relation; None when boxes cannot decide it."""
    sx = (subject[0] + subject[2]) / 2.0
    ax = (anchor[0] + anchor[2]) / 2.0
    if relation == "left":
        return sx < ax
    if relation == "right":
        return sx > ax
    if relation == "beside":
        anchor_w = max(1, anchor[2] - anchor[0])
        gap = max(subject[0] - anchor[2], anchor[0] - subject[2], 0)
        overlap_y = min(subject[3], anchor[3]) - max(subject[1], anchor[1])
        min_h = max(1, min(subject[3] - subject[1], anchor[3] - anchor[1]))
        return gap <= anchor_w and overlap_y > 0.3 * min_h
    return None  # behind / in front of: depth is not decidable from boxes


def placement_box(
    anchor: Box,
    relation: str,
    hw: tuple[int, int],
    *,
    prefer_side: str | None = None,
    occupied: Sequence[Box] = (),
) -> Box | None:
    """A person-sized box next to ``anchor`` where an absent entity should appear.

    Same height as the anchor, one anchor-width wide, separated by a small gap.
    ``left``/``right`` are fixed; ``beside``/``behind``/``front`` pick the
    preferred side (where the entity was last seen) or the side with more free
    room.  ``behind`` shrinks the box slightly and lifts it.  When the frame
    edge clips the box below half its width (anchor standing near the edge, the
    ns290 shape) the box is instead laid flush against that edge at 0.6 of the
    width, overlapping the anchor's bbox: the anchor's own segmentation track is
    excluded from the repaint mask, so only background between the two is
    touched.  A side with less than 0.08 W of room is still rejected.
    """
    H, W = int(hw[0]), int(hw[1])
    x1, y1, x2, y2 = anchor
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    gap = max(1, int(round(0.15 * w)))
    scale = 0.85 if relation == "behind" else 1.0
    bw, bh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    dy = -int(round(0.10 * h)) if relation == "behind" else 0

    def side_box(side: str) -> Box | None:
        if side == "left":
            bx2 = x1 - gap
            bx1 = bx2 - bw
        else:
            bx1 = x2 + gap
            bx2 = bx1 + bw
        by2 = y2 + dy
        by1 = by2 - bh
        cx1, cx2 = max(0, bx1), min(W, bx2)
        cy1, cy2 = max(0, by1), min(H, by2)
        if cy2 - cy1 < 0.5 * bh:
            return None
        if cx2 - cx1 < 0.5 * bw:
            room = x1 if side == "left" else W - x2
            if room < 0.08 * W:
                return None
            edge_w = max(int(round(0.6 * bw)), int(room))
            cx1, cx2 = (0, min(W, edge_w)) if side == "left" else (max(0, W - edge_w), W)
        return (int(cx1), int(cy1), int(cx2), int(cy2))

    if relation in ("left", "right"):
        return side_box(relation)

    def room(side: str) -> int:
        free = x1 if side == "left" else W - x2
        for ox1, _, ox2, _ in occupied:
            if side == "left" and ox2 <= x1:
                free = min(free, x1 - ox2)
            if side == "right" and ox1 >= x2:
                free = min(free, ox1 - x2)
        return free

    sides = ["left", "right"]
    if prefer_side in sides:
        sides = [prefer_side] + [s for s in sides if s != prefer_side]
    else:
        sides.sort(key=room, reverse=True)
    for side in sides:
        box = side_box(side)
        if box is not None:
            return box
    return None


def box_iou(a: Box, b: Box) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def scale_box_norm1000(box: Sequence[int], width: int, height: int) -> Box | None:
    """Qwen3-VL emits 0..1000 normalised coordinates; clamp to pixels."""
    x1, y1, x2, y2 = [float(v) for v in box]
    x1, x2 = sorted((max(0.0, x1 / 1000 * width), min(width - 1.0, x2 / 1000 * width)))
    y1, y2 = sorted((max(0.0, y1 / 1000 * height), min(height - 1.0, y2 / 1000 * height)))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return int(x1), int(y1), int(x2), int(y2)


# --------------------------------------------------------------------------
# Localisation / comparison backends (VLM + SAM2) behind small interfaces


class Localizer(Protocol):
    def assign_subjects(self, entities: Sequence[EntitySpec], reference_crops: Mapping[str, Path], frame: Path,
                        boxes: Sequence[Box], *, caption: str, previous_boxes: Mapping[str, Box],
                        shot_index: int, inputs: Any) -> Mapping[str, Mapping[str, Any]]: ...

    def enumerate_subjects(self, frame: Path, *, shot_index: int, inputs: Any) -> list[Box]: ...


class Comparer(Protocol):
    def compare(self, entity: EntitySpec, reference_crop: Path | None, crops: Sequence[Path], *, shot_index: int, inputs: Any) -> Mapping[str, Any]: ...


class Tracker(Protocol):
    def tracks(self, frames_dir: Path, prompts: Mapping[str, Mapping[int, Box]], num_frames: int, hw: tuple[int, int]) -> dict[str, np.ndarray]: ...


def validate_role_assignments(entities, boxes, raw):
    """Reject invalid or competing claims without inventing another assignment."""
    result = {}
    claims: dict[int, list[str]] = {}
    for entity in entities:
        value = raw.get(entity.name, {})
        row = dict(value) if isinstance(value, Mapping) else {}
        status, index = row.get("status"), row.get("subject_index")
        valid_index = isinstance(index, int) and not isinstance(index, bool)
        valid = (
            (status == "assigned" and valid_index and 0 <= index < len(boxes))
            or (status in ("absent", "uncertain") and index == -1)
        )
        if not valid:
            row = {"status": "uncertain", "subject_index": -1, "reason": "invalid or missing role assignment"}
        result[entity.name] = row
        if row["status"] == "assigned":
            claims.setdefault(row["subject_index"], []).append(entity.name)
    # Duplicate candidate geometry cannot evade mutual exclusion with two IDs.
    conflict_names = set()
    for index, names in claims.items():
        if len(names) > 1:
            conflict_names.update(names)
        for other, other_names in claims.items():
            if other > index and box_iou(boxes[index], boxes[other]) >= 0.85:
                conflict_names.update(names + other_names)
    for name in conflict_names:
        result[name] = {"status": "uncertain", "subject_index": -1, "reason": "conflicting candidate claims"}
    return result


class VLMLocalizer:
    """Qwen3-VL grounding through the fingerprinted client (cached, audited)."""

    def __init__(self, client: FingerprintedVLMClient) -> None:
        self.client = client

    @staticmethod
    def _size(frame: Path) -> tuple[int, int]:
        from PIL import Image

        return Image.open(frame).size

    def _numbered_candidates(self, frame: Path, boxes: Sequence[Box]) -> Path:
        """Make candidate IDs visible; enumeration order has no spatial meaning."""
        from PIL import Image, ImageDraw, ImageFont
        key = fingerprint({"image": file_sha256(frame), "boxes": boxes, "overlay_version": 1})
        target = self.client.cache_dir / "assignment_inputs" / f"{key}.png"
        if not target.is_file():
            with Image.open(frame) as source:
                canvas = source.convert("RGB")
            draw = ImageDraw.Draw(canvas)
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", max(20, canvas.width // 40))
            except OSError:
                font = ImageFont.load_default()
            for index, box in enumerate(boxes):
                draw.rectangle(box, outline="#00ff99", width=max(2, canvas.width // 320))
                label = f"ID {index}"
                x, y = box[:2]
                bounds = draw.textbbox((x + 3, y + 2), label, font=font)
                draw.rectangle((bounds[0] - 3, bounds[1] - 2, bounds[2] + 3, bounds[3] + 2), fill="black")
                draw.text((x + 3, y + 2), label, font=font, fill="white")
            target.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(target)
        return target

    def assign_subjects(self, entities, reference_crops, frame, boxes, *, caption, previous_boxes, shot_index, inputs):
        if not boxes:
            return {e.name: {"status": "absent", "subject_index": -1, "reason": "no enumerated subject"} for e in entities}
        W, H = self._size(frame)
        numbered_frame = self._numbered_candidates(frame, boxes)
        references = [(e.name, reference_crops[e.name]) for e in entities if e.name in reference_crops]
        item_schema = {
            "type": "object", "additionalProperties": False,
            "properties": {
                "status": {"type": "string", "enum": ["assigned", "absent", "uncertain"]},
                "subject_index": {"type": "integer", "minimum": -1, "maximum": len(boxes) - 1},
                "reason": {"type": "string", "maxLength": 80},
            },
            "required": ["status", "subject_index", "reason"],
        }
        schema = {"type": "object", "additionalProperties": False,
                  "properties": {e.name: item_schema for e in entities}, "required": [e.name for e in entities]}
        # Appearance requirements belong to the comparer. Remove verbatim
        # appearance appositives from this judge-only caption view; the authored
        # caption used for generation and repair remains untouched.
        role_caption = caption
        for entity in entities:
            role_caption = re.sub(
                rf"(\b{re.escape(entity.name)},\s*){re.escape(entity.description)}(?=\s*[,.;])",
                r"\1the scripted character", role_caption, flags=re.I,
            )
        prompt = {
            "roles": [{"name": e.name, "presence": e.presence_in(shot_index)} for e in entities],
            "role_caption": role_caption,
            "candidate_boxes_pixels": {str(i): list(box) for i, box in enumerate(boxes)},
            "frame_size_pixels": [W, H],
            "previous_chunk_role_boxes_pixels": {name: list(box) for name, box in previous_boxes.items()},
            "images": {"0": "current frame with printed candidate IDs", **{str(i + 1): f"earlier accepted reference of {name}" for i, (name, _) in enumerate(references)}},
            "task": (
                "Assign ALL scripted roles jointly to the enumerated candidate indices. Decide which visible subject "
                "is marked by each printed ID before selecting it. Candidate order is arbitrary: index 0 does NOT "
                "necessarily mean leftmost. The subject_index must match the ID PRINTED ON THE IMAGE. Then decide who "
                "is PLAYING each role using spatial relations, activity, continuity and the other roles together. "
                "References help establish continuity, but hair, clothing, age, face and even gender may have drifted: "
                "a wrong appearance is NOT absence. Appearance correctness is judged by a separate comparer. "
                "If a single visible subject performs the sole scripted role's activity, assign that role even when "
                "its appearance is entirely wrong. Appearance differences alone do not justify uncertain either. "
                "For example a wrong-haired woman in Clara's established place can still be Clara. Never assign Nora "
                "to Clara just because Clara is missing. One role gets at most one index and each index belongs to at "
                "most one role. Do not draw or change any box. Use status=absent, index=-1 only when the role has no "
                "visible instance; use uncertain, index=-1 when the available evidence cannot resolve who is who. "
                "Previous boxes are hints only: camera cuts and movement can change positions. "
                "An ambiguous scripted presence does not mean the entity must be absent. Give a short reason about "
                "position, activity or continuity; do not assess appearance correctness."
            ),
            "output": "strict JSON matching the supplied schema",
        }
        res = self.client.call(
            prompt=json.dumps(prompt, sort_keys=True, ensure_ascii=False),
            image_paths=[numbered_frame, *(path for _, path in references)],
            schema=schema,
            call_kind="entity_assign",
            shot_index=int(shot_index),
            temporal_inputs=inputs,
        )
        return validate_role_assignments(entities, boxes, res["parsed"])

    def enumerate_subjects(self, frame, *, shot_index, inputs):
        W, H = self._size(frame)
        prompt = (
            f"Image is {W}x{H} pixels. Return one bounding box bbox_2d=[x1,y1,x2,y2] for EVERY main subject in the "
            "image: each person, animal or prominent object that a viewer would call a subject of the shot (not "
            "furniture or background). Output strict JSON matching the supplied schema."
        )
        res = self.client.call(
            prompt=prompt, image_paths=[frame], schema=ENUMERATE_SCHEMA, call_kind="entity_enumerate",
            shot_index=int(shot_index), temporal_inputs=inputs,
        )
        boxes = [scale_box_norm1000(s["bbox_2d"], W, H) for s in res["parsed"]["subjects"]]
        return [b for b in boxes if b]


class VLMComparer:
    def __init__(self, client: FingerprintedVLMClient) -> None:
        self.client = client

    def compare(self, entity, reference_crop, crops, *, shot_index, inputs):
        """Three separate crop judgments; agreement is measured, never self-reported."""
        votes = [self._compare_crop(entity, reference_crop, crop, shot_index=shot_index, inputs=inputs) for crop in crops]
        return self._aggregate_votes(votes, crops)

    @staticmethod
    def _aggregate_votes(votes, crops):
        matches = [bool(v["matches_reference"]) for v in votes]
        positive = sum(matches)
        agreement = max(positive, len(matches) - positive) / len(matches) if matches else 0.0
        return {
            "matches_reference": positive > len(matches) / 2,
            "differences": "; ".join(dict.fromkeys(v["differences"] for v in votes if v["differences"])),
            "target_phrase": "",  # diagnostic prose is never a repair instruction
            "confidence": agreement,
            "confidence_basis": "separate_crop_vote_agreement",
            "crop_votes": matches,
            "crop_paths": [str(p) for p in crops],
            "reference_registration_eligible": len(matches) == 3 and all(matches),
        }

    def _compare_crop(self, entity, reference_crop, crop, *, shot_index, inputs):
        prompt = {
            "entity": entity.name,
            "script_description": entity.description,
            "images": (
                "First image: reference crop (the entity as established). Remaining images: the same entity in the "
                "current shot."
                if reference_crop
                else "All images: the entity in the current shot; compare against the script description only."
            ),
            "task": (
                "Does the entity in the current shot still match its reference identity and visible appearance? "
                "For a person, compare face, hair, build, clothing, colours and distinguishing marks. For an animal, "
                "compare species, coat/skin/plumage, colours, build and distinguishing marks. For an object, compare "
                "object type, shape, material, colour, scale and distinguishing marks. Judge appearance only: this "
                "call does not verify actions, spatial relations or scene state. Ignore pose, camera, framing and lighting. "
                "differences: one short sentence naming what changed, empty if it matches. target_phrase: a short "
                "POSITIVE phrase describing how the changed aspect should look (e.g. 'curly red hair, moss green "
                "coat'); never mention the wrong appearance. target_phrase is diagnostic metadata only; it is not "
                "trusted as a generation instruction. Judge only this current crop; other crops are judged separately."
            ),
            "output": "strict JSON matching the supplied schema",
        }
        res = self.client.call(
            prompt=json.dumps(prompt, sort_keys=True, ensure_ascii=False),
            image_paths=[p for p in (reference_crop, crop) if p],
            schema=COMPARE_SCHEMA,
            call_kind="entity_compare_crop",
            shot_index=int(shot_index),
            temporal_inputs=inputs,
        )
        return dict(res["parsed"])


class Sam2Tracker:
    """Shot-level SAM2 video propagation from per-chunk box prompts."""

    def __init__(self, device: str = "cuda:0", model: str = "facebook/sam2.1-hiera-large") -> None:
        from sam2.sam2_video_predictor import SAM2VideoPredictor

        self.pred = SAM2VideoPredictor.from_pretrained(model, device=device)

    def tracks(self, frames_dir, prompts, num_frames, hw):
        H, W = hw
        out = {key: np.zeros((num_frames, H, W), dtype=bool) for key in prompts}
        if not any(prompts.values()):
            return out
        ids = {key: i + 1 for i, key in enumerate(prompts)}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = self.pred.init_state(video_path=str(frames_dir))
            first = num_frames
            for key, per_frame in prompts.items():
                for frame_idx, box in sorted(per_frame.items()):
                    self.pred.add_new_points_or_box(state, frame_idx=int(frame_idx), obj_id=ids[key], box=np.array(box, dtype=np.float32))
                    first = min(first, int(frame_idx))
            by_id = {v: k for k, v in ids.items()}
            for reverse in (False, True):
                if reverse and first == 0:
                    continue
                for frame_idx, obj_ids, logits in self.pred.propagate_in_video(state, start_frame_idx=first, reverse=reverse):
                    masks = (logits > 0).cpu().numpy()
                    for j, oid in enumerate(obj_ids):
                        out[by_id[int(oid)]][frame_idx] |= masks[j, 0]
            self.pred.reset_state(state)
        return out


# --------------------------------------------------------------------------
# The judge


@dataclass
class EntityStatus:
    name: str
    expected: bool
    located_chunks: list[int] = field(default_factory=list)
    matches: bool | None = None
    differences: str = ""
    target_phrase: str = ""  # untrusted comparer output; diagnostic only
    confidence: float = 0.0
    comparison_basis: str | None = None
    assignment_uncertain: bool = False
    crop_votes: list[bool] | None = None
    registration_eligible: bool = True
    coverage: float = 0.0

    @property
    def present(self) -> bool:
        return bool(self.located_chunks)


class EntityDriftVerifier(Verifier):
    """Verifier producing edit_shot plans from located entity tracks.

    The backend must expose, in ``observation.telemetry``: ``shot_video`` (path of
    the candidate render), ``chunk_start``/``chunk_end`` (inclusive chunk range of
    the shot) and ``caption``-bearing normative state.  Reference crops are taken
    from accepted candidates (first accepted, high-confidence script match of
    each entity).  A first appearance cannot pass until its script comparison
    reaches the registration threshold; registration provenance is emitted in
    verdict metrics.
    """

    def __init__(
        self,
        entities: Sequence[EntitySpec],
        *,
        localizer: Localizer,
        comparer: Comparer,
        tracker: Tracker,
        work_dir: str | Path,
        edit_entry_sigma: float = 1.0,
        min_track_coverage: float = 0.01,
        unassigned_iou: float = 0.3,
        reference_registration_min_confidence: float = 0.75,
        place_absent: str = "relation_or_history",
        placement_prompt: str = "caption_and_spec",
        grid: tuple[int, int] = LATENT_GRID,
        frames_per_chunk: int = FRAMES_PER_CHUNK,
    ) -> None:
        self.entities = list(entities)
        if len({e.name for e in self.entities}) != len(self.entities):
            raise ValueError("scripted entity names must be unique for joint role assignment")
        self.localizer = localizer
        self.comparer = comparer
        self.tracker = tracker
        self.work_dir = Path(work_dir)
        self.edit_entry_sigma = float(edit_entry_sigma)
        self.min_track_coverage = float(min_track_coverage)
        self.unassigned_iou = float(unassigned_iou)
        self.reference_registration_min_confidence = float(reference_registration_min_confidence)
        if not 0.0 <= self.reference_registration_min_confidence <= 1.0:
            raise ValueError("reference_registration_min_confidence must be between 0 and 1")
        if place_absent not in ("never", "relation_or_history", "always"):
            raise ValueError("place_absent must be never, relation_or_history or always")
        self.place_absent = str(place_absent)
        if placement_prompt not in ("caption_and_spec", "spec_only"):
            raise ValueError("placement_prompt must be caption_and_spec or spec_only")
        # spec_only: a placement repaint is prompted with the absent entity's
        # description alone (plus the scene's global caption if any) instead of
        # the shot caption, which mostly describes the entities already present.
        self.placement_prompt = str(placement_prompt)
        self.grid = tuple(grid)
        self.fpc = int(frames_per_chunk)
        self.reference_crops: dict[str, Path] = {}
        self.reference_provenance: dict[str, dict[str, Any]] = {}
        # Per entity: pixel boxes (by local chunk) from its last accepted shot.
        # Placement falls back to this when the caption names no relation.
        self.last_tracks: dict[str, dict[str, Any]] = {}

    def snapshot_state(self) -> dict[str, Any]:
        """Return the persistent identity state for tentative-branch rollback."""
        return {
            "reference_crops": dict(self.reference_crops),
            "reference_provenance": copy.deepcopy(self.reference_provenance),
            "last_tracks": copy.deepcopy(self.last_tracks),
        }

    def restore_state(self, state: Mapping[str, Any]) -> None:
        """Restore identity references after a rejected speculative branch."""
        self.reference_crops = {
            str(name): Path(path)
            for name, path in dict(state.get("reference_crops", {})).items()
        }
        self.reference_provenance = copy.deepcopy(
            dict(state.get("reference_provenance", {}))
        )
        self.last_tracks = copy.deepcopy(dict(state.get("last_tracks", {})))

    # -- placement of absent entities -------------------------------------
    def _track_boxes(self, track: np.ndarray, num_chunks: int) -> dict[int, Box]:
        """Tight per-chunk bbox of a SAM2 track (first frame of each chunk).

        The VLM locator's boxes are loose (on ns290 Nora's box often spanned
        Ethan too); geometry for placement and relation checks uses these.
        """
        boxes: dict[int, Box] = {}
        for c in range(num_chunks):
            frame = track[c * self.fpc]
            ys, xs = np.nonzero(frame)
            if len(xs) == 0:
                continue
            boxes[c] = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        return boxes

    def _entity_geometry(
        self,
        name: str,
        tracks: Mapping[str, np.ndarray],
        prompts: Mapping[str, Mapping[int, Box]],
        num_chunks: int,
    ) -> dict[int, Box]:
        """Per-chunk boxes for a located entity: track bbox, else locator box."""
        if name in tracks:
            boxes = self._track_boxes(tracks[name], num_chunks)
            if boxes:
                return boxes
        return {int(f) // self.fpc: tuple(box) for f, box in prompts.get(name, {}).items()}

    @staticmethod
    def _nearest_chunk_box(boxes: Mapping[int, Box], chunk: int) -> Box | None:
        if not boxes:
            return None
        if chunk in boxes:
            return boxes[chunk]
        nearest = min(boxes, key=lambda c: abs(int(c) - chunk))
        return boxes[nearest]

    def _placement_plan(
        self,
        name: str,
        relations: Sequence[Mapping[str, str]],
        prompts: Mapping[str, Mapping[int, Box]],
        statuses: Mapping[str, "EntityStatus"],
        tracks: Mapping[str, np.ndarray],
        hw: tuple[int, int],
        num_chunks: int,
    ) -> dict[str, Any] | None:
        """Where should the absent entity ``name`` be painted, chunk by chunk?"""
        H, W = int(hw[0]), int(hw[1])
        located = [n for n, s in statuses.items() if s.present]
        geometry = {n: self._entity_geometry(n, tracks, prompts, num_chunks) for n in located}
        history = self.last_tracks.get(name)
        hist_boxes: dict[int, Box] = {}
        if history:
            hist_boxes = {int(c): tuple(int(v) for v in b) for c, b in dict(history.get("boxes", {})).items()}

        for relation in relations:
            if relation["subject"] != name:
                continue
            anchor = relation["anchor"]
            if anchor not in statuses or not statuses[anchor].present:
                continue
            anchor_boxes = geometry.get(anchor, {})
            boxes: dict[int, Box] = {}
            for c in range(num_chunks):
                abox = self._nearest_chunk_box(anchor_boxes, c)
                if abox is None:
                    continue
                prefer = None
                hbox = self._nearest_chunk_box(hist_boxes, c)
                if hbox is not None:
                    prefer = "left" if (hbox[0] + hbox[2]) < (abox[0] + abox[2]) else "right"
                occupied = [
                    self._nearest_chunk_box(geometry[other], c)
                    for other in located
                    if other != anchor and self._nearest_chunk_box(geometry[other], c) is not None
                ]
                box = placement_box(abox, relation["relation"], (H, W), prefer_side=prefer, occupied=occupied)
                if box is not None:
                    boxes[c] = box
            if boxes:
                return {"source": "relation", "relation": dict(relation), "anchor": anchor, "boxes": {int(c): list(b) for c, b in boxes.items()}}

        if hist_boxes:
            boxes = {}
            for c in range(num_chunks):
                hbox = self._nearest_chunk_box(hist_boxes, c)
                x1, y1, x2, y2 = hbox
                box = (max(0, x1), max(0, y1), min(W, x2), min(H, y2))
                if box[2] - box[0] > 0 and box[3] - box[1] > 0:
                    boxes[c] = box
            if boxes:
                return {"source": "history", "relation": None, "history_shot": int(history.get("shot_index", -1)), "boxes": {int(c): list(b) for c, b in boxes.items()}}

        if self.place_absent != "always":
            return None
        # Free-space fallback: the widest column band no located entity occupies.
        occupied_all = [tuple(b) for n in located for b in geometry[n].values()]
        edges = sorted({0, W} | {b[0] for b in occupied_all} | {b[2] for b in occupied_all})
        best: tuple[int, int] | None = None
        for lo, hi in zip(edges, edges[1:]):
            if any(b[0] < hi and b[2] > lo for b in occupied_all):
                continue
            if best is None or hi - lo > best[1] - best[0]:
                best = (lo, hi)
        if best is None or best[1] - best[0] < 0.1 * W:
            return None
        widths = [b[2] - b[0] for b in occupied_all] or [W // 4]
        heights = [b[3] - b[1] for b in occupied_all] or [int(0.7 * H)]
        bottoms = [b[3] for b in occupied_all] or [H]
        bw = min(int(np.median(widths)), best[1] - best[0])
        bh = int(np.median(heights))
        cx = (best[0] + best[1]) // 2
        y2 = int(np.median(bottoms))
        box = (max(0, cx - bw // 2), max(0, y2 - bh), min(W, cx + bw // 2), min(H, y2))
        return {"source": "free_space", "relation": None, "boxes": {c: list(box) for c in range(num_chunks)}}

    def _rasterize_boxes(
        self,
        boxes: Mapping[int, Sequence[int]],
        exclude: Sequence[np.ndarray],
        n_frames: int,
        hw: tuple[int, int],
    ) -> np.ndarray:
        H, W = int(hw[0]), int(hw[1])
        track = np.zeros((n_frames, H, W), dtype=bool)
        for c, (x1, y1, x2, y2) in boxes.items():
            track[int(c) * self.fpc : (int(c) + 1) * self.fpc, int(y1):int(y2), int(x1):int(x2)] = True
        for other in exclude:
            track &= ~other.astype(bool)
        return track

    def _assess_relations(
        self,
        relations: Sequence[Mapping[str, str]],
        statuses: Mapping[str, "EntityStatus"],
        prompts: Mapping[str, Mapping[int, Box]],
        tracks: Mapping[str, np.ndarray],
        num_chunks: int,
    ) -> list[dict[str, Any]]:
        """Geometric status of every parsed relation (does not gate acceptance)."""
        out: list[dict[str, Any]] = []
        geometry: dict[str, dict[int, Box]] = {}
        for relation in relations:
            subject, anchor = relation["subject"], relation["anchor"]
            row: dict[str, Any] = {**relation, "chunks_checked": 0, "chunks_satisfied": 0, "status": "unknown"}
            if any(name in statuses and not statuses[name].expected for name in (subject, anchor)):
                row["excluded_ambiguous_presence"] = True
                out.append(row)
                continue
            if (
                subject in statuses and anchor in statuses
                and statuses[subject].present and statuses[anchor].present
            ):
                for n in (subject, anchor):
                    if n not in geometry:
                        geometry[n] = self._entity_geometry(n, tracks, prompts, num_chunks)
                for c in range(num_chunks):
                    sbox = geometry[subject].get(c)
                    abox = geometry[anchor].get(c)
                    if sbox is None or abox is None:
                        continue
                    holds = spatial_relation_holds(sbox, abox, relation["relation"])
                    if holds is None:
                        continue
                    row["chunks_checked"] += 1
                    row["chunks_satisfied"] += int(holds)
                if row["chunks_checked"]:
                    ratio = row["chunks_satisfied"] / row["chunks_checked"]
                    row["status"] = "satisfied" if ratio >= 0.5 else "violated"
                    row["score"] = ratio
            out.append(row)
        return out

    # -- frame plumbing ---------------------------------------------------
    def _shot_frames(self, video: Path, chunk_start: int, chunk_end: int) -> tuple[Path, int, tuple[int, int]]:
        """Dump the shot's frames as 0..N-1 JPEGs (SAM2 input) + chunk-first PNGs."""
        key = f"{file_sha256(video)[:16]}_c{chunk_start:02d}-{chunk_end:02d}"
        d = self.work_dir / "frames" / key
        n = (chunk_end - chunk_start + 1) * self.fpc
        if not (d / f"{n - 1}.jpg").exists():
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)
            frames = read_video_frames(video, chunk_start * self.fpc, n)
            if len(frames) != n:
                raise RuntimeError(f"expected {n} frames from {video}, got {len(frames)}")
            for i, im in enumerate(frames):
                im.save(d / f"{i}.jpg", quality=95)
                if i % self.fpc == 0:
                    im.save(d / f"chunk{i // self.fpc:02d}.png")
        from PIL import Image

        with Image.open(d / "0.jpg") as im:
            W, H = im.size
        return d, n, (H, W)

    @staticmethod
    def _crop(frame: Path, box: Box, out: Path, margin: float = 0.1) -> Path:
        from PIL import Image

        im = Image.open(frame)
        W, H = im.size
        x1, y1, x2, y2 = box
        mx, my = int((x2 - x1) * margin), int((y2 - y1) * margin)
        out.parent.mkdir(parents=True, exist_ok=True)
        im.crop((max(0, x1 - mx), max(0, y1 - my), min(W, x2 + mx), min(H, y2 + my))).save(out)
        return out

    def _soft(self, track: np.ndarray, num_chunks: int) -> dict[int, torch.Tensor]:
        t = torch.from_numpy(track.astype(np.float32)).unsqueeze(1)
        soft = torch.nn.functional.adaptive_avg_pool2d(t, self.grid).squeeze(1)
        return {c: soft[c * self.fpc : (c + 1) * self.fpc] for c in range(num_chunks)}

    def _repair_dependencies(
        self,
        *,
        shot_index: int,
        entity_names: Sequence[str],
        metrics: Mapping[str, Any],
        repairs_unassigned_subject: bool,
    ) -> dict[str, Any]:
        """Describe which generic predicates and future appearances an edit can affect.

        Explicit ``EntitySpec.shots`` values are safe authored dependencies.  A
        ``shots=None`` spec deliberately stays symbolic: expanding it requires
        the scene's shot count, which belongs to the runtime rather than this
        verifier's causal view.
        """

        entity_ids = sorted(set(str(name) for name in entity_names))
        entity_set = set(entity_ids)
        constraint_ids: set[str] = set()
        for raw in metrics.get("constraint_assessments", ()):
            if not isinstance(raw, Mapping) or raw.get("status") == "satisfied":
                continue
            constraint = raw.get("constraint")
            if not isinstance(constraint, Mapping):
                continue
            subjects = {str(item) for item in constraint.get("subjects", ())}
            parameters = constraint.get("parameters")
            is_unassigned_count = (
                repairs_unassigned_subject
                and constraint.get("kind") == "count"
                and isinstance(parameters, Mapping)
                and parameters.get("category") == "unassigned_subject"
            )
            if subjects & entity_set or is_unassigned_count:
                constraint_ids.add(str(constraint["id"]))

        future_shots: set[int] = set()
        all_future = False
        by_name = {entity.name: entity for entity in self.entities}
        for name in entity_ids:
            spec = by_name[name]
            if spec.shots is None:
                all_future = True
            else:
                future_shots.update(int(index) for index in spec.shots if int(index) > shot_index)
        dependencies: dict[str, Any] = {
            "entity_ids": entity_ids,
            "constraint_ids": sorted(constraint_ids),
            "future_shots": sorted(future_shots),
        }
        if all_future:
            dependencies["all_future"] = True
        return dependencies

    # -- the verdict ------------------------------------------------------
    def judge(self, observation: ShotObservation, normative: NormativeView) -> Verdict:
        k = int(observation.shot_index)
        tele = observation.telemetry
        video = Path(str(tele["shot_video"]))
        c0, c1 = int(tele["chunk_start"]), int(tele["chunk_end"])
        num_chunks = c1 - c0 + 1
        frames_dir, n_frames, hw = self._shot_frames(video, c0, c1)
        chunk_pngs = [frames_dir / f"chunk{c:02d}.png" for c in range(num_chunks)]
        caption = str(normative.shot_state[-1].get("caption", "")) if normative.shot_state else ""
        inputs = observation.inputs
        candidate_key = observation.input_fingerprint[:12] or file_sha256(video)[:12]
        attempt_dir = self.work_dir / (
            f"shot{k:02d}_attempt{observation.attempt}_{candidate_key}"
        )

        # 1. Enumerate once per frame, then assign all roles jointly. Ambiguous
        # scripted presences can claim/protect a subject but are never scored.
        relevant = [e for e in self.entities if e.presence_in(k) != "not_required"]
        statuses = {e.name: EntityStatus(name=e.name, expected=e.expected_in(k)) for e in self.entities}
        prompts: dict[str, dict[int, Box]] = {e.name: {} for e in relevant}
        enumerated, assignments, claimed = {}, {}, {}
        previous_boxes: dict[str, Box] = {}
        for c, png in enumerate(chunk_pngs):
            boxes = self.localizer.enumerate_subjects(png, shot_index=k, inputs=inputs)
            enumerated[c] = boxes
            raw = self.localizer.assign_subjects(
                relevant, self.reference_crops, png, boxes, caption=caption,
                previous_boxes=previous_boxes, shot_index=k, inputs=inputs,
            ) if relevant else {}
            rows = validate_role_assignments(relevant, boxes, raw)
            assignments[c] = rows
            claimed[c] = set()
            previous_boxes = {}
            for name, row in rows.items():
                if row["status"] == "assigned":
                    index = row["subject_index"]
                    prompts[name][c * self.fpc] = boxes[index]
                    statuses[name].located_chunks.append(c)
                    claimed[c].add(index)
                    previous_boxes[name] = boxes[index]
                elif row["status"] == "uncertain":
                    statuses[name].assignment_uncertain = True

        # 2. SAM2 gets only the selected enumeration boxes. Segmentation never
        # decides role ownership; a thin/empty track cannot unassign Nora.
        tracks = self.tracker.tracks(frames_dir, prompts, n_frames, hw)
        softs = {key: self._soft(track, num_chunks) for key, track in tracks.items()}
        for key, soft in softs.items():
            statuses[key].coverage = float(np.mean([(s >= 0.35).float().mean().item() for s in soft.values()]))

        # 3. Unclaimed candidate IDs, not mask-box overlap, identify extras.
        unassigned: dict[str, dict[int, Box]] = {}
        uncertain_chunks = set()
        for c, boxes in enumerated.items():
            if any(row["status"] == "uncertain" for row in assignments[c].values()):
                uncertain_chunks.add(c)
            for index, box in enumerate(boxes):
                if index in claimed[c]:
                    continue
                key = None
                for ukey, per_frame in unassigned.items():  # same subject across chunks
                    if box_iou(box, per_frame[max(per_frame)]) >= self.unassigned_iou:
                        key = ukey
                        break
                if key is None:
                    key = f"unassigned_{len(unassigned)}"
                    unassigned[key] = {}
                unassigned[key][c * self.fpc] = box
        unassigned_live: set[str] = set()
        unresolved = [s for s in statuses.values() if s.expected and not s.present and s.assignment_uncertain]
        # An unresolved role (including ambiguous cast) could own an unclaimed
        # subject. Do not repaint those subjects based on an uncertain match.
        ambiguous_unlocated = any(e.presence_in(k) == "ambiguous" and not statuses[e.name].present for e in relevant)
        if unassigned and not uncertain_chunks and not ambiguous_unlocated and any(not s.present for s in statuses.values() if s.expected):
            # impostors only matter when a scripted entity is absent; track them
            extra = self.tracker.tracks(frames_dir, unassigned, n_frames, hw)
            for key, track in extra.items():
                softs[key] = self._soft(track, num_chunks)
                if float(np.mean([(s >= 0.35).float().mean().item() for s in softs[key].values()])) >= self.min_track_coverage:
                    unassigned_live.add(key)

        # 4. compare every located entity with its reference.  With no frozen
        #    reference yet, the comparison basis is the EntitySpec description.
        comparison_sources = {}
        for ent in self.entities:
            status = statuses[ent.name]
            if not status.present or not status.expected:
                continue
            crops = []
            chunks = sorted(status.located_chunks)
            samples = [(c * self.fpc, chunk_pngs[c], prompts[ent.name][c * self.fpc]) for c in chunks]
            # Short appearances still need three distinct frame observations,
            # never three repeated votes on the same frame.
            if len(samples) < 3:
                for c in chunks:
                    for offset in (self.fpc // 2, self.fpc - 1):
                        frame_index = c * self.fpc + offset
                        track = tracks.get(ent.name)
                        if track is None:
                            continue
                        yy, xx = np.where(track[frame_index])
                        if len(xx):
                            samples.append((frame_index, frames_dir / f"{frame_index}.jpg", (int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1)))
            samples.sort(key=lambda item: item[0])
            selected = [samples[int(i)] for i in np.linspace(0, len(samples) - 1, min(3, len(samples)), dtype=int)]
            comparison_sources[ent.name] = selected
            for frame_index, frame, box in selected:
                crops.append(self._crop(frame, box, attempt_dir / f"{ent.name}_frame{frame_index:03d}.png", margin=0.0))
            reference = self.reference_crops.get(ent.name)
            status.comparison_basis = "reference_crop" if reference is not None else "script_description"
            result = self.comparer.compare(ent, reference, crops, shot_index=k, inputs=inputs)
            status.matches = bool(result.get("matches_reference", False))
            status.differences = str(result.get("differences", ""))
            status.target_phrase = str(result.get("target_phrase", ""))
            try:
                confidence = float(result.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            status.confidence = confidence if np.isfinite(confidence) and 0.0 <= confidence <= 1.0 else 0.0
            status.crop_votes = result.get("crop_votes")
            status.registration_eligible = bool(result.get("reference_registration_eligible", True))

        # 5. decision from the data structure
        expected = [s for s in statuses.values() if s.expected]
        drifted = [s for s in expected if s.present and s.matches is False]
        absent = [s for s in expected if not s.present and not s.assignment_uncertain]
        good = [s for s in expected if s.present and s.matches]
        # The first accepted appearance becomes the durable identity anchor.
        # A merely positive but low-confidence script comparison is therefore
        # not good enough to commit: repaint/review it while the branch is still
        # tentative instead of letting a weak reference govern every later shot.
        uncertain_first_appearance = [
            s
            for s in good
            if s.name not in self.reference_crops
            and s.comparison_basis == "script_description"
            and (not s.registration_eligible or s.confidence < self.reference_registration_min_confidence)
        ]
        uncertain_names = {s.name for s in uncertain_first_appearance}
        strict_good = [s for s in good if s.name not in uncertain_names]
        # An unassigned subject is an impostor only when it can be standing in
        # for an absent scripted entity (enforced above: they are only tracked
        # then).  With every expected entity located and matching, extra
        # subjects (a bookend, a passer-by) are outside the judge's mandate --
        # the mandate is the identity of scripted entities.
        n_exp = max(1, len(expected))
        repair_score = (len(strict_good) - len(unassigned_live)) / n_exp if expected else None
        metrics = {
            "entities": {
                s.name: {
                    "expected": s.expected, "located_chunks": s.located_chunks, "matches": s.matches,
                    "differences": s.differences, "target_phrase": s.target_phrase,
                    "target_phrase_used_for_repair": False, "confidence": s.confidence,
                    "comparison_basis": s.comparison_basis, "coverage": round(s.coverage, 4),
                    "presence": next(e.presence_in(k) for e in self.entities if e.name == s.name),
                    "assignment_uncertain": s.assignment_uncertain,
                    "crop_votes": s.crop_votes,
                    "reference_registration_eligible": s.registration_eligible,
                    "confidence_basis": "separate_crop_vote_agreement" if s.crop_votes is not None else "legacy_comparer",
                    "boxes": {int(f) // self.fpc: list(b) for f, b in prompts.get(s.name, {}).items()},
                }
                for s in statuses.values()
            },
            "unassigned_subjects": sorted(unassigned_live),
            "enumerated_boxes": {c: [list(b) for b in boxes] for c, boxes in enumerated.items()},
            "role_assignments": assignments,
            "uncertain_assignment_chunks": sorted(uncertain_chunks),
            "excluded_ambiguous_entities": [e.name for e in relevant if e.presence_in(k) == "ambiguous"],
            "evaluation_status": "evaluated" if expected else "excluded_no_required_entities",
            "unassigned_boxes": {
                key: {int(f) // self.fpc: list(b) for f, b in per_frame.items()}
                for key, per_frame in unassigned.items()
            },
            "repair_score": repair_score,
            "reference_available": {s.name: s.name in self.reference_crops for s in expected},
            "reference_registration_policy": {
                "requires_accepted_verdict": True,
                "requires_script_description_match": True,
                "min_confidence": self.reference_registration_min_confidence,
                "requires_three_positive_crop_votes": isinstance(self.comparer, VLMComparer),
            },
            "reference_registration": {},
        }
        relations = parse_spatial_relations(caption, [e.name for e in self.entities])
        metrics["spatial_relations"] = self._assess_relations(relations, statuses, prompts, tracks, num_chunks)
        # Keep the legacy entity metrics during migration, and expose the same
        # evidence as neutral presence/identity/attribute/count/relation predicates.
        metrics = attach_constraint_assessments(metrics, shot_index=k)
        accept = (
            not drifted
            and not absent
            and not unassigned_live
            and not uncertain_first_appearance
            and not unresolved
        )
        if accept:
            # Freeze only a first appearance that was accepted while being
            # compared directly against the frozen script description and met
            # the separately configurable registration threshold.
            for s in good:
                if s.name in self.reference_crops:
                    metrics["reference_registration"][s.name] = {
                        "status": "already_registered",
                        "provenance": self.reference_provenance.get(s.name),
                    }
                    continue
                if s.comparison_basis != "script_description":
                    metrics["reference_registration"][s.name] = {"status": "deferred_not_script_match"}
                    continue
                if not s.registration_eligible or s.confidence < self.reference_registration_min_confidence:
                    raise RuntimeError(
                        "low-confidence first appearance reached reference registration"
                    )
                samples = comparison_sources[s.name]
                source_index, source_frame, box = samples[len(samples) // 2]
                reference = self._crop(
                    source_frame,
                    box,
                    self.work_dir
                    / "references"
                    / f"{s.name}_shot{k:02d}_attempt{observation.attempt}_{candidate_key}.png",
                    margin=0.0,
                )
                spec = next(e for e in self.entities if e.name == s.name)
                provenance = {
                    "source": "accepted_script_description_match",
                    "entity": spec.name,
                    "script_description": spec.description,
                    "shot_index": k,
                    "attempt": int(observation.attempt),
                    "input_fingerprint": observation.input_fingerprint,
                    "candidate_video": str(video),
                    "source_chunk": source_index // self.fpc + c0,
                    "source_frame": str(source_frame),
                    "source_bbox_pixels": list(box),
                    "reference_crop": str(reference),
                    "reference_crop_sha256": file_sha256(reference),
                    "comparison_basis": s.comparison_basis,
                    "confidence": s.confidence,
                    "crop_votes": s.crop_votes,
                    "confidence_basis": "separate_crop_vote_agreement" if s.crop_votes is not None else "legacy_comparer",
                    "verdict_accept": True,
                }
                self.reference_crops[s.name] = reference
                self.reference_provenance[s.name] = provenance
                metrics["reference_registration"][s.name] = {
                    "status": "registered",
                    "provenance": provenance,
                }
            for s in good:
                self.last_tracks[s.name] = {
                    "shot_index": k,
                    "hw": [int(hw[0]), int(hw[1])],
                    "boxes": {int(c): list(box) for c, box in self._entity_geometry(s.name, tracks, prompts, num_chunks).items()},
                }
            return Verdict(accept=True, reason="accept_entities_match" if expected else "exclude_ambiguous_presence", metrics=metrics, input_fingerprint=observation.input_fingerprint)

        # Rejected/quarantinable candidates are never allowed to seed a future
        # reference, even if one entity happened to receive a positive match.
        for s in expected:
            if s.name in self.reference_crops:
                metrics["reference_registration"][s.name] = {
                    "status": "already_registered",
                    "provenance": self.reference_provenance.get(s.name),
                }
            elif s.name in uncertain_names:
                metrics["reference_registration"][s.name] = {
                    "status": "blocked_low_confidence_candidate",
                    "confidence": s.confidence,
                }
            else:
                metrics["reference_registration"][s.name] = {"status": "blocked_rejected_candidate"}

        # repaint region = drifted entity tracks ∪ unassigned subject tracks
        mask_keys = (
            [s.name for s in drifted]
            + [s.name for s in uncertain_first_appearance]
            + sorted(unassigned_live)
        )
        # ... ∪ synthesised placement boxes for absent entities when no impostor
        # track exists to repaint (an impostor in the entity's place keeps the
        # existing path: repaint the impostor, prompt names the entity).
        placements: dict[str, dict[str, Any]] = {}
        if absent and not unassigned_live and self.place_absent != "never":
            located_tracks = [tracks[key] for key in tracks if key in statuses and statuses[key].present]
            for s in absent:
                plan = self._placement_plan(s.name, relations, prompts, statuses, tracks, hw, num_chunks)
                if plan is None:
                    continue
                key = f"place_{s.name}"
                softs[key] = self._soft(
                    self._rasterize_boxes(plan["boxes"], located_tracks, n_frames, hw), num_chunks
                )
                placements[s.name] = plan
                mask_keys.append(key)
        metrics["placement"] = placements
        union: dict[int, torch.Tensor] = {}
        for c in range(num_chunks):
            parts = [softs[key][c] for key in mask_keys if key in softs]
            union[c + c0] = torch.stack(parts, 0).max(0).values if parts else torch.zeros((self.fpc, *self.grid))
        coverage = float(np.mean([(m >= 0.35).float().mean().item() for m in union.values()]))
        metrics["mask_keys"] = mask_keys
        metrics["mask_coverage"] = coverage
        if coverage < self.min_track_coverage:
            # Nothing to repaint (entity absent, no impostor): no edit plan.
            if unresolved:
                reason = "reject_entity_assignment_uncertain"
            elif absent:
                reason = "reject_entity_absent"
            elif uncertain_first_appearance:
                reason = "reject_identity_reference_low_confidence_unmaskable"
            else:
                reason = "reject_entity_drift_unmaskable"
            return Verdict(accept=False, reason=reason, metrics=metrics, input_fingerprint=observation.input_fingerprint)

        mask_path = save_mask_payload(union, attempt_dir / "repaint_masks.pt", phrase=json.dumps(mask_keys), video=video)
        placement_only = bool(placements) and not drifted and not unassigned_live and not uncertain_first_appearance
        if placement_only and self.placement_prompt == "spec_only":
            global_caption = str(normative.global_state.get("global_caption", "")) if normative.global_state else ""
            prompt_parts = [global_caption.strip()]
        else:
            prompt_parts = [caption.strip()]
        repair_targets = {
            s.name: s for s in drifted + absent + uncertain_first_appearance
        }
        for s in repair_targets.values():
            spec = next(e for e in self.entities if e.name == s.name)
            if placement_only and self.placement_prompt == "spec_only":
                prompt_parts.append(f"{spec.name}, {spec.description}, stands in the scene.")
            else:
                prompt_parts.append(f"{spec.name} is {spec.description}.")
        metrics["repair_prompt_mode"] = (
            self.placement_prompt if placement_only else "caption_and_spec"
        )
        action = Action(
            "edit_shot",
            {"shot": k, "prompt": " ".join(p for p in prompt_parts if p), "mask_path": str(mask_path), "entry_sigma": self.edit_entry_sigma},
            reason=(
                "entity_placement_masked_renoise_repair"
                if placements and not drifted and not unassigned_live
                else "entity_tracks_masked_renoise_repair"
            ),
        ).normalized(shot_index=k)
        if drifted:
            reason = "reject_entity_drift"
        elif uncertain_first_appearance:
            reason = "reject_identity_reference_low_confidence"
        elif not absent:
            reason = "reject_unassigned_subject"
        elif placements and not unassigned_live:
            reason = "reject_entity_absent_place"
        else:
            reason = "reject_entity_absent_impostor"
        metrics["repair_dependencies"] = self._repair_dependencies(
            shot_index=k,
            entity_names=tuple(repair_targets),
            metrics=metrics,
            repairs_unassigned_subject=bool(unassigned_live),
        )
        return Verdict(accept=False, reason=reason, metrics=metrics, directed_actions=(action,), input_fingerprint=observation.input_fingerprint)
