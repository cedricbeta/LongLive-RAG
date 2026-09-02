"""Entity-level drift judge for the edit_shot repair loop.

Replaces the single free-form comparative call (``ComparativeDriftVerifier``)
with three narrow questions whose answers form a data structure; the verdict
and the repaint region are derived from that structure instead of from a
judge-written phrase.  Nothing here is specific to people: an entity is
whatever the script names (a man, a fish, a pocket watch).

    locate    for each scripted entity and each chunk-first frame of the shot:
              "where is THIS entity (reference crop + script description) in
              this frame, even if its appearance drifted; absent otherwise"
    enumerate for each chunk-first frame: "box every main subject" -- boxes
              that overlap no located entity are *unassigned subjects*
              (an impostor sitting in an absent entity's place lands here)
    compare   for each located entity: "does it still match its reference?"
              plus a positive phrase naming the aspect to restore

SAM2 turns the per-chunk boxes into shot-level tracks (forward + reverse
propagation from every prompted frame, so a chunk the locator missed is filled
from its neighbours).  Repaint mask = drifted-entity tracks ∪ unassigned-subject
tracks; repaint prompt = authored caption + script description of every drifted
or absent entity + the comparer's positive fix phrases.  An absent entity with
nothing unassigned yields no mask, so no edit is directed and the session's
restore+retry path runs instead -- "nothing to mask" is a data state, not a
rule.

Verdict metrics carry ``repair_score`` (matching expected entities minus
unassigned subjects, over expected entities) which the session uses for
best-of-N delivery when every round is rejected.

Heavy dependencies (SAM2, PIL, cv2) are imported lazily; the decision logic is
pure and unit-testable with stub localiser/comparer objects.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import torch

from utils.agent_edit import FRAMES_PER_CHUNK, LATENT_GRID, read_video_frames, save_mask_payload
from utils.agent_runtime import Action, NormativeView, ShotObservation, Verdict, Verifier, file_sha256
from utils.agent_vlm import FingerprintedVLMClient

Box = tuple[int, int, int, int]

LOCATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "present": {"type": "boolean"},
        "bbox_2d": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "integer"}},
    },
    "required": ["present", "bbox_2d"],
}

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
        "confidence": {"type": "number", "enum": [0, 0.25, 0.5, 0.75, 1]},
    },
    "required": ["matches_reference", "differences", "target_phrase", "confidence"],
}


@dataclass(frozen=True)
class EntitySpec:
    """One scripted entity: name, script description, shots in which it is expected."""

    name: str
    description: str
    shots: tuple[int, ...] | None = None  # None = expected in every shot

    def expected_in(self, shot_index: int) -> bool:
        return self.shots is None or int(shot_index) in self.shots

    @classmethod
    def from_value(cls, value: Mapping[str, Any]) -> "EntitySpec":
        shots = value.get("shots")
        return cls(
            name=str(value["name"]),
            description=str(value.get("description", value["name"])),
            shots=None if shots is None else tuple(int(s) for s in shots),
        )


def load_entity_specs(path: str | Path, scene: str) -> list[EntitySpec]:
    """Read ``{scene: [{name, description, shots}]}`` and return the scene's specs."""
    table = json.loads(Path(path).read_text(encoding="utf-8"))
    if scene not in table:
        raise KeyError(f"no entity specs for scene {scene!r} in {path}")
    return [EntitySpec.from_value(item) for item in table[scene]]


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
    def locate(self, entity: EntitySpec, reference_crop: Path | None, frame: Path, *, shot_index: int, inputs: Any) -> Box | None: ...

    def enumerate_subjects(self, frame: Path, *, shot_index: int, inputs: Any) -> list[Box]: ...


class Comparer(Protocol):
    def compare(self, entity: EntitySpec, reference_crop: Path | None, crops: Sequence[Path], *, shot_index: int, inputs: Any) -> Mapping[str, Any]: ...


class Tracker(Protocol):
    def tracks(self, frames_dir: Path, prompts: Mapping[str, Mapping[int, Box]], num_frames: int, hw: tuple[int, int]) -> dict[str, np.ndarray]: ...


class VLMLocalizer:
    """Qwen3-VL grounding through the fingerprinted client (cached, audited)."""

    def __init__(self, client: FingerprintedVLMClient) -> None:
        self.client = client

    @staticmethod
    def _size(frame: Path) -> tuple[int, int]:
        from PIL import Image

        return Image.open(frame).size

    def locate(self, entity, reference_crop, frame, *, shot_index, inputs):
        W, H = self._size(frame)
        prompt = {
            "entity": entity.name,
            "script_description": entity.description,
            "images": (
                "First image: reference crop of this entity from an earlier accepted shot. Last image: the current frame."
                if reference_crop
                else "Only image: the current frame."
            ),
            "task": (
                f"In the current frame ({W}x{H} pixels) locate THIS SAME entity — the instance playing this role in the "
                "scene — even if its appearance has drifted (hair, clothing, colours or build may differ from the "
                "reference). Return present=false and a zero box if no instance of this entity is in the frame. Never "
                "box a different character. bbox_2d=[x1,y1,x2,y2], whole body including hair."
            ),
            "output": "strict JSON matching the supplied schema",
        }
        res = self.client.call(
            prompt=json.dumps(prompt, sort_keys=True, ensure_ascii=False),
            image_paths=[p for p in (reference_crop, frame) if p],
            schema=LOCATE_SCHEMA,
            call_kind="entity_locate",
            shot_index=int(shot_index),
            temporal_inputs=inputs,
        )
        parsed = res["parsed"]
        if not parsed["present"]:
            return None
        return scale_box_norm1000(parsed["bbox_2d"], W, H)

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
                "Does the entity in the current shot still match its reference identity and appearance (species, "
                "face, hair, colours, clothing, distinguishing marks)? Ignore pose, camera, framing and lighting. "
                "differences: one short sentence naming what changed, empty if it matches. target_phrase: a short "
                "POSITIVE phrase describing how the changed aspect should look (e.g. 'curly red hair, moss green "
                "coat'); never mention the wrong appearance."
            ),
            "output": "strict JSON matching the supplied schema",
        }
        res = self.client.call(
            prompt=json.dumps(prompt, sort_keys=True, ensure_ascii=False),
            image_paths=[p for p in (reference_crop, *crops) if p],
            schema=COMPARE_SCHEMA,
            call_kind="entity_compare",
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
    target_phrase: str = ""
    coverage: float = 0.0

    @property
    def present(self) -> bool:
        return bool(self.located_chunks)


class EntityDriftVerifier(Verifier):
    """Verifier producing edit_shot plans from located entity tracks.

    The backend must expose, in ``observation.telemetry``: ``shot_video`` (path of
    the candidate render), ``chunk_start``/``chunk_end`` (inclusive chunk range of
    the shot) and ``caption``-bearing normative state.  Reference crops are taken
    from accepted candidates (first accepted appearance of each entity), so shot 0
    doubles as the fidelity gate: entities the script places in shot 0 must be
    locatable there, or the shot is rejected (with no edit plan).
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
        grid: tuple[int, int] = LATENT_GRID,
        frames_per_chunk: int = FRAMES_PER_CHUNK,
    ) -> None:
        self.entities = list(entities)
        self.localizer = localizer
        self.comparer = comparer
        self.tracker = tracker
        self.work_dir = Path(work_dir)
        self.edit_entry_sigma = float(edit_entry_sigma)
        self.min_track_coverage = float(min_track_coverage)
        self.unassigned_iou = float(unassigned_iou)
        self.grid = tuple(grid)
        self.fpc = int(frames_per_chunk)
        self.reference_crops: dict[str, Path] = {}

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
        attempt_dir = self.work_dir / f"shot{k:02d}_attempt{observation.attempt}"

        # 1. locate every expected entity on every chunk-first frame
        statuses: dict[str, EntityStatus] = {}
        prompts: dict[str, dict[int, Box]] = {}
        entity_boxes: dict[int, list[Box]] = {c: [] for c in range(num_chunks)}
        for ent in self.entities:
            status = EntityStatus(name=ent.name, expected=ent.expected_in(k))
            statuses[ent.name] = status
            if not status.expected:
                continue
            per_frame: dict[int, Box] = {}
            for c, png in enumerate(chunk_pngs):
                box = self.localizer.locate(ent, self.reference_crops.get(ent.name), png, shot_index=k, inputs=inputs)
                if box:
                    per_frame[c * self.fpc] = box
                    entity_boxes[c].append(box)
                    status.located_chunks.append(c)
            prompts[ent.name] = per_frame

        # 2. SAM2 tracks of the located entities (the locator's boxes can be
        #    loose; the segmentation is what assignment and masks rely on)
        tracks = self.tracker.tracks(frames_dir, prompts, n_frames, hw)
        softs = {key: self._soft(track, num_chunks) for key, track in tracks.items()}
        for key, soft in softs.items():
            statuses[key].coverage = float(np.mean([(s >= 0.35).float().mean().item() for s in soft.values()]))

        # 3. enumerate subjects; a subject is assigned to an entity when the
        #    entity's segmentation fills a fair share of the subject's box,
        #    otherwise it is an unassigned subject (tracked across chunks)
        def assigned(box: Box, c: int) -> bool:
            x1, y1, x2, y2 = box
            area = max(1, (x2 - x1) * (y2 - y1))
            for key, track in tracks.items():
                if track[c * self.fpc, y1:y2, x1:x2].sum() / area >= self.unassigned_iou:
                    return True
            return False

        unassigned: dict[str, dict[int, Box]] = {}
        for c, png in enumerate(chunk_pngs):
            for box in self.localizer.enumerate_subjects(png, shot_index=k, inputs=inputs):
                if assigned(box, c):
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
        if unassigned and any(not statuses[e.name].present for e in self.entities if e.expected_in(k)):
            # impostors only matter when a scripted entity is absent; track them
            extra = self.tracker.tracks(frames_dir, unassigned, n_frames, hw)
            for key, track in extra.items():
                softs[key] = self._soft(track, num_chunks)
                if float(np.mean([(s >= 0.35).float().mean().item() for s in softs[key].values()])) >= self.min_track_coverage:
                    unassigned_live.add(key)

        # 4. compare every located entity with its reference
        for ent in self.entities:
            status = statuses[ent.name]
            if not status.present:
                continue
            crops = []
            for j, c in enumerate(sorted(status.located_chunks)[:: max(1, len(status.located_chunks) // 3)][:3]):
                crops.append(self._crop(chunk_pngs[c], prompts[ent.name][c * self.fpc], attempt_dir / f"{ent.name}_chunk{c:02d}.png"))
            result = self.comparer.compare(ent, self.reference_crops.get(ent.name), crops, shot_index=k, inputs=inputs)
            status.matches = bool(result.get("matches_reference", False))
            status.differences = str(result.get("differences", ""))
            status.target_phrase = str(result.get("target_phrase", ""))

        # 5. decision from the data structure
        expected = [s for s in statuses.values() if s.expected]
        drifted = [s for s in expected if s.present and s.matches is False]
        absent = [s for s in expected if not s.present]
        good = [s for s in expected if s.present and s.matches]
        # An unassigned subject is an impostor only when it can be standing in
        # for an absent scripted entity (enforced above: they are only tracked
        # then).  With every expected entity located and matching, extra
        # subjects (a bookend, a passer-by) are outside the judge's mandate --
        # the mandate is the identity of scripted entities.
        n_exp = max(1, len(expected))
        repair_score = (len(good) - len(unassigned_live)) / n_exp
        metrics = {
            "entities": {
                s.name: {
                    "expected": s.expected, "located_chunks": s.located_chunks, "matches": s.matches,
                    "differences": s.differences, "target_phrase": s.target_phrase, "coverage": round(s.coverage, 4),
                }
                for s in statuses.values()
            },
            "unassigned_subjects": sorted(unassigned_live),
            "repair_score": repair_score,
            "reference_available": {s.name: s.name in self.reference_crops for s in expected},
        }
        accept = not drifted and not absent and not unassigned_live
        if accept:
            # first accepted appearance becomes the reference crop
            for s in good:
                if s.name not in self.reference_crops:
                    c = sorted(s.located_chunks)[len(s.located_chunks) // 2]
                    self.reference_crops[s.name] = self._crop(
                        chunk_pngs[c], prompts[s.name][c * self.fpc], self.work_dir / "references" / f"{s.name}.png"
                    )
            return Verdict(accept=True, reason="accept_entities_match", metrics=metrics, input_fingerprint=observation.input_fingerprint)

        # repaint region = drifted entity tracks ∪ unassigned subject tracks
        mask_keys = [s.name for s in drifted] + sorted(unassigned_live)
        union: dict[int, torch.Tensor] = {}
        for c in range(num_chunks):
            parts = [softs[key][c] for key in mask_keys if key in softs]
            union[c + c0] = torch.stack(parts, 0).max(0).values if parts else torch.zeros((self.fpc, *self.grid))
        coverage = float(np.mean([(m >= 0.35).float().mean().item() for m in union.values()]))
        metrics["mask_keys"] = mask_keys
        metrics["mask_coverage"] = coverage
        if coverage < self.min_track_coverage:
            # Nothing to repaint (entity absent, no impostor): no edit plan.
            reason = "reject_entity_absent" if absent else "reject_entity_drift_unmaskable"
            return Verdict(accept=False, reason=reason, metrics=metrics, input_fingerprint=observation.input_fingerprint)

        mask_path = save_mask_payload(union, attempt_dir / "repaint_masks.pt", phrase=json.dumps(mask_keys), video=video)
        prompt_parts = [caption.strip()]
        for s in drifted + absent:
            spec = next(e for e in self.entities if e.name == s.name)
            prompt_parts.append(f"{spec.name} is {spec.description}.")
            if s.target_phrase.strip():
                prompt_parts.append(f"{spec.name}: {s.target_phrase.strip()}.")
        action = Action(
            "edit_shot",
            {"shot": k, "prompt": " ".join(p for p in prompt_parts if p), "mask_path": str(mask_path), "entry_sigma": self.edit_entry_sigma},
            reason="entity_tracks_masked_renoise_repair",
        ).normalized(shot_index=k)
        reason = "reject_entity_drift" if drifted else ("reject_unassigned_subject" if not absent else "reject_entity_absent_impostor")
        return Verdict(accept=False, reason=reason, metrics=metrics, directed_actions=(action,), input_fingerprint=observation.input_fingerprint)
