"""Opt-in offline visual-feedback repair over the existing bounded edit tools.

No weight training and no change to visual acceptance criteria. Planner input
has an explicit observation horizon, separate from the shot being repaired.
All masks come from the original parent; rejected children never supply masks
for a different parent or become trusted reference images.
"""
from __future__ import annotations

import dataclasses
import json
import re
import time
from pathlib import Path
from typing import Any, Mapping

from utils.agent_prompt_rewrite import validate_rewrite
from utils.agent_runtime import (
    Action, CausalShotSession, ShotObservation, ShotResult, TemporalInput,
    Verdict, file_sha256, fingerprint,
)


PLAN_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "option_id": {"type": "string"},
        "prompt": {"type": "string", "maxLength": 900},
        "entry_sigma": {"type": "number", "enum": [1.0]},
        "decision": {"type": "string", "enum": ["revise", "resample", "stop"]},
        "reason": {"type": "string", "minLength": 1, "maxLength": 700},
        "change_from_previous": {"type": "string", "minLength": 1, "maxLength": 700},
        "expected_observation": {"type": "string", "minLength": 1, "maxLength": 700},
    },
    "required": ["option_id", "prompt", "entry_sigma", "decision", "reason", "change_from_previous", "expected_observation"],
}


def effective_action(action, options):
    """Compare executed inputs, excluding model-written claims and filenames."""
    params = action["params"]
    mask_path = params.get("mask_path")
    option = next((o for o in options if o["action"]["params"].get("mask_path") == mask_path), {}) if mask_path else {}
    mask = option.get("mask_content_sha256") or option.get("mask_sha256")
    if mask is None:
        mask = file_sha256(mask_path) if mask_path else params.get("mask_phrase")
    return {"mask": mask, "prompt": " ".join(re.findall(r"\w+", params["prompt"].casefold())),
            "entry_sigma": float(params["entry_sigma"])}


def review_evidence(verdict: Verdict) -> dict[str, Any]:
    """Keep actual judge language and predicate evidence, not an LLM summary."""
    metrics = verdict.metrics
    lookahead = dict(metrics.get("lookahead", {}))
    if lookahead:
        lookahead["checked"] = [
            {"shot_index": c.get("shot_index"), "decision": c.get("decision"),
             "comparison": c.get("comparison", {}),
             **{side: {key: c.get(side, {}).get(key) for key in ("accept", "reason", "input_fingerprint")}
                for side in ("parent", "candidate")}}
            for c in lookahead.get("checked", [])
        ]
    return {
        "accept": verdict.accept, "reason": verdict.reason,
        "input_fingerprint": verdict.input_fingerprint,
        "constraints": metrics.get("constraint_assessments", []),
        "entities": metrics.get("entities", {}),
        "lookahead": lookahead,
    }


def classify_failure(verdict: Verdict) -> str:
    lookahead = verdict.metrics.get("lookahead", {})
    if lookahead.get("error"):
        return "evaluation_error"
    deltas = [d for c in lookahead.get("checked", [])
              for d in c.get("comparison", {}).get("regressions", [])]
    if any(str(d.get("reason", "")).startswith("comparison error:") for d in deltas):
        return "comparison_error"
    if verdict.reason == "reject_commit_lookahead":
        if deltas and all(d.get("candidate_status") == "unknown" for d in deltas):
            return "insufficient_evidence"
        return "future_regression"
    assessments = verdict.metrics.get("constraint_assessments", [])
    if assessments:
        # Explicit predicate evidence takes precedence over aggregate reason
        # labels. Low registration confidence is uncertainty, not a defect.
        failures = [a for a in assessments if a.get("status") == "violated"
                    and not any(v.get("code") in ("low_confidence_first_appearance", "not_observable", "assignment_uncertain")
                                for v in a.get("violations", []))]
        if failures:
            return "current_defect"
        if any(a.get("status") == "unknown" or any(v.get("code") == "low_confidence_first_appearance"
               for v in a.get("violations", [])) for a in assessments):
            return "insufficient_evidence"
    if "uncertain" in verdict.reason or "low_confidence" in verdict.reason:
        return "insufficient_evidence"
    return "current_defect"


class VLMRepairPlanner:
    def __init__(self, client: Any):
        self.client = client

    def plan(self, packet: Mapping[str, Any], *, image_paths: list[str],
             inputs: tuple[TemporalInput, ...]) -> tuple[Action | None, dict[str, Any]]:
        from utils.agent_vlm import _validate_json

        options = {option["id"]: option for option in packet["options"]}
        schema = json.loads(json.dumps(PLAN_SCHEMA))
        schema["properties"]["option_id"]["enum"] = [*options, "stop"]
        # Keep executable artifacts private. Remove the ready-written answer
        # that may have encouraged copying in the v1 run.
        public_packet = {**packet, "options": [
            {key: value for key, value in option.items() if key in (
                "id", "targets", "description", "parts", "allowed_sigmas", "coverage", "preview_images",
            )} for option in options.values()
        ]}
        request = {
            "instruction": (
                "Plan one video repair using the attached, labelled images and the actual review evidence. "
                "The reviewed failure may be imperfect; distinguish missing evidence from a visible defect. "
                "Choose an offered option or stop. All options act on the ORIGINAL parent, not the rejected "
                "child. Compare the labelled region previews: red shows repaint weight, blue shows protected "
                "pixels. Different options provide different parent-aligned masks; do not invent a mask or tool. "
                "Repair strength is fixed: entry_sigma must be 1.0 for every option. "
                "For a missing person choose the whole-shot option. For appearance drift use "
                "the offered regions. Preserve every required character, "
                "their authored attributes, setting and actions. State what changes from the previous attempt "
                "and what should be observable if it works. The prompt is a positive visual description in "
                "English, at most 90 words, naming all required characters exactly. Do not paste criticism "
                "into the rendering prompt or simplify away requested actions. A stop is an unresolved task, "
                "not a successful repair. Use decision=revise for a first or changed action. Identify the visible "
                "failure and explain why changing this region or prompt may address it. If you keep the same "
                "mask and prompt, use decision=resample and describe only a new noise draw: this is allowed "
                "once per identical action. Rewording the explanation is not a changed action. If evidence "
                "cannot establish the defect or no useful option remains, use option_id=stop and decision=stop. "
                "Never claim a repair has succeeded before it is rendered and checked. Strict JSON."
            ),
            "packet": public_packet,
        }
        failures = []
        calls = []
        for _ in range(2):
            response = self.client.call(
                prompt=json.dumps(request, ensure_ascii=False, sort_keys=True),
                image_paths=image_paths, schema=schema, call_kind="offline_repair_plan_v2",
                # This is the actual decision horizon, not the target shot.
                shot_index=int(packet["observed_through_shot"]), temporal_inputs=inputs,
            )
            calls.append({k: response.get(k) for k in ("call_id", "cached", "wall_seconds", "transcript")})
            plan = response["parsed"]
            try:
                _validate_json(plan, schema)
                if plan["option_id"] == "stop":
                    if plan["decision"] != "stop":
                        raise ValueError("option_id=stop requires decision=stop")
                    return None, {"plan": plan, "calls": calls, "rejected_plans": failures}
                if plan["decision"] == "stop":
                    raise ValueError("decision=stop requires option_id=stop")
                option = options[plan["option_id"]]
                if plan["entry_sigma"] not in option["allowed_sigmas"]:
                    raise ValueError("selected strength is not allowed for this option")
                problems = validate_rewrite(
                    plan["prompt"], original=packet["caption"],
                    absent_names=list(packet["required_characters"]),
                    present_names=list(packet["required_characters"]),
                )
                if problems:
                    raise ValueError("; ".join(problems))
                raw = option["action"]
                params = {**raw["params"], "prompt": plan["prompt"], "entry_sigma": plan["entry_sigma"]}
                mask = params.get("mask_path")
                if mask and file_sha256(mask) != option["mask_sha256"]:
                    raise RuntimeError("offered parent mask changed after preparation")
                action = Action("edit_shot", params, reason="offline_feedback_" + option["id"]).normalized(
                    shot_index=int(packet["target_shot"])
                )
                actual = effective_action(action.to_dict(), packet["options"])
                previous = [row.get("planning", {}).get("effective_action") or effective_action(row["action"], packet["options"])
                            for row in packet.get("history", [])]
                repeats = sum(value == actual for value in previous)
                if repeats >= 2:
                    raise ValueError("identical mask and prompt already rendered twice; select a different region/prompt or stop")
                if repeats and plan["decision"] != "resample":
                    raise ValueError("mask and prompt are unchanged; label this resample, change the actual action, or stop")
                if not repeats and plan["decision"] == "resample":
                    raise ValueError("resample must repeat an earlier action; a new region/prompt is revise")
                changes = [key for key in actual if not previous or actual[key] != previous[-1][key]]
                return action, {"plan": plan, "calls": calls, "rejected_plans": failures,
                                "effective_action": actual, "actual_changes": changes,
                                "identical_prior_attempts": repeats}
            except ValueError as error:
                failures.append({"plan": plan, "error": str(error)})
                request["invalid_plan_feedback"] = failures[-1]
        raise ValueError(f"planner returned two invalid plans: {failures}")


class OfflineFeedbackShotSession(CausalShotSession):
    """Bounded search with fresh planner calls and explicit offline provenance."""
    def __init__(self, *args, planner: VLMRepairPlanner, max_repair_seconds: float = 1800,
                 **kwargs):
        if max_repair_seconds <= 0:
            raise ValueError("max_repair_seconds must be positive")
        self.planner = planner
        self.max_repair_seconds = float(max_repair_seconds)
        self.observations: dict[str, ShotObservation] = {}
        super().__init__(*args, **kwargs)
        self.ledger.append("repair_mode", mode="offline_feedback_v2",
                           max_repair_seconds=max_repair_seconds,
                           budget_semantics="no new attempt after deadline; in-flight operations finish")

    def _observe_and_judge(self, *args, **kwargs):
        observation, verdict = super()._observe_and_judge(*args, **kwargs)
        self.observations[verdict.input_fingerprint] = observation
        return observation, verdict

    @staticmethod
    def _verdict_summary(verdict):
        return {**CausalShotSession._verdict_summary(verdict), "input_fingerprint": verdict.input_fingerprint}

    def _planner_packet(self, context, original, latest, history, attempt):
        k = context.shot_index
        source = self.observations[original.input_fingerprint]
        images: list[str] = []
        labels: list[dict[str, Any]] = []
        temporal = list(context.inputs)
        horizon = k

        def add_observation(digest, label, limit):
            nonlocal horizon
            observation = self.observations.get(digest)
            if observation is None:
                if digest:
                    raise RuntimeError(f"repair feedback has no audited observation: {digest}")
                return
            horizon = max(horizon, observation.shot_index)
            temporal.extend(observation.inputs)
            frames = list(observation.frames or [])
            selected = frames if len(frames) <= limit else [frames[round(i * (len(frames)-1)/(limit-1))] for i in range(limit)]
            for path in selected:
                if not isinstance(path, (str, Path)) or not Path(path).is_file():
                    raise ValueError("offline planner needs materialized, auditable frame paths")
                images.append(str(path))
                labels.append({"image": len(images), "role": label, "shot_index": observation.shot_index})
                temporal.append(TemporalInput(
                    kind="offline_repair_frame", identifier=str(path), sha256=file_sha256(path),
                    available_after_shot=observation.shot_index,
                ))

        add_observation(original.input_fingerprint, "original_parent", 3)
        if latest.input_fingerprint != original.input_fingerprint and latest.reason != "reject_commit_lookahead":
            add_observation(latest.input_fingerprint, "latest_candidate", 3)
        # At most the first failing paired future, with its full original
        # comparison report. Every earlier review remains in the history.
        lookahead = latest.metrics.get("lookahead", {})
        for checked in lookahead.get("checked", []):
            if checked.get("comparison", {}).get("allow_commit", True):
                continue
            for side in ("parent", "candidate"):
                add_observation(checked.get(side, {}).get("input_fingerprint", ""), "future_" + side, 2)
            break
        for name, reference in list(original.metrics.get("repair_references", {}).items())[:4]:
            path = reference["path"]
            provenance = reference.get("provenance", {})
            if provenance.get("reference_crop_sha256") and file_sha256(path) != provenance["reference_crop_sha256"]:
                raise RuntimeError("trusted repair reference changed")
            available = int(provenance.get("shot_index", k))
            horizon = max(horizon, available)
            images.append(str(path))
            labels.append({"image": len(images), "role": "trusted_reference", "entity": name, "shot_index": available})
            temporal.append(TemporalInput(kind="repair_reference", identifier=str(path),
                                          sha256=file_sha256(path), available_after_shot=available))
        # Failed current images still matter when a locally accepted candidate
        # failed its future guard.
        if history and latest.reason == "reject_commit_lookahead":
            add_observation(history[-1]["current_review"]["input_fingerprint"], "latest_candidate", 3)
        options = []
        for original_option in original.metrics.get("repair_options", []):
            option = dict(original_option)
            mask = option["action"]["params"].get("mask_path")
            if mask:
                from utils.agent_repair_regions import region_preview
                preview = region_preview(option, video=source.telemetry["shot_video"],
                                         output_dir=Path(mask).parent / "planner_previews")
                option.update(mask_content_sha256=preview["mask_content_sha256"], coverage=preview["coverage"],
                              preview_images=[len(images) + 1])
                images.append(preview["path"])
                labels.append({"image": len(images), "role": "region_preview", "option_id": option["id"],
                               "shot_index": k, "source_frame_indices": preview["frame_indices"]})
                for kind, path, digest in (
                    ("repair_region_preview", preview["path"], preview["sha256"]),
                    ("repair_region_mask", mask, option["mask_sha256"]),
                    ("repair_region_source", source.telemetry["shot_video"], preview["source_video_sha256"]),
                ):
                    temporal.append(TemporalInput(kind=kind, identifier=str(path), sha256=digest, available_after_shot=k))
                if option.get("protected_mask_path"):
                    temporal.append(TemporalInput(kind="repair_protected_mask", identifier=option["protected_mask_path"],
                                                  sha256=option["protected_mask_sha256"], available_after_shot=k))
            options.append(option)
        horizon = max(horizon, *(int(row["observed_through_shot"]) for row in history),
                      int(lookahead.get("decision_as_of_shot", k)))
        packet = {
            "mode": "offline_feedback_v2", "target_shot": k, "observed_through_shot": horizon,
            "caption": str(context.normative.shot_state[-1].get("caption", "")),
            "required_characters": original.metrics.get("repair_specs", {}),
            "options": options,
            "original_review": review_evidence(original), "latest_review": review_evidence(latest),
            "history": history, "attempt": attempt,
            "remaining_render_attempts": self.config.max_edits_per_shot - attempt + 1,
            "images": labels, "parent_observation": source.input_fingerprint,
        }
        temporal.append(TemporalInput.from_value(
            kind="offline_repair_feedback", identifier=f"{self.scene.scene}:{k}:{attempt}",
            value=packet, available_after_shot=horizon,
        ))
        audited = self._inputs_or_fail(temporal, shot_index=horizon, stage="offline_repair_planner")
        return packet, images, audited

    def run_shot(self, shot_index):
        k = int(shot_index)
        context = self.pre_shot(k)
        program = self.controller.decide(context)
        if program.input_fingerprint != context.input_fingerprint:
            raise RuntimeError("controller fingerprint mismatch")
        self.apply(k, program, retry=False)
        boundary = self.backend.snapshot(k)
        boundary_digest = fingerprint(boundary)
        self.ledger.append("snapshot", shot_index=k, inventory_fingerprint=boundary_digest)
        parent = self.backend.generate(k, 0)
        parent_state = self.backend.snapshot(k)
        parent_digest = fingerprint(parent_state)
        original = self._stage_and_judge(context, parent, expected_attempt=0)
        if original.accept:
            return self._commit_accepted(parent, original, boundary_snapshot=boundary,
                                         boundary_snapshot_digest=boundary_digest)
        self._reject_candidate(parent, original)
        latest = original
        history = []
        started = time.monotonic()
        attempts = 0
        stop_reason = "render_budget_exhausted"
        for attempt in range(1, self.config.max_edits_per_shot + 1):
            if time.monotonic() - started >= self.max_repair_seconds:
                stop_reason = "wall_budget_exhausted"
                break
            if not original.metrics.get("repair_options"):
                stop_reason = "no_supported_repair_option"
                break
            if classify_failure(latest) == "insufficient_evidence":
                stop_reason = "insufficient_evidence"
                break
            plan_started = time.monotonic()
            try:
                packet, images, inputs = self._planner_packet(context, original, latest, history, attempt)
                action, planning = self.planner.plan(packet, image_paths=images, inputs=inputs)
            except (ValueError, RuntimeError, OSError) as error:
                self.ledger.append("repair_planner_error", shot_index=k, attempt=attempt,
                                   error=f"{type(error).__name__}: {error}")
                stop_reason = "planner_error"
                break
            self.ledger.append("repair_plan", shot_index=k, attempt=attempt,
                               decision_as_of_shot=packet["observed_through_shot"],
                               input_fingerprint=fingerprint(packet), packet=packet,
                               image_paths=images, planning=planning,
                               wall_seconds=time.monotonic()-plan_started)
            if action is None:
                stop_reason = "planner_stopped_unresolved"
                break
            if time.monotonic() - started >= self.max_repair_seconds:
                stop_reason = "wall_budget_exhausted"
                break
            # Existing renderer uses the verdict fingerprint for provenance.
            # Its dependency set belongs to the parent option, never to a
            # rejected child's newly observed mask coordinates.
            directed = dataclasses.replace(original, directed_actions=(action,))
            render_started = time.monotonic()
            child, current_review = self._edit_and_review(
                context, directed, action, boundary, boundary_digest, attempt,
            )
            attempts = attempt
            latest = current_review
            if current_review.accept:
                committed = self._commit_accepted(
                    child, current_review, boundary_snapshot=boundary,
                    boundary_snapshot_digest=boundary_digest,
                    fallback=(parent, parent_state, parent_digest), defer_rollback_delivery=True,
                )
                if isinstance(committed, ShotResult):
                    self.ledger.append("repair_finished", shot_index=k, success=True,
                                       render_attempts=attempts, wall_seconds=time.monotonic()-started)
                    return committed
                latest = committed
            else:
                self._reject_candidate(child, current_review)
            row = {
                "attempt": attempt, "action": action.to_dict(), "planning": planning,
                "current_review": review_evidence(current_review), "failure": review_evidence(latest),
                "failure_kind": classify_failure(latest),
                "observed_through_shot": max(k, int(latest.metrics.get("lookahead", {}).get("decision_as_of_shot", k))),
                "render_and_review_seconds": time.monotonic()-render_started,
            }
            history.append(row)
            self.ledger.append("repair_feedback", shot_index=k, **row)
            # An evaluator error gives no semantic learning signal. Preserve
            # evidence and stop this shot; never spend a render to fix a crash.
            if row["failure_kind"] in ("evaluation_error", "comparison_error", "insufficient_evidence"):
                stop_reason = row["failure_kind"]
                break
        self._restore_checked(k, parent_state, parent_digest, purpose="offline_repair_fallback")
        delivered = self.backend.commit(parent, quarantined=True)
        self.committed_shots.add(k)
        self.ledger.append("commit", shot_index=k, attempt=parent.attempt, quarantined=True,
                           payload_fingerprint=self.memory.quarantined_payload(k, parent.attempt).payload_fingerprint,
                           selection="nearest_stable_parent", reason=stop_reason,
                           partition_counts=self.memory.partition_counts())
        self.ledger.append("repair_finished", shot_index=k, success=False, reason=stop_reason,
                           render_attempts=attempts, wall_seconds=time.monotonic()-started)
        self.scene.assert_unchanged()
        errors = self.memory.audit_invisible()
        if errors:
            raise RuntimeError("; ".join(errors))
        return ShotResult(k, attempts + 1, False, True, delivered, latest)
