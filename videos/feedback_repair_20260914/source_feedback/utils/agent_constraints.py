"""Generic script-constraint assessments and paired no-regression decisions.

The agent runtime originally exposed one entity-specific ``repair_score``.  This
module provides the neutral data boundary underneath future verifiers: a verifier
states *what the script requires*, records evidence for that requirement, and
reports a pass/fail/unknown assessment.  The commit guard can then compare a
stable parent and a repaired candidate constraint by constraint without knowing
whether the subject is a person, prop, animal, action, relation, or scene state.

All public data objects round-trip through JSON-compatible dictionaries.  The
comparison routine is pure and accepts either objects, their serialized forms,
verdict metrics containing ``constraint_assessments``, or Verdict-like objects
with a ``metrics`` attribute.  Old metrics without assessments retain a narrow
``accept``/``repair_score`` compatibility path during migration.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote


CONSTRAINT_ASSESSMENTS_KEY = "constraint_assessments"


class ConstraintKind(str, Enum):
    """Script predicates supported by the generic constraint layer."""

    PRESENCE = "presence"
    IDENTITY = "identity"
    ATTRIBUTE = "attribute"
    COUNT = "count"
    RELATION = "relation"
    ACTION = "action"
    STATE = "state"


class AssessmentStatus(str, Enum):
    SATISFIED = "satisfied"
    VIOLATED = "violated"
    UNKNOWN = "unknown"


def _jsonable(value: Any) -> Any:
    """Return a deterministic JSON-compatible copy of ``value``."""

    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value):
        return {
            item.name: _jsonable(getattr(value, item.name))
            for item in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        converted = [_jsonable(item) for item in value]
        return sorted(converted, key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("constraint data cannot contain NaN or infinity")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"constraint data is not JSON-compatible: {type(value).__name__}")


class _JSONRecord:
    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        )


@dataclass(frozen=True)
class Constraint(_JSONRecord):
    """One authored predicate, independent of the model used to verify it.

    ``subjects`` names the persistent instances involved.  A relation normally
    has two or more subjects; an action can include the actor and object.  The
    open-ended ``expected`` and ``parameters`` fields carry JSON values so this
    schema does not need a new Python class for every script vocabulary item.
    """

    id: str
    kind: ConstraintKind | str
    description: str
    subjects: tuple[str, ...] = ()
    shot_indices: tuple[int, ...] = ()
    expected: Any = True
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.id).strip():
            raise ValueError("constraint id must be non-empty")
        object.__setattr__(self, "id", str(self.id))
        object.__setattr__(self, "kind", ConstraintKind(self.kind))
        object.__setattr__(self, "description", str(self.description))
        object.__setattr__(self, "subjects", tuple(str(item) for item in self.subjects))
        shots = tuple(int(item) for item in self.shot_indices)
        if any(item < 0 for item in shots):
            raise ValueError("constraint shot indices must be non-negative")
        object.__setattr__(self, "shot_indices", shots)
        # Validate extension payloads eagerly; to_dict still returns a fresh copy.
        _jsonable(self.expected)
        _jsonable(self.parameters)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Constraint":
        return cls(
            id=str(value["id"]),
            kind=ConstraintKind(value["kind"]),
            description=str(value.get("description", "")),
            subjects=tuple(value.get("subjects", ())),
            shot_indices=tuple(value.get("shot_indices", ())),
            expected=value.get("expected", True),
            parameters=dict(value.get("parameters", {})),
        )


@dataclass(frozen=True)
class ConstraintEvidence(_JSONRecord):
    """An observation supporting an assessment.

    ``confidence`` is confidence in the observation, not a reward.  Continuous
    repair quality belongs in :class:`ConstraintAssessment.score`.
    """

    source: str
    observed: Any = None
    confidence: float | None = None
    details: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "details", str(self.details))
        if self.confidence is not None:
            confidence = float(self.confidence)
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError("evidence confidence must be between 0 and 1")
            object.__setattr__(self, "confidence", confidence)
        _jsonable(self.observed)
        _jsonable(self.metadata)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ConstraintEvidence":
        return cls(
            source=str(value.get("source", "unknown")),
            observed=value.get("observed"),
            confidence=value.get("confidence"),
            details=str(value.get("details", "")),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True)
class Violation(_JSONRecord):
    """A machine-readable and human-readable explanation of a failed predicate."""

    code: str
    message: str
    severity: str = "error"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.code).strip():
            raise ValueError("violation code must be non-empty")
        object.__setattr__(self, "code", str(self.code))
        object.__setattr__(self, "message", str(self.message))
        object.__setattr__(self, "severity", str(self.severity))
        _jsonable(self.metadata)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Violation":
        return cls(
            code=str(value["code"]),
            message=str(value.get("message", "")),
            severity=str(value.get("severity", "error")),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True)
class ConstraintAssessment(_JSONRecord):
    """The verifier's result for exactly one :class:`Constraint`."""

    constraint: Constraint
    status: AssessmentStatus | str
    evidence: tuple[ConstraintEvidence, ...] = ()
    violations: tuple[Violation, ...] = ()
    # Optional, comparable quality within this same constraint: 0 worst, 1 best.
    score: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.constraint, Constraint):
            object.__setattr__(self, "constraint", Constraint.from_dict(self.constraint))
        object.__setattr__(self, "status", AssessmentStatus(self.status))
        object.__setattr__(
            self, "evidence",
            tuple(
                item if isinstance(item, ConstraintEvidence) else ConstraintEvidence.from_dict(item)
                for item in self.evidence
            ),
        )
        object.__setattr__(
            self, "violations",
            tuple(item if isinstance(item, Violation) else Violation.from_dict(item) for item in self.violations),
        )
        if self.score is not None:
            score = float(self.score)
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError("assessment score must be between 0 and 1")
            object.__setattr__(self, "score", score)

    @property
    def id(self) -> str:
        return self.constraint.id

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ConstraintAssessment":
        return cls(
            constraint=Constraint.from_dict(value["constraint"]),
            status=AssessmentStatus(value["status"]),
            evidence=tuple(ConstraintEvidence.from_dict(item) for item in value.get("evidence", ())),
            violations=tuple(Violation.from_dict(item) for item in value.get("violations", ())),
            score=value.get("score"),
        )


def _entity_constraint_id(shot_index: int, entity: str, predicate: str) -> str:
    escaped = quote(str(entity), safe="")
    return f"shot:{int(shot_index)}:entity:{escaped}:{predicate}"


def _status_assessment(
    constraint: Constraint,
    status: AssessmentStatus,
    evidence: ConstraintEvidence,
    *,
    score: float | None,
    violation: Violation | None = None,
) -> ConstraintAssessment:
    return ConstraintAssessment(
        constraint=constraint,
        status=status,
        evidence=(evidence,),
        violations=() if violation is None else (violation,),
        score=score,
    )


def entity_judge_metrics_to_assessments(
    metrics: Mapping[str, Any],
    *,
    shot_index: int,
    source: str = "entity_judge",
) -> tuple[ConstraintAssessment, ...]:
    """Adapt legacy ``EntityDriftVerifier`` metrics to generic assessments.

    The adapter deliberately preserves what the old judge actually measured:
    expected-entity presence, reference identity (when a reference exists),
    script-description attributes (on first appearance), the number of
    unassigned subjects, and caption-parsed spatial relations checked on the
    located boxes.  It does not invent action or scene-state evidence from a
    judge that never evaluated those predicates.
    """

    if int(shot_index) < 0:
        raise ValueError("shot_index must be non-negative")
    entities = metrics.get("entities", {})
    if not isinstance(entities, Mapping):
        raise TypeError("entity judge metrics['entities'] must be a mapping")
    reference_available = metrics.get("reference_available", {})
    if not isinstance(reference_available, Mapping):
        reference_available = {}
    policy = metrics.get("reference_registration_policy", {})
    min_registration_confidence = 0.75
    if isinstance(policy, Mapping):
        try:
            min_registration_confidence = float(policy.get("min_confidence", 0.75))
        except (TypeError, ValueError):
            min_registration_confidence = 0.75

    assessments: list[ConstraintAssessment] = []
    for raw_name, raw_entity in sorted(entities.items(), key=lambda item: str(item[0])):
        name = str(raw_name)
        entity = dict(raw_entity) if isinstance(raw_entity, Mapping) else {}
        if not bool(entity.get("expected", False)):
            # The old judge skips non-expected entities, so it provides no
            # trustworthy evidence that they are absent.
            continue

        chunks = tuple(int(item) for item in entity.get("located_chunks", ()) or ())
        present = bool(chunks)
        coverage = _optional_unit_float(entity.get("coverage"))
        presence = Constraint(
            id=_entity_constraint_id(shot_index, name, "presence"),
            kind=ConstraintKind.PRESENCE,
            description=f"{name} is present in shot {shot_index}",
            subjects=(name,), shot_indices=(int(shot_index),), expected=True,
        )
        presence_evidence = ConstraintEvidence(
            source=source,
            observed={"present": present, "located_chunks": list(chunks), "coverage": coverage},
            details="located by the legacy entity judge" if present else "not located by the legacy entity judge",
        )
        assignment_unknown = bool(entity.get("assignment_uncertain")) and not present
        assessments.append(_status_assessment(
            presence,
            AssessmentStatus.UNKNOWN if assignment_unknown else AssessmentStatus.SATISFIED if present else AssessmentStatus.VIOLATED,
            presence_evidence,
            score=None if assignment_unknown else 1.0 if present else 0.0,
            violation=None if present else Violation("assignment_uncertain" if assignment_unknown else "missing", f"expected entity {name!r} was not reliably located"),
        ))

        basis = str(entity.get("comparison_basis") or "")
        has_reference = bool(reference_available.get(name, False)) or basis == "reference_crop"
        appearance_kind = ConstraintKind.IDENTITY if has_reference else ConstraintKind.ATTRIBUTE
        predicate = "identity" if has_reference else "script_attributes"
        appearance = Constraint(
            id=_entity_constraint_id(shot_index, name, predicate),
            kind=appearance_kind,
            description=(
                f"{name} matches its persistent reference"
                if has_reference else f"{name} matches its authored script description"
            ),
            subjects=(name,), shot_indices=(int(shot_index),), expected=True,
            # The requirement is fixed before either branch is observed. A
            # missing person has no observed comparison_basis; that must not
            # redefine the predicate when the person becomes visible again.
            parameters={"comparison_basis": "reference_crop" if has_reference else "script_description"},
        )
        confidence = _optional_unit_float(entity.get("confidence"))
        matches = entity.get("matches")
        evidence = ConstraintEvidence(
            source=source,
            observed={
                "present": present,
                "matches": matches if isinstance(matches, bool) else None,
                "comparison_basis": basis or None,
                "differences": str(entity.get("differences", "")),
            },
            confidence=confidence,
            details=str(entity.get("differences", "")),
        )
        if not present or not isinstance(matches, bool):
            assessments.append(_status_assessment(
                appearance, AssessmentStatus.UNKNOWN, evidence, score=None,
                violation=Violation(
                    "not_observable",
                    f"cannot assess {predicate} for {name!r} because the entity was not reliably located",
                    severity="warning",
                ),
            ))
            continue

        low_confidence_first = (
            not has_reference
            and matches
            and (
                entity.get("reference_registration_eligible") is False
                or (confidence is not None and confidence < min_registration_confidence)
            )
        )
        appearance_ok = matches and not low_confidence_first
        if appearance_ok:
            violation = None
        elif low_confidence_first:
            violation = Violation(
                "low_confidence_first_appearance",
                f"{name!r} matched the script below the registration confidence threshold",
                metadata={"confidence": confidence, "min_confidence": min_registration_confidence},
            )
        else:
            violation = Violation(
                "identity_mismatch" if has_reference else "attribute_mismatch",
                f"{name!r} does not match " + ("its reference" if has_reference else "its script description"),
            )
        assessments.append(_status_assessment(
            appearance,
            AssessmentStatus.SATISFIED if appearance_ok else AssessmentStatus.VIOLATED,
            evidence,
            # The legacy comparer exposes a categorical match plus confidence
            # in that judgment, not a calibrated similarity/severity score.
            # Do not pretend two different mismatches are equally bad: without
            # a comparable score their fail->fail delta is explicitly
            # inconclusive, while pass->fail and fail->pass remain comparable.
            score=1.0 if appearance_ok else None,
            violation=violation,
        ))

    # Spatial relations parsed from the caption and checked geometrically on the
    # located boxes.  Emitted for every parsed relation (unknown when a party is
    # not located) so parent and candidate always expose the same constraint set.
    for raw in metrics.get("spatial_relations", ()) or ():
        if not isinstance(raw, Mapping):
            continue
        if raw.get("excluded_ambiguous_presence"):
            continue
        subject, anchor, relation = str(raw.get("subject", "")), str(raw.get("anchor", "")), str(raw.get("relation", ""))
        if not subject or not anchor or not relation:
            continue
        status_raw = str(raw.get("status", "unknown"))
        status = AssessmentStatus(status_raw) if status_raw in {s.value for s in AssessmentStatus} else AssessmentStatus.UNKNOWN
        wording = {"left": "is left of", "right": "is right of", "beside": "is beside", "behind": "is behind", "front": "is in front of"}.get(relation, relation)
        constraint = Constraint(
            id=f"shot:{int(shot_index)}:relation:{quote(subject, safe='')}:{relation}:{quote(anchor, safe='')}",
            kind=ConstraintKind.RELATION,
            description=f"{subject} {wording} {anchor} in shot {shot_index}",
            subjects=(subject, anchor), shot_indices=(int(shot_index),), expected=True,
            parameters={"relation": relation, "geometry": "bbox_centres_screen_convention"},
        )
        checked = int(raw.get("chunks_checked", 0) or 0)
        evidence = ConstraintEvidence(
            source=source,
            observed={"chunks_checked": checked, "chunks_satisfied": int(raw.get("chunks_satisfied", 0) or 0), "text": str(raw.get("text", ""))},
            details="geometric check on located boxes" if checked else "a party was not located or the relation is not decidable from boxes",
        )
        if status == AssessmentStatus.UNKNOWN:
            violation = Violation("not_observable", f"cannot assess {subject!r} {relation} {anchor!r}", severity="warning")
        elif status == AssessmentStatus.VIOLATED:
            violation = Violation("spatial_relation_violated", f"{subject!r} is not {wording} {anchor!r}")
        else:
            violation = None
        assessments.append(_status_assessment(
            constraint, status, evidence,
            score=_optional_unit_float(raw.get("score")),
            violation=violation,
        ))

    unassigned_raw = metrics.get("unassigned_subjects", ()) or ()
    if isinstance(unassigned_raw, str) or not isinstance(unassigned_raw, Sequence):
        unassigned = (str(unassigned_raw),) if unassigned_raw else ()
    else:
        unassigned = tuple(str(item) for item in unassigned_raw)
    count = len(unassigned)
    count_unobservable = (
        bool(metrics.get("uncertain_assignment_chunks"))
        or metrics.get("evaluation_status") == "excluded_no_required_entities"
    )
    no_extra = Constraint(
        id=f"shot:{int(shot_index)}:count:unassigned_subjects",
        kind=ConstraintKind.COUNT,
        description=f"shot {shot_index} contains no unassigned subject standing in for a scripted entity",
        shot_indices=(int(shot_index),), expected=0,
        parameters={"category": "unassigned_subject"},
    )
    assessments.append(_status_assessment(
        no_extra,
        AssessmentStatus.UNKNOWN if count_unobservable else AssessmentStatus.SATISFIED if count == 0 else AssessmentStatus.VIOLATED,
        ConstraintEvidence(
            source=source,
            observed={"count": count, "instances": list(unassigned)},
            details=f"legacy entity judge found {count} unassigned subject(s)",
        ),
        # Gives useful ordering when both branches still violate count=0,
        # without conflating this predicate with unrelated entity predicates.
        score=None if count_unobservable else 1.0 / (1.0 + count),
        violation=None if count == 0 else Violation(
            "unexpected_count", f"found {count} unassigned subject(s); expected none",
            metadata={"instances": list(unassigned)},
        ),
    ))
    return tuple(assessments)


def _optional_unit_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and 0.0 <= number <= 1.0 else None


def attach_constraint_assessments(
    metrics: Mapping[str, Any],
    *,
    shot_index: int,
    source: str = "entity_judge",
    key: str = CONSTRAINT_ASSESSMENTS_KEY,
) -> dict[str, Any]:
    """Return a JSON-safe metrics copy with serialized generic assessments."""

    result = _jsonable(metrics)
    result[str(key)] = [
        item.to_dict()
        for item in entity_judge_metrics_to_assessments(metrics, shot_index=shot_index, source=source)
    ]
    return result


@dataclass(frozen=True)
class ConstraintDelta(_JSONRecord):
    constraint_id: str
    kind: str
    classification: str
    parent_status: str | None
    candidate_status: str | None
    parent_score: float | None
    candidate_score: float | None
    reason: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ConstraintDelta":
        return cls(
            constraint_id=str(value["constraint_id"]),
            kind=str(value.get("kind", "unknown")),
            classification=str(value["classification"]),
            parent_status=value.get("parent_status"),
            candidate_status=value.get("candidate_status"),
            parent_score=value.get("parent_score"),
            candidate_score=value.get("candidate_score"),
            reason=str(value.get("reason", "")),
        )


@dataclass(frozen=True)
class NoRegressionDecision(_JSONRecord):
    """Paired guard result.  ``allow_commit`` means no clear regression."""

    allow_commit: bool
    comparison_mode: str
    regressions: tuple[ConstraintDelta, ...] = ()
    improvements: tuple[ConstraintDelta, ...] = ()
    unchanged: tuple[ConstraintDelta, ...] = ()
    inconclusive: tuple[ConstraintDelta, ...] = ()
    reason: str = ""

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NoRegressionDecision":
        def deltas(name: str) -> tuple[ConstraintDelta, ...]:
            return tuple(ConstraintDelta.from_dict(item) for item in value.get(name, ()))

        return cls(
            allow_commit=bool(value["allow_commit"]),
            comparison_mode=str(value.get("comparison_mode", "constraints")),
            regressions=deltas("regressions"), improvements=deltas("improvements"),
            unchanged=deltas("unchanged"), inconclusive=deltas("inconclusive"),
            reason=str(value.get("reason", "")),
        )


def _assessment_map(value: Any) -> tuple[dict[str, ConstraintAssessment] | None, bool | None, float | None]:
    """Coerce supported comparison inputs and retain legacy verdict signals."""

    accept: bool | None = None
    metrics: Any = value
    if hasattr(value, "metrics"):
        metrics = getattr(value, "metrics")
        raw_accept = getattr(value, "accept", None)
        accept = raw_accept if isinstance(raw_accept, bool) else None

    payload: Any = None
    repair_score: float | None = None
    if isinstance(metrics, Mapping):
        raw_accept = metrics.get("accept")
        if accept is None and isinstance(raw_accept, bool):
            accept = raw_accept
        repair_score = _finite_float(metrics.get("repair_score"))
        if CONSTRAINT_ASSESSMENTS_KEY in metrics:
            payload = metrics[CONSTRAINT_ASSESSMENTS_KEY]
        elif "constraint" in metrics and "status" in metrics:
            payload = (metrics,)
    elif isinstance(metrics, Iterable) and not isinstance(metrics, (str, bytes)):
        payload = metrics

    if payload is None:
        return None, accept, repair_score
    result: dict[str, ConstraintAssessment] = {}
    for raw in payload:
        item = raw if isinstance(raw, ConstraintAssessment) else ConstraintAssessment.from_dict(raw)
        if item.id in result:
            raise ValueError(f"duplicate constraint assessment id: {item.id}")
        result[item.id] = item
    return result, accept, repair_score


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def compare_constraint_assessments(
    parent: Any,
    candidate: Any,
    *,
    score_tolerance: float = 0.0,
    unknown_is_regression: bool = False,
) -> NoRegressionDecision:
    """Compare a repaired candidate against its stable parent.

    A clear semantic downgrade (satisfied -> violated), or a score decline
    beyond ``score_tolerance`` while both versions remain violated, blocks the
    commit.  Losing evidence for a previously satisfied constraint also blocks:
    that candidate cannot support a no-regression claim.  Unknown evidence on
    an already-violated parent remains inconclusive by default; it does not
    pretend that an unobserved failure was caused by the repair.

    Parent and candidate must expose the same non-empty set of immutable
    constraint definitions.  A missing, added, or redefined constraint is a
    comparison error and fails closed instead of silently comparing different
    requirements.

    During migration, if either side has no ``constraint_assessments``, the
    function compares available Verdict ``accept`` flags and then the legacy
    ``repair_score``.  This path is labelled in ``comparison_mode`` so papers
    and logs cannot confuse it with the constraint-level guard.  With no
    comparable evidence at all the comparison fails closed.
    """

    tolerance = float(score_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("score_tolerance must be finite and non-negative")
    parent_items, parent_accept, parent_score = _assessment_map(parent)
    candidate_items, candidate_accept, candidate_score = _assessment_map(candidate)
    if parent_items is None or candidate_items is None:
        return _legacy_decision(
            parent_accept, candidate_accept, parent_score, candidate_score,
            score_tolerance=tolerance,
        )

    regressions: list[ConstraintDelta] = []
    improvements: list[ConstraintDelta] = []
    unchanged: list[ConstraintDelta] = []
    inconclusive: list[ConstraintDelta] = []
    constraint_ids = sorted(set(parent_items) | set(candidate_items))
    if not constraint_ids:
        regressions.append(ConstraintDelta(
            "comparison:empty_constraint_set", "schema", "regression",
            None, None, None, None,
            "comparison error: parent and candidate constraint sets are empty",
        ))

    for constraint_id in constraint_ids:
        before = parent_items.get(constraint_id)
        after = candidate_items.get(constraint_id)
        if before is None or after is None:
            present = after or before
            regressions.append(ConstraintDelta(
                constraint_id, present.constraint.kind.value, "regression",
                None if before is None else before.status.value,
                None if after is None else after.status.value,
                None if before is None else before.score,
                None if after is None else after.score,
                (
                    "comparison error: constraint is missing from the parent"
                    if before is None
                    else "comparison error: constraint is missing from the candidate"
                ),
            ))
            continue
        before_definition = before.constraint.to_dict()
        after_definition = after.constraint.to_dict()
        changed_fields = [
            key
            for key in ("kind", "description", "subjects", "shot_indices", "expected", "parameters")
            if before_definition[key] != after_definition[key]
        ]
        if changed_fields:
            regressions.append(ConstraintDelta(
                constraint_id, after.constraint.kind.value, "regression",
                before.status.value, after.status.value, before.score, after.score,
                "comparison error: constraint definition changed in "
                + ", ".join(changed_fields),
            ))
            continue
        delta = _classify_pair(before, after, tolerance, unknown_is_regression)
        {
            "regression": regressions,
            "improvement": improvements,
            "unchanged": unchanged,
            "inconclusive": inconclusive,
        }[delta.classification].append(delta)

    allow = not regressions
    if regressions:
        ids = ", ".join(item.constraint_id for item in regressions)
        if any(item.reason.startswith("comparison error:") for item in regressions):
            reason = f"reject: constraint comparison error or regression: {ids}"
        else:
            reason = f"reject: {len(regressions)} constraint(s) clearly regressed: {ids}"
    else:
        reason = (
            "accept: no constraint clearly regressed "
            f"({len(improvements)} improved, {len(unchanged)} unchanged, "
            f"{len(inconclusive)} inconclusive)"
        )
    return NoRegressionDecision(
        allow_commit=allow, comparison_mode="constraints",
        regressions=tuple(regressions), improvements=tuple(improvements),
        unchanged=tuple(unchanged), inconclusive=tuple(inconclusive), reason=reason,
    )


def _classify_pair(
    before: ConstraintAssessment,
    after: ConstraintAssessment,
    tolerance: float,
    unknown_is_regression: bool,
) -> ConstraintDelta:
    common = dict(
        constraint_id=before.id,
        kind=before.constraint.kind.value,
        parent_status=before.status.value,
        candidate_status=after.status.value,
        parent_score=before.score,
        candidate_score=after.score,
    )
    if before.status == AssessmentStatus.SATISFIED and after.status == AssessmentStatus.VIOLATED:
        return ConstraintDelta(**common, classification="regression", reason="satisfied constraint became violated")
    if before.status == AssessmentStatus.VIOLATED and after.status == AssessmentStatus.SATISFIED:
        return ConstraintDelta(**common, classification="improvement", reason="violated constraint became satisfied")
    if before.status == AssessmentStatus.SATISFIED and after.status == AssessmentStatus.UNKNOWN:
        return ConstraintDelta(
            **common,
            classification="regression",
            reason="candidate lost evidence for a previously satisfied constraint",
        )
    if after.status == AssessmentStatus.UNKNOWN or before.status == AssessmentStatus.UNKNOWN:
        if unknown_is_regression and before.status != AssessmentStatus.UNKNOWN:
            return ConstraintDelta(**common, classification="regression", reason="candidate evidence became unknown under strict policy")
        return ConstraintDelta(**common, classification="inconclusive", reason="at least one assessment is unknown")
    if before.status == after.status == AssessmentStatus.VIOLATED:
        if before.score is None or after.score is None:
            return ConstraintDelta(
                **common,
                classification="inconclusive",
                reason="both constraints remain violated but comparable quality scores are unavailable",
            )
        change = after.score - before.score
        if change < -tolerance:
            return ConstraintDelta(**common, classification="regression", reason=f"violated-constraint score decreased by {-change:.6g}")
        if change > tolerance:
            return ConstraintDelta(**common, classification="improvement", reason=f"violated-constraint score increased by {change:.6g}")
        return ConstraintDelta(**common, classification="unchanged", reason="constraint remains violated without a material score change")
    # Both are satisfied.  Confidence is evidence reliability rather than
    # quality, and a score wobble must not manufacture a regression here.
    return ConstraintDelta(**common, classification="unchanged", reason="constraint remains satisfied")


def _legacy_decision(
    parent_accept: bool | None,
    candidate_accept: bool | None,
    parent_score: float | None,
    candidate_score: float | None,
    *,
    score_tolerance: float,
) -> NoRegressionDecision:
    common = dict(
        constraint_id="legacy:verdict",
        kind="legacy",
        parent_status=None if parent_accept is None else ("satisfied" if parent_accept else "violated"),
        candidate_status=None if candidate_accept is None else ("satisfied" if candidate_accept else "violated"),
        parent_score=parent_score,
        candidate_score=candidate_score,
    )
    if parent_accept is True and candidate_accept is False:
        delta = ConstraintDelta(**common, classification="regression", reason="legacy verdict changed from accept to reject")
        return NoRegressionDecision(False, "legacy_accept_repair_score", regressions=(delta,), reason="reject: legacy verdict regressed")
    if parent_accept is False and candidate_accept is True:
        delta = ConstraintDelta(**common, classification="improvement", reason="legacy verdict changed from reject to accept")
        return NoRegressionDecision(True, "legacy_accept_repair_score", improvements=(delta,), reason="accept: legacy verdict improved")
    if parent_score is not None and candidate_score is not None:
        change = candidate_score - parent_score
        if change < -score_tolerance:
            delta = ConstraintDelta(**common, classification="regression", reason=f"legacy repair_score decreased by {-change:.6g}")
            return NoRegressionDecision(False, "legacy_accept_repair_score", regressions=(delta,), reason="reject: legacy repair_score regressed")
        if change > score_tolerance:
            delta = ConstraintDelta(**common, classification="improvement", reason=f"legacy repair_score increased by {change:.6g}")
            return NoRegressionDecision(True, "legacy_accept_repair_score", improvements=(delta,), reason="accept: legacy repair_score improved")
        delta = ConstraintDelta(**common, classification="unchanged", reason="legacy repair_score is unchanged within tolerance")
        return NoRegressionDecision(True, "legacy_accept_repair_score", unchanged=(delta,), reason="accept: legacy metrics did not regress")
    # No constraint assessments, no accept flags and no comparable scores:
    # nothing supports a no-regression claim, so fail closed like the
    # constraint path does for an empty constraint set.
    delta = ConstraintDelta(**common, classification="regression", reason="comparison error: constraint assessments and comparable legacy metrics are missing")
    return NoRegressionDecision(
        False, "inconclusive", regressions=(delta,),
        reason="reject: no comparable evidence on either side (comparison error)",
    )


__all__ = [
    "CONSTRAINT_ASSESSMENTS_KEY",
    "AssessmentStatus",
    "Constraint",
    "ConstraintAssessment",
    "ConstraintDelta",
    "ConstraintEvidence",
    "ConstraintKind",
    "NoRegressionDecision",
    "Violation",
    "attach_constraint_assessments",
    "compare_constraint_assessments",
    "entity_judge_metrics_to_assessments",
]
