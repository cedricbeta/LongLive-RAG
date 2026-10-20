import json
import unittest
from dataclasses import dataclass

from utils.agent_constraints import (
    AssessmentStatus,
    Constraint,
    ConstraintAssessment,
    ConstraintEvidence,
    ConstraintKind,
    NoRegressionDecision,
    Violation,
    attach_constraint_assessments,
    compare_constraint_assessments,
    entity_judge_metrics_to_assessments,
)


def _assessment(
    constraint_id: str,
    status: AssessmentStatus,
    *,
    kind: ConstraintKind = ConstraintKind.PRESENCE,
    score: float | None = None,
    description: str | None = None,
    subjects: tuple[str, ...] = ("subject",),
    shot_indices: tuple[int, ...] = (1,),
    expected=True,
    parameters=None,
) -> ConstraintAssessment:
    return ConstraintAssessment(
        constraint=Constraint(
            id=constraint_id,
            kind=kind,
            description=description or f"check {constraint_id}",
            subjects=subjects,
            shot_indices=shot_indices,
            expected=expected,
            parameters={} if parameters is None else parameters,
        ),
        status=status,
        evidence=(ConstraintEvidence("stub", observed=status.value, confidence=0.75),),
        violations=() if status != AssessmentStatus.VIOLATED else (
            Violation("failed", f"{constraint_id} failed"),
        ),
        score=score,
    )


class ConstraintSerializationTests(unittest.TestCase):
    def test_every_kind_round_trips_as_json(self):
        for kind in ConstraintKind:
            original = ConstraintAssessment(
                constraint=Constraint(
                    id=f"c:{kind.value}",
                    kind=kind,
                    description=f"a {kind.value} predicate",
                    subjects=("Iris", "laptop"),
                    shot_indices=(1, 2),
                    expected={"value": True},
                    parameters={"threshold": 0.5},
                ),
                status=AssessmentStatus.VIOLATED,
                evidence=(ConstraintEvidence(
                    "test", observed={"value": False}, confidence=0.75,
                    metadata={"boxes": [[1, 2, 3, 4]]},
                ),),
                violations=(Violation("mismatch", "did not match"),),
                score=0.25,
            )
            encoded = json.loads(original.to_json())
            self.assertEqual(encoded["constraint"]["kind"], kind.value)
            self.assertEqual(ConstraintAssessment.from_dict(encoded), original)

    def test_invalid_confidence_score_and_payload_are_rejected(self):
        with self.assertRaises(ValueError):
            ConstraintEvidence("x", confidence=1.01)
        with self.assertRaises(ValueError):
            _assessment("x", AssessmentStatus.SATISFIED, score=float("nan"))
        with self.assertRaises(TypeError):
            Constraint("x", "presence", "x", expected=object())


class EntityMetricsAdapterTests(unittest.TestCase):
    def setUp(self):
        self.metrics = {
            "entities": {
                "Iris": {
                    "expected": True,
                    "located_chunks": [0, 1],
                    "matches": True,
                    "differences": "",
                    "confidence": 1.0,
                    "comparison_basis": "reference_crop",
                    "coverage": 0.42,
                },
                "Noah": {
                    "expected": True,
                    "located_chunks": [],
                    "matches": None,
                    "confidence": 0.0,
                    "comparison_basis": None,
                    "coverage": 0.0,
                },
                "Offstage": {"expected": False, "located_chunks": []},
            },
            "reference_available": {"Iris": True, "Noah": False},
            "reference_registration_policy": {"min_confidence": 0.75},
            "unassigned_subjects": ["unassigned_0", "unassigned_1"],
            "repair_score": 0.0,
        }

    def test_adapter_emits_presence_appearance_and_count_constraints(self):
        items = entity_judge_metrics_to_assessments(self.metrics, shot_index=2)
        by_id = {item.id: item for item in items}

        iris_presence = by_id["shot:2:entity:Iris:presence"]
        self.assertEqual(iris_presence.status, AssessmentStatus.SATISFIED)
        iris_identity = by_id["shot:2:entity:Iris:identity"]
        self.assertEqual(iris_identity.constraint.kind, ConstraintKind.IDENTITY)
        self.assertEqual(iris_identity.status, AssessmentStatus.SATISFIED)

        self.assertEqual(
            by_id["shot:2:entity:Noah:presence"].status,
            AssessmentStatus.VIOLATED,
        )
        self.assertEqual(
            by_id["shot:2:entity:Noah:script_attributes"].status,
            AssessmentStatus.UNKNOWN,
        )
        self.assertNotIn("shot:2:entity:Offstage:presence", by_id)

        count = by_id["shot:2:count:unassigned_subjects"]
        self.assertEqual(count.constraint.kind, ConstraintKind.COUNT)
        self.assertEqual(count.status, AssessmentStatus.VIOLATED)
        self.assertAlmostEqual(count.score, 1 / 3)

    def test_low_confidence_first_appearance_is_an_attribute_violation(self):
        metrics = {
            "entities": {"Iris": {
                "expected": True, "located_chunks": [0], "matches": True,
                "confidence": 0.5, "comparison_basis": "script_description",
            }},
            "reference_available": {"Iris": False},
            "reference_registration_policy": {"min_confidence": 0.75},
            "unassigned_subjects": [],
        }
        by_id = {
            item.id: item
            for item in entity_judge_metrics_to_assessments(metrics, shot_index=0)
        }
        appearance = by_id["shot:0:entity:Iris:script_attributes"]
        self.assertEqual(appearance.constraint.kind, ConstraintKind.ATTRIBUTE)
        self.assertEqual(appearance.status, AssessmentStatus.VIOLATED)
        self.assertEqual(appearance.violations[0].code, "low_confidence_first_appearance")

    def test_attach_returns_ledger_safe_copy_without_mutating_input(self):
        attached = attach_constraint_assessments(self.metrics, shot_index=2)
        self.assertNotIn("constraint_assessments", self.metrics)
        self.assertIsInstance(attached["constraint_assessments"], list)
        self.assertEqual(
            attached["constraint_assessments"][0]["constraint"]["kind"],
            "presence",
        )
        json.dumps(attached, allow_nan=False)


class NoRegressionTests(unittest.TestCase):
    def test_empty_constraint_sets_fail_closed_as_comparison_error(self):
        decision = compare_constraint_assessments([], [])
        self.assertFalse(decision.allow_commit)
        self.assertEqual(decision.regressions[0].constraint_id, "comparison:empty_constraint_set")
        self.assertIn("comparison error", decision.regressions[0].reason)

    def test_missing_added_and_disjoint_constraint_sets_fail_closed(self):
        satisfied = _assessment("kept", AssessmentStatus.SATISFIED, score=1.0)
        added = _assessment("added", AssessmentStatus.SATISFIED, score=1.0)
        cases = {
            "missing_from_candidate": ([satisfied], []),
            "added_to_candidate": ([], [added]),
            "disjoint": ([satisfied], [added]),
        }
        for name, (parent, candidate) in cases.items():
            with self.subTest(name=name):
                decision = compare_constraint_assessments(parent, candidate)
                self.assertFalse(decision.allow_commit)
                self.assertTrue(decision.regressions)
                self.assertTrue(all("comparison error" in row.reason for row in decision.regressions))

    def test_same_id_with_changed_definition_fails_closed(self):
        parent = _assessment("stable-id", AssessmentStatus.SATISFIED, score=1.0)
        variants = {
            "kind": {"kind": ConstraintKind.ACTION},
            "description": {"description": "a different predicate"},
            "subjects": {"subjects": ("different-subject",)},
            "shot_indices": {"shot_indices": (2,)},
            "expected": {"expected": False},
            "parameters": {"parameters": {"threshold": 0.5}},
        }
        for field, overrides in variants.items():
            with self.subTest(field=field):
                candidate = _assessment(
                    "stable-id", AssessmentStatus.SATISFIED, score=1.0, **overrides,
                )
                decision = compare_constraint_assessments([parent], [candidate])
                self.assertFalse(decision.allow_commit)
                self.assertEqual(len(decision.regressions), 1)
                self.assertIn("constraint definition changed", decision.regressions[0].reason)
                self.assertIn(field, decision.regressions[0].reason)

    def test_satisfied_to_violated_is_a_clear_regression(self):
        decision = compare_constraint_assessments(
            [_assessment("identity", AssessmentStatus.SATISFIED, kind=ConstraintKind.IDENTITY, score=1.0)],
            [_assessment("identity", AssessmentStatus.VIOLATED, kind=ConstraintKind.IDENTITY, score=0.0)],
        )
        self.assertFalse(decision.allow_commit)
        self.assertEqual(decision.regressions[0].constraint_id, "identity")
        self.assertIn("clearly regressed", decision.reason)

    def test_existing_failure_may_improve_without_becoming_perfect(self):
        decision = compare_constraint_assessments(
            [_assessment("extras", AssessmentStatus.VIOLATED, kind=ConstraintKind.COUNT, score=1 / 3)],
            [_assessment("extras", AssessmentStatus.VIOLATED, kind=ConstraintKind.COUNT, score=1 / 2)],
        )
        self.assertTrue(decision.allow_commit)
        self.assertEqual([item.constraint_id for item in decision.improvements], ["extras"])

    def test_existing_failure_that_gets_worse_blocks_commit(self):
        decision = compare_constraint_assessments(
            [_assessment("extras", AssessmentStatus.VIOLATED, kind=ConstraintKind.COUNT, score=1 / 2)],
            [_assessment("extras", AssessmentStatus.VIOLATED, kind=ConstraintKind.COUNT, score=1 / 3)],
            score_tolerance=0.01,
        )
        self.assertFalse(decision.allow_commit)
        self.assertIn("score decreased", decision.regressions[0].reason)

    def test_satisfied_to_unknown_blocks_even_without_strict_policy(self):
        parent = [_assessment("action", AssessmentStatus.SATISFIED, kind=ConstraintKind.ACTION, score=1.0)]
        candidate = [_assessment("action", AssessmentStatus.UNKNOWN, kind=ConstraintKind.ACTION)]
        default = compare_constraint_assessments(parent, candidate)
        strict = compare_constraint_assessments(parent, candidate, unknown_is_regression=True)
        self.assertFalse(default.allow_commit)
        self.assertEqual(len(default.regressions), 1)
        self.assertFalse(strict.allow_commit)

    def test_violated_to_unknown_is_inconclusive_unless_strict(self):
        parent = [_assessment("action", AssessmentStatus.VIOLATED, kind=ConstraintKind.ACTION, score=0.0)]
        candidate = [_assessment("action", AssessmentStatus.UNKNOWN, kind=ConstraintKind.ACTION)]
        default = compare_constraint_assessments(parent, candidate)
        strict = compare_constraint_assessments(parent, candidate, unknown_is_regression=True)
        self.assertTrue(default.allow_commit)
        self.assertEqual(len(default.inconclusive), 1)
        self.assertFalse(strict.allow_commit)

    def test_two_violations_without_comparable_scores_are_inconclusive(self):
        parent = [_assessment("identity", AssessmentStatus.VIOLATED, kind=ConstraintKind.IDENTITY)]
        candidate = [_assessment("identity", AssessmentStatus.VIOLATED, kind=ConstraintKind.IDENTITY)]
        decision = compare_constraint_assessments(parent, candidate)
        self.assertTrue(decision.allow_commit)
        self.assertEqual(len(decision.inconclusive), 1)
        self.assertEqual(decision.unchanged, ())
        self.assertIn("scores are unavailable", decision.inconclusive[0].reason)

    def test_any_regression_blocks_even_when_another_constraint_improves(self):
        parent = [
            _assessment("fixed", AssessmentStatus.VIOLATED, score=0.0),
            _assessment("broken", AssessmentStatus.SATISFIED, score=1.0),
        ]
        candidate = [
            _assessment("fixed", AssessmentStatus.SATISFIED, score=1.0),
            _assessment("broken", AssessmentStatus.VIOLATED, score=0.0),
        ]
        decision = compare_constraint_assessments(parent, candidate)
        self.assertFalse(decision.allow_commit)
        self.assertEqual([item.constraint_id for item in decision.improvements], ["fixed"])
        self.assertEqual([item.constraint_id for item in decision.regressions], ["broken"])

    def test_duplicate_constraint_ids_are_rejected(self):
        duplicate = _assessment("duplicate", AssessmentStatus.SATISFIED, score=1.0)
        with self.assertRaisesRegex(ValueError, "duplicate constraint assessment id"):
            compare_constraint_assessments([duplicate, duplicate], [duplicate])

    def test_serialized_metrics_are_supported_and_decision_round_trips(self):
        parent = {"constraint_assessments": [
            _assessment("presence", AssessmentStatus.VIOLATED, score=0.0).to_dict(),
        ]}
        candidate = {"constraint_assessments": [
            _assessment("presence", AssessmentStatus.SATISFIED, score=1.0).to_dict(),
        ]}
        decision = compare_constraint_assessments(parent, candidate)
        self.assertTrue(decision.allow_commit)
        self.assertEqual(decision.comparison_mode, "constraints")
        self.assertEqual(NoRegressionDecision.from_dict(json.loads(decision.to_json())), decision)

    def test_legacy_verdict_and_repair_score_fallback(self):
        @dataclass
        class LegacyVerdict:
            accept: bool
            metrics: dict

        better = compare_constraint_assessments(
            LegacyVerdict(False, {"repair_score": 0.0}),
            LegacyVerdict(False, {"repair_score": 0.333}),
        )
        worse = compare_constraint_assessments(
            LegacyVerdict(True, {"repair_score": 1.0}),
            LegacyVerdict(False, {"repair_score": 1.0}),
        )
        self.assertTrue(better.allow_commit)
        self.assertEqual(better.comparison_mode, "legacy_accept_repair_score")
        self.assertFalse(worse.allow_commit)

    def test_no_comparable_evidence_fails_closed(self):
        decision = compare_constraint_assessments({}, {})
        self.assertFalse(decision.allow_commit)
        self.assertEqual(decision.comparison_mode, "inconclusive")
        self.assertIn("comparison error", decision.regressions[0].reason)


if __name__ == "__main__":
    unittest.main()
