"""Fixed-action comparator contracts, not video efficacy results."""
import importlib.util
from pathlib import Path

from tests.test_agent_bounded_feedback import choices, menu, executed

spec = importlib.util.spec_from_file_location('fixed_noise_control',
    Path(__file__).resolve().parents[1] / 'scripts/run_native_fixed_noise_control.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_reviewed_mix_gets_exactly_one_seed_change_then_stops(choices):
    policy = module.FixedNoiseControlPolicy(allow_mixed_evidence_revision=True)
    first = menu(policy, choices)[0][0]
    following, _ = menu(policy, choices, history=[executed(first)], seed=4244, failure='target_evidence_mixed')
    assert len(following) == 1 and following[0]['kind'] == 'resample'
    second, report = policy.choose(following)
    assert not report['calls']
    assert second['actual']['family_fingerprint'] == first['actual']['family_fingerprint']
    assert second['actual']['noise_seed'] != first['actual']['noise_seed']
    exhausted, _ = menu(policy, choices, history=[executed(first), executed(second)], seed=4245, failure='target_evidence_mixed')
    assert not exhausted


def test_fixed_control_never_broadens_narrows_or_redraws_missing_evidence(choices):
    policy = module.FixedNoiseControlPolicy(allow_mixed_evidence_revision=True)
    first = menu(policy, choices)[0][0]
    for failure in ['localization_miss','protection_regression','future_regression','insufficient_evidence']:
        offered, _ = menu(policy, choices, history=[executed(first)], seed=4244, failure=failure)
        assert not offered
