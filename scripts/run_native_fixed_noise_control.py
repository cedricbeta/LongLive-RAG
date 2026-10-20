#!/usr/bin/env python3
"""Matched native integration control: retain the first recipe, only change seed.

This freezes the same reviewed mixed-evidence eligibility as the experimental
route. It does not label unknown as pass. There is no visual planner call, and
all diagnosis, localization, acceptance, rollback and future guards are shared.
"""
import runpy
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import utils.agent_bounded_feedback as feedback

OriginalPolicy = feedback.BoundedFeedbackPolicy


class FixedNoiseControlPolicy(OriginalPolicy):
    def menu(self, first, options, requirements, *, parent_video, parent_latents,
             seed, history, failure):
        # The shared session already enforces reviewed-target eligibility.
        translated = 'target_not_improved' if failure == 'target_evidence_mixed' else failure
        offered, reason = super().menu(first, options, requirements, parent_video=parent_video,
            parent_latents=parent_latents, seed=seed, history=history, failure=translated)
        offered = [r for r in offered if r['purpose'] == 'fixed_recipe']
        return offered, 'menu_ready' if offered else ('fixed_noise_budget_exhausted' if reason == 'menu_ready' else reason)

    def choose(self, menu, **kwargs):
        if not menu:
            return None, {'stop': 'empty_menu', 'calls': []}
        if len(menu) != 1 or menu[0]['purpose'] != 'fixed_recipe':
            raise ValueError('Fixed-noise control cannot choose a changed prompt or region')
        return menu[0], {'selection': 'program_fixed_noise_control', 'calls': []}


if __name__ == '__main__':
    feedback.BoundedFeedbackPolicy = FixedNoiseControlPolicy
    # The shared runner snapshots itself and all utilities. Keep this explicit
    # policy wrapper in the trial's separately frozen plan as well.
    runpy.run_path(str(BASE / 'scripts/run_native_progressive.py'), run_name='__main__')
