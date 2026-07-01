# Round 17 VLM Closed-Judge Ablation

- Generated UTC: 2026-06-19T02:32:05+00:00
- Repo hash before artifact commit: `844ece081474b163b272d65b7ccc5b58779a55d3`
- Result: `HONEST NULL`
- Winner: `None`
- Gate power: `REDUCED-POWER single-seed exploratory gate; no paired noise floor`
- Baseline path: `videos/round17_20260618/stage_A/baseline_seed0`
- Baseline config: `videos/round17_20260618/stage_A/configs/baseline_seed0.yaml`
- Full Round17 manifest: `round17_manifest.json`

## Pipeline
Orchestrator command:

```bash
CUDA_VISIBLE_DEVICES=<idle-gpu> python scripts/run_round17_vlm_loop.py --adherence_tolerance 0.02 --admission_diversity_floor 0.03 --admission_json_override docs/round17_20260618_fast/fast5_admission.json --baseline_drift_floor 0.02 --baseline_seeds 0 --clip_device cuda:0 --config_path configs/inference_kv_rag_long_multishot.yaml --diversity_tolerance 0.1 --docs_root docs/round17_20260618_fast --extra_logit_bias_lambdas  --kv_anchor_cap 4 --min_admitted_main_scenes 5 --motion_tolerance 0.2 --primary_logit_bias_lambda 0.0 --prompts_dir example/long_multishot_prompts:example/multiview_prompts --rounds 1 --single_seed_consistency_delta_threshold 0.02 --video_root videos/round17_20260618_fast
```

Per-arm reproduction commands for the final primary round:
- `prompt_only`: `CUDA_VISIBLE_DEVICES=<generator-gpu> python inference.py --config_path videos/round17_20260618_fast/iter01/configs/prompt_only.yaml`
- `kv_only`: `CUDA_VISIBLE_DEVICES=<generator-gpu> python inference.py --config_path videos/round17_20260618_fast/iter01/configs/kv_only.yaml`
- `both`: `CUDA_VISIBLE_DEVICES=<generator-gpu> python inference.py --config_path videos/round17_20260618_fast/iter01/configs/both.yaml`

## Scene Admission
- Candidate scenes: 13
- Admitted main scenes: 5
- Negative controls: []
- Synthetic negative control: `degraded_copy_cheat` from `shimmering_puzzle_surface`
- Synthetic negative result: passed_guard=`True` fixture=`degraded_copy_cheat` flags=`{'copy_cheat': True, 'freeze_cheat': True, 'position_bias_suspected': False, 'prompt_collapse': True, 'rationale': 'The same frame is reused for all six captioned shots, eliminating required action, camera variation, and scene changes.'}`
- Deferred from fast run, not rejected: ['aquamarine_underwater', 'cozy_red_room', 'frying_egg_closeup', 'frying_egg_same_event', 'indoor_tender_moment', 'sandy_beach_driftwood', 'vintage_archival_scene', 'warm_indoor_dining']
- Exploratory scored main scenes: ['shimmering_puzzle_surface']
- Consistency-delta gate: `single_seed_fixed_absolute_delta` threshold=`0.02` seeds=`[0]`
- Ledger note: REDUCED-POWER single-seed exploratory gate: no paired baseline noise floor; consistency wins use a fixed absolute delta threshold and are not definitive.

| scene | admitted | negative | reject reason | diversity | drift |
| --- | --- | --- | --- | --- | --- |
| african_savanna | True | False | - | 0.1746 | 0.4004 |
| brown_bear_river | True | False | - | 0.1241 | 0.4543 |
| skateboarder_high_motion | True | False | - | 0.0634 | 0.2487 |
| sunlit_balcony_tour | True | False | - | 0.1058 | 0.1950 |
| shimmering_puzzle_surface | True | False | - | 0.1222 | 0.2672 |
| aquamarine_underwater | False | False | deferred from fast single-pass Round17 run, not rejected | 0.1110 | 0.4339 |
| cozy_red_room | False | False | deferred from fast single-pass Round17 run, not rejected | 0.1681 | 0.1108 |
| frying_egg_closeup | False | False | deferred from fast single-pass Round17 run, not rejected | 0.1240 | 0.4167 |
| frying_egg_same_event | False | False | deferred from fast single-pass Round17 run, not rejected | 0.1264 | 0.2470 |
| indoor_tender_moment | False | False | deferred from fast single-pass Round17 run, not rejected | 0.2185 | 0.5202 |
| sandy_beach_driftwood | False | False | deferred from fast single-pass Round17 run, not rejected | 0.1624 | 0.6369 |
| vintage_archival_scene | False | False | deferred from fast single-pass Round17 run, not rejected | 0.1710 | 0.4502 |
| warm_indoor_dining | False | False | deferred from fast single-pass Round17 run, not rejected | 0.1280 | 0.3992 |

Authored missing `global.json` invariant/contrast captions in the prompt subset for 7 scene(s).

## Final Primary Round
| arm | pass | judge wins | mean judge delta | numeric pass | attention mass | logit calls | guard reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| prompt_only | False | 2/3 | -0.0320 | False | 0.000000 | 0 | numeric guard failed: consistency_ok, diversity_ok, invariant_ok, consistency_wins=0/3 |
| kv_only | False | 2/3 | +0.0060 | False | 0.017277 | 0 | numeric guard failed: consistency_ok, diversity_ok, motion_ok, invariant_ok, consistency_wins=1/3 |
| both | False | 4/3 | +0.0220 | False | 0.018145 | 0 | numeric guard failed: consistency_ok, diversity_ok, motion_ok, invariant_ok, consistency_wins=1/3 |

## Grid Videos
- `african_savanna`: [african_savanna_seed0_4up.mp4](../../videos/round17_20260618_fast/grid_comparisons/african_savanna_seed0_4up.mp4) cuts=[96, 192, 240, 288, 336, 384, 432]
- `brown_bear_river`: [brown_bear_river_seed0_4up.mp4](../../videos/round17_20260618_fast/grid_comparisons/brown_bear_river_seed0_4up.mp4) cuts=[96, 192, 240, 288, 336, 384, 432]
- `skateboarder_high_motion`: [skateboarder_high_motion_seed0_4up.mp4](../../videos/round17_20260618_fast/grid_comparisons/skateboarder_high_motion_seed0_4up.mp4) cuts=[96, 192, 288, 384, 432]
- `sunlit_balcony_tour`: [sunlit_balcony_tour_seed0_4up.mp4](../../videos/round17_20260618_fast/grid_comparisons/sunlit_balcony_tour_seed0_4up.mp4) cuts=[96, 192, 288, 384, 432]
- `shimmering_puzzle_surface`: [shimmering_puzzle_surface_seed0_4up.mp4](../../videos/round17_20260618_fast/grid_comparisons/shimmering_puzzle_surface_seed0_4up.mp4) cuts=[96, 192, 288, 384, 432]

## Per-Scene Details
### african_savanna
- Shared manual plan differs from per-scene KV plan: `True`
- Prompt diffs:
- `videos/round17_20260618_fast/iter01/prompt_refined/african_savanna/global.json`
  - before: One continuous wildlife scene in the same open African savanna under warm daylight. The invariant environment is dry golden grass, scattered acacia trees, distant low hills, and a wide natural horizon. The invariant subject is the same group of large savann...
  - after: One continuous wildlife scene in the same open African savanna under warm daylight. The invariant environment is dry golden grass, scattered acacia trees, distant low hills, and a wide natural horizon. The invariant subject is the same group of large savanna animals moving through the same grassland without changing species or setting. Added invariant: Th...
- `videos/round17_20260618_fast/iter01/prompt_refined/african_savanna/0.json`
  - before: The video presents a warm, golden-hued African savanna scene under a soft, diffused daylight, suggesting late afternoon or early evening. The atmosphere is serene and natural, with gentle winds rustling the grasses and leaves of scattered acacia trees. The ...
  - after: The video presents a warm, golden-hued African savanna scene under a soft, diffused daylight, suggesting late afternoon or early evening. The atmosphere is serene and natural, with gentle winds rustling the grasses and leaves of scattered acacia trees. The style is documentary-like, emphasizing realism with a shallow depth of field that subtly blurs the f...
- `videos/round17_20260618_fast/iter01/prompt_refined/african_savanna/1.json`
  - before: The video presents a warm, golden-hued tone under natural daylight, with soft, diffused sunlight filtering through a partly cloudy sky, casting gentle shadows across the savanna. The atmosphere is tranquil and immersive, evoking a sense of quiet observation...
  - after: The video presents a warm, golden-hued tone under natural daylight, with soft, diffused sunlight filtering through a partly cloudy sky, casting gentle shadows across the savanna. The atmosphere is tranquil and immersive, evoking a sense of quiet observation and awe. The style is documentary realism, with a handheld or mounted camera capturing intimate, ob...
- `videos/round17_20260618_fast/iter01/prompt_refined/african_savanna/2.json`
  - before: The video showcases a serene, natural African savanna landscape under a partly cloudy sky, with soft, diffused daylight illuminating the scene. The color tone is warm and earthy — golden yellows of the grass, deep browns of the elephant’s skin, and muted gr...
  - after: The video showcases a serene, natural African savanna landscape under a partly cloudy sky, with soft, diffused daylight illuminating the scene. The color tone is warm and earthy — golden yellows of the grass, deep browns of the elephant’s skin, and muted greens of scattered shrubs — contrasted against the purplish-gray tones of distant hills and the pale,...
- `videos/round17_20260618_fast/iter01/prompt_refined/african_savanna/3.json`
  - before: The video opens with a warm, earthy-toned palette, bathed in soft, natural daylight that filters through the dappled canopy above. The lighting is gentle and diffused, casting subtle shadows that accentuate the textured, wrinkled skin of the elephant. The a...
  - after: The video opens with a warm, earthy-toned palette, bathed in soft, natural daylight that filters through the dappled canopy above. The lighting is gentle and diffused, casting subtle shadows that accentuate the textured, wrinkled skin of the elephant. The atmosphere is serene and immersive, evoking a sense of quiet wilderness. The style is documentary rea...
- `videos/round17_20260618_fast/iter01/prompt_refined/african_savanna/4.json`
  - before: The video presents a serene, naturalistic portrayal of an elephant in its wild habitat, captured under warm, golden-hour lighting that bathes the scene in a soft, earthy tone. The color palette is dominated by deep browns and muted greens, evoking a tranqui...
  - after: The video presents a serene, naturalistic portrayal of an elephant in its wild habitat, captured under warm, golden-hour lighting that bathes the scene in a soft, earthy tone. The color palette is dominated by deep browns and muted greens, evoking a tranquil, sun-dappled savanna atmosphere. The lighting suggests late afternoon or early evening, with gentl...
- `videos/round17_20260618_fast/iter01/prompt_refined/african_savanna/5.json`
  - before: The video presents a vivid, naturalistic portrayal of an African elephant traversing a savanna landscape under a dynamic sky. The color tone is warm and earthy, dominated by golden-brown grasses, deep greens of scattered acacia trees, and the gray-brown ski...
  - after: The video presents a vivid, naturalistic portrayal of an African elephant traversing a savanna landscape under a dynamic sky. The color tone is warm and earthy, dominated by golden-brown grasses, deep greens of scattered acacia trees, and the gray-brown skin of the elephant. The lighting is soft and diffused, suggesting either early morning or late aftern...
- `videos/round17_20260618_fast/iter01/prompt_refined/african_savanna/6.json`
  - before: The video opens with a warm, golden-hued tone under bright, natural daylight, casting soft, diffused light across the savanna. The atmosphere is serene yet dynamic, evoking the wildness of an African safari. The style is documentary-like, with handheld came...
  - after: The video opens with a warm, golden-hued tone under bright, natural daylight, casting soft, diffused light across the savanna. The atmosphere is serene yet dynamic, evoking the wildness of an African safari. The style is documentary-like, with handheld camera movement that conveys immediacy and immersion. The camera starts with a close-up of an adult Afri...
- `videos/round17_20260618_fast/iter01/prompt_refined/african_savanna/7.json`
  - before: The video presents a serene, naturalistic scene in a savanna environment under a dynamic sky. The color tone is warm and earthy, with golden-brown grasses, muted greens of scattered shrubs, and a soft gradient of blue and white clouds overhead. The lighting...
  - after: The video presents a serene, naturalistic scene in a savanna environment under a dynamic sky. The color tone is warm and earthy, with golden-brown grasses, muted greens of scattered shrubs, and a soft gradient of blue and white clouds overhead. The lighting is natural daylight, with midday sun casting even illumination and minimal harsh shadows, creating ...
- KV-selected decoded frames:
  - boundary `1` frames `[0]`: Frame 0 from Shot 0 provides the foundational view of the savanna and elephant from inside the vehicle, establishing the consistent perspective.
  - boundary `2` frames `[0, 48]`: Frames 0 and 48 from Shot 0 provide a stable anchor for the transition into Shot 2, reinforcing the vehicle's interior perspective.
  - boundary `3` frames `[0, 48, 95]`: Frames 0, 48, and 95 from Shot 0 provide a robust anchor for the transition into Shot 3, reinforcing the vehicle's interior perspective.
  - boundary `4` frames `[0, 48, 95, 96]`: Frames 0, 48, 95, and 96 from Shot 0 provide a robust anchor for the transition into Shot 4, reinforcing the vehicle's interior perspective.
  - boundary `5` frames `[0, 48, 95, 96]`: Frames 0, 48, 95, 96, and 144 from Shot 0 provide a robust anchor for the transition into Shot 5, reinforcing the vehicle's interior perspective.
  - boundary `6` frames `[0, 48, 95, 96]`: Frames 0, 48, 95, 96, 144, and 191 from Shot 0 provide a robust anchor for the transition into Shot 6, reinforcing the vehicle's interior perspective.
  - boundary `7` frames `[0, 48, 95, 96]`: Frames 0, 48, 95, 96, 144, 191, and 192 from Shot 0 provide a robust anchor for the transition into Shot 7, reinforcing the vehicle's interior perspective.
- Selected/anchor frame gallery:
  - ![african_savanna shot00_frame00_idx0000](iter01/optimizer_frames/african_savanna/shot00_frame00_idx0000.png)
  - ![african_savanna shot00_frame01_idx0048](iter01/optimizer_frames/african_savanna/shot00_frame01_idx0048.png)
  - ![african_savanna shot00_frame02_idx0095](iter01/optimizer_frames/african_savanna/shot00_frame02_idx0095.png)
  - ![african_savanna shot01_frame00_idx0096](iter01/optimizer_frames/african_savanna/shot01_frame00_idx0096.png)

### brown_bear_river
- Shared manual plan differs from per-scene KV plan: `True`
- Prompt diffs:
- `videos/round17_20260618_fast/iter01/prompt_refined/brown_bear_river/global.json`
  - before: One continuous wilderness scene in the same shallow river and forest bank under natural daylight. The invariant subject is the same large brown bear with thick brown fur moving near the same water, mud, stones, grass, and dense green vegetation.
  - after: One continuous wilderness scene in the same shallow river and forest bank under natural daylight. The invariant subject is the same large brown bear with thick brown fur moving near the same water, mud, stones, grass, and dense green vegetation. Added invariant: The bear’s movement is continuous and natural, without artificial digital interventions.
- `videos/round17_20260618_fast/iter01/prompt_refined/brown_bear_river/4.json`
  - before: The video presents a serene, naturalistic close-up of a brown bear’s face, bathed in soft, diffused daylight that suggests either early morning or late afternoon. The color tone is warm and earthy, dominated by browns and muted greens, with a shallow depth ...
  - after: The video presents a serene, naturalistic close-up of a brown bear’s face, bathed in soft, diffused daylight that suggests either early morning or late afternoon. The color tone is warm and earthy, dominated by browns and muted greens, with a shallow depth of field that blurs the background into a soft bokeh of foliage, emphasizing the bear’s facial featu...
- KV-selected decoded frames:
  - boundary `1` frames `[0, 48]`: Frames 0 and 48 establish the bear emerging from the river under natural daylight, setting the baseline for the scene’s lighting and environment.
  - boundary `2` frames `[0, 48, 95]`: Frames 0, 48, and 95 capture the bear’s transition from water to land, reinforcing the riverbank habitat and the bear’s movement.
  - boundary `3` frames `[0, 48, 95, 96]`: Frames 0, 48, 95, and 96 show the bear walking away from the water, maintaining continuity of the riverbank environment.
  - boundary `4` frames `[0, 48, 95, 96]`: Frames 0, 48, 95, and 96 are sufficient to anchor the transition into shot 4, as the bear is still in the riverbank environment before the digital overlays appear.
- Selected/anchor frame gallery:
  - ![brown_bear_river shot00_frame00_idx0000](iter01/optimizer_frames/brown_bear_river/shot00_frame00_idx0000.png)
  - ![brown_bear_river shot00_frame01_idx0048](iter01/optimizer_frames/brown_bear_river/shot00_frame01_idx0048.png)
  - ![brown_bear_river shot00_frame02_idx0095](iter01/optimizer_frames/brown_bear_river/shot00_frame02_idx0095.png)
  - ![brown_bear_river shot01_frame00_idx0096](iter01/optimizer_frames/brown_bear_river/shot01_frame00_idx0096.png)

### skateboarder_high_motion
- Shared manual plan differs from per-scene KV plan: `True`
- Prompt diffs:
- `videos/round17_20260618_fast/iter01/prompt_refined/skateboarder_high_motion/global.json`
  - before: One continuous high-motion skatepark sequence in the same concrete bowl under clear afternoon light. The invariant subject is the same skateboarder wearing a red hoodie, black helmet, dark pants, and white sneakers, riding the same natural-wood skateboard w...
  - after: One continuous high-motion skatepark sequence in the same concrete bowl under clear afternoon light. The invariant subject is the same skateboarder wearing a red hoodie, black helmet, dark pants, and white sneakers, riding the same natural-wood skateboard with black grip tape. The invariant environment is the same graffiti-marked concrete skate bowl with ...
- `videos/round17_20260618_fast/iter01/prompt_refined/skateboarder_high_motion/2.json`
  - before: Shot 2 switches to an overhead angle. The skateboarder snaps a small ollie near the lip and lands back into the bowl.
  - after: Shot 2 switches to an overhead angle. The skateboarder snaps a small ollie near the lip and lands back into the bowl. Maintain invariant: The skateboarder wears a black helmet and a black t-shirt with white lettering.
- KV-selected decoded frames:
  - boundary `1` frames `[48]`: Frame 96 is the first frame of Shot 1 and captures the rider's helmet and hoodie clearly, providing a stable anchor for Shot 2.
- Selected/anchor frame gallery:
  - ![skateboarder_high_motion shot00_frame01_idx0048](iter01/optimizer_frames/skateboarder_high_motion/shot00_frame01_idx0048.png)

### sunlit_balcony_tour
- Shared manual plan differs from per-scene KV plan: `True`
- Prompt diffs:
- `videos/round17_20260618_fast/iter01/prompt_refined/sunlit_balcony_tour/global.json`
  - before: One continuous apartment tour in the same sunlit modern living room and balcony area. The invariant subject is the same young adult male host wearing a white V-neck T-shirt, dark trousers, and a small clipped microphone. The invariant environment is warm da...
  - after: One continuous apartment tour in the same sunlit modern living room and balcony area. The invariant subject is the same young adult male host wearing a white V-neck T-shirt, dark trousers, and a small clipped microphone. The invariant environment is warm daylight, wood wall panels, a blue armchair, a round coffee table, framed wall art, and a glass balcon...
- `videos/round17_20260618_fast/iter01/prompt_refined/sunlit_balcony_tour/0.json`
  - before: Shot 0 opens in a medium view near the living room doorway. The host faces the camera and gestures toward the seating area while daylight enters from the balcony side.
  - after: Shot 0 opens in a medium view near the living room doorway. The host faces the camera and gestures toward the seating area while daylight enters from the balcony side. Maintain invariant: The host wears a dark blue T-shirt with a green logo.
- `videos/round17_20260618_fast/iter01/prompt_refined/sunlit_balcony_tour/1.json`
  - before: Shot 1 cuts wider. The host walks a few steps past the blue armchair and sweeps one hand across the room to show the layout.
  - after: Shot 1 cuts wider. The host walks a few steps past the blue armchair and sweeps one hand across the room to show the layout. Maintain invariant: The host wears a dark blue T-shirt with a green logo.
- `videos/round17_20260618_fast/iter01/prompt_refined/sunlit_balcony_tour/2.json`
  - before: Shot 2 tracks toward the glass balcony door. The host turns his shoulders toward the bright doorway and points outside.
  - after: Shot 2 tracks toward the glass balcony door. The host turns his shoulders toward the bright doorway and points outside. Maintain invariant: The host wears a dark blue T-shirt with a green logo.
- `videos/round17_20260618_fast/iter01/prompt_refined/sunlit_balcony_tour/3.json`
  - before: Shot 3 moves to a closer side angle by the balcony door. The host opens the door partway and steps into the light.
  - after: Shot 3 moves to a closer side angle by the balcony door. The host opens the door partway and steps into the light. Maintain invariant: The host wears a dark blue T-shirt with a green logo.
- `videos/round17_20260618_fast/iter01/prompt_refined/sunlit_balcony_tour/4.json`
  - before: Shot 4 looks back from near the balcony into the room. The host gestures from the doorway toward the chair, table, and wall art.
  - after: Shot 4 looks back from near the balcony into the room. The host gestures from the doorway toward the chair, table, and wall art. Maintain invariant: The host wears a dark blue T-shirt with a green logo.
- `videos/round17_20260618_fast/iter01/prompt_refined/sunlit_balcony_tour/5.json`
  - before: Shot 5 finishes with a steady medium shot. The host stands beside the balcony door, nods to the camera, and lowers his hands.
  - after: Shot 5 finishes with a steady medium shot. The host stands beside the balcony door, nods to the camera, and lowers his hands. Maintain invariant: The host wears a dark blue T-shirt with a green logo.
- KV-selected decoded frames:
  - boundary `1` frames `[0, 48]`: Frames 0 and 48 establish the host's initial appearance and shirt color before the cut to shot 1.
  - boundary `2` frames `[0, 48, 95, 96]`: Frames 0, 48, 95, and 96 provide a stable anchor for the transition into shot 2, maintaining visual continuity.
  - boundary `3` frames `[0, 48, 95, 96]`: Frames 0, 48, 95, 96, 144, 191, and 192 anchor the transition into shot 3, ensuring the host's appearance remains consistent.
  - boundary `4` frames `[0, 48, 95, 96]`: Frames 0, 48, 95, 96, 144, 191, 192, 240, 287, and 288 anchor the transition into shot 4, maintaining visual continuity.
  - boundary `5` frames `[0, 48, 95, 96]`: Frames 0, 48, 95, 96, 144, 191, 192, 240, 287, 288, 336, 383, and 384 anchor the transition into shot 5, ensuring the host's appearance remains consistent.
- Selected/anchor frame gallery:
  - ![sunlit_balcony_tour shot00_frame00_idx0000](iter01/optimizer_frames/sunlit_balcony_tour/shot00_frame00_idx0000.png)
  - ![sunlit_balcony_tour shot00_frame01_idx0048](iter01/optimizer_frames/sunlit_balcony_tour/shot00_frame01_idx0048.png)
  - ![sunlit_balcony_tour shot00_frame02_idx0095](iter01/optimizer_frames/sunlit_balcony_tour/shot00_frame02_idx0095.png)
  - ![sunlit_balcony_tour shot01_frame00_idx0096](iter01/optimizer_frames/sunlit_balcony_tour/shot01_frame00_idx0096.png)

### shimmering_puzzle_surface
- Exploratory note: ill-posed/self-contradicting prompt promoted to scored main scene for this fast run.
- Shared manual plan differs from per-scene KV plan: `True`
- Prompt diffs:
- `videos/round17_20260618_fast/iter01/prompt_refined/shimmering_puzzle_surface/global.json`
  - before: Negative control: this folder preserves an intentionally conflicting abstract scene. The invariant claims one coherent shimmering puzzle surface with stable interlocking pieces and a consistent reflective tabletop, while shot captions may pull toward incomp...
  - after: Negative control: this folder preserves an intentionally conflicting abstract scene. The invariant claims one coherent shimmering puzzle surface with stable interlocking pieces and a consistent reflective tabletop, while shot captions may pull toward incompatible object layouts. A consistency win without adherence loss here is text-override evidence, neve...
- `videos/round17_20260618_fast/iter01/prompt_refined/shimmering_puzzle_surface/0.json`
  - before: The video opens with a shimmering, textured surface resembling water or snow, dotted with scattered green fragments that resemble puzzle pieces or leaves. The lighting is soft and diffused, casting gentle shadows and creating a serene, almost ethereal atmos...
  - after: The video opens with a shimmering, textured surface resembling water or snow, dotted with scattered green fragments that resemble puzzle pieces or leaves. The lighting is soft and diffused, casting gentle shadows and creating a serene, almost ethereal atmosphere. The color tone leans toward muted greens and grays, suggesting a calm, quiet environment, pos...
- `videos/round17_20260618_fast/iter01/prompt_refined/shimmering_puzzle_surface/1.json`
  - before: The video features a festive, cartoonish 3D animation with a warm, holiday-themed color tone dominated by reds, greens, and golds. The lighting is soft and diffused, evoking a cozy indoor atmosphere, likely during the daytime, as no shadows suggest harsh su...
  - after: The video features a festive, cartoonish 3D animation with a warm, holiday-themed color tone dominated by reds, greens, and golds. The lighting is soft and diffused, evoking a cozy indoor atmosphere, likely during the daytime, as no shadows suggest harsh sunlight or nighttime. The style is playful and exaggerated, typical of animated holiday specials, wit...
- `videos/round17_20260618_fast/iter01/prompt_refined/shimmering_puzzle_surface/2.json`
  - before: The video features a stylized, animated Santa Claus character in a festive, indoor setting, likely a decorated room or workshop. The color tone is warm and festive, dominated by rich reds, whites, and greens, complemented by soft, diffused lighting that evo...
  - after: The video features a stylized, animated Santa Claus character in a festive, indoor setting, likely a decorated room or workshop. The color tone is warm and festive, dominated by rich reds, whites, and greens, complemented by soft, diffused lighting that evokes a cozy, holiday atmosphere. The style is 3D computer animation with a slightly exaggerated, cart...
- `videos/round17_20260618_fast/iter01/prompt_refined/shimmering_puzzle_surface/3.json`
  - before: The video presents a festive, animated scene with a warm, slightly saturated color tone, dominated by rich reds, greens, and whites, evoking a cheerful holiday atmosphere. Lighting is soft and diffused, suggesting an indoor or artificially lit environment, ...
  - after: The video presents a festive, animated scene with a warm, slightly saturated color tone, dominated by rich reds, greens, and whites, evoking a cheerful holiday atmosphere. Lighting is soft and diffused, suggesting an indoor or artificially lit environment, possibly during evening or nighttime, with no visible natural light sources. The style is cartoonish...
- KV-selected decoded frames:
  - boundary `1` frames `[0, 48]`: Frames 0 and 48 establish the initial environment and Santa's interaction with puzzle pieces.
  - boundary `2` frames `[0, 48, 95]`: Frames 0, 48, and 95 reinforce the consistent indoor setting with puzzle pieces and tree.
  - boundary `3` frames `[0, 48, 95, 96]`: Frame 96 continues the same environment, ensuring continuity into Shot 3.
- Selected/anchor frame gallery:
  - ![shimmering_puzzle_surface shot00_frame00_idx0000](iter01/optimizer_frames/shimmering_puzzle_surface/shot00_frame00_idx0000.png)
  - ![shimmering_puzzle_surface shot00_frame01_idx0048](iter01/optimizer_frames/shimmering_puzzle_surface/shot00_frame01_idx0048.png)
  - ![shimmering_puzzle_surface shot00_frame02_idx0095](iter01/optimizer_frames/shimmering_puzzle_surface/shot00_frame02_idx0095.png)
  - ![shimmering_puzzle_surface shot01_frame00_idx0096](iter01/optimizer_frames/shimmering_puzzle_surface/shot01_frame00_idx0096.png)

