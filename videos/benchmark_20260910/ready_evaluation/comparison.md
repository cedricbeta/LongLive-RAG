# 当前 C：已交付场景的配对诊断（非完整 benchmark）

已评分 30/33 场景。四列使用完全相同的 sample ID；缺失/失败另列在 population.json，未冒充成功。
没有完整覆盖、fallback 与独立裁判审核前，不宣称方法有效。

| 指标 | A（配对） | B（配对） | 旧 C（配对） | 当前 C |
|---|---:|---:|---:|---:|
| subject_consistency | 91.51 | 91.87 | 91.04 | 91.53 |
| background_consistency | 90.78 | 91.03 | 90.63 | 90.70 |
| temporal_flickering | 84.11 | 83.72 | 83.65 | 83.95 |
| motion_smoothness | 28.89 | 28.93 | 27.18 | 28.58 |
| vtss | 62.75 | 63.12 | 60.51 | 63.40 |
| boundary_smoothness | 1.10 | 0.38 | 0.94 | 0.93 |
| conditional_adjacent | 62.56 | 65.44 | 62.54 | 62.41 |
| conditional_longrange | 75.29 | 77.32 | 73.78 | 75.09 |
| entity_grounding | 58.86 | 56.88 | 59.37 | 60.89 |
| dynamic_trajectory | 56.28 | 56.40 | 56.12 | 56.56 |
| vlm_score | 80.88 | 76.68 | 80.11 | 79.21 |
| quality_score | 71.61 | 71.74 | 70.60 | 71.63 |
| temporal_score | 46.32 | 47.71 | 45.76 | 46.14 |
| instruction_score | 65.34 | 63.32 | 65.20 | 65.55 |
| total_score | 61.09 | 60.92 | 60.52 | 61.11 |
