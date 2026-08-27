#!/usr/bin/env python3
"""Convert NarraStream-Bench prompt suite to MultiTextConcatDataset scene dirs.

NarraStream-Bench (github.com/Eddie0521/NarraStream-Bench) ships
`streambench_324.jsonl`: 324 lines, each `{"prompts": [six 10-second segment
prompts]}`, block-ordered by scene anchor (12 anchors x 9 blueprints x 3
variants). This script materializes each selected record as a scene folder
our dataset already understands:

    <out>/ns<idx>_<slug>/{0..5}.json   (caption per segment)
    <out>/ns<idx>_<slug>/shot_durations.txt

No global.json is written: NarraStream prompts are self-contained by design
(each segment restates the scene anchor), and the benchmark tests exactly
whether the model resolves cross-segment references without a global crutch.

Native segment length is 10s. At 24 fps with 8-pixel-frame blocks that is 30
blocks/segment (60s total per video). Use --blocks-per-segment 8 (~2.7s) for
iteration renders; 30 for benchmark-faithful runs.

Because the jsonl carries no per-record dimension metadata, subset selection
uses a stride over the file order, which walks across the anchor blocks and
so covers all 12 scenes; --indices lets you pin an exact list instead.
"""

import argparse
import json
import re
from pathlib import Path

DEFAULT_JSONL = Path(__file__).resolve().parent.parent / "example/narrastream_bench/streambench_324.jsonl"


def slugify(text: str, max_words: int = 4) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return "_".join(words[:max_words]) or "scene"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--blocks-per-segment", type=int, default=8,
                    help="8 (~2.7s/segment) for iteration; 30 for the native 10s/segment")
    ap.add_argument("--stride", type=int, default=1,
                    help="take every Nth record (file is anchor-block-ordered, so a stride samples across scenes)")
    ap.add_argument("--offset", type=int, default=0, help="start index for the stride")
    ap.add_argument("--indices", type=int, nargs="*", default=None,
                    help="explicit record indices; overrides --stride/--offset")
    args = ap.parse_args()

    records = [json.loads(line) for line in args.jsonl.open()]
    if args.indices is not None:
        chosen = args.indices
    else:
        chosen = list(range(args.offset, len(records), args.stride))

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for idx in chosen:
        rec = records[idx]
        prompts = rec["prompts"]
        if len(prompts) != 6:
            raise ValueError(f"record {idx} has {len(prompts)} prompts, expected 6")
        name = f"ns{idx:03d}_{slugify(prompts[0].split(',')[0])}"
        scene = args.out / name
        scene.mkdir(exist_ok=True)
        for seg, caption in enumerate(prompts):
            payload = {
                "caption": caption,
                "title": f"Segment {seg}",
                "source": f"narrastream_bench/streambench_324.jsonl#{idx}",
            }
            (scene / f"{seg}.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        (scene / "shot_durations.txt").write_text(
            "\n".join([str(args.blocks_per_segment)] * 6) + "\n", encoding="utf-8"
        )
        manifest.append({"index": idx, "sample_name": name})

    (args.out / "manifest.json").write_text(
        json.dumps(
            {
                "source": str(args.jsonl),
                "blocks_per_segment": args.blocks_per_segment,
                "selection": {"indices": args.indices, "stride": args.stride, "offset": args.offset},
                "samples": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {len(manifest)} scenes to {args.out}")


if __name__ == "__main__":
    main()
