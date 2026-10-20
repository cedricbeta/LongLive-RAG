#!/usr/bin/env python3
"""Compile scripted entity specs for the entity judge from a scene's shot captions.

For each scene directory (0.json..N.json with captions) one text-only VLM call
returns the recurring entities (people, animals, prominent objects that the
script establishes), a positive appearance description, and the shots whose
caption places each entity in frame.  Output matches
example/narrastream_bench/entity_specs_pilot3.json.

    python scripts/extract_entity_specs.py --data_root example/narrastream_bench/multitext_subset33_b8 \
        --base_url http://127.0.0.1:53456/v1 --out example/narrastream_bench/entity_specs_subset33.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.agent_vlm import FingerprintedVLMClient  # noqa: E402

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "entities": {
            "type": "array",
            "minItems": 1,
            "maxItems": 8,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string", "maxLength": 40},
                    "description": {"type": "string", "maxLength": 160},
                    "shots": {"type": "array", "items": {"type": "integer"}, "maxItems": 12},
                },
                "required": ["name", "description", "shots"],
            },
        }
    },
    "required": ["entities"],
}


def compile_scene(client: FingerprintedVLMClient, scene_dir: Path) -> list[dict]:
    caps = {int(p.stem): json.load(open(p)).get("caption", "") for p in scene_dir.glob("*.json") if p.stem.isdigit()}
    prompt = json.dumps(
        {
            "role": "normative entity compiler",
            "rule": (
                "From the numbered shot captions, list the recurring scripted CHARACTERS (people or animals; never "
                "objects, furniture or props) whose identity must stay consistent across shots. For each: name (as in "
                "the script); description = APPEARANCE ONLY (gender/age, hair, clothing, colours, distinguishing marks) "
                "using only what the captions state, never actions, locations or roles — if the captions give no "
                "appearance, write just 'a man', 'a woman' or 'a person' as the captions imply; and the list of shot "
                "indices whose caption places the character IN FRAME (a caption merely mentioning someone absent, e.g. "
                "'reflecting on the interaction with Iris', does not count). A character present in every shot lists "
                "every shot index."
            ),
            "captions": {str(k): v for k, v in sorted(caps.items())},
            "output": "strict JSON matching the supplied schema",
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    res = client.call(prompt=prompt, image_paths=[], schema=SCHEMA, call_kind="entity_specs", shot_index=0, temporal_inputs=())
    ents = []
    for e in res["parsed"]["entities"]:
        shots = sorted({int(s) for s in e["shots"] if int(s) in caps})
        item = {"name": e["name"].strip(), "description": e["description"].strip()}
        if shots and len(shots) < len(caps):
            item["shots"] = shots
        if not shots:
            continue
        ents.append(item)
    return ents


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--base_url", default="http://127.0.0.1:53456/v1")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache_dir", default="videos/entity_specs_cache")
    args = ap.parse_args()
    client = FingerprintedVLMClient(base_url=args.base_url, model=args.model, cache_dir=ROOT / args.cache_dir, max_tokens=1024)
    table = {"_comment": "Scripted entities compiled from shot captions by scripts/extract_entity_specs.py; review before use."}
    for scene_dir in sorted(p for p in (ROOT / args.data_root).iterdir() if p.is_dir()):
        table[scene_dir.name] = compile_scene(client, scene_dir)
        print(scene_dir.name, json.dumps(table[scene_dir.name], ensure_ascii=False))
    Path(ROOT / args.out).write_text(json.dumps(table, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
