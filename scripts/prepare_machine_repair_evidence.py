#!/usr/bin/env python3
"""Extract fixed before/after samples, without scoring or choosing winners.

Each first shot uses all eight chunk starts; each later shot uses its first,
middle, and last frame. Labels retain baseline/repaired provenance: this is
assistant inspection, not an independent blinded evaluation or motion metric.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
from PIL import Image, ImageDraw, ImageFont

from utils.agent_runtime import file_sha256
from scripts.run_oracle_entity_repair import write_json


def sample_frames(path, indices):
    capture = cv2.VideoCapture(str(path))
    wanted = set(indices)
    result = {}
    try:
        for index in range(max(indices) + 1):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"missing frame {index}: {path}")
            if index in wanted:
                result[index] = Image.fromarray(frame[:, :, ::-1])
    finally:
        capture.release()
    return result


def prepare(root):
    packet_path = root / "packet.json"
    packet = json.loads(packet_path.read_text())
    output = root / "machine_evidence"
    output.mkdir(exist_ok=False)
    cases = []
    font = ImageFont.truetype("DejaVuSans.ttf", 16)
    for case in packet["cases"]:
        result_path = root / case["id"] / "result.json"
        if not result_path.exists():
            failure = root / case["id"] / "failure.json"
            cases.append(dict(id=case["id"], status="not_rendered", failure=
                json.loads(failure.read_text()) if failure.exists() else None))
            continue
        result = json.loads(result_path.read_text())
        if result["packet_sha256"] != file_sha256(packet_path):
            raise ValueError("result packet mismatch")
        after = result["result"]["edited_video"]
        if file_sha256(case["video"]) != case["video_sha256"] or file_sha256(after) != result["result"]["edited_video_sha256"]:
            raise ValueError("video changed after rendering")
        durations = [int(n) for n in (root / case["id"] / "data" / case["scene"] / "shot_durations.txt").read_text().split()]
        shots, start = [], 0
        for shot, chunks in enumerate(durations):
            count = chunks * 8
            indices = list(range(start, start + count, 8)) if shot == 0 else [start, start + count // 2, start + count - 1]
            shots.append(dict(shot=shot, indices=indices, caption=json.loads(
                (root / case["id"] / "data" / case["scene"] / f"{shot}.json").read_text())["caption"]))
            start += count
        indices = [i for shot in shots for i in shot["indices"]]
        sampled = [sample_frames(path, indices) for path in (case["video"], after)]
        for shot in shots:
            columns = 4 if shot["shot"] == 0 else 3
            width = 1280 // columns
            height = round(width * 704 / 1280)
            rows = (len(shot["indices"]) + columns - 1) // columns
            sheet = Image.new("RGB", (columns * width, 2 * rows * (height + 26)), "#202020")
            for side, frames in enumerate(sampled):
                for i, index in enumerate(shot["indices"]):
                    x, y = i % columns * width, (side * rows + i // columns) * (height + 26)
                    sheet.paste(frames[index].resize((width, height)), (x, y + 26))
                    label = f"{'BASE' if side == 0 else 'EDIT'} s{shot['shot']} frame {index}"
                    ImageDraw.Draw(sheet).text((x + 3, y + 3), label, font=font, fill="white")
            path = output / f"{case['id']}_s{shot['shot']:02d}.png"
            sheet.save(path)
            shot.update(sheet=str(path), sha256=file_sha256(path))
        cases.append(dict(id=case["id"], status="awaiting_assistant_inspection", entity=case["entity"],
            spec=case["spec"], defect=case["defect_proposal"], before=case["video"], after=after, shots=shots))
    write_json(output / "index.json", dict(packet_sha256=file_sha256(packet_path), cases=cases,
        limitation="Sparse frames cannot establish smooth motion, exact action preservation, or absence of between-sample defects. No human or independent blinded validation."))
    print(json.dumps({"evidence": str(output), "cases": len(cases)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    prepare(parser.parse_args().root.resolve())
