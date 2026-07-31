"""VLM guidance for LongLive-RAG same-shot retrieval.

Two roles, both served by a local Qwen3-VL (SGLang OpenAI-compatible API):
  1. Admission gate: judge each generated block; rejected blocks are barred
     from the retrieval memory pool (mode="guide") or merely labeled
     (mode="observe", for pollution diagnostics with identical compute).
  2. Golden-frame election: early in the shot, pick the frames that show the
     subject most clearly; those frames are force-included in every later
     retrieval (they ride the same position-0 sink-phase injection).
"""
import base64
import io
import json
import re
import urllib.request

import torch
from PIL import Image


def _to_data_uri(frame: torch.Tensor, width: int = 320) -> str:
    """frame: [C, H, W] float in [0, 1] -> base64 jpeg data URI."""
    arr = (frame.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    img = Image.fromarray(arr)
    h = int(img.height * width / img.width)
    img = img.resize((width, h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON in VLM reply: {text[:200]}")
    return json.loads(m.group(0))


class VLMGuide:
    def __init__(self, base_url="http://127.0.0.1:30000/v1",
                 model="Qwen/Qwen3-VL-8B-Instruct", timeout=60):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.calls = 0

    def _chat(self, content, max_tokens=300):
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer none"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            reply = json.loads(r.read())
        self.calls += 1
        return reply["choices"][0]["message"]["content"]

    def judge_block(self, frames, video_prompt: str) -> dict:
        """frames: list of [C,H,W] tensors in [0,1] from one generated block."""
        content = [{"type": "image_url", "image_url": {"url": _to_data_uri(f)}} for f in frames]
        content.append({"type": "text", "text": (
            "These frames come from one short segment of a long AI-generated video. "
            f"The video is supposed to show: \"{video_prompt}\"\n\n"
            "Judge whether this segment is a TRUSTWORTHY visual reference for keeping "
            "the rest of the video consistent. Reject ONLY for clear failures:\n"
            "- the main subject described in the prompt is missing or unrecognizable\n"
            "- the subject's appearance has clearly changed (different clothing/color/identity)\n"
            "- severe visual corruption (heavy smearing, dissolving shapes, garbage textures)\n"
            "Ordinary motion blur, small artifacts, or viewpoint changes are fine.\n\n"
            "Answer with JSON only: {\"subject_ok\": bool, \"appearance_ok\": bool, "
            "\"quality_ok\": bool, \"admit\": bool, \"reason\": \"<15 words\", "
            "\"subject_bbox\": [x1, y1, x2, y2]}\n"
            "subject_bbox: tight bounding box of the MAIN subject in the FIRST image, "
            "in 0-1000 normalized coordinates (x across width, y down height). "
            "Use null if no clear subject."
        )})
        err = None
        for attempt in range(3):
            try:
                out = _extract_json(self._chat(content))
                checks = [out.get(k) for k in ("subject_ok", "appearance_ok", "quality_ok")]
                if all(isinstance(c, bool) for c in checks):
                    out["admit"] = all(checks)
                else:
                    out["admit"] = bool(out.get("admit", True))
                bbox = out.get("subject_bbox")
                if (isinstance(bbox, (list, tuple)) and len(bbox) == 4
                        and all(isinstance(v, (int, float)) for v in bbox)
                        and bbox[0] < bbox[2] and bbox[1] < bbox[3]):
                    out["subject_bbox"] = [max(0, min(1000, int(v))) for v in bbox]
                else:
                    out["subject_bbox"] = None
                return out
            except Exception as e:
                err = e
        return {"admit": True, "reason": f"vlm_error: {err}", "error": True, "subject_bbox": None}

    def rewrite_prompt(self, original_prompt: str, reject_reasons, frames) -> dict:
        """Prompt surgery after block-0 adherence failure.

        frames: sample frames of what the model actually drew instead.
        Returns {"rewritten_prompt": ..., "reason": ...}.
        """
        content = [{"type": "image_url", "image_url": {"url": _to_data_uri(f)}} for f in frames]
        reasons = "; ".join(sorted(set(str(r) for r in reject_reasons)))[:400]
        content.append({"type": "text", "text": (
            "A text-to-video model was asked to generate:\n"
            f"\"{original_prompt}\"\n\n"
            "But across many sampling attempts it kept drawing the WRONG subject. "
            f"The attached frames show what it drew instead. Reviewer notes: {reasons}\n\n"
            "Rewrite the prompt so the model draws the REQUIRED subject:\n"
            "- Keep the scene, action, mood and camera wording unchanged where possible.\n"
            "- Expand the subject with its discriminative visual anatomy — the features "
            "that distinguish it from what was wrongly drawn.\n"
            "- Anchor rare or ambiguous concepts to visually similar COMMON concepts the "
            "model is likely to know well (e.g. describe an unusual animal via a related "
            "familiar animal plus differences).\n"
            "- Do not mention the wrong subject by name in a way that could invite it; "
            "phrase differences positively (what IS there), not negatively.\n\n"
            "Answer with JSON only: {\"rewritten_prompt\": \"...\", \"reason\": \"<20 words\"}"
        )})
        err = None
        for attempt in range(3):
            try:
                out = _extract_json(self._chat(content, max_tokens=500))
                rp = str(out.get("rewritten_prompt", "")).strip()
                if len(rp) > 20:
                    return {"rewritten_prompt": rp, "reason": out.get("reason", "")}
            except Exception as e:
                err = e
        return {"rewritten_prompt": original_prompt, "reason": f"rewrite_failed: {err}", "error": True}

    def elect_golden(self, labeled_frames, video_prompt: str, num_golden=2) -> dict:
        """labeled_frames: list of (global_frame_idx, [C,H,W] tensor)."""
        content = []
        for idx, f in labeled_frames:
            content.append({"type": "text", "text": f"Candidate F{idx}:"})
            content.append({"type": "image_url", "image_url": {"url": _to_data_uri(f)}})
        content.append({"type": "text", "text": (
            f"These are candidate frames from the start of an AI-generated video of: \"{video_prompt}\"\n\n"
            f"Pick the {num_golden} frames that BEST show the main subject clearly and "
            "match the prompt (subject fully visible, well-framed, sharp). These become "
            "permanent reference anchors for the rest of the generation, so prefer frames "
            "showing identity-defining appearance (face, clothing, colors).\n\n"
            "Answer with JSON only: {\"golden\": [frame numbers], \"reason\": \"<20 words\"}"
        )})
        err = None
        for attempt in range(3):
            try:
                out = _extract_json(self._chat(content))
                valid = {i for i, _ in labeled_frames}
                golden = []
                for g in out.get("golden", []):
                    gi = int(re.sub(r"[^0-9]", "", str(g)) or -1)
                    if gi in valid and gi not in golden:
                        golden.append(gi)
                if golden:
                    return {"golden": golden[:num_golden], "reason": out.get("reason", "")}
            except Exception as e:
                err = e
        return {"golden": [labeled_frames[0][0]], "reason": f"fallback_first: {err}", "error": True}
