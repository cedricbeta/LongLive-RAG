# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
"""Closed-source perceptual judge harness for cross-shot consistency.

This module deliberately separates the held-out judge from any local VLM
optimizer. If the OpenAI judge cannot be reached or does not pass the fixture
suite, callers must stop before gating or prompt/KV optimization.
"""

from __future__ import annotations

import base64
import datetime as _dt
import getpass
import hashlib
import inspect
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "subject_identity_preserved": {
            "type": "object",
            "properties": {
                "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "verdict": {"type": "string", "enum": ["low", "medium", "high"]},
                "rationale": {"type": "string"},
            },
            "required": ["score", "verdict", "rationale"],
            "additionalProperties": False,
        },
        "background_coherent": {
            "type": "object",
            "properties": {
                "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "verdict": {"type": "string", "enum": ["low", "medium", "high"]},
                "rationale": {"type": "string"},
            },
            "required": ["score", "verdict", "rationale"],
            "additionalProperties": False,
        },
        "per_shot_adherence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "shot_index": {"type": "integer", "minimum": 0},
                    "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "issue": {"type": "string"},
                },
                "required": ["shot_index", "score", "issue"],
                "additionalProperties": False,
            },
        },
        "inter_shot_diversity": {
            "type": "object",
            "properties": {
                "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "verdict": {"type": "string", "enum": ["collapsed", "borderline", "healthy"]},
                "rationale": {"type": "string"},
            },
            "required": ["score", "verdict", "rationale"],
            "additionalProperties": False,
        },
        "motion_continuity": {
            "type": "object",
            "properties": {
                "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "verdict": {"type": "string", "enum": ["frozen", "borderline", "active"]},
                "rationale": {"type": "string"},
            },
            "required": ["score", "verdict", "rationale"],
            "additionalProperties": False,
        },
        "which_cut_broke": {
            "type": "object",
            "properties": {
                "cut_index": {"anyOf": [{"type": "integer", "minimum": 0}, {"type": "null"}]},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "how": {"type": "string"},
            },
            "required": ["cut_index", "confidence", "how"],
            "additionalProperties": False,
        },
        "cheat_flags": {
            "type": "object",
            "properties": {
                "copy_cheat": {"type": "boolean"},
                "freeze_cheat": {"type": "boolean"},
                "prompt_collapse": {"type": "boolean"},
                "position_bias_suspected": {"type": "boolean"},
                "rationale": {"type": "string"},
            },
            "required": [
                "copy_cheat",
                "freeze_cheat",
                "prompt_collapse",
                "position_bias_suspected",
                "rationale",
            ],
            "additionalProperties": False,
        },
        "overall_consistency_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": [
        "subject_identity_preserved",
        "background_coherent",
        "per_shot_adherence",
        "inter_shot_diversity",
        "motion_continuity",
        "which_cut_broke",
        "cheat_flags",
        "overall_consistency_score",
    ],
    "additionalProperties": False,
}


JUDGE_INSTRUCTIONS = """You are a held-out video consistency judge.

You will see 1-3 decoded frames per shot, in temporal shot order unless the
fixture explicitly says it is a shuffled-order probe. Judge only perceptual
cross-shot consistency:
- subject_identity_preserved: whether the same central subject/object continues
  across shot cuts.
- background_coherent: whether the same scene/place/event persists.
- per_shot_adherence: whether each shot still follows its own caption.
- inter_shot_diversity: whether the shots preserve legitimate requested camera,
  action, composition, and scene variation instead of collapsing toward the same
  looking shot.
- motion_continuity: whether the sampled sequence shows plausible motion/change
  for the captions instead of becoming frozen or near-static.
- which_cut_broke: the cut index i for transition shot i -> i+1 that most
  clearly broke consistency, or null if no cut broke.
- cheat_flags: flag copy/freeze/prompt-collapse cheats. Do not reward a copied
  frame sequence when captions ask for distinct actions or camera views.

Do not penalize a legitimate camera/viewpoint/framing change by itself. Do
penalize any apparent consistency gain that comes from reducing inter-shot
diversity or motion. Return only the strict JSON object required by the
response schema."""


@dataclass(frozen=True)
class JudgeFixture:
    name: str
    description: str
    shots: list[list[Path]]
    captions: list[str]
    expected: str


def utc_timestamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def load_openai_api_key(auth_path: str | Path | None = None) -> tuple[str | None, dict[str, Any]]:
    """Load an API key from env or ``~/.codex/auth.json`` without logging it."""
    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key:
        return env_key, {"source": "env:OPENAI_API_KEY", "present": True}

    path = Path(auth_path or "~/.codex/auth.json").expanduser()
    if not path.exists():
        return None, {"source": str(path), "present": False, "reason": "auth file not found"}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, {"source": str(path), "present": False, "reason": f"auth file unreadable: {exc}"}

    value = data.get("OPENAI_API_KEY") if isinstance(data, dict) else None
    if isinstance(value, str) and value.strip():
        return value.strip(), {"source": str(path), "present": True}
    return None, {
        "source": str(path),
        "present": False,
        "reason": "OPENAI_API_KEY missing or null",
        "auth_keys": sorted(data.keys()) if isinstance(data, dict) else [],
    }


def load_claude_oauth_token(
    credentials_path: str | Path | None = None,
    *,
    claude_bin: str | None = None,
) -> tuple[str | None, dict[str, Any]]:
    """Load Claude Code OAuth metadata, refreshing through the official CLI if expired.

    The returned metadata is safe to log; the token value is returned only for the
    caller's Authorization header and is never included in metadata.
    """
    path = Path(credentials_path or "~/.claude/.credentials.json").expanduser()
    meta: dict[str, Any] = {
        "provider": "claude_oauth",
        "credentials_path": str(path),
        "auth_header_format": "Authorization: Bearer <redacted>",
        "anthropic_beta": "oauth-2025-04-20",
        "token_present": False,
        "expiresAt": None,
        "unexpired": False,
        "refresh_attempted": False,
    }

    def read_file() -> tuple[str | None, int | None, dict[str, Any] | None, str | None]:
        if not path.exists():
            return None, None, None, "credentials file not found"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            return None, None, None, f"credentials file unreadable: {exc}"
        oauth = data.get("claudeAiOauth") if isinstance(data, dict) else None
        if not isinstance(oauth, dict):
            return None, None, data if isinstance(data, dict) else None, "claudeAiOauth missing"
        token = oauth.get("accessToken")
        exp = oauth.get("expiresAt")
        return (
            token if isinstance(token, str) and token else None,
            int(exp) if isinstance(exp, int) else None,
            data if isinstance(data, dict) else None,
            None,
        )

    token, expires_at, data, error = read_file()
    meta["token_present"] = bool(token)
    meta["expiresAt"] = expires_at
    if data is not None:
        meta["top_level_keys"] = sorted(data.keys())
    if error:
        meta["reason"] = error
        return None, meta

    now_ms = int(time.time() * 1000)
    meta["now_ms"] = now_ms
    meta["unexpired"] = bool(expires_at is not None and now_ms < expires_at)
    if token and meta["unexpired"]:
        return token, meta

    meta["refresh_attempted"] = True
    cli = claude_bin or shutil.which("claude")
    meta["refresh_command"] = "claude auth status --json"
    if not cli:
        meta["reason"] = "Claude OAuth token expired and claude CLI is unavailable"
        return None, meta
    refresh = subprocess.run(
        [cli, "auth", "status", "--json"],
        text=True,
        capture_output=True,
        timeout=60,
    )
    meta["refresh_returncode"] = refresh.returncode
    if refresh.returncode != 0:
        meta["reason"] = "Claude OAuth token expired and claude CLI refresh/status failed"
        meta["refresh_stderr_prefix"] = refresh.stderr[:500]
        return None, meta

    token, expires_at, data, error = read_file()
    now_ms = int(time.time() * 1000)
    meta["now_ms_after_refresh"] = now_ms
    meta["token_present"] = bool(token)
    meta["expiresAt"] = expires_at
    meta["unexpired"] = bool(expires_at is not None and now_ms < expires_at)
    if data is not None:
        meta["top_level_keys"] = sorted(data.keys())
    if error:
        meta["reason"] = error
        return None, meta
    if not token or not meta["unexpired"]:
        meta["reason"] = "Claude OAuth token missing or still expired after claude CLI refresh/status"
        return None, meta
    return token, meta


def _write_png(path: Path, rgb: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"failed to write fixture image: {path}")
    return path


def _frame(
    *,
    subject: str,
    background: str,
    viewpoint: int = 0,
    action: int = 0,
    size: int = 160,
) -> np.ndarray:
    bg_colors = {
        "studio": np.array([210, 214, 220], dtype=np.uint8),
        "kitchen": np.array([205, 175, 122], dtype=np.uint8),
        "park": np.array([126, 174, 112], dtype=np.uint8),
    }
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = bg_colors.get(background, bg_colors["studio"])
    offset = 15 * viewpoint
    cv2.rectangle(img, (6 + offset, 10), (70 + offset, 42), (80, 80, 85), -1)
    cv2.circle(img, (126 - offset, 34), 18, (245, 235, 160), -1)
    cv2.line(img, (0, 132 - offset // 3), (size, 104 - offset // 4), (90, 95, 100), 3)

    center = (78 + offset // 2, 88 + action * 4)
    if subject == "red_robot":
        color = (210, 35, 45)
        cv2.rectangle(img, (center[0] - 22, center[1] - 24), (center[0] + 22, center[1] + 24), color, -1)
        cv2.circle(img, (center[0], center[1] - 34), 16, color, -1)
        cv2.circle(img, (center[0] - 6, center[1] - 38), 3, (255, 255, 255), -1)
        cv2.circle(img, (center[0] + 6, center[1] - 38), 3, (255, 255, 255), -1)
    elif subject == "blue_drone":
        color = (40, 90, 220)
        pts = np.array([
            (center[0], center[1] - 32),
            (center[0] + 34, center[1] + 18),
            (center[0] - 34, center[1] + 18),
        ])
        cv2.fillPoly(img, [pts], color)
        cv2.circle(img, (center[0], center[1]), 10, (230, 230, 240), -1)
    else:
        cv2.circle(img, center, 28, (120, 120, 120), -1)

    if action:
        cv2.arrowedLine(
            img,
            (center[0] - 44, center[1] + 38),
            (center[0] + 36, center[1] + 38),
            (30, 30, 30),
            3,
            tipLength=0.2,
        )
    return img


def build_judge_fixtures(root: str | Path) -> list[JudgeFixture]:
    """Create a small deterministic validation suite of PNG frames."""
    root = Path(root)

    def write_set(name: str, specs: list[tuple[str, str, int, int]]) -> list[list[Path]]:
        shots: list[list[Path]] = []
        for shot_idx, (subject, background, viewpoint, action) in enumerate(specs):
            frame_path = root / name / f"shot{shot_idx}_frame0.png"
            _write_png(
                frame_path,
                _frame(subject=subject, background=background, viewpoint=viewpoint, action=action),
            )
            shots.append([frame_path])
        return shots

    viewpoint = write_set(
        "viewpoint_only",
        [
            ("red_robot", "studio", 0, 0),
            ("red_robot", "studio", 2, 1),
            ("red_robot", "studio", 4, 0),
        ],
    )
    subject_swap = write_set(
        "subject_swap",
        [
            ("red_robot", "studio", 0, 0),
            ("blue_drone", "studio", 1, 1),
            ("blue_drone", "studio", 2, 0),
        ],
    )
    copy_cheat = write_set(
        "copy_cheat",
        [("red_robot", "kitchen", 0, 0)] * 3,
    )
    shuffled = [viewpoint[i] for i in [2, 0, 1]]

    return [
        JudgeFixture(
            name="viewpoint_only",
            description="Same red robot and same studio, only camera/framing/action changes.",
            shots=viewpoint,
            captions=[
                "Wide shot of the red robot standing in the gray studio.",
                "Side view of the same red robot moving across the same gray studio.",
                "Closer angled view of the same red robot returning to the studio mark.",
            ],
            expected="viewpoint_change_not_penalized",
        ),
        JudgeFixture(
            name="subject_swap",
            description="Subject changes from red robot to blue drone after the first cut.",
            shots=subject_swap,
            captions=[
                "The red robot starts in the gray studio.",
                "The same red robot should continue moving in the studio.",
                "The same red robot should still be visible in the studio.",
            ],
            expected="subject_identity_low",
        ),
        JudgeFixture(
            name="copy_cheat",
            description="All shots are pixel-identical despite captions asking for different views/actions.",
            shots=copy_cheat,
            captions=[
                "Wide kitchen view of the red robot standing still.",
                "Close-up side view of the red robot moving right.",
                "Overhead view of the red robot reaching the counter.",
            ],
            expected="copy_cheat_flagged",
        ),
        JudgeFixture(
            name="viewpoint_only_shuffled",
            description="The same viewpoint-only frames, deliberately shown in a shuffled shot order.",
            shots=shuffled,
            captions=[
                "Closer angled view of the same red robot returning to the studio mark.",
                "Wide shot of the red robot standing in the gray studio.",
                "Side view of the same red robot moving across the same gray studio.",
            ],
            expected="shuffle_position_bias_probe",
        ),
    ]


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fixture_video_hash(fixture: JudgeFixture) -> str:
    h = hashlib.sha256()
    for shot in fixture.shots:
        for frame in shot:
            h.update(file_sha256(frame).encode("ascii"))
    return h.hexdigest()


def prompt_hash(captions: list[str], instructions: str = JUDGE_INSTRUCTIONS) -> str:
    blob = json.dumps({"instructions": instructions, "captions": captions}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _image_data_url(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def build_judge_input(fixture: JudgeFixture) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": (
                f"Fixture: {fixture.name}\n"
                f"Description: {fixture.description}\n"
                "Shots and captions are listed below. Judge the frames in the order shown."
            ),
        }
    ]
    for shot_idx, (frames, caption) in enumerate(zip(fixture.shots, fixture.captions)):
        content.append({"type": "input_text", "text": f"SHOT {shot_idx} CAPTION: {caption}"})
        for frame_idx, frame_path in enumerate(frames):
            content.append(
                {
                    "type": "input_image",
                    "image_url": _image_data_url(frame_path),
                    "detail": "high",
                }
            )
            content.append(
                {
                    "type": "input_text",
                    "text": f"Frame marker: shot={shot_idx}, frame={frame_idx}, file={frame_path.name}",
                }
            )
    return [{"role": "user", "content": content}]


def build_responses_payload(fixture: JudgeFixture, *, model: str) -> dict[str, Any]:
    return {
        "model": model,
        "store": False,
        "temperature": 0,
        "instructions": JUDGE_INSTRUCTIONS,
        "input": build_judge_input(fixture),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "cross_shot_consistency_judge",
                "strict": True,
                "schema": JUDGE_SCHEMA,
            }
        },
    }


def build_anthropic_messages_payload(fixture: JudgeFixture, *, model: str) -> dict[str, Any]:
    """Build a Claude Messages request with a forced verdict tool call."""
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"Fixture: {fixture.name}\n"
                f"Description: {fixture.description}\n"
                "Shots and captions are listed below. Judge the frames in the order shown."
            ),
        }
    ]
    for shot_idx, (frames, caption) in enumerate(zip(fixture.shots, fixture.captions)):
        content.append({"type": "text", "text": f"SHOT {shot_idx} CAPTION: {caption}"})
        for frame_idx, frame_path in enumerate(frames):
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(frame_path.read_bytes()).decode("ascii"),
                    },
                }
            )
            content.append(
                {
                    "type": "text",
                    "text": f"Frame marker: shot={shot_idx}, frame={frame_idx}, file={frame_path.name}",
                }
            )
    return {
        "model": model,
        "max_tokens": 1200,
        "system": JUDGE_INSTRUCTIONS,
        "tools": [
            {
                "name": "record_consistency_judgment",
                "description": "Record the strict JSON cross-shot consistency verdict.",
                "input_schema": JUDGE_SCHEMA,
            }
        ],
        "tool_choice": {"type": "tool", "name": "record_consistency_judgment"},
        "messages": [{"role": "user", "content": content}],
    }


def build_codex_judge_prompt(fixture: JudgeFixture) -> str:
    lines = [
        JUDGE_INSTRUCTIONS,
        "",
        f"Fixture: {fixture.name}",
        f"Description: {fixture.description}",
        "Images are attached in exact shot order. Return only JSON matching the output schema.",
    ]
    for shot_idx, (frames, caption) in enumerate(zip(fixture.shots, fixture.captions)):
        lines.append(f"SHOT {shot_idx} CAPTION: {caption}")
        for frame_idx, frame_path in enumerate(frames):
            lines.append(f"Frame marker: shot={shot_idx}, frame={frame_idx}, file={frame_path.name}")
    return "\n".join(lines)


def _redact_images_for_log(value: Any) -> Any:
    """Deep-copy a request payload while omitting inline base64 image payloads."""
    safe = json.loads(json.dumps(value))
    if isinstance(safe, dict):
        for item in safe.get("input", []) or []:
            for content in item.get("content", []) or []:
                if isinstance(content, dict) and content.get("type") == "input_image":
                    content["image_url"] = "<base64 image omitted>"
        for message in safe.get("messages", []) or []:
            for content in message.get("content", []) or []:
                if isinstance(content, dict) and content.get("type") == "image":
                    source = content.get("source")
                    if isinstance(source, dict) and "data" in source:
                        source["data"] = "<base64 image omitted>"
    return safe


class OpenAIResponsesError(RuntimeError):
    pass


class OpenAIResponsesClient:
    client_kind = "openai_responses_api_key"

    def __init__(self, api_key: str, *, endpoint: str = "https://api.openai.com/v1/responses"):
        self.api_key = api_key
        self.endpoint = endpoint

    def payload_for_fixture(self, fixture: JudgeFixture, *, model: str) -> dict[str, Any]:
        return build_responses_payload(fixture, model=model)

    def metadata(self) -> dict[str, Any]:
        return {
            "provider": self.client_kind,
            "endpoint": self.endpoint,
            "auth_header_format": "Authorization: Bearer <redacted>",
        }

    def create(self, payload: dict[str, Any], *, timeout: int = 120) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:2000]
            raise OpenAIResponsesError(f"OpenAI HTTP {exc.code}: {detail}") from exc
        except Exception as exc:
            raise OpenAIResponsesError(f"OpenAI request failed: {exc}") from exc


class ClaudeOAuthMessagesClient:
    client_kind = "claude_oauth_messages"

    def __init__(
        self,
        *,
        credentials_path: str | Path | None = None,
        claude_bin: str | None = None,
        endpoint: str = "https://api.anthropic.com/v1/messages",
    ):
        self.credentials_path = credentials_path
        self.claude_bin = claude_bin
        self.endpoint = endpoint
        self.last_auth_meta: dict[str, Any] | None = None

    def payload_for_fixture(self, fixture: JudgeFixture, *, model: str) -> dict[str, Any]:
        return build_anthropic_messages_payload(fixture, model=model)

    def metadata(self) -> dict[str, Any]:
        return {
            "provider": self.client_kind,
            "endpoint": self.endpoint,
            "auth_header_format": "Authorization: Bearer <redacted>",
            "anthropic-beta": "oauth-2025-04-20",
            "anthropic-version": "2023-06-01",
            "credentials_path": str(Path(self.credentials_path or "~/.claude/.credentials.json").expanduser()),
            "last_auth": self.last_auth_meta,
        }

    def create(self, payload: dict[str, Any], *, timeout: int = 120) -> dict[str, Any]:
        token, meta = load_claude_oauth_token(self.credentials_path, claude_bin=self.claude_bin)
        self.last_auth_meta = meta
        if not token:
            raise OpenAIResponsesError(f"Claude OAuth unavailable: {meta.get('reason', 'missing token')}")
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {token}",
                "anthropic-beta": "oauth-2025-04-20",
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:2000]
            raise OpenAIResponsesError(f"Claude OAuth HTTP {exc.code}: {detail}") from exc
        except Exception as exc:
            raise OpenAIResponsesError(f"Claude OAuth request failed: {exc}") from exc
        verdict = extract_anthropic_tool_input(raw)
        usage = raw.get("usage") if isinstance(raw, dict) else {}
        input_tokens = int((usage or {}).get("input_tokens", 0) or 0)
        output_tokens = int((usage or {}).get("output_tokens", 0) or 0)
        return {
            "output_text": json.dumps(verdict),
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
            "raw_response": raw,
            "client": self.metadata(),
        }


def extract_anthropic_tool_input(response: dict[str, Any]) -> dict[str, Any]:
    for content in response.get("content", []) or []:
        if isinstance(content, dict) and content.get("type") == "tool_use":
            if content.get("name") == "record_consistency_judgment":
                value = content.get("input")
                if isinstance(value, dict):
                    validate_verdict_shape(value)
                    return value
    # Some beta/model combinations can return JSON text despite a tool choice.
    text_chunks = [
        c.get("text", "")
        for c in response.get("content", []) or []
        if isinstance(c, dict) and c.get("type") == "text"
    ]
    if text_chunks:
        try:
            value = json.loads("\n".join(text_chunks))
            validate_verdict_shape(value)
            return value
        except Exception as exc:
            raise ValueError("Claude response did not contain the required tool input") from exc
    raise ValueError("Claude response did not contain the required tool input")


class CodexExecJudgeClient:
    client_kind = "codex_exec_oauth"

    def __init__(
        self,
        *,
        codex_bin: str | None = None,
        model: str = "gpt-5.5",
        cwd: str | Path | None = None,
    ):
        self.codex_bin = codex_bin or shutil.which("codex")
        self.model = model
        self.cwd = Path(cwd or Path(__file__).resolve().parents[1])
        self.last_command_meta: dict[str, Any] | None = None

    def payload_for_fixture(self, fixture: JudgeFixture, *, model: str) -> dict[str, Any]:
        payload = build_responses_payload(fixture, model=model)
        payload["codex_prompt"] = build_codex_judge_prompt(fixture)
        payload["codex_images"] = [str(p) for shot in fixture.shots for p in shot]
        return payload

    def metadata(self) -> dict[str, Any]:
        return {
            "provider": self.client_kind,
            "command": "codex exec --output-schema ... --image ...",
            "auth": "reused Codex login through codex CLI; no token file extraction",
            "codex_bin": self.codex_bin,
            "model": self.model,
            "last_command": self.last_command_meta,
        }

    def create(
        self,
        payload: dict[str, Any],
        *,
        fixture: JudgeFixture | None = None,
        output_dir: str | Path | None = None,
        timeout: int = 240,
    ) -> dict[str, Any]:
        if not self.codex_bin:
            raise OpenAIResponsesError("codex CLI is unavailable")
        if fixture is None:
            raise OpenAIResponsesError("Codex exec judge requires fixture paths for --image attachments")
        work = Path(output_dir or tempfile.mkdtemp(prefix="codex-judge-"))
        work.mkdir(parents=True, exist_ok=True)
        schema_path = work / "codex_judge_schema.json"
        output_path = work / f"codex_{fixture.name}_last_message.json"
        schema_path.write_text(json.dumps(JUDGE_SCHEMA, indent=2), encoding="utf-8")
        cmd = [
            self.codex_bin,
            "exec",
            "-C",
            str(self.cwd),
            "--sandbox",
            "read-only",
            "--ephemeral",
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(output_path),
            "--model",
            self.model,
        ]
        for shot in fixture.shots:
            for frame in shot:
                cmd.extend(["--image", str(frame)])
        prompt = build_codex_judge_prompt(fixture)
        cmd.append("-")
        self.last_command_meta = {
            "argv_redacted": [
                part if part not in {str(p) for shot in fixture.shots for p in shot} else "<image-path>"
                for part in cmd
            ],
            "schema_path": str(schema_path),
            "output_path": str(output_path),
        }
        proc = subprocess.run(
            cmd,
            cwd=str(self.cwd),
            text=True,
            input=prompt,
            capture_output=True,
            timeout=timeout,
        )
        self.last_command_meta["returncode"] = proc.returncode
        self.last_command_meta["stderr_prefix"] = proc.stderr[:1000]
        if proc.returncode != 0:
            raise OpenAIResponsesError(f"codex exec judge failed with code {proc.returncode}: {proc.stderr[:1000]}")
        text = output_path.read_text(encoding="utf-8") if output_path.exists() else proc.stdout
        try:
            verdict = json.loads(text)
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                verdict = json.loads(text[start:end + 1])
            else:
                raise ValueError(f"codex exec judge did not return JSON: {text[:500]}")
        validate_verdict_shape(verdict)
        return {
            "output_text": json.dumps(verdict),
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "raw_response": {"stdout_prefix": proc.stdout[:1000], "stderr_prefix": proc.stderr[:1000]},
            "client": self.metadata(),
        }


def extract_response_text(response: dict[str, Any]) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    chunks: list[str] = []
    for item in response.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            text = content.get("text") or content.get("output_text")
            if isinstance(text, str):
                chunks.append(text)
    if chunks:
        return "\n".join(chunks)
    raise ValueError("response did not contain output text")


def parse_judge_json(response: dict[str, Any]) -> dict[str, Any]:
    text = extract_response_text(response)
    try:
        verdict = json.loads(text)
    except Exception as exc:
        raise ValueError(f"judge output was not JSON: {text[:300]}") from exc
    validate_verdict_shape(verdict)
    return verdict


def validate_verdict_shape(verdict: dict[str, Any]) -> None:
    required = set(JUDGE_SCHEMA["required"])
    missing = sorted(required - set(verdict))
    extra = sorted(set(verdict) - required)
    if missing:
        raise ValueError(f"judge verdict missing required fields: {missing}")
    if extra:
        raise ValueError(f"judge verdict has unexpected fields: {extra}")
    for key in ("subject_identity_preserved", "background_coherent"):
        node = verdict[key]
        if not isinstance(node, dict):
            raise ValueError(f"{key} must be an object")
        for child in ("score", "verdict", "rationale"):
            if child not in node:
                raise ValueError(f"{key} missing {child}")
        score = float(node["score"])
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"{key}.score outside [0,1]")
    for key in ("inter_shot_diversity", "motion_continuity"):
        node = verdict[key]
        if not isinstance(node, dict):
            raise ValueError(f"{key} must be an object")
        for child in ("score", "verdict", "rationale"):
            if child not in node:
                raise ValueError(f"{key} missing {child}")
        score = float(node["score"])
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"{key}.score outside [0,1]")
    flags = verdict["cheat_flags"]
    if not isinstance(flags, dict):
        raise ValueError("cheat_flags must be an object")
    for key in ("copy_cheat", "freeze_cheat", "prompt_collapse", "position_bias_suspected"):
        if not isinstance(flags.get(key), bool):
            raise ValueError(f"cheat_flags.{key} must be boolean")
    if not isinstance(verdict["per_shot_adherence"], list):
        raise ValueError("per_shot_adherence must be a list")
    if not isinstance(verdict["which_cut_broke"], dict):
        raise ValueError("which_cut_broke must be an object")


def evaluate_fixture_expectation(fixture: JudgeFixture, verdict: dict[str, Any]) -> tuple[bool, str]:
    subject_score = float(verdict["subject_identity_preserved"]["score"])
    background_score = float(verdict["background_coherent"]["score"])
    flags = verdict["cheat_flags"]
    if fixture.expected == "subject_identity_low":
        ok = subject_score <= 0.35 or verdict["subject_identity_preserved"]["verdict"] == "low"
        return ok, f"subject swap expected low identity, got score={subject_score:.3f}"
    if fixture.expected == "copy_cheat_flagged":
        ok = bool(flags["copy_cheat"] or flags["freeze_cheat"] or flags["prompt_collapse"])
        return ok, f"copy/freeze cheat expected, got flags={flags}"
    if fixture.expected == "viewpoint_change_not_penalized":
        ok = subject_score >= 0.65 and background_score >= 0.55 and not flags["copy_cheat"]
        return ok, (
            "viewpoint-only expected preserved identity/background without copy flag, "
            f"got subject={subject_score:.3f}, background={background_score:.3f}, flags={flags}"
        )
    if fixture.expected == "shuffle_position_bias_probe":
        ok = subject_score >= 0.55 and not flags["position_bias_suspected"]
        return ok, (
            "shuffled-order probe expected no strong position bias, "
            f"got subject={subject_score:.3f}, flags={flags}"
        )
    return False, f"unknown expectation {fixture.expected!r}"


def _stable_repeat_ok(a: dict[str, Any], b: dict[str, Any], *, tolerance: float = 0.10) -> tuple[bool, str]:
    fields = [
        "subject_identity_preserved",
        "background_coherent",
    ]
    diffs = {
        f: abs(float(a[f]["score"]) - float(b[f]["score"]))
        for f in fields
    }
    cut_a = a["which_cut_broke"].get("cut_index")
    cut_b = b["which_cut_broke"].get("cut_index")
    ok = all(v <= tolerance for v in diffs.values()) and cut_a == cut_b
    return ok, f"repeat stability diffs={diffs}, cuts=({cut_a}, {cut_b}), tolerance={tolerance}"


def _safe_payload_for_log(payload: dict[str, Any]) -> dict[str, Any]:
    safe = json.loads(json.dumps(payload))
    for item in safe.get("input", []) or []:
        for content in item.get("content", []) or []:
            if isinstance(content, dict) and content.get("type") == "input_image":
                content["image_url"] = "<base64 image omitted>"
    return safe


def _usage_from_response(response: dict[str, Any]) -> dict[str, Any]:
    usage = response.get("usage") if isinstance(response, dict) else None
    return usage if isinstance(usage, dict) else {}


def run_judge_validation(
    *,
    output_dir: str | Path,
    model: str = "gpt-5.5",
    auth_path: str | Path | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    """Validate the closed judge before any optimizer/gating work."""
    output_dir = Path(output_dir)
    fixtures_dir = output_dir / "judge_validation_frames"
    transcript_dir = output_dir / "judge_validation_transcripts"
    cache_dir = output_dir / "judge_cache"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if client is None:
        api_key, auth = load_openai_api_key(auth_path)
        api: Any = OpenAIResponsesClient(api_key or "")
    else:
        api_key, auth = None, getattr(client, "metadata", lambda: {"provider": "custom"})()
        api = client
    result: dict[str, Any] = {
        "stage": "judge_validation",
        "timestamp_utc": utc_timestamp(),
        "model": model,
        "schema_required_fields": list(JUDGE_SCHEMA["required"]),
        "passed": False,
        "blocked_reason": None,
        "auth": auth,
        "fixtures": [],
        "cache_dir": str(cache_dir),
        "transcript_dir": str(transcript_dir),
        "api_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "client": getattr(api, "metadata", lambda: {"provider": "custom"})(),
    }
    if client is None and not api_key:
        result["blocked_reason"] = f"OpenAI API key unavailable: {auth.get('reason', 'missing key')}"
        return result

    fixtures = build_judge_fixtures(fixtures_dir)
    verdicts: dict[str, dict[str, Any]] = {}
    failures: list[str] = []

    for fixture in fixtures:
        calls = 2 if fixture.name == "viewpoint_only" else 1
        for call_idx in range(calls):
            if hasattr(api, "payload_for_fixture"):
                payload = api.payload_for_fixture(fixture, model=model)
            else:
                payload = build_responses_payload(fixture, model=model)
            video_hash = fixture_video_hash(fixture)
            phash = prompt_hash(fixture.captions)
            cache_path = cache_dir / f"{video_hash}_{phash}_{call_idx}.json"
            transcript_path = transcript_dir / f"{fixture.name}_run{call_idx}.json"
            started = time.time()
            try:
                if cache_path.exists():
                    cached = json.loads(cache_path.read_text(encoding="utf-8"))
                    response = cached["response"]
                    from_cache = True
                else:
                    sig = inspect.signature(api.create)
                    kwargs: dict[str, Any] = {}
                    if "fixture" in sig.parameters:
                        kwargs["fixture"] = fixture
                    if "output_dir" in sig.parameters:
                        kwargs["output_dir"] = transcript_dir
                    if "timeout" in sig.parameters:
                        kwargs["timeout"] = 240
                    response = api.create(payload, **kwargs)
                    cache_path.write_text(json.dumps({"response": response}, indent=2), encoding="utf-8")
                    from_cache = False
                verdict = parse_judge_json(response)
                ok, message = evaluate_fixture_expectation(fixture, verdict)
                if not ok:
                    failures.append(f"{fixture.name}: {message}")
                usage = _usage_from_response(response)
                for key in ("input_tokens", "output_tokens", "total_tokens"):
                    result["api_usage"][key] += int(usage.get(key, 0) or 0)
                elapsed = time.time() - started
                transcript = {
                    "fixture": fixture.name,
                    "run": call_idx,
                    "model": model,
                    "from_cache": from_cache,
                    "elapsed_seconds": elapsed,
                    "video_hash": video_hash,
                    "prompt_hash": phash,
                    "frames_shown": [[str(p) for p in shot] for shot in fixture.shots],
                    "request": _redact_images_for_log(payload),
                    "client": getattr(api, "metadata", lambda: {"provider": "custom"})(),
                    "response": response,
                    "verdict": verdict,
                    "expectation_ok": ok,
                    "expectation_message": message,
                }
                transcript_path.write_text(json.dumps(transcript, indent=2), encoding="utf-8")
                result["fixtures"].append({
                    "name": fixture.name,
                    "run": call_idx,
                    "passed": ok,
                    "message": message,
                    "transcript": str(transcript_path),
                    "cache": str(cache_path),
                    "frames_shown": [[str(p) for p in shot] for shot in fixture.shots],
                })
                verdicts[f"{fixture.name}:{call_idx}"] = verdict
            except Exception as exc:
                failures.append(f"{fixture.name}: API/schema failure: {exc}")

    if "viewpoint_only:0" in verdicts and "viewpoint_only:1" in verdicts:
        stable, message = _stable_repeat_ok(verdicts["viewpoint_only:0"], verdicts["viewpoint_only:1"])
        if not stable:
            failures.append(f"repeat-run stability: {message}")
        result["repeat_run_stability"] = {"passed": stable, "message": message}
    else:
        result["repeat_run_stability"] = {
            "passed": False,
            "message": "repeat fixture did not produce two verdicts",
        }
        failures.append("repeat-run stability: repeat fixture did not produce two verdicts")

    result["passed"] = not failures
    if failures:
        result["blocked_reason"] = "closed judge failed validation: " + "; ".join(failures)
    result["client"] = getattr(api, "metadata", lambda: {"provider": "custom"})()
    return result


def run_oauth_judge_validation(
    *,
    output_dir: str | Path,
    primary_model: str = "claude-opus-4-8",
    codex_model: str = "gpt-5.5",
    provider: str = "auto",
    claude_credentials_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the independent closed judge through reused OAuth paths.

    ``auto`` tries Claude OAuth first, then Codex exec. Both paths avoid paid API
    keys and avoid reading/replaying Codex's OAuth token file.
    """
    root = Path(output_dir)
    attempts: list[dict[str, Any]] = []
    choices = [provider] if provider != "auto" else ["claude", "codex"]
    for choice in choices:
        if choice == "claude":
            client = ClaudeOAuthMessagesClient(credentials_path=claude_credentials_path)
            model = primary_model
            subdir = root / "claude_oauth"
        elif choice == "codex":
            client = CodexExecJudgeClient(model=codex_model)
            model = codex_model
            subdir = root / "codex_exec"
        else:
            return {
                "stage": "judge_validation",
                "timestamp_utc": utc_timestamp(),
                "model": primary_model,
                "passed": False,
                "blocked_reason": f"unknown judge provider {provider!r}",
                "attempts": [],
            }
        validation = run_judge_validation(output_dir=subdir, model=model, client=client)
        validation["provider_choice"] = choice
        attempts.append(validation)
        if validation.get("passed"):
            merged = dict(validation)
            merged["attempts"] = attempts
            merged["selected_provider"] = choice
            return merged
    reasons = [
        f"{a.get('provider_choice')}: {a.get('blocked_reason') or 'validation failed'}"
        for a in attempts
    ]
    return {
        "stage": "judge_validation",
        "timestamp_utc": utc_timestamp(),
        "model": primary_model,
        "schema_required_fields": list(JUDGE_SCHEMA["required"]),
        "passed": False,
        "blocked_reason": "all OAuth judge paths failed validation: " + "; ".join(reasons),
        "attempts": attempts,
        "selected_provider": None,
        "api_usage": {
            "input_tokens": sum(int((a.get("api_usage") or {}).get("input_tokens", 0) or 0) for a in attempts),
            "output_tokens": sum(int((a.get("api_usage") or {}).get("output_tokens", 0) or 0) for a in attempts),
            "total_tokens": sum(int((a.get("api_usage") or {}).get("total_tokens", 0) or 0) for a in attempts),
        },
    }


def _default_sglang_python() -> str:
    return str(Path.home() / "miniconda3/envs/sglang/bin/python")


def _process_users_by_gpu() -> dict[int, list[str]]:
    users: dict[int, list[str]] = {}
    try:
        pmon = subprocess.run(["nvidia-smi", "pmon", "-c", "1"], text=True, capture_output=True, timeout=20)
        rows: list[tuple[int, str]] = []
        pids: set[str] = set()
        for line in pmon.stdout.splitlines():
            if line.lstrip().startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                rows.append((int(parts[0]), parts[1]))
                pids.add(parts[1])
        pid_to_user: dict[str, str] = {}
        if pids:
            ps = subprocess.run(
                ["ps", "-o", "user=,pid=", "-p", ",".join(sorted(pids))],
                text=True,
                capture_output=True,
                timeout=20,
            )
            for line in ps.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    pid_to_user[parts[1]] = parts[0]
        for gpu, pid in rows:
            users.setdefault(gpu, []).append(pid_to_user.get(pid, "unknown"))
    except Exception:
        pass
    return users


def select_free_gpu(
    *,
    max_memory_used_mib: int | None = None,
    max_utilization: int = 15,
    min_free_memory_mib: int = 45000,
) -> dict[str, Any]:
    """Select an idle-headroom GPU.

    This intentionally allows small idle allocations from other users. The
    guard is compute utilization plus free memory, not complete process absence.
    """
    inv = gpu_inventory()
    current_user = getpass.getuser()
    users_by_gpu = _process_users_by_gpu()
    candidates: list[dict[str, Any]] = []
    blockers: list[str] = []
    for gpu in inv.get("gpus", []):
        idx = int(gpu["index"])
        users = sorted(set(users_by_gpu.get(idx, [])))
        free_mib = int(gpu["memory_total_mib"]) - int(gpu["memory_used_mib"])
        ok = (
            (max_memory_used_mib is None or int(gpu["memory_used_mib"]) <= max_memory_used_mib)
            and int(gpu["utilization_gpu_percent"]) <= max_utilization
            and free_mib >= int(min_free_memory_mib)
        )
        row = {
            **gpu,
            "memory_free_mib": free_mib,
            "process_users": users,
            "free_for_optimizer": ok,
            "selection_rule": (
                f"utilization <= {max_utilization}% and free memory >= "
                f"{int(min_free_memory_mib)} MiB; idle shared allocations allowed"
            ),
        }
        if ok:
            candidates.append(row)
        else:
            blockers.append(
                f"gpu{idx}: used={gpu['memory_used_mib']}MiB free={free_mib}MiB "
                f"util={gpu['utilization_gpu_percent']}% users={users}"
            )
    candidates.sort(
        key=lambda row: (
            int(row["utilization_gpu_percent"]),
            -int(row["memory_free_mib"]),
            int(row["index"]),
        )
    )
    return {
        "selected_gpu": candidates[0]["index"] if candidates else None,
        "candidates": candidates,
        "gpu_inventory": inv,
        "blocked_reason": None if candidates else "no idle-headroom GPU for Qwen3 optimizer: " + "; ".join(blockers),
    }


def probe_qwen3_optimizer(
    *,
    sglang_python: str | Path | None = None,
    model: str = "Qwen/Qwen3-VL-8B-Instruct",
) -> dict[str, Any]:
    """Probe the isolated SGLang/Qwen3 optimizer environment without importing it here."""
    py = str(sglang_python or _default_sglang_python())
    out: dict[str, Any] = {
        "optimizer_model": model,
        "required": "Qwen3-VL-8B via SGLang OpenAI-compatible server",
        "qwen2_5_substitution_allowed": False,
        "sglang_python": py,
        "env_ok": False,
        "server_started": False,
        "selected_gpu": None,
        "blocked_reason": None,
        "details": {},
    }
    if not Path(py).exists():
        out["blocked_reason"] = f"SGLang python not found: {py}"
        return out
    code = f"""
import importlib.metadata as m, json
from transformers import AutoConfig
pkgs = {{}}
for pkg in ['sglang','transformers','torch','qwen-vl-utils','openai']:
    try: pkgs[pkg] = m.version(pkg)
    except Exception as e: pkgs[pkg] = 'NOT_INSTALLED'
cfg = AutoConfig.from_pretrained({model!r}, local_files_only=True, trust_remote_code=True)
print(json.dumps({{'packages': pkgs, 'model_type': getattr(cfg, 'model_type', None), 'architectures': getattr(cfg, 'architectures', None)}}))
"""
    proc = subprocess.run([py, "-c", code], text=True, capture_output=True, timeout=60)
    out["details"]["env_probe_returncode"] = proc.returncode
    out["details"]["env_probe_stderr_prefix"] = proc.stderr[:1000]
    if proc.returncode != 0:
        out["blocked_reason"] = f"Qwen3/SGLang env probe failed: {proc.stderr[:1000] or proc.stdout[:1000]}"
        return out
    try:
        out["details"].update(json.loads(proc.stdout.strip().splitlines()[-1]))
    except Exception:
        out["details"]["env_probe_stdout_prefix"] = proc.stdout[:1000]
    if out["details"].get("model_type") != "qwen3_vl":
        out["blocked_reason"] = f"Qwen3 model type not recognized in SGLang env: {out['details'].get('model_type')}"
        return out
    gpu = select_free_gpu()
    out["gpu_selection"] = gpu
    out["selected_gpu"] = gpu.get("selected_gpu")
    if out["selected_gpu"] is None:
        out["blocked_reason"] = gpu.get("blocked_reason")
        return out
    out["env_ok"] = True
    return out


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def qwen3_sglang_launch_command(
    *,
    port: int,
    gpu: int,
    sglang_python: str | Path | None = None,
    model: str = "Qwen/Qwen3-VL-8B-Instruct",
    mem_fraction_static: float = 0.45,
) -> dict[str, Any]:
    py = str(sglang_python or _default_sglang_python())
    mem = min(float(mem_fraction_static), 0.45)
    cmd = [
        py,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--enable-multimodal",
        "--trust-remote-code",
        "--mem-fraction-static",
        str(mem),
        "--served-model-name",
        model,
    ]
    return {
        "cmd": cmd,
        "env": {"CUDA_VISIBLE_DEVICES": str(gpu)},
        "base_url": f"http://127.0.0.1:{port}/v1",
        "mem_fraction_static": mem,
    }


def validate_qwen3_sglang_server(
    *,
    output_dir: str | Path,
    sglang_python: str | Path | None = None,
    model: str = "Qwen/Qwen3-VL-8B-Instruct",
    mem_fraction_static: float = 0.45,
    startup_timeout: int = 300,
) -> dict[str, Any]:
    """Bounded Qwen3 optimizer server validation.

    Starts the server only after the environment and free-GPU preflight passes.
    The process is stopped after `/v1/models` responds; the actual optimizer
    loop should launch its own server run and log transcripts separately.
    """
    started = time.time()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = probe_qwen3_optimizer(sglang_python=sglang_python, model=model)
    result["stage"] = "qwen3_sglang_server_validation"
    result["gpu_time_seconds"] = 0.0
    if result.get("blocked_reason"):
        return result
    port = find_free_port()
    launch = qwen3_sglang_launch_command(
        port=port,
        gpu=int(result["selected_gpu"]),
        sglang_python=sglang_python,
        model=model,
        mem_fraction_static=mem_fraction_static,
    )
    result["launch"] = {
        "cmd": launch["cmd"],
        "env": launch["env"],
        "base_url": launch["base_url"],
        "mem_fraction_static": launch["mem_fraction_static"],
    }
    log_path = out_dir / "qwen3_sglang_server.log"
    env = os.environ.copy()
    env.update(launch["env"])
    proc = None
    try:
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                launch["cmd"],
                cwd=str(Path(__file__).resolve().parents[1]),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        deadline = time.time() + int(startup_timeout)
        models_url = f"{launch['base_url']}/models"
        last_error = None
        while time.time() < deadline:
            if proc.poll() is not None:
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-3000:] if log_path.exists() else ""
                result["blocked_reason"] = (
                    f"SGLang Qwen3 server exited before readiness with code {proc.returncode}: {tail}"
                )
                return result
            try:
                with urllib.request.urlopen(models_url, timeout=5) as response:
                    data = json.loads(response.read().decode("utf-8"))
                result["server_started"] = True
                result["models_response"] = data
                result["base_url"] = launch["base_url"]
                result["log_path"] = str(log_path)
                result["gpu_time_seconds"] = time.time() - started
                return result
            except Exception as exc:
                last_error = str(exc)
                time.sleep(5)
        result["blocked_reason"] = f"SGLang Qwen3 server did not become ready within {startup_timeout}s: {last_error}"
        result["log_path"] = str(log_path)
        return result
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=30)
        result["gpu_time_seconds"] = result.get("gpu_time_seconds") or (time.time() - started)


def probe_local_vlm() -> dict[str, Any]:
    """Backward-compatible name for the required Qwen3/SGLang optimizer probe."""
    return probe_qwen3_optimizer()


def gpu_inventory() -> dict[str, Any]:
    """Record visible GPU occupancy and processes for the ledger."""
    result: dict[str, Any] = {
        "available": False,
        "gpus": [],
        "processes": [],
        "process_users": [],
        "error": None,
    }
    try:
        q = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        for line in q.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                result["gpus"].append(
                    {
                        "index": int(parts[0]),
                        "name": parts[1],
                        "memory_used_mib": int(parts[2]),
                        "memory_total_mib": int(parts[3]),
                        "utilization_gpu_percent": int(parts[4]),
                    }
                )
        pmon = subprocess.run(["nvidia-smi", "pmon", "-c", "1"], text=True, capture_output=True)
        process_lines = [line for line in pmon.stdout.splitlines() if line.strip()]
        result["processes"] = process_lines
        pids: list[str] = []
        for line in process_lines:
            if line.lstrip().startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                pids.append(parts[1])
        if pids:
            ps = subprocess.run(
                ["ps", "-o", "user=,pid=,cmd=", "-p", ",".join(sorted(set(pids)))],
                text=True,
                capture_output=True,
            )
            result["process_users"] = [line for line in ps.stdout.splitlines() if line.strip()]
        result["available"] = True
    except Exception as exc:
        result["error"] = str(exc)
    return result


def build_fail_closed_ledger(
    *,
    validation: dict[str, Any],
    output_dir: str | Path,
    numeric_metrics_path: str | Path | None = None,
) -> dict[str, Any]:
    blocked = validation.get("blocked_reason")
    pass_possible = bool(validation.get("passed"))
    return {
        "date_utc": utc_timestamp().split("T", 1)[0],
        "stage": "closed_judge_pre_gate_validation",
        "pass": False,
        "is_null_result": True,
        "blocked_reason": None if pass_possible else blocked,
        "judge_validation": validation,
        "separation": {
            "judge_model": validation.get("model"),
            "optimizer_probe": probe_local_vlm(),
            "shared_transcript": False,
            "optimizer_used": False,
            "reason": "optimizer is not invoked before closed judge validation passes",
        },
        "numeric_guards": {
            "evaluated": False,
            "reason": "pre-gate did not pass; no ablation gating run",
            "prior_numeric_metrics_path": str(numeric_metrics_path) if numeric_metrics_path else None,
        },
        "ablation": {
            "evaluated": False,
            "arms": ["baseline", "prompt_refine_only", "kv_select_only", "both"],
            "reason": "blocked before optimizer/generator work",
        },
        "repro": {
            "output_dir": str(output_dir),
            "rerun": (
                "python scripts/run_closed_judge_loop.py --output_root "
                f"{output_dir} --model {validation.get('model', 'gpt-5.5')}"
            ),
        },
        "resources": {
            "gpu_inventory": gpu_inventory(),
            "api_tokens": validation.get("api_usage", {}),
            "api_cost_usd": None,
            "api_cost_note": "OAuth judge path; no paid API key configured. Cost is recorded as null."
            if not pass_possible
            else "OAuth judge path; no paid API key configured. Cost is recorded as null.",
            "wall_clock_seconds": None,
            "gpu_time_seconds": 0,
        },
        "next_hypothesis": (
            "Fix the named OAuth judge or Qwen3/SGLang pre-gate blocker, then rerun validation; "
            "only after the closed judge and Qwen3 optimizer pass, admit long-regime scenes and run "
            "the four-arm ablation."
        ),
    }


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
