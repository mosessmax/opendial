"""Accent and line packs: data files that shape the caller and the line.

An accent pack tells the caller's LLM how to speak and lists spelling
variants to forgive when scoring transcription. A line pack is a named set of
audio conditions. Other packages can ship packs by pointing an
`opendial.packs` entry point at a directory of YAML files.
"""

from __future__ import annotations

from functools import cache
from importlib.metadata import entry_points
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

from opendial.scenario import Audio


class AccentPack(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["accent"]
    id: str
    name: str
    description: str
    tts_locale: str
    style: str
    """Instructions for the caller's LLM."""
    examples: list[str] = []
    normalize: dict[str, str] = {}
    """Spelling variant -> canonical form, applied before computing WER."""


class LinePack(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["line"]
    id: str
    name: str
    description: str
    audio: Audio


Pack = AccentPack | LinePack


def _dirs() -> list[Path]:
    extra = (Path(ep.load()) for ep in entry_points(group="opendial.packs"))
    return [Path(__file__).parent, *extra]


@cache
def packs() -> dict[str, Pack]:
    found: dict[str, Pack] = {}
    for root in _dirs():
        for path in sorted(root.glob("*.yaml")):
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            pack = AccentPack(**data) if data.get("kind") == "accent" else LinePack(**data)
            if pack.id in found:
                raise ValueError(f"duplicate pack id {pack.id!r} in {path}")
            found[pack.id] = pack
    return found


def accent(pack_id: str) -> AccentPack:
    pack = packs().get(pack_id)
    if not isinstance(pack, AccentPack):
        raise KeyError(f"unknown accent pack {pack_id!r}")
    return pack


def line(pack_id: str) -> LinePack:
    pack = packs().get(pack_id)
    if not isinstance(pack, LinePack):
        raise KeyError(f"unknown line pack {pack_id!r}")
    return pack


def resolve_audio(audio: Audio) -> Audio:
    """Apply a scenario's line pack, letting fields set on the scenario win."""
    if audio.line is None:
        return audio
    base = line(audio.line).audio.model_dump(exclude_none=True)
    return Audio(**{**base, **audio.model_dump(exclude_none=True)})
