"""Loader for genre and theory knowledge YAML files.

Provides formatted text blocks ready to inject into CompositionPrep output.
Keeps token budget under control by selecting only the relevant genre.
"""

import json
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_KNOWLEDGE_DIR = Path(__file__).parent

_theory_cache: dict | None = None
_genres_cache: dict | None = None


def _load_theory() -> dict:
    global _theory_cache
    if _theory_cache is None:
        with open(_KNOWLEDGE_DIR / "theory.yaml") as f:
            _theory_cache = yaml.safe_load(f)
    return _theory_cache


def _load_genres() -> dict:
    global _genres_cache
    if _genres_cache is None:
        with open(_KNOWLEDGE_DIR / "genres.yaml") as f:
            _genres_cache = yaml.safe_load(f)
    return _genres_cache


def get_theory_block() -> str:
    """Return universal composition theory as a compact text block."""
    theory = _load_theory()
    lines: list[str] = []

    # Voice leading
    lines.append("VOICE LEADING RULES:")
    for rule in theory["voice_leading"]["rules"]:
        lines.append(f"  - {rule}")

    # Cadences
    lines.append("CADENCES:")
    cad = theory["cadences"]
    lines.append(f"  Authentic: {cad['authentic']}")
    lines.append(f"  Half: {cad['half']}")
    lines.append(f"  Deceptive: {cad['deceptive']}")
    for p in cad["placement"]:
        lines.append(f"  - {p}")

    # Motif development
    lines.append("MOTIF DEVELOPMENT TECHNIQUES:")
    techs = theory["motif_development"]["techniques"]
    for name, desc in techs.items():
        lines.append(f"  {name}: {desc}")

    # Motif workflow (bar-by-bar roadmap)
    lines.append("BAR-BY-BAR COMPOSITION ROADMAP:")
    for step in theory["motif_development"]["workflow"]:
        lines.append(f"  {step}")

    # Melodic principles
    lines.append("MELODIC PRINCIPLES:")
    for item in theory["melodic_principles"]["contour"]:
        lines.append(f"  - {item}")
    for item in theory["melodic_principles"]["intervals"]:
        lines.append(f"  - {item}")

    # Bass independence
    lines.append("BASS VOICE RULES:")
    for rule in theory["bass_independence"]["rules"]:
        lines.append(f"  - {rule}")

    return "\n".join(lines)


def get_genre_block(style: str) -> str:
    """Return genre-specific knowledge as a compact text block.

    Includes form, structure directives, rhythm patterns, voicing,
    melody style, and 2-bar example measures in JSON format.
    Fuzzy-matches style names (e.g. "light jazzy" → "jazz").
    """
    genres = _load_genres()
    style = style.strip().lower()
    genre = genres.get(style)
    if not genre:
        # Fuzzy match: check if any genre key is a substring of the style
        # or vice versa (handles "jazzy" → "jazz", "bluesy" → "blues", etc.)
        for key in genres:
            if key in style or style in key:
                genre = genres[key]
                style = key
                break
        # Also try stripping common suffixes
        if not genre:
            for suffix in ("y", "ish", "-like", " style", "ey"):
                stripped = style.rstrip(suffix).rstrip()
                if stripped in genres:
                    genre = genres[stripped]
                    style = stripped
                    break
    if not genre:
        # Fallback to classical if unknown style
        genre = genres.get("classical")
        if not genre:
            return ""
        style = "classical"
        logger.warning("Unknown genre '%s' — falling back to classical", style)

    lines: list[str] = []
    lines.append(f"GENRE ({style.upper()}):")
    lines.append(f"  Form: {genre['form']}")

    lines.append("  Structure:")
    for s in genre["structure"]:
        lines.append(f"    {s}")

    lines.append(f"  Treble rhythm: {genre['rhythm']['treble']}")
    lines.append(f"  Bass rhythm: {genre['rhythm']['bass']}")
    lines.append(f"  Voicing: {genre['voicing']}")
    lines.append(f"  Melody style: {genre['melody_style']}")

    # Example measures — compact JSON
    if "example_bars" in genre:
        lines.append("  EXAMPLE BARS (imitate this style):")
        for i, bar in enumerate(genre["example_bars"], 1):
            compact = json.dumps(bar, separators=(",", ":"))
            lines.append(f"    Bar {i}: {compact}")

    return "\n".join(lines)
