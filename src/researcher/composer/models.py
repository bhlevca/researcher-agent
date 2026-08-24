"""Pydantic models for the Music Composer feature."""

from pydantic import BaseModel


# --- Request models ---


class ComposerSessionCreate(BaseModel):
    """Create a new composer session."""
    name: str
    genre: str = "classical"        # classical, jazz, pop, film, folk, etc.
    key_signature: str = "C major"  # e.g. "C major", "A minor", "Bb major"
    time_signature: str = "4/4"     # e.g. "4/4", "3/4", "6/8"
    tempo: int = 120                # BPM


class ComposerSessionUpdate(BaseModel):
    """Update an existing composer session."""
    name: str
    messages: list


class ComposeChatRequest(BaseModel):
    """Send a message in a composer session (conversational mode)."""
    message: str
    session_id: str
    mode: str = "compose"  # compose | arrange | harmonize | analyze


class ComposeScoreRequest(BaseModel):
    """Request score generation."""
    session_id: str
    description: str           # natural language description of what to compose
    instruments: list[str] = ["Piano"]
    measures: int = 16
    style: str = ""            # additional style hints
    # Target length in minutes. Above ~0.5 the piece is ASSEMBLED from recurring
    # sections rather than asked for in one go: a single model call holds
    # together for 8-16 bars and then repeats or drifts, so 80 bars of
    # continuous request produces 80 bars of mush.
    minutes: float = 0.0


class ComposeHarmonizeRequest(BaseModel):
    """Request harmonization of a melody."""
    session_id: str
    melody: str                # melody description or notes
    style: str = "classical"   # harmonization style


class ComposeAnalyzeRequest(BaseModel):
    """Request analysis of a musical piece or passage."""
    session_id: str
    content: str               # musical content to analyze
