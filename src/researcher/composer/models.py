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


class ComposeSongRequest(BaseModel):
    """Render a song as audio with ACE-Step.

    Two modes, chosen by whether *response* carries a score:

    - **cover** — the composer's own score is rendered to audio through
      fluidsynth and handed to ACE-Step as source audio, which keeps the
      structure and changes the production. This is the point of the feature:
      the MusicXML stays the thing that is correct, and the audio is a
      performance of it.
    - **text2music** — no score, just *prompt*. A parallel generator rather
      than a renderer, which is a weaker fit but what people expect from a
      prompt box.
    """

    session_id: str
    prompt: str = ""            # style / production description
    lyrics: str = ""            # blank for an instrumental
    instrumental: bool = False  # force no vocals even with lyrics present
    response: str = ""          # a score-bearing composer reply, for cover mode
    composition_id: str = ""    # or a saved composition, whose MusicXML is exact
    duration: float = 0.0       # seconds; 0 lets the model choose
    cover_strength: float = 0.75   # 1.0 holds the score, 0.2 is loose
    vocal_language: str = "en"
