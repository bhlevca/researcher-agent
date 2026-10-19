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
    """Render takes of a song with ACE-Step, from a description and lyrics.

    Text to audio only. The Song tab used to offer to perform a score, and to
    write one; both produced scores unrelated to what ACE-Step generated, which
    was worse than none. Notation lives in the Score tab.

    Musical control is mostly in *prompt* (genre, instruments, mood,
    production, voice type) and *lyrics*, whose section tags — [Verse],
    [Chorus], [Bridge], [Instrumental], [Outro], with descriptors such as
    [Chorus - anthemic] — set the song's structure.
    """

    session_id: str
    prompt: str = ""                # genre, instruments, mood, voice, production
    lyrics: str = ""                # with section tags; blank for instrumental
    instrumental: bool = False      # no vocals, whatever the lyrics say
    vocal_language: str = "auto"    # "auto" reads it from the lyrics
    duration: float = 0.0           # seconds; 0 lets the model choose
    takes: int = 2                  # 1-4; two cost the same as one
    seed: int | None = None         # set to hear the same song with one change
    model: str = ""                 # "" = Fast; validated against an allowlist
    bpm: int | None = None          # overrides the session's tempo
    key_scale: str = ""             # overrides the session's key
    time_signature: str = ""        # overrides the session's time signature


class SongSaveRequest(BaseModel):
    """Keep one take in the library, under a name."""

    session_id: str
    file: str                       # the take's path, e.g. "scratch/song-….mp3"
    title: str = ""
    prompt: str = ""
    seed: int | None = None


class CompositionRename(BaseModel):
    title: str
