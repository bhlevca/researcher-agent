"""Pydantic models for the Language Tutor feature."""

from pydantic import BaseModel, Field


# --- Request models ---


class TutorSessionCreate(BaseModel):
    """Create a new tutor session."""
    name: str
    target_lang: str          # e.g. "French", "Spanish", "German"
    native_lang: str = "English"
    level: str = "A1"         # CEFR level: A1, A2, B1, B2, C1, C2


class TutorSessionUpdate(BaseModel):
    """Update an existing tutor session."""
    name: str
    messages: list


class TutorChatRequest(BaseModel):
    """Send a message in a tutor session."""
    message: str
    session_id: str
    mode: str = "conversation"  # conversation | scenario
    scenario_id: str | None = None  # required when mode == "scenario"
    save_transcript: bool = False   # auto-save scenario dialog to lesson_plans


class LessonPlanRequest(BaseModel):
    """Request a new lesson plan."""
    session_id: str
    topic: str                  # e.g. "ordering food", "past tense", "travel"
    lesson_type: str = "mixed"  # grammar | vocabulary | situation | mixed


class PassageRequest(BaseModel):
    """Request a levelled reading passage (comprehensible input)."""
    session_id: str
    topic: str = ""             # blank = use the session's current curriculum topic
    word_count: int = 180
    seed_due_vocab: bool = True  # weave in words that are due for review
    max_seed_words: int = 8


class QuizRequest(BaseModel):
    """Request a quiz."""
    session_id: str
    quiz_type: str = "mixed"    # vocabulary | grammar | translation | mixed
    num_questions: int = 10
    lesson_id: str | None = None  # optionally scope to a lesson


class VocabularyAddRequest(BaseModel):
    """Add a vocabulary entry manually."""
    session_id: str
    word: str
    translation: str
    context: str = ""
    phonetic: str = ""
    part_of_speech: str = ""
    target_lang: str = ""


class QuizSubmitRequest(BaseModel):
    """Submit quiz answers for grading."""
    session_id: str
    quiz_id: str
    answers: list[dict]  # [{"question_id": 0, "answer": "..."}, ...]


class AppraisalRequest(BaseModel):
    """Request an appraisal of student progress."""
    session_id: str


# ---------------------------------------------------------------------------
# Printable sheets
# ---------------------------------------------------------------------------


class SheetSection(BaseModel):
    """One part of a printable sheet."""

    skill: str
    count: int = 5
    marks_each: int = 1
    title: str = ""
    instruction: str = ""


class SheetRequest(BaseModel):
    """Generate (or regenerate) a printable sheet.

    *seed* is optional on the way in and always returned: a sheet cannot be
    reprinted or marked without it.
    """

    language: str = "French"
    level: str = "A1"
    kind: str = "homework"              # homework | test | exam | challenge
    title: str = ""
    student: str = ""
    seed: int | None = None
    sections: list[SheetSection] = Field(default_factory=list)
    native: str = "English"


class SheetSaveRequest(SheetRequest):
    notes: str = ""


class StoryRequest(BaseModel):
    """Ask the model for a story with comprehension and summary tasks."""

    language: str = "French"
    level: str = "A1"
    topic: str = ""
    num_questions: int = 5
    word_count: int = 200
    summary_words: int = 40


class StorySheetRequest(BaseModel):
    """Print a story that has already been generated.

    The content is passed back in rather than regenerated: a second model call
    writes a different story, so regenerating would print a sheet that does not
    match the one the child was given.
    """

    content: dict
    language: str = "French"
    level: str = "A1"
    kind: str = "homework"
    title: str = ""
    student: str = ""
    notes: str = ""
    native: str = "English"


class LessonSheetRequest(BaseModel):
    """Print a Markdown lesson as a sheet or as its answer key."""

    markdown: str
    title: str = ""
    language: str = ""
    level: str = ""
    student: str = ""
    # The words belonging to the page — Name, Date, Answer Key — follow the
    # student's NATIVE language, not the one being taught.
    native: str = "English"
    # Leave writing space after each numbered exercise.
    practice_space: bool = True


class LessonRequest(BaseModel):
    """Ask the model for a lesson.

    *full* selects the teacher's scheme of work — thorough, and about four
    minutes to write. The default is the compact version sized for one printed
    sheet, which is what someone waiting at a Print button wants.
    """

    language: str = "French"
    level: str = "A1"
    topic: str = ""
    full: bool = False
