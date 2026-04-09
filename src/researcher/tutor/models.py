"""Pydantic models for the Language Tutor feature."""

from pydantic import BaseModel


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
    mode: str = "conversation"  # conversation | lesson | quiz


class LessonPlanRequest(BaseModel):
    """Request a new lesson plan."""
    session_id: str
    topic: str                  # e.g. "ordering food", "past tense", "travel"
    lesson_type: str = "mixed"  # grammar | vocabulary | situation | mixed


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
