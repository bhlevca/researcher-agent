"""Request models for the Math Tutor API."""

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Generate (or regenerate) a sheet.

    *seed* is optional on the way in and always returned on the way out: a sheet
    cannot be reprinted or marked without it.
    """

    grade: str = "grade8"
    topics: list[str] = Field(default_factory=list)
    count: int = 10
    seed: int | None = None
    title: str = "Mathematics"
    student: str = ""
    # Spread the count across the chosen topics rather than generating *count*
    # of each, which is what a mixed-review sheet wants.
    mix: bool = True
    # "en", "fr", "ro", or a pair such as "fr+en" for a bilingual sheet.
    lang: str = "en"


class SaveRequest(GenerateRequest):
    notes: str = ""


class MarkRequest(BaseModel):
    """A marked sheet coming back. Positions are 1-based, as printed."""

    worksheet_id: str
    student: str = ""
    marks: list[dict] = Field(default_factory=list)
