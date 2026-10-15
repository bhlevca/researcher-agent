"""Math Tutor: IB-level problem generation, Grade 1 to Grade 12.

Answers are computed with SymPy and figures are drawn with matplotlib, both
from the same parameters. No language model is involved in producing a question
or an answer — see generators.py for why.
"""

from researcher.mathtutor.generators import Problem, generate, available_topics

__all__ = ["Problem", "generate", "available_topics", "curriculum"]


def curriculum() -> dict:
    """The grade -> topics map, read from curriculum.yaml."""
    from pathlib import Path

    import yaml

    path = Path(__file__).parent / "curriculum.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))
