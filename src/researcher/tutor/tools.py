"""Tutor-specific CrewAI tools.

These tools are available to the language tutor agent during task execution.
They provide grammar checking, vocabulary lookup, and pronunciation guidance.
"""

import json
import logging

from crewai.tools import tool

logger = logging.getLogger(__name__)


@tool("GrammarCheck")
def grammar_check(text: str) -> str:
    """Check a sentence in the target language for grammar errors.

    Input: a sentence to check.
    Returns a brief analysis of any grammar errors found, with corrections.
    If the sentence is correct, says so.

    NOTE: This tool uses the LLM's own knowledge — it structures the request
    so the agent can reason about grammar systematically.
    """
    return (
        f"Please analyze this sentence for grammar errors:\n\n"
        f'"{text}"\n\n'
        "List each error (if any), the correction, and a brief rule explanation. "
        "If the sentence is grammatically correct, state that."
    )


@tool("VocabularyLookup")
def vocabulary_lookup(word: str) -> str:
    """Look up a word and provide its translation, part of speech, conjugation
    (if a verb), common usage examples, and pronunciation guide.

    Input: a word or short phrase in any language.
    Returns detailed vocabulary information.
    """
    return (
        f"Provide a detailed vocabulary entry for: {word}\n\n"
        "Include:\n"
        "- Translation (both directions if relevant)\n"
        "- Part of speech\n"
        "- Gender (if applicable)\n"
        "- Plural form (if noun)\n"
        "- Conjugation summary (if verb — present, past, future)\n"
        "- 3 example sentences with translations\n"
        "- Common collocations or phrases\n"
        "- IPA pronunciation\n"
        "- Any irregular forms or exceptions"
    )


@tool("PronunciationGuide")
def pronunciation_guide(text: str) -> str:
    """Provide pronunciation guidance for a word or phrase.

    Input: a word or phrase in the target language.
    Returns IPA transcription and native-language sound approximations.
    """
    return (
        f"Provide pronunciation guidance for: {text}\n\n"
        "Include:\n"
        "- IPA transcription\n"
        "- Syllable breakdown with stress markers\n"
        "- Sound approximations using English words\n"
        "- Common pronunciation mistakes to avoid\n"
        "- Tips for difficult sounds"
    )


@tool("ConjugationTable")
def conjugation_table(verb: str) -> str:
    """Generate a conjugation table for a verb.

    Input: a verb in its infinitive form.
    Returns a markdown table with conjugations in major tenses.
    """
    return (
        f"Generate a complete conjugation table for the verb: {verb}\n\n"
        "Include these tenses (as applicable to the language):\n"
        "- Present indicative\n"
        "- Past (preterite / passé composé / Präteritum)\n"
        "- Imperfect\n"
        "- Future\n"
        "- Conditional\n"
        "- Subjunctive (present)\n\n"
        "Format as a markdown table with pronouns in the first column.\n"
        "Note any irregular forms with an asterisk (*)."
    )


# All tools exported for use by the TutorCrew
TUTOR_TOOLS = [
    grammar_check,
    vocabulary_lookup,
    pronunciation_guide,
    conjugation_table,
]
