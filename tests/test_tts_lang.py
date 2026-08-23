"""Tests for language-aware TTS voice switching."""

import pytest
from researcher.tts import _detect_lang, _pick_voice_for_lang, _split_multilingual


# Sample voice list mimicking edge-tts structure
SAMPLE_VOICES = [
    {"name": "en-US-AriaNeural", "gender": "Female", "locale": "en-US"},
    {"name": "en-US-GuyNeural", "gender": "Male", "locale": "en-US"},
    {"name": "en-GB-SoniaNeural", "gender": "Female", "locale": "en-GB"},
    {"name": "fr-FR-DeniseNeural", "gender": "Female", "locale": "fr-FR"},
    {"name": "fr-FR-HenriNeural", "gender": "Male", "locale": "fr-FR"},
    {"name": "de-DE-KatjaNeural", "gender": "Female", "locale": "de-DE"},
    {"name": "de-DE-ConradNeural", "gender": "Male", "locale": "de-DE"},
    {"name": "es-ES-ElviraNeural", "gender": "Female", "locale": "es-ES"},
    {"name": "es-ES-AlvaroNeural", "gender": "Male", "locale": "es-ES"},
    {"name": "pt-BR-FranciscaNeural", "gender": "Female", "locale": "pt-BR"},
    {"name": "ro-RO-AlinaNeural", "gender": "Female", "locale": "ro-RO"},
    {"name": "ro-RO-EmilNeural", "gender": "Male", "locale": "ro-RO"},
]


class TestDetectLang:
    def test_english(self):
        assert _detect_lang("Hello, how are you today?") == "en"

    def test_french(self):
        assert _detect_lang("Bonjour, comment allez-vous aujourd'hui?") == "fr"

    def test_german(self):
        assert _detect_lang("Guten Tag, wie geht es Ihnen?") == "de"

    def test_spanish(self):
        assert _detect_lang("Buenos días, cómo estás hoy?") == "es"

    def test_short_text_fallback(self):
        # Very short text may not detect reliably, but shouldn't crash
        result = _detect_lang("Hi")
        assert isinstance(result, str)

    def test_empty_defaults_to_en(self):
        assert _detect_lang("") == "en"


class TestPickVoiceForLang:
    def test_keeps_base_voice_for_matching_lang(self):
        result = _pick_voice_for_lang("en", "en-US-AriaNeural", SAMPLE_VOICES)
        assert result == "en-US-AriaNeural"

    def test_switches_to_french_female(self):
        result = _pick_voice_for_lang("fr", "en-US-AriaNeural", SAMPLE_VOICES)
        assert result == "fr-FR-DeniseNeural"  # Female, matching gender

    def test_switches_to_french_male(self):
        result = _pick_voice_for_lang("fr", "en-US-GuyNeural", SAMPLE_VOICES)
        assert result == "fr-FR-HenriNeural"  # Male, matching gender

    def test_switches_to_german_female(self):
        result = _pick_voice_for_lang("de", "en-US-AriaNeural", SAMPLE_VOICES)
        assert result == "de-DE-KatjaNeural"

    def test_switches_to_german_male(self):
        result = _pick_voice_for_lang("de", "en-US-GuyNeural", SAMPLE_VOICES)
        assert result == "de-DE-ConradNeural"

    def test_unknown_lang_falls_back_to_base(self):
        result = _pick_voice_for_lang("xx", "en-US-AriaNeural", SAMPLE_VOICES)
        assert result == "en-US-AriaNeural"

    def test_empty_voice_list_falls_back(self):
        result = _pick_voice_for_lang("fr", "en-US-AriaNeural", [])
        assert result == "en-US-AriaNeural"

    def test_romanian(self):
        result = _pick_voice_for_lang("ro", "en-US-AriaNeural", SAMPLE_VOICES)
        assert result == "ro-RO-AlinaNeural"


class TestSplitMultilingual:
    def test_pure_english(self):
        segs = _split_multilingual("Hello, this is a nice day.", "en")
        assert len(segs) == 1
        assert segs[0][0] == "en"

    def test_quoted_french_extracted(self):
        text = 'In French you say "Bonjour, comment allez-vous?" to greet someone.'
        segs = _split_multilingual(text, "en")
        langs = [s[0] for s in segs]
        # French must be detected (whole text or quoted phrase)
        assert "fr" in langs

    def test_quoted_german_extracted(self):
        text = 'The phrase "Guten Tag, wie geht es Ihnen?" means hello.'
        segs = _split_multilingual(text, "en")
        langs = [s[0] for s in segs]
        assert "de" in langs

    def test_asterisk_emphasis_french(self):
        text = "Try saying *Je voudrais un café, s'il vous plaît* for ordering."
        segs = _split_multilingual(text, "en")
        langs = [s[0] for s in segs]
        assert "fr" in langs

    def test_double_asterisk_german(self):
        text = "Say **Wo ist der Bahnhof bitte?** to find the station."
        segs = _split_multilingual(text, "en")
        langs = [s[0] for s in segs]
        assert "de" in langs

    def test_no_duplicate_segments(self):
        text = 'Say "Bonjour, comment allez-vous?" when greeting.'
        segs = _split_multilingual(text, "en")
        # No segment text should appear twice
        all_text = " ".join(s[1] for s in segs)
        assert all_text.count("Bonjour") == 1

    def test_multiple_quoted_phrases(self):
        text = 'Say "Bonjour tout le monde" and then "Au revoir mes amis" when leaving.'
        segs = _split_multilingual(text, "en")
        fr_segs = [s for s in segs if s[0] == "fr"]
        assert len(fr_segs) >= 1  # at least one French segment detected

    def test_guillemets_french(self):
        text = 'Répétez: «Je ne sais pas du tout» means I do not know.'
        segs = _split_multilingual(text, "en")
        langs = [s[0] for s in segs]
        assert "fr" in langs


# ---------------------------------------------------------------------------
# Declared language overrides detection (T-D)
# ---------------------------------------------------------------------------


class TestDeclaredLanguage:
    """The tutor knows its target language, so TTS must not have to guess.

    Regression: short, punctuation-heavy opening lines like
    "Bonjour ! Ça va très bien, merci ! Et toi ?" split into 2-4 word fragments,
    none long enough to beat the base-language bias. With an English voice
    selected that resolved to English, so the FIRST tutor reply was spoken with
    an English accent while later, longer replies were fine.
    """

    @pytest.mark.parametrize("value,expected", [
        ("French", "fr"), ("french", "fr"), ("fr", "fr"), ("fr-FR", "fr"),
        ("Romanian", "ro"), ("Chinese", "zh"), ("Japanese", "ja"),
        ("", ""), ("  ", ""),
    ])
    def test_lang_code_accepts_names_and_codes(self, value, expected):
        from researcher.tts import _lang_code
        assert _lang_code(value) == expected

    def test_short_french_opener_is_misdetected_without_a_declared_language(self):
        """Pins the actual bug: this is why the fix is needed at all."""
        from researcher.tts import _split_multilingual
        text = "Bonjour ! Ça va très bien, merci ! Et toi ?"
        langs = {lang for lang, _ in _split_multilingual(text, "en")}
        assert "fr" not in langs, (
            "If this now detects French unaided, langdetect changed and the "
            "declared-language path is no longer load-bearing for this case."
        )

    def test_declaring_french_fixes_the_same_text(self):
        from researcher.tts import _split_multilingual, _lang_code
        text = "Bonjour ! Ça va très bien, merci ! Et toi ?"
        langs = {lang for lang, _ in _split_multilingual(text, _lang_code("French"))}
        assert "fr" in langs and "en" not in langs

    def test_request_model_carries_lang_and_defaults_blank(self):
        from researcher.tts import TTSRequest
        assert TTSRequest(text="hi").lang == ""
        assert TTSRequest(text="hi", lang="French").lang == "French"
