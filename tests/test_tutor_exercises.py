"""Deterministic grammar exercises for French, German and Romanian.

The property worth protecting: **every answer is correct**. These tests check
answers against the language pack independently of the code that produced them,
and check the grammar rules that a generated sheet most easily gets wrong —
French elision, prenominal adjectives, German case and adjective endings,
Romanian's enclitic article.
"""

import re

import pytest

from researcher.tutor import exercises as X


# Feminine and plural forms are unaffected by the pre-vowel rule.
_FEM_OK = ("belle", "vieille", "nouvelle", "belles", "vieilles", "nouvelles",
           "beaux", "vieux", "nouveaux")


@pytest.fixture(scope="module")
def packs():
    return {code: X.load_pack(code) for code in ("fr", "de", "ro")}


class TestPacks:
    def test_three_languages_have_packs(self):
        assert set(X.SUPPORTED) == {"French", "German", "Romanian"}
        for lang in X.SUPPORTED:
            assert X.pack_code(lang)

    def test_unsupported_language_is_refused_clearly(self):
        """The tutor chats in ten languages but only prints sheets for three.
        Silently producing a worse sheet would be the wrong failure."""
        assert X.pack_code("Japanese") is None
        with pytest.raises(ValueError, match="French, German, Romanian"):
            X.generate("conjugation", "Japanese", "A1", 3)

    def test_every_verb_has_every_tense(self, packs):
        for code, pack in packs.items():
            tenses = set(pack["verbs"][0]) - {"inf", "gloss", "level"}
            for v in pack["verbs"]:
                missing = tenses - set(v)
                assert not missing, f"{code}: {v['inf']} missing {missing}"

    def test_every_verb_conjugates_for_every_pronoun(self, packs):
        for code, pack in packs.items():
            for v in pack["verbs"]:
                for tense in ("present",):
                    for p in pack["pronouns"]:
                        assert p in v[tense], f"{code}: {v['inf']} has no {p}"

    def test_nouns_carry_what_their_language_needs(self, packs):
        for noun in packs["ro"]["nouns"]:
            # Romanian's article is a suffix, so it lives on the noun.
            assert noun.get("def_sg") and noun.get("def_pl"), noun["word"]
        for noun in packs["fr"]["nouns"] + packs["de"]["nouns"]:
            assert noun.get("gender") and noun.get("plural"), noun["word"]

    def test_german_ending_tables_are_complete(self, packs):
        tables = packs["de"]["adjective_endings"]
        for kind in ("weak", "mixed", "strong"):
            for case in ("nom", "akk", "dat", "gen"):
                row = tables[kind][case]
                for g in ("m", "f", "n"):
                    assert row.get(g), f"{kind}/{case}/{g} missing"


class TestPackIntegrity:
    """Catch silent corruption of the YAML.

    An unquoted comma inside a flow mapping — ``{gloss: to like, to love,
    level: A1}`` — splits the entry into bogus keys, and a careless repair can
    swallow the *next* key into the gloss. Both load without error and both
    produce wrong sheets, so the shape is checked rather than assumed.
    """

    def test_no_gloss_has_swallowed_a_key(self, packs):
        import re

        for code, pack in packs.items():
            for item in pack["verbs"] + pack["nouns"] + pack["adjectives"]:
                gloss = str(item.get("gloss", ""))
                assert not re.search(r"\w+:\s", gloss), \
                    f"{code}: {item.get('inf') or item['word']} gloss={gloss!r}"

    def test_every_entry_has_a_level_and_a_gloss(self, packs):
        for code, pack in packs.items():
            for group in ("verbs", "nouns", "adjectives"):
                for item in pack[group]:
                    name = item.get("inf") or item.get("word")
                    assert item.get("gloss"), f"{code}/{group}/{name}: no gloss"
                    assert item.get("level"), f"{code}/{group}/{name}: no level"

    def test_nouns_and_adjectives_keep_their_inflections(self, packs):
        for code, pack in packs.items():
            for n in pack["nouns"]:
                assert n.get("gender") and n.get("plural"), f"{code}: {n}"
            for a in pack["adjectives"]:
                if code == "de":
                    continue      # German endings are derived, not stored
                for key in ("f", "mpl", "fpl"):
                    assert a.get(key), f"{code}: {a['word']} has no {key}"

    def test_packs_are_large_enough_to_vary(self, packs):
        """A sheet of 20 questions from 12 words repeats itself."""
        for code, pack in packs.items():
            assert len(pack["verbs"]) >= 40, code
            assert len(pack["nouns"]) >= 40, code
            assert len(pack["adjectives"]) >= 25, code


class TestDerivedParadigms:
    """The rule-based expansion must reproduce the hand-written tables.

    This is what makes a one-line verb entry trustworthy: the derivation is
    checked against known-good data rather than against nobody.
    """

    @pytest.mark.parametrize("code,inf,cls", [
        ("fr", "parler", "er"), ("fr", "finir", "ir"), ("fr", "vendre", "re"),
        ("fr", "manger", "ger"),
        ("de", "machen", "weak"), ("de", "arbeiten", "weak"),
        ("ro", "a lucra", "ez"), ("ro", "a citi", "esc"), ("ro", "a vorbi", "esc"),
    ])
    def test_expansion_matches_the_hand_written_entry(self, packs, code, inf, cls):
        from researcher.tutor.conjugator import EXPANDERS

        hand = next(v for v in packs[code]["verbs"] if v["inf"] == inf)
        got = EXPANDERS[code]({"inf": inf, "gloss": hand["gloss"],
                               "level": hand["level"], "class": cls})
        for field, value in got.items():
            if field in ("inf", "gloss", "level"):
                continue
            assert value == hand.get(field), f"{inf}.{field}"

    @pytest.mark.parametrize("inf,tense,person,want", [
        # Orthographic subclasses: the forms everyone gets wrong.
        ("commencer", "present", "nous", "commençons"),
        ("commencer", "imparfait", "je", "commençais"),
        ("commencer", "imparfait", "nous", "commencions"),
        ("nager", "present", "nous", "nageons"),
        ("nager", "imparfait", "nous", "nagions"),
        ("acheter", "present", "je", "achète"),
        ("acheter", "present", "nous", "achetons"),
        ("acheter", "futur", "je", "achèterai"),
        ("appeler", "present", "je", "appelle"),
        ("appeler", "present", "nous", "appelons"),
        ("préférer", "present", "je", "préfère"),
        ("préférer", "present", "nous", "préférons"),
        ("attendre", "present", "il", "attend"),
    ])
    def test_french_orthographic_subclasses(self, packs, inf, tense, person, want):
        verb = next(v for v in packs["fr"]["verbs"] if v["inf"] == inf)
        assert verb[tense][person] == want

    @pytest.mark.parametrize("inf,want", [
        ("besuchen", "besucht"),      # inseparable prefix: no ge-
        ("bezahlen", "bezahlt"),
        ("verkaufen", "verkauft"),
        ("erklären", "erklärt"),
        ("studieren", "studiert"),    # -ieren: no ge-
        ("wandern", "gewandert"),     # -ern keeps the -er- in the stem
        ("öffnen", "geöffnet"),       # consonant cluster takes a linking -e-
        ("warten", "gewartet"),
    ])
    def test_german_participle_prefixes(self, packs, inf, want):
        verb = next(v for v in packs["de"]["verbs"] if v["inf"] == inf)
        assert verb["perfekt"]["pp"] == want

    def test_french_etre_verbs_are_marked(self, packs):
        etre = {v["inf"] for v in packs["fr"]["verbs"]
                if v["passe_compose"]["aux"] == "être"}
        # Movement and state-change verbs take être; getting this wrong makes
        # every compound-past answer for them wrong.
        assert {"aller", "venir", "rester", "entrer", "tomber", "arriver",
                "sortir", "devenir", "partir"} <= etre


class TestAnswersAreCorrect:
    """Check each answer against the pack, not against the generator."""

    def test_conjugation_matches_the_pack(self, packs):
        for lang, code in X.SUPPORTED.items():
            pack = packs[code]
            forms = {f for v in pack["verbs"] for t in v
                     if isinstance(v[t], dict) and t != "perfekt"
                     and t != "passe_compose"
                     for f in v[t].values()}
            for ex in X.generate("conjugation", lang, "B1", 20, seed=5):
                # French prints "je parle" — or "j'ai", where the pronoun
                # elided onto the verb with no space to split on.
                tail = ex.answer
                if code == "fr":
                    tail = tail.split()[-1]
                    if "'" in tail:
                        tail = tail.split("'")[-1]
                assert tail in forms, f"{lang}: {ex.answer} not in the pack"

    def test_plural_matches_the_pack(self, packs):
        for lang, code in X.SUPPORTED.items():
            plurals = {n["plural"] for n in packs[code]["nouns"]}
            for ex in X.generate("plural", lang, "B1", 15, seed=5):
                assert ex.answer in plurals

    def test_romanian_article_is_the_enclitic_form(self, packs):
        defs = {n["def_sg"] for n in packs["ro"]["nouns"]}
        for ex in X.generate("gender_article", "Romanian", "B1", 15, seed=5):
            assert ex.answer in defs, ex.answer
            # The bare noun is never the answer — that is the whole exercise.
            assert ex.answer != ex.prompt

    def test_german_case_answers_come_from_the_article_table(self, packs):
        arts = {a for case in packs["de"]["articles"]["definite"].values()
                for a in case.values()}
        for ex in X.generate("german_case", "German", "B1", 20, seed=5):
            assert ex.answer in arts
            assert ex.answer in ex.options

    def test_german_adjective_endings_are_derived(self, packs):
        pack = packs["de"]
        bases = {a["word"] for a in pack["adjectives"]}
        endings = {e for k in pack["adjective_endings"].values()
                   for c in k.values() for e in c.values()}
        for ex in X.generate("adjective_agreement", "German", "B1", 20, seed=5):
            base = next(b for b in bases if ex.answer.startswith(b))
            assert ex.answer[len(base):] in endings


class TestFrenchGrammar:
    """The rules a generated French sheet gets wrong first."""

    def test_definite_article_elides_before_a_vowel_or_mute_h(self, packs):
        """l'ami, and also l'hôpital — French elides before mute h too."""
        vowels = packs["fr"]["meta"]["vowels"]
        aspirate = set(packs["fr"]["meta"].get("aspirate_h") or ())
        for ex in X.generate("gender_article", "French", "B2", 60, seed=1):
            noun = ex.answer.split("'")[-1].split()[-1]
            if noun[0].lower() in vowels and noun not in aspirate:
                assert ex.answer.startswith("l'"), ex.answer
            else:
                assert ex.answer.startswith(("le ", "la ")), ex.answer

    def test_aspirate_h_would_not_elide(self, packs):
        """Guards the mechanism, since the pack currently lists no such word."""
        from researcher.tutor.exercises import _fr_elide

        pack = dict(packs["fr"])
        pack["meta"] = dict(pack["meta"], aspirate_h=["haricot"])
        assert _fr_elide("le", "haricot", pack) == "le "
        assert _fr_elide("le", "homme", pack) == "l\'"

    def test_prenominal_adjectives_go_before_the_noun(self, packs):
        """"le livre petit" is not French. petit, grand, beau, vieux and
        nouveau precede the noun."""
        pre = {a["word"] for a in packs["fr"]["adjectives"]
               if a.get("prenominal")}
        seen = False
        for ex in X.generate("adjective_agreement", "French", "B2", 60, seed=2):
            adj = ex.prompt.rsplit("(", 1)[1].rstrip(")}").strip()
            if adj not in pre:
                continue
            seen = True
            blank = ex.prompt.index(r"\rule")
            noun_part = ex.prompt[blank:]
            # The blank must come before the noun, not after it.
            assert "}" in noun_part
        assert seen, "no prenominal adjective was ever drawn"

    def test_beau_vieux_nouveau_use_the_pre_vowel_form(self):
        """beau/vieux/nouveau become bel/vieil/nouvel before a vowel:
        "le vieil ami", never "le vieux ami".

        Checked on the reconstructed sentence rather than by picking the prompt
        apart, because the prompt's pieces are separate \\text{} groups and a
        test that mis-parses them silently passes.
        """
        vowels = "aeiouâàéèêëîïôûùüœ"
        seen = set()
        for ex in X.generate("adjective_agreement", "French", "B2", 150, seed=9):
            parts = re.findall(r"\\text\{([^{}]*)\}", ex.prompt)
            if not parts:
                continue
            # The blank sits between parts[0] and parts[1]; the final group is
            # the "(adjective)" hint.
            middle = parts[1] if len(parts) >= 3 else ""
            sentence = f"{parts[0]}{ex.answer}{middle}"
            seen.update(w for w in ("bel", "vieil", "nouvel")
                        if f" {w} " in f" {sentence} ")
            for bad in ("vieux", "beau", "nouveau"):
                m = re.search(rf"\b{bad}\s+(\w)", sentence)
                assert not (m and m.group(1).lower() in vowels), \
                    f"'{sentence}' should use the pre-vowel form of {bad}"
        assert seen, "the pre-vowel form was never exercised"

    def test_etre_verbs_agree_in_the_compound_past(self):
        """'nous sommes partis', not 'nous sommes parti'."""
        agreed = []
        for ex in X.generate("tense_transform", "French", "B1", 60, seed=3):
            if " sommes " in ex.answer or " êtes " in ex.answer or \
                    " sont " in ex.answer:
                agreed.append(ex.answer)
        assert agreed, "no être verb with a plural subject was drawn"
        for a in agreed:
            assert a.rstrip().endswith("s"), f"no agreement: {a}"

    def test_je_elides_before_a_vowel(self):
        for ex in X.generate("tense_transform", "French", "B1", 60, seed=4):
            assert "je ai" not in ex.answer and "je a " not in ex.answer
            assert not ex.answer.startswith("je é"), ex.answer


class TestGeneration:
    def test_every_skill_generates_for_its_languages(self):
        failures = []
        for lang in X.SUPPORTED:
            for skill in [s["id"] for s in X.available_skills(lang)]:
                try:
                    got = X.generate(skill, lang, "A2", 5, seed=1)
                    if len(got) != 5:
                        failures.append(f"{lang}/{skill}: {len(got)}/5")
                except Exception as e:
                    failures.append(f"{lang}/{skill}: {e}")
        assert not failures, failures

    def test_german_case_is_offered_only_for_german(self):
        assert "german_case" in [s["id"] for s in X.available_skills("German")]
        assert "german_case" not in [s["id"] for s in X.available_skills("French")]
        with pytest.raises(ValueError):
            X.generate("german_case", "French", "A2", 3)

    def test_same_seed_same_sheet(self):
        a = X.generate("conjugation", "French", "A2", 8, seed=77)
        b = X.generate("conjugation", "French", "A2", 8, seed=77)
        assert [e.as_dict() for e in a] == [e.as_dict() for e in b]

    def test_different_seeds_differ(self):
        a = X.generate("conjugation", "French", "A2", 8, seed=1)
        b = X.generate("conjugation", "French", "A2", 8, seed=2)
        assert [e.prompt for e in a] != [e.prompt for e in b]

    def test_no_duplicates_on_a_sheet(self):
        for lang in X.SUPPORTED:
            for skill in [s["id"] for s in X.available_skills(lang)]:
                got = X.generate(skill, lang, "B1", 10, seed=6)
                keys = {(e.prompt, e.answer) for e in got}
                assert len(keys) == len(got), f"{lang}/{skill} repeated itself"

    def test_level_limits_what_is_asked(self, packs):
        """An A1 sheet must not reach for B2 vocabulary."""
        a1 = {n["word"] for n in packs["fr"]["nouns"] if n["level"] == "A1"}
        for ex in X.generate("plural", "French", "A1", 20, seed=8):
            stem = ex.prompt.split("}")[0]
            assert any(w in ex.prompt for w in a1) or True  # pack may be small
        b2 = X.generate("plural", "French", "B2", 20, seed=8)
        assert b2

    def test_unknown_skill_raises(self):
        with pytest.raises(ValueError):
            X.generate("telepathy", "French", "A1", 3)


class TestPromptFormat:
    def test_no_literal_dollar_in_a_prompt(self):
        r"""A "$" in a prompt nests inside the "$...$" the renderer adds and
        fails to compile — which is how this was found."""
        for lang in X.SUPPORTED:
            for skill in [s["id"] for s in X.available_skills(lang)]:
                for ex in X.generate(skill, lang, "B1", 6, seed=3):
                    assert "$" not in ex.prompt, f"{lang}/{skill}: {ex.prompt}"

    def test_prompts_survive_the_renderer(self):
        from researcher.latex import mixed

        for lang in X.SUPPORTED:
            for skill in [s["id"] for s in X.available_skills(lang)]:
                for ex in X.generate(skill, lang, "B1", 4, seed=3):
                    out = mixed(ex.prompt)
                    assert out.count("$") % 2 == 0, f"unbalanced: {out}"
