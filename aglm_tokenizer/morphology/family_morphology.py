"""
Multilingual Morphology System by Language Family.
Implements Section 12 of Mandatory Specifications:
Provides family-specific morphological factorization rather than single English heuristics.
Families:
- Germanic (Compounding & Affixes)
- Romance (Clitics & Inflections)
- Slavic (Case endings & Verb prefixes)
- Semitic (Triliteral roots & Affixes/Harakat)
- Indo-Aryan (Sandhi, Vibhakti, Postpositions)
- Dravidian (Agglutinative suffix chains)
- Turkic (Vowel harmony & Agglutinative chains)
- Korean/Japanese (Particles, Postpositions, Auxiliary verbs)
"""

from typing import List, Tuple, Dict, Set, Optional
from aglm_tokenizer.corpus.language_registry import LanguageFamily, ScriptFamily


class FamilyMorphologyAnalyzer:
    """Analyzes and decomposes words into morphological components based on language family."""

    # Germanic Affixes
    GERMANIC_SUFFIXES = ["ing", "ed", "er", "est", "ly", "tion", "ness", "ment", "able", "ible", "ung", "keit", "heit", "schaft", "lich", "isch", "baar", "lijk", "heid"]
    GERMANIC_PREFIXES = ["un", "re", "in", "im", "dis", "en", "non", "ver", "ent", "be", "ge", "zer", "aus", "auf", "ein"]

    # Romance Affixes
    ROMANCE_SUFFIXES = ["mente", "ción", "tion", "ção", "zione", "idad", "idade", "ità", "able", "ible", "ando", "endo", "ieron", "amos", "emos", "aran", "ieron"]
    ROMANCE_CLITICS = ["le", "la", "les", "los", "me", "te", "se", "nos", "os", "lo", "gliel", "m'", "l'", "d'", "qu'"]

    # Slavic Affixes
    SLAVIC_PREFIXES = ["при", "пере", "под", "раз", "рас", "от", "до", "за", "на", "по", "про", "с", "из", "вз", "prz", "roz", "od", "do", "za", "na", "po", "prze"]
    SLAVIC_SUFFIXES = ["ость", "ение", "ание", "тель", "ник", "щик", "ович", "евна", "ский", "ный", "ский", "ość", "enie", "anie", "nik", "owy", "ski"]

    # Semitic Affixes (Arabic / Hebrew)
    SEMITIC_PREFIXES = ["ال", "و", "ف", "ب", "ك", "ل", "لل", "ي", "ت", "ن", "أ", "م", "ה", "ו", "ב", "כ", "ל", "מ", "ש"]
    SEMITIC_SUFFIXES = ["ين", "ون", "ات", "ان", "ية", "هم", "هن", "ها", "كم", "نا", "ني", "تك", "ים", "ות", "יה", "כם", "נו", "ני"]

    # Indo-Aryan (Hindi, Marathi, Bengali, etc.)
    INDO_ARYAN_POSTPOSITIONS = ["ने", "को", "से", "का", "के", "की", "में", "पर", "तक", "एर", "ते", "র", "তে", "কে", "थे", "तून", "चे", "च्या"]
    INDO_ARYAN_SUFFIXES = ["वाला", "वाली", "वाले", "दार", "कार", "ता", "पन", "वान", "शील", "কারী", "শীল", "વાળા", "ਵਾਲਾ"]

    # Dravidian Agglutinative Suffixes (Tamil, Telugu, Kannada, Malayalam)
    DRAVIDIAN_SUFFIXES = [
        "gal", "kal", "kku", "il", "oda", "ai", "um", "in", "odu", "aal",  # Tamil
        "lu", "ki", "ku", "lo", "tho", "ni", "nu", "gaa", "gari", "la", "undi",  # Telugu
        "galu", "ge", "ge", "alli", "inda", "annu", "ige", "ondige",  # Kannada
        "kal", "il", "kku", "odu", "nte", "e", "ude", "aayi"  # Malayalam
    ]

    # Turkic Suffixes (Turkish)
    TURKIC_SUFFIXES = ["lar", "ler", "den", "dan", "de", "da", "e", "a", "i", "ı", "u", "ü", "in", "ın", "un", "ün", "imiz", "ımız", "umuz", "ümüz", "lik", "lık", "luk", "lük", "ci", "cı", "cu", "cü"]

    # Korean / Japanese Particles
    KOREAN_PARTICLES = ["은", "는", "이", "가", "을", "를", "에", "에서", "에게", "의", "로", "으로", "과", "와", "도", "만", "까지", "부터"]
    JAPANESE_PARTICLES = ["は", "が", "を", "に", "で", "と", "へ", "から", "まで", "より", "も", "ね", "よ", "か", "の"]

    @classmethod
    def extract_morphological_candidates(cls, word: str, family: LanguageFamily) -> List[Tuple[str, str]]:
        """
        Extracts morphological factorizations (stem, affix) appropriate for the language family.
        Returns list of (unit, unit_type) candidates.
        """
        candidates: List[Tuple[str, str]] = []
        if len(word) <= 2:
            return [(word, "ROOT")]

        # Family-guided extraction
        if family == LanguageFamily.GERMANIC:
            for pref in cls.GERMANIC_PREFIXES:
                if word.lower().startswith(pref) and len(word) > len(pref) + 2:
                    candidates.append((pref, "PREFIX"))
                    candidates.append((word[len(pref):], "STEM"))
            for suff in cls.GERMANIC_SUFFIXES:
                if word.lower().endswith(suff) and len(word) > len(suff) + 2:
                    candidates.append((word[:-len(suff)], "STEM"))
                    candidates.append((suff, "SUFFIX"))

        elif family == LanguageFamily.ROMANCE:
            for clitic in cls.ROMANCE_CLITICS:
                if word.lower().startswith(clitic) and len(word) > len(clitic) + 2:
                    candidates.append((clitic, "CLITIC"))
                    candidates.append((word[len(clitic):], "STEM"))
            for suff in cls.ROMANCE_SUFFIXES:
                if word.lower().endswith(suff) and len(word) > len(suff) + 2:
                    candidates.append((word[:-len(suff)], "STEM"))
                    candidates.append((suff, "SUFFIX"))

        elif family == LanguageFamily.SLAVIC:
            for pref in cls.SLAVIC_PREFIXES:
                if word.lower().startswith(pref) and len(word) > len(pref) + 2:
                    candidates.append((pref, "PREFIX"))
                    candidates.append((word[len(pref):], "STEM"))
            for suff in cls.SLAVIC_SUFFIXES:
                if word.lower().endswith(suff) and len(word) > len(suff) + 2:
                    candidates.append((word[:-len(suff)], "STEM"))
                    candidates.append((suff, "SUFFIX"))

        elif family == LanguageFamily.SEMITIC:
            for pref in cls.SEMITIC_PREFIXES:
                if word.startswith(pref) and len(word) > len(pref) + 2:
                    candidates.append((pref, "PREFIX"))
                    candidates.append((word[len(pref):], "ROOT_PATTERN"))
            for suff in cls.SEMITIC_SUFFIXES:
                if word.endswith(suff) and len(word) > len(suff) + 2:
                    candidates.append((word[:-len(suff)], "ROOT_PATTERN"))
                    candidates.append((suff, "SUFFIX"))

        elif family == LanguageFamily.INDO_ARYAN:
            for suff in cls.INDO_ARYAN_SUFFIXES + cls.INDO_ARYAN_POSTPOSITIONS:
                if word.endswith(suff) and len(word) > len(suff) + 1:
                    candidates.append((word[:-len(suff)], "STEM"))
                    candidates.append((suff, "POSTPOSITION_SUFFIX"))

        elif family == LanguageFamily.DRAVIDIAN:
            for suff in cls.DRAVIDIAN_SUFFIXES:
                if word.endswith(suff) and len(word) > len(suff) + 2:
                    candidates.append((word[:-len(suff)], "STEM"))
                    candidates.append((suff, "AGGLUTINATIVE_SUFFIX"))

        elif family == LanguageFamily.TURKIC:
            for suff in cls.TURKIC_SUFFIXES:
                if word.lower().endswith(suff) and len(word) > len(suff) + 2:
                    candidates.append((word[:-len(suff)], "STEM"))
                    candidates.append((suff, "AGGLUTINATIVE_SUFFIX"))

        elif family == LanguageFamily.KOREANIC:
            for p in cls.KOREAN_PARTICLES:
                if word.endswith(p) and len(word) > len(p):
                    candidates.append((word[:-len(p)], "NOUN_STEM"))
                    candidates.append((p, "PARTICLE"))

        elif family == LanguageFamily.JAPONIC:
            for p in cls.JAPANESE_PARTICLES:
                if word.endswith(p) and len(word) > len(p):
                    candidates.append((word[:-len(p)], "STEM"))
                    candidates.append((p, "PARTICLE"))

        return candidates
