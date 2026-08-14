"""
Language and Script Registry for Universal Multilingual Tokenizer.
Covers 50+ languages across all major families, scripts, romanized variants, and code-switched combinations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set


class ScriptFamily(str, Enum):
    LATIN = "Latin"
    CYRILLIC = "Cyrillic"
    ARABIC = "Arabic"
    DEVANAGARI = "Devanagari"
    BENGALI = "Bengali"
    GUJARATI = "Gujarati"
    GURMUKHI = "Gurmukhi"
    ODIA = "Odia"
    TAMIL = "Tamil"
    TELUGU = "Telugu"
    KANNADA = "Kannada"
    MALAYALAM = "Malayalam"
    HAN_SIMPLIFIED = "Han (Simplified)"
    HAN_TRADITIONAL = "Han (Traditional)"
    JAPANESE = "Japanese (Kanji+Kana)"
    HANGUL = "Hangul"
    THAI = "Thai"
    MYANMAR = "Myanmar"
    KHMER = "Khmer"
    LAO = "Lao"
    HEBREW = "Hebrew"
    GREEK = "Greek"
    ARMENIAN = "Armenian"
    GEORGIAN = "Georgian"
    ETHIOPIC = "Ethiopic (Ge'ez)"
    NATIVE_AFRICAN_LATIN = "African (Latin)"
    ROMANIZED = "Romanized/Transliterated"
    CODE_SWITCHED = "Code-Switched"


class LanguageFamily(str, Enum):
    GERMANIC = "Germanic"
    ROMANCE = "Romance"
    SLAVIC = "Slavic"
    SEMITIC = "Semitic"
    INDO_ARYAN = "Indo-Aryan"
    DRAVIDIAN = "Dravidian"
    SINO_TIBETAN = "Sino-Tibetan"
    JAPONIC = "Japonic"
    KOREANIC = "Koreanic"
    AUSTROASIATIC = "Austroasiatic"
    AUSTRONESIAN = "Austronesian"
    KRA_DAI = "Kra-Dai"
    TURKIC = "Turkic"
    NIGER_CONGO = "Niger-Congo"
    AFROASIATIC = "Afroasiatic"
    HELLENIC = "Hellenic"
    ARMENIAN = "Armenian"
    KARTVELIAN = "Kartvelian"
    MIXED_CODE_SWITCH = "Code-Switched"


@dataclass(frozen=True)
class LanguageSpec:
    code: str
    name: str
    script: ScriptFamily
    family: LanguageFamily
    is_romanized: bool = False
    is_code_switched: bool = False
    native_name: Optional[str] = None
    sample_texts: List[str] = field(default_factory=list)
    has_spaces: bool = True
    agglutinative: bool = False


# Comprehensive Registry of 50+ Languages meeting Section 1-3 Requirements
LANGUAGES: Dict[str, LanguageSpec] = {
    # 1. LATIN-SCRIPT
    "en": LanguageSpec("en", "English", ScriptFamily.LATIN, LanguageFamily.GERMANIC, native_name="English"),
    "es": LanguageSpec("es", "Spanish", ScriptFamily.LATIN, LanguageFamily.ROMANCE, native_name="Español"),
    "fr": LanguageSpec("fr", "French", ScriptFamily.LATIN, LanguageFamily.ROMANCE, native_name="Français"),
    "de": LanguageSpec("de", "German", ScriptFamily.LATIN, LanguageFamily.GERMANIC, native_name="Deutsch"),
    "pt": LanguageSpec("pt", "Portuguese", ScriptFamily.LATIN, LanguageFamily.ROMANCE, native_name="Português"),
    "it": LanguageSpec("it", "Italian", ScriptFamily.LATIN, LanguageFamily.ROMANCE, native_name="Italiano"),
    "nl": LanguageSpec("nl", "Dutch", ScriptFamily.LATIN, LanguageFamily.GERMANIC, native_name="Nederlands"),
    "pl": LanguageSpec("pl", "Polish", ScriptFamily.LATIN, LanguageFamily.SLAVIC, native_name="Polski"),
    "cs": LanguageSpec("cs", "Czech", ScriptFamily.LATIN, LanguageFamily.SLAVIC, native_name="Čeština"),
    "ro": LanguageSpec("ro", "Romanian", ScriptFamily.LATIN, LanguageFamily.ROMANCE, native_name="Română"),
    "tr": LanguageSpec("tr", "Turkish", ScriptFamily.LATIN, LanguageFamily.TURKIC, native_name="Türkçe", agglutinative=True),
    "vi": LanguageSpec("vi", "Vietnamese", ScriptFamily.LATIN, LanguageFamily.AUSTROASIATIC, native_name="Tiếng Việt"),
    "id": LanguageSpec("id", "Indonesian/Malay", ScriptFamily.LATIN, LanguageFamily.AUSTRONESIAN, native_name="Bahasa Indonesia"),
    "sw": LanguageSpec("sw", "Swahili", ScriptFamily.LATIN, LanguageFamily.NIGER_CONGO, native_name="Kiswahili", agglutinative=True),

    # 2. CYRILLIC
    "ru": LanguageSpec("ru", "Russian", ScriptFamily.CYRILLIC, LanguageFamily.SLAVIC, native_name="Русский"),
    "uk": LanguageSpec("uk", "Ukrainian", ScriptFamily.CYRILLIC, LanguageFamily.SLAVIC, native_name="Українська"),
    "bg": LanguageSpec("bg", "Bulgarian", ScriptFamily.CYRILLIC, LanguageFamily.SLAVIC, native_name="Български"),
    "sr": LanguageSpec("sr", "Serbian", ScriptFamily.CYRILLIC, LanguageFamily.SLAVIC, native_name="Српски"),

    # 3. ARABIC-SCRIPT
    "ar": LanguageSpec("ar", "Arabic", ScriptFamily.ARABIC, LanguageFamily.SEMITIC, native_name="العربية"),
    "fa": LanguageSpec("fa", "Persian/Farsi", ScriptFamily.ARABIC, LanguageFamily.INDO_ARYAN, native_name="فارسی"),
    "ur": LanguageSpec("ur", "Urdu", ScriptFamily.ARABIC, LanguageFamily.INDO_ARYAN, native_name="اردو"),

    # 4. INDIC (Indo-Aryan)
    "hi": LanguageSpec("hi", "Hindi", ScriptFamily.DEVANAGARI, LanguageFamily.INDO_ARYAN, native_name="हिन्दी"),
    "mr": LanguageSpec("mr", "Marathi", ScriptFamily.DEVANAGARI, LanguageFamily.INDO_ARYAN, native_name="मराठी"),
    "bn": LanguageSpec("bn", "Bengali", ScriptFamily.BENGALI, LanguageFamily.INDO_ARYAN, native_name="বাংলা"),
    "gu": LanguageSpec("gu", "Gujarati", ScriptFamily.GUJARATI, LanguageFamily.INDO_ARYAN, native_name="ગુજરાતી"),
    "pa": LanguageSpec("pa", "Punjabi", ScriptFamily.GURMUKHI, LanguageFamily.INDO_ARYAN, native_name="ਪੰਜਾਬੀ"),
    "or": LanguageSpec("or", "Odia", ScriptFamily.ODIA, LanguageFamily.INDO_ARYAN, native_name="ଓଡ଼ିଆ"),
    "as": LanguageSpec("as", "Assamese", ScriptFamily.BENGALI, LanguageFamily.INDO_ARYAN, native_name="অসমীয়া"),
    "ne": LanguageSpec("ne", "Nepali", ScriptFamily.DEVANAGARI, LanguageFamily.INDO_ARYAN, native_name="नेपाली"),

    # 5. DRAVIDIAN
    "ta": LanguageSpec("ta", "Tamil", ScriptFamily.TAMIL, LanguageFamily.DRAVIDIAN, native_name="தமிழ்", agglutinative=True),
    "te": LanguageSpec("te", "Telugu", ScriptFamily.TELUGU, LanguageFamily.DRAVIDIAN, native_name="తెలుగు", agglutinative=True),
    "kn": LanguageSpec("kn", "Kannada", ScriptFamily.KANNADA, LanguageFamily.DRAVIDIAN, native_name="ಕನ್ನಡ", agglutinative=True),
    "ml": LanguageSpec("ml", "Malayalam", ScriptFamily.MALAYALAM, LanguageFamily.DRAVIDIAN, native_name="മലയാളം", agglutinative=True),

    # 6. EAST ASIAN
    "zh-Hans": LanguageSpec("zh-Hans", "Simplified Chinese", ScriptFamily.HAN_SIMPLIFIED, LanguageFamily.SINO_TIBETAN, native_name="简体中文", has_spaces=False),
    "zh-Hant": LanguageSpec("zh-Hant", "Traditional Chinese", ScriptFamily.HAN_TRADITIONAL, LanguageFamily.SINO_TIBETAN, native_name="繁體中文", has_spaces=False),
    "ja": LanguageSpec("ja", "Japanese", ScriptFamily.JAPANESE, LanguageFamily.JAPONIC, native_name="日本語", has_spaces=False, agglutinative=True),
    "ko": LanguageSpec("ko", "Korean", ScriptFamily.HANGUL, LanguageFamily.KOREANIC, native_name="한국어", agglutinative=True),

    # 7. SOUTHEAST ASIAN
    "th": LanguageSpec("th", "Thai", ScriptFamily.THAI, LanguageFamily.KRA_DAI, native_name="ไทย", has_spaces=False),
    "my": LanguageSpec("my", "Burmese/Myanmar", ScriptFamily.MYANMAR, LanguageFamily.SINO_TIBETAN, native_name="မြန်မာဘာသာ", has_spaces=False),
    "km": LanguageSpec("km", "Khmer", ScriptFamily.KHMER, LanguageFamily.AUSTROASIATIC, native_name="ភាសាខ្មែរ", has_spaces=False),
    "lo": LanguageSpec("lo", "Lao", ScriptFamily.LAO, LanguageFamily.KRA_DAI, native_name="ພາສາລາວ", has_spaces=False),
    "tl": LanguageSpec("tl", "Filipino/Tagalog", ScriptFamily.LATIN, LanguageFamily.AUSTRONESIAN, native_name="Tagalog"),

    # 8. OTHER IMPORTANT SCRIPTS / FAMILIES / AFRICAN
    "he": LanguageSpec("he", "Hebrew", ScriptFamily.HEBREW, LanguageFamily.SEMITIC, native_name="עברית"),
    "el": LanguageSpec("el", "Greek", ScriptFamily.GREEK, LanguageFamily.HELLENIC, native_name="Ελληνικά"),
    "hy": LanguageSpec("hy", "Armenian", ScriptFamily.ARMENIAN, LanguageFamily.ARMENIAN, native_name="Հայերեն"),
    "ka": LanguageSpec("ka", "Georgian", ScriptFamily.GEORGIAN, LanguageFamily.KARTVELIAN, native_name="ქართული"),
    "am": LanguageSpec("am", "Amharic", ScriptFamily.ETHIOPIC, LanguageFamily.SEMITIC, native_name="አማርኛ"),
    "yo": LanguageSpec("yo", "Yoruba", ScriptFamily.NATIVE_AFRICAN_LATIN, LanguageFamily.NIGER_CONGO, native_name="Èdè Yorùbá"),
    "ha": LanguageSpec("ha", "Hausa", ScriptFamily.NATIVE_AFRICAN_LATIN, LanguageFamily.AFROASIATIC, native_name="Harshen Hausa"),
    "zu": LanguageSpec("zu", "Zulu", ScriptFamily.NATIVE_AFRICAN_LATIN, LanguageFamily.NIGER_CONGO, native_name="isiZulu", agglutinative=True),
    "so": LanguageSpec("so", "Somali", ScriptFamily.NATIVE_AFRICAN_LATIN, LanguageFamily.AFROASIATIC, native_name="Soomaaliga"),

    # 9. ROMANIZED / TRANSLITERATED LANGUAGES (Section 2)
    "hi-Latn": LanguageSpec("hi-Latn", "Hinglish / Roman Hindi", ScriptFamily.ROMANIZED, LanguageFamily.INDO_ARYAN, is_romanized=True),
    "ur-Latn": LanguageSpec("ur-Latn", "Roman Urdu", ScriptFamily.ROMANIZED, LanguageFamily.INDO_ARYAN, is_romanized=True),
    "te-Latn": LanguageSpec("te-Latn", "Roman Telugu", ScriptFamily.ROMANIZED, LanguageFamily.DRAVIDIAN, is_romanized=True),
    "ta-Latn": LanguageSpec("ta-Latn", "Tanglish / Roman Tamil", ScriptFamily.ROMANIZED, LanguageFamily.DRAVIDIAN, is_romanized=True),
    "kn-Latn": LanguageSpec("kn-Latn", "Roman Kannada", ScriptFamily.ROMANIZED, LanguageFamily.DRAVIDIAN, is_romanized=True),
    "ml-Latn": LanguageSpec("ml-Latn", "Roman Malayalam", ScriptFamily.ROMANIZED, LanguageFamily.DRAVIDIAN, is_romanized=True),
    "bn-Latn": LanguageSpec("bn-Latn", "Roman Bengali", ScriptFamily.ROMANIZED, LanguageFamily.INDO_ARYAN, is_romanized=True),
    "ar-Latn": LanguageSpec("ar-Latn", "Arabizi / Roman Arabic", ScriptFamily.ROMANIZED, LanguageFamily.SEMITIC, is_romanized=True),
    "fa-Latn": LanguageSpec("fa-Latn", "Roman Persian (Fingilish)", ScriptFamily.ROMANIZED, LanguageFamily.INDO_ARYAN, is_romanized=True),
    "ru-Latn": LanguageSpec("ru-Latn", "Roman Russian (Translit)", ScriptFamily.ROMANIZED, LanguageFamily.SLAVIC, is_romanized=True),
    "ja-Latn": LanguageSpec("ja-Latn", "Roman Japanese (Romaji)", ScriptFamily.ROMANIZED, LanguageFamily.JAPONIC, is_romanized=True),

    # 10. CODE-SWITCHED LANGUAGE COMBINATIONS (Section 3)
    "cs-en-hi": LanguageSpec("cs-en-hi", "English + Hindi (Mixed)", ScriptFamily.CODE_SWITCHED, LanguageFamily.MIXED_CODE_SWITCH, is_code_switched=True),
    "cs-en-es": LanguageSpec("cs-en-es", "English + Spanish (Spanglish)", ScriptFamily.CODE_SWITCHED, LanguageFamily.MIXED_CODE_SWITCH, is_code_switched=True),
    "cs-en-ar": LanguageSpec("cs-en-ar", "English + Arabic (Mixed)", ScriptFamily.CODE_SWITCHED, LanguageFamily.MIXED_CODE_SWITCH, is_code_switched=True),
    "cs-en-ja": LanguageSpec("cs-en-ja", "English + Japanese (Mixed)", ScriptFamily.CODE_SWITCHED, LanguageFamily.MIXED_CODE_SWITCH, is_code_switched=True),
    "cs-en-ko": LanguageSpec("cs-en-ko", "English + Korean (Mixed)", ScriptFamily.CODE_SWITCHED, LanguageFamily.MIXED_CODE_SWITCH, is_code_switched=True),
    "cs-en-pt": LanguageSpec("cs-en-pt", "English + Portuguese (Mixed)", ScriptFamily.CODE_SWITCHED, LanguageFamily.MIXED_CODE_SWITCH, is_code_switched=True),
    "cs-hi-en": LanguageSpec("cs-hi-en", "Hindi + English (Devanagari+Latin)", ScriptFamily.CODE_SWITCHED, LanguageFamily.MIXED_CODE_SWITCH, is_code_switched=True),
    "cs-ta-en": LanguageSpec("cs-ta-en", "Tamil + English (Tamil+Latin)", ScriptFamily.CODE_SWITCHED, LanguageFamily.MIXED_CODE_SWITCH, is_code_switched=True),
    "cs-te-en": LanguageSpec("cs-te-en", "Telugu + English (Telugu+Latin)", ScriptFamily.CODE_SWITCHED, LanguageFamily.MIXED_CODE_SWITCH, is_code_switched=True),
    "cs-ar-fr": LanguageSpec("cs-ar-fr", "Arabic + French (Franco-Arabe)", ScriptFamily.CODE_SWITCHED, LanguageFamily.MIXED_CODE_SWITCH, is_code_switched=True),
}


def get_languages_by_family(family: LanguageFamily) -> List[LanguageSpec]:
    return [lang for lang in LANGUAGES.values() if lang.family == family]


def get_languages_by_script(script: ScriptFamily) -> List[LanguageSpec]:
    return [lang for lang in LANGUAGES.values() if lang.script == script]


def get_romanized_languages() -> List[LanguageSpec]:
    return [lang for lang in LANGUAGES.values() if lang.is_romanized]


def get_code_switched_languages() -> List[LanguageSpec]:
    return [lang for lang in LANGUAGES.values() if lang.is_code_switched]


def get_native_languages() -> List[LanguageSpec]:
    return [lang for lang in LANGUAGES.values() if not lang.is_romanized and not lang.is_code_switched]
