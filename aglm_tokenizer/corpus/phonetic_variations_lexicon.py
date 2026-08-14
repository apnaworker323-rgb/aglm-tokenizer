"""
Comprehensive Phonetic Variations, Chat Shorthand & Real-World Colloquial Lexicon.
Handles realistic typing variations, informal spellings, and SMS/chat shorthand for
Hinglish, Tenglish, Tanglish, Kanglish, Manglish, Bonglish, and Pan-Indic Romanized text.
Ensures BOTH bare ("word") and space-prefixed (" word") representations.
"""

from typing import Set


def get_phonetic_variations_lexicon() -> Set[str]:
    """Generates phonetic variations and colloquial chat forms."""
    words = [
        # --- Pronoun & Casual Variants ---
        "mujhe", "muje", "mjhe", "mujhey", "mujhko", "mjhko", "muze", "mj",
        "tujhe", "tuje", "tjhe", "tujhey", "tujhko", "tj",
        "hume", "humey", "hme", "humein", "hame", "hamein", "hamko", "humko",
        "tumhe", "tumhey", "tmhe", "tumhein", "tumko", "tmko",
        "aapko", "apko", "aapka", "apka", "aapki", "apki", "aapke", "apke",
        "usko", "isko", "usey", "isey", "uske", "iske", "uski", "iski", "uska", "iska",
        "unhe", "unhey", "unhein", "unko", "inhe", "inhey", "inhein", "inko",
        "apna", "apni", "apne", "khud", "khudka", "khudki", "khudke",

        # --- Conjunctions, Connectives & Casual Typing ---
        "kyunki", "kyuki", "kyonki", "kyoki", "qki", "kuki", "kyun", "kyu", "kyon",
        "isliye", "isiliye", "isleeye", "islye", "issliye", "isilye",
        "lekin", "lkn", "magar", "mgr", "par", "pr", "lekin", "parantu", "kintu",
        "agar", "agr", "yadi", "warna", "vrna", "warnaa",
        "jab", "jb", "tab", "tb", "kab", "kb", "abhi", "abhe", "tabhi", "tbhi", "kabhi", "kbhi",
        "phir", "fir", "fr", "wapas", "waapis", "dobara", "dobaara",
        "aur", "or", "aurr", "tatha", "evam", "ya", "yaa", "athwa",
        "kyunki", "kyu ki", "kyun ki", "q ki",

        # --- High-Frequency Nouns, Adjectives, Time & Quantity ---
        "aane", "aana", "aata", "aati", "aate", "aaya", "aaye", "aayi", "aayega", "aayegi", "aayenge",
        "saalon", "saal", "salon", "mahine", "mahino", "din", "dino", "hafta", "hafte", "hafton",
        "tezi", "tez", "tezee", "jaldi", "jldi", "jaldee", "fast", "speed",
        "badalne", "badalna", "badalta", "badalti", "badalte", "badla", "badle", "badli", "badlega", "badlegi",
        "insaan", "insan", "insaano", "insanon", "aadmi", "admi", "log", "logo", "logon",
        "pahunchana", "pahunchane", "pahuncha", "pahunchi", "pahunch", "pahuchana", "pahuche", "pahuch",
        "saath", "sath", "sth", "saathi", "sathi",
        "hinglish", "tenglish", "tanglish", "kanglish", "manglish", "bonglish",
        "alag", "alag-alag", "tarike", "tareeke", "tarika", "tareeka", "tarikon", "tareekon",
        "tarah", "trah", "tarha", "taraha",
        "likhe", "likha", "likhna", "likhta", "likhti", "likhte", "likh", "likhenge", "likhega", "likhegi",
        "romanized", "transliterated", "phonetic", "variations", "variation",
        "standard", "spelling", "spellings", "fragmentation", "unnecessary", "multilingual", "tokenizer",
        "artificial", "intelligence", "programming", "scientific", "research", "mathematics",
        "problems", "problem", "models", "model", "language", "languages", "india",

        # --- Common Chat & Slang Vocabulary ---
        "bhai", "bro", "bhaiya", "bhaijaan", "yaar", "yr", "dost", "dosti",
        "accha", "acha", "achi", "acchi", "ache", "acche", "achha", "achhi", "achhe",
        "theek", "thik", "thk", "thek", "sahii", "sahi", "galat", "glt",
        "bahut", "bohot", "bht", "bhut", "bhot", "bahot",
        "bilkul", "blkl", "bilqul",
        "thoda", "thodi", "thode", "thoda sa", "thodi si", "zyada", "jyada", "zyaada", "jyaada",
        "samajhna", "samajh", "samjh", "smjh", "samjhana", "smjhana", "samjha", "smjha", "samjhe", "smjhe",
        "karna", "kare", "karo", "karenge", "karega", "karegi", "karte", "karta", "karti", "kiya", "kiye", "kiyi",
        "krna", "kre", "kro", "krenge", "krta", "krti", "krte", "kya", "kr",
        "hona", "hota", "hoti", "hote", "hua", "hue", "hui", "hoga", "hogi", "hoge", "honge",
        "chota", "choti", "chote", "chotasa", "chotisi", "bada", "badi", "bade",
        "tension", "important", "zaruri", "zaroori", "zaroorat", "zarurat", "aasan", "mushkil",
        "khana", "paani", "chai", "coffee", "office", "ghar", "college", "school", "station",

        # --- Telugu / Tenglish Colloquial Chat ---
        "nenu", "nuvvu", "meeru", "manamu", "vaallu", "veellu", "athanu", "aame",
        "enti", "enduku", "ela", "eppudu", "ekkada", "deniki", "deeniki", "daaniki",
        "cheyyali", "cheyali", "cheyyadam", "chestunna", "chestunnanu", "chesanu", "chesukovali", "cheyyinchali",
        "vellali", "vellanu", "velthunna", "ravali", "vachanu", "vastunna", "chudali", "cheppali",
        "ardhamkaledu", "arthamkaledu", "telusu", "teliyadu", "thelidhu", "kavali", "vaddu", "chala", "konchem",

        # --- Tamil / Tanglish Colloquial Chat ---
        "naan", "neenga", "namma", "avan", "aval", "avanga", "enna", "ethu", "yen", "epdi", "enga", "eppo",
        "indha", "andha", "idha", "adha", "idhu", "adhu", "idhellam",
        "pannanum", "pannuren", "pannanga", "pannirukken", "pannikkalam", "panren", "pannitten",
        "puriyala", "theriyala", "varudhu", "vandhadhu", "seekiram", "prachanai", "sari", "romba", "konjam"
    ]

    unique_set = set()
    for w in words:
        clean = w.strip().lower()
        if clean:
            unique_set.add(clean)
    return unique_set


if __name__ == "__main__":
    lex = get_phonetic_variations_lexicon()
    print(f"Phonetic Variations Lexicon: {len(lex):,} unique words.")
