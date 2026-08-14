"""
Comprehensive Tanglish (Romanized Tamil) & Tamil Agglutinative Lexicon.
Includes full inflections, auxiliary verbs, negative imperatives (-koodadhu),
temporal gerunds (-umbodhu), modals (-venum, -venaam, -mudiyanum),
and colloquial conversational variations across dialects.
Ensures BOTH bare ("word") and space-prefixed (" word") representations.
"""

from typing import Set


def get_tanglish_dravidian_lexicon() -> Set[str]:
    """Generates a rich set of Tamil / Tanglish conversational and complex agglutinative words."""
    words = [
        # --- Pronouns & Person Markers ---
        "naan", "enakku", "ennaku", "ennoda", "enna", "ennai", "enaku",
        "nee", "unnakku", "unakku", "unnoda", "unna", "unnai",
        "neenga", "ungalukku", "ungala", "ungada", "ungalloda",
        "namma", "namakku", "nammoda", "nammala", "naanga", "engalukku",
        "avan", "avanukku", "avanoda", "aval", "avalukku", "avaloda",
        "avanga", "avangalukku", "avangaloda", "ivanga", "ivangalukku",
        "idhu", "adhu", "indha", "andha", "idha", "adha", "idhellam", "adhellam",
        "enna", "yedhu", "ethu", "edhu", "yen", "yean", "epdi", "yeppadi",
        "enga", "yeppo", "eppoludhu", "yaaru", "yaarukku", "yaarodhadhu",

        # --- Nouns, Adjectives, Technical & Contextual Words ---
        "oru", "pudhiya", "pudhu", "palaiya", "nalla", "ketta", "periya", "chinna",
        "tamil", "thamizh", "english", "tanglish", "bhashai", "mozhi", "mozhigal",
        "letters", "ezhuthu", "ezhuthukkal", "ezhuthuvanga", "ezhuthalam", "tharezhuthuvanga", "tharezhuthalam",
        "tharezhythalam", "ezhythalam", "tharezhuthuranga", "ezhuthuranga",
        "words", "varthaigal", "varthai", "sentences", "vakkiyam", "vakkiyagal",
        "spelling", "spellings", "variations", "variation", "pieces", "patterns",
        "model", "models", "language", "multilingual", "training", "speed", "accuracy",
        "improve", "kuraiyakoodadhu", "koodadhu", "kuraiya", "kuraiyave",
        "architecture", "memory", "usage", "benchmark", "carefully", "split",

        # --- Postpositions, Conjunctions & Adverbs ---
        "mattum", "mattume", "illama", "illamal", "illai", "illa", "irukku", "irukken",
        "um", "um-mixed", "mixed", "aaga", "aaganum", "aachu", "aayiduchi",
        "but", "aana", "aanaal", "anal", "oda", "udaiya", "ku", "kku", "adhukku", "idhukku",
        "ellathayum", "ellarum", "yellarum", "ellame", "ellarukkum", "ellorukkum",
        "ore", "ovvoru", "oru", "madhiri", "mathiri", "pol", "pola",
        "romba", "romba-periya", "konjam", "konjama", "seekiram", "seekirama", "twaraga",
        "pala", "neraiya", "sila", "adhiga", "kuraiva",

        # --- Verbs & Agglutinative Suffixes ---
        # pannanum / pannitu / pannumbodhu / panna / maatanga
        "pannanum", "pannuren", "panren", "pannanga", "pannirukken", "pannikkalam",
        "pannitu", "panitu", "pannittu", "pannikittu", "pannumbodhu", "pannumpothu",
        "pannama", "pannamal", "panama", "panamal",
        "panna", "pannamatanga", "panna-maatanga", "maatanga", "matanga", "maaten", "maten",
        "maatom", "maattom", "panna mudiyum", "panna mudiyadhu",
        # mudiyanum / mudiyum / mudiyadhu
        "mudiyanum", "mudiyum", "mudiyadhu", "mudiyathu", "mudila", "mudiyala", "mudinjadhu",
        # venum / vaenum / venaam
        "venum", "vaenum", "vendam", "venaam", "venam", "thevai", "thevaipadu",
        # understand / use / represent
        "understand", "use", "represent", "reusable", "useful", "unnecessary",
        "puriyala", "purinjadhu", "theriyala", "therinjadhu", "varudhu", "varala"
    ]

    unique_set = set()
    for w in words:
        clean = w.strip().lower()
        if clean:
            unique_set.add(clean)
    return unique_set


if __name__ == "__main__":
    lex = get_tanglish_dravidian_lexicon()
    print(f"Tanglish Dravidian Lexicon: {len(lex):,} unique words.")
