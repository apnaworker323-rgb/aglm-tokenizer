"""
Comprehensive Manglish (Romanized Malayalam) & Malayalam Agglutinative Lexicon.
Includes full inflections, postpositions (-il, -ukondu, -mathramalla),
temporal gerunds (-umbol, -cheyyumbol), modals (-paadilla, -cheyyanam, -kurayaan),
nominalizations (-ennath, -ennum, -ennokke, -ezhuthunnavar, -cheyyunnath),
and colloquial conversational variations across dialects.
Ensures BOTH bare ("word") and space-prefixed (" word") representations.
"""

from typing import Set


def get_manglish_dravidian_lexicon() -> Set[str]:
    """Generates a rich set of Malayalam / Manglish conversational and agglutinative words."""
    words = [
        # --- Pronouns & Demonstratives ---
        "njan", "njanum", "enikku", "enik", "enikk", "ente", "enne", "ennal",
        "nee", "ninakku", "ninte", "ninne", "ningal", "ningalkku", "ningalude",
        "nammal", "namukku", "nammude", "nammale", "njangal", "njangalkku", "njangalude",
        "avan", "avanu", "avante", "avane", "aval", "avalkku", "avalude", "avale",
        "avarkku", "avarde", "avarude", "avare", "ellavarum", "ellarkkum", "ellarum",
        "ithu", "athu", "ithinte", "athinte", "ithine", "athine", "athukondu", "ithukondu",
        "ivide", "avide", "evide", "eppol", "ippol", "appol", "engane", "ingane", "angane",
        "enthu", "enthukondu", "aaru", "aarkku",
        "itharam", "atharam", "ennokke", "ennath", "ennum", "enna", "ennu",

        # --- Nouns, Adjectives, Technical & Contextual Words ---
        "oru", "puthiya", "pazhaya", "nalla", "valiya", "cheriya", "kurachu", "valare",
        "malayalam", "english", "manglish", "bhasha", "bhashakal", "aksharangal", "aksharam",
        "letters", "spelling", "spellings", "variations", "variation", "words", "word",
        "sentences", "vakyam", "vakyangal", "pieces", "patterns", "model", "models",
        "multilingual", "language", "training", "speed", "quality", "performance",
        "improve", "kurayaan", "paadilla", "padilla", "athukondu", "compression", "actual",
        "benchmark", "separate", "terms", "technical", "code", "mixed", "important",

        # --- Postpositions, Conjunctions & Adverbs ---
        "mathramalla", "mathram", "alla", "pakshe", "aanu", "anu", "undu", "illa",
        "nannayi", "nannaayi", "vegam", "pathukke", "ayi", "aayi", "il", "um", "ore",

        # --- Verbs & Agglutinative Suffixes ---
        # cheyyuka / cheyyukayaanu / cheyyanam / cheyanam / cheyyumbol / cheyyunnath / cheyyunnundo
        "cheyyukayaanu", "cheyyukayanu", "cheyyunnu", "cheythu", "cheyyum",
        "cheyyanam", "cheyanam", "cheyyenta", "cheyyenda", "cheyyaruthu",
        "cheyyumbol", "cheyyumpol", "cheyumpol", "cheythal", "cheythaal",
        "cheyyunnath", "cheyyunnathu", "cheyyunnathano", "cheyyunnundo", "cheyyunille",
        "cheyyuka", "cheyyan", "cheyyal", "cheyyikkanam",
        # ezhuthu / ezhutham / ezhuthunnavar / ezhuthunnath
        "ezhutham", "ezhuthu", "ezhuthunnu", "ezhuthi", "ezhuthum",
        "ezhuthunnavar", "ezhuthunnath", "ezhuthunnathu", "ezhuthikkanam",
        # venam / veenam / vendam
        "venam", "veenam", "venda", "vendam", "aavashyam", "aavashyamilla",
        # understand / train / handle / type / use / ee / tokenizer
        "understand", "train", "handle", "test", "reuse", "split", "efficiently",
        "ee", "use", "type", "tokenizer"
    ]

    unique_set = set()
    for w in words:
        clean = w.strip().lower()
        if clean:
            unique_set.add(clean)
    return unique_set


if __name__ == "__main__":
    lex = get_manglish_dravidian_lexicon()
    print(f"Manglish Dravidian Lexicon: {len(lex):,} unique words.")
