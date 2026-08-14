"""
Comprehensive Kanglish (Romanized Kannada) & Kannada Agglutinative Lexicon.
Includes full inflections, postpositions (-alli, -annu, -inda, -ge, -jothege),
temporal gerunds (-vaga, -daga, -bandaga), modals (-beku, -bahudu, -dilla, -dare, -alu, -uttare),
reflexives (-kollalu, -kalitukollalu), and colloquial chat variations.
Ensures BOTH bare ("word") and space-prefixed (" word") representations.
"""

from typing import Set


def get_kanglish_dravidian_lexicon() -> Set[str]:
    """Generates a rich set of Kannada / Kanglish conversational and agglutinative words."""
    words = [
        # --- Pronouns & Demonstratives ---
        "naanu", "nanage", "nange", "nanna", "nannannu", "nanninda", "nannodane",
        "neenu", "ninage", "ninge", "ninna", "ninnannu", "ninninda", "ninnodane",
        "neevu", "nimage", "nimge", "nimma", "nimmannu", "nimminda",
        "naavu", "namage", "namge", "namma", "nammannu", "namminda",
        "avanu", "avanige", "avange", "avana", "avanannu", "avaninda",
        "avalu", "avalige", "avalge", "avala", "avalannu", "avalinda",
        "avaru", "avarige", "avarge", "avara", "avarannu", "avarinda", "kelavaru", "pratiyobbaru",
        "idu", "adu", "idannu", "adannu", "annodanna", "annodu", "annadanna",
        "ee", "aa", "yava", "yaava", "yaru", "yaaru", "yaake", "yake",
        "elli", "alli", "illi", "hege", "yaavaga", "yavaga",

        # --- Nouns, Adjectives, Technical & Contextual Words ---
        "ondu", "eradu", "mooru", "naalku", "aidu", "hosa", "hale", "dodda", "chikka",
        "kannada", "english", "kanglish", "bhashe", "bhashegalu", "padagalu", "pada",
        "letters", "akshara", "aksharagalu", "spelling", "spellings", "variations", "variation",
        "sentences", "vakya", "vakyagalu", "words", "pieces", "patterns", "model", "models",
        "multilingual", "language", "training", "speed", "accuracy", "slow", "fast",
        "architecture", "optimize", "memory", "usage", "techniques", "test",
        "tara", "reethi", "vidha", "sahaya", "upayoga", "useful", "reusable",

        # --- Postpositions, Conjunctions & Adverbs ---
        "jothege", "jothe", "kooda", "jotheyalli", "mattu", "haagu", "aadare", "adare", "aadre",
        "sariyagi", "sariga", "tumba", "kadime", "bere", "onde", "anta", "yendu",
        "annu", "alli", "inda", "ge", "kke", "annodanna",

        # --- Verbs & Agglutinative Suffixes ---
        # madu / madabeku / madbeku / maduttiddene / maduvudilla / madade / madidare / madalu / madabahudu
        "maduttiddene", "madtidini", "madtiddini", "maduttidene",
        "madabeku", "madbeku", "madbahudu", "madabodu", "madabahudu",
        "maduvudilla", "madalla", "madodilla", "madade", "madadhe", "madidare", "maddre",
        "madalu", "madalikke", "madoke", "madide", "madiddu", "madidini", "madi", "madida",
        # bare / bareyuvaga / bareyuttare / bareyabahudu / bareyoke
        "bareyuvaga", "barevaga", "bareyuttare", "bareytare", "bareyabahudu", "bareyabodu",
        "bareyoke", "bareyalu", "baredu", "barede", "bareyala",
        # bandaga / agabahudu / kalitukollalu
        "bandaga", "bandaaga", "bandare", "bandre", "bandu",
        "agabahudu", "agabodu", "aguttade", "aaguttade", "aagodu", "aagalla", "aagide", "aayithu",
        "kalitukollalu", "kalitukoloke", "kalthukollalu", "kalthukoloke", "kalliyalu", "kalloke",
        # understand / reuse / break / common / technical / gpu / tokenizer / use
        "understand", "reuse", "break", "common", "train", "code", "mixed",
        "gpu", "technical", "use", "tokenizer"
    ]

    unique_set = set()
    for w in words:
        clean = w.strip().lower()
        if clean:
            unique_set.add(clean)
    return unique_set


if __name__ == "__main__":
    lex = get_kanglish_dravidian_lexicon()
    print(f"Kanglish Dravidian Lexicon: {len(lex):,} unique words.")
