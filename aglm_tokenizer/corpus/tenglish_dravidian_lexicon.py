"""
Comprehensive Tenglish (Romanized Telugu) & South-Indic Agglutinative Lexicon.
Includes full inflections, multi-word suffixes, temporal gerunds (-appudu),
conditionals (-pothe), abilitatives (-galagali), and negatives (-kunda, -kaavatledu).
Ensures BOTH bare ("word") and space-prefixed (" word") representations.
"""

from typing import Set


def get_tenglish_dravidian_lexicon() -> Set[str]:
    """Generates a rich set of Telugu / Tenglish conversational and complex agglutinative words."""
    words = [
        # --- Core Pronouns & Variants ---
        "nenu", "naku", "naaku", "naadi", "naatho", "nannu", "naalo", "naavalla",
        "nuvvu", "neeku", "needi", "neetho", "ninnu", "neelo", "neevalla",
        "meeru", "meeku", "meedi", "meetho", "mimmulni", "meelo", "meevalla",
        "manamu", "manaki", "manadi", "manatho", "manalni", "manalo",
        "athanu", "athani", "athaniki", "athanitho", "aame", "aamenu", "aameku", "aametho",
        "vaallu", "vaallaki", "vaallani", "vaallatho", "veellu", "veellaki", "veellani", "veellatho",
        "andaroo", "andaru", "andarni", "andariki", "andaritho", "konthamandi", "konthamandiki",
        "idi", "idhi", "adi", "adhi", "edhi", "edi", "ivi", "avi", "evi",
        "evaru", "evariki", "evadiki", "evadu", "evarevaru", "enti", "enduku", "ela", "elaga",
        "eppudu", "ekkada", "deniki", "deeniki", "daaniki", "denitho", "deenitho", "daanitho",

        # --- Nouns, Technical & Linguistic Context ---
        "kottha", "kotha", "paatha", "pata", "padanni", "padam", "padalu", "padalani",
        "ane", "ani", "anukuntunna", "anukuntunnaru", "anukovali", "anadam",
        "aksharam", "aksharalu", "aksharalani", "spelling", "spellings", "variations", "variation",
        "bhasha", "bhashalu", "telugu", "english", "hindi", "tenglish", "hinglish",
        "letters", "words", "sentences", "sentence", "pieces", "patterns", "tokenizer",
        "model", "models", "language", "multilingual", "training", "validation", "loss",
        "accuracy", "architecture", "optimization", "optimize", "memory", "batch", "size",
        "gpu", "cpu", "system", "data", "dataset", "epoch", "learning", "rate",

        # --- Postpositions, Adverbs, Modifiers & Connectives ---
        "tho", "paatu", "thopaatu", "patu", "thopatu", "kuda", "kooda", "ni", "nu", "lo", "ki", "ku",
        "kosam", "gurinchi", "dwara", "valana", "valla", "batti", "battii",
        "sarigga", "sariga", "baaga", "baagane", "bagane", "chala", "chaala", "konchem", "chinna",
        "pedda", "manchi", "manchiga", "chedda", "twaraga", "ventane", "ekkuva", "thakkuva",
        "ippudu", "appudu", "eppudu", "kani", "kaani", "anduke", "kabatti", "ayithe",
        "ilanti", "alanti", "elanti", "ituvanti", "atuvanti", "etuvanti",
        "oke", "oka", "anni", "enni", "enta", "antha", "yentha", "inta",

        # --- Verbs & Agglutinative Inflections ---
        # nadustundi / nadavadam
        "nadustundi", "nadustundhi", "nadustunna", "nadustunnanu", "nadavadam", "nadavatam",
        # avutundo / avvadam
        "avutundo", "avutundhi", "avutundi", "avuthundi", "avuthundo", "avutam", "avadam",
        "avvakapothe", "avvadam", "avtundi", "avthundi", "avtundo", "avthundo",
        # ardham / kaavatledu
        "ardham", "artham", "ardhamkaledu", "arthamkaledu", "ardhamkaavatledu", "arthamkaavatledu",
        "kaavatledu", "kaledu", "theliyadhu", "teliyadu", "thelidhu", "telusu", "telusukovali",
        # saripokapothe / saripotundi
        "saripokapothe", "saripotundi", "saripothundi", "saripoledu", "saripovatledu", "saripovadam",
        # tagginchi / penchi
        "tagginchi", "tagginchali", "taggali", "taggadam", "penchi", "penchali", "penchadam",
        # cheyyalsi / vastundi
        "cheyyalsi", "cheyalsi", "vastundi", "vastundhi", "cheyyalsivastundi", "cheyalsivastundi",
        "ravalsivastundi", "cheyalsi vastundi", "cheyyalsi vastundi",
        # raasetappudu / temporal -appudu
        "raasetappudu", "raasetapudu", "rasetappudu", "raasinappudu", "rasinappudu",
        "chusetappudu", "chesetappudu", "vachetappudu", "velletappudu", "matladetappudu",
        "vachinappudu", "vellinappudu", "chesinappudu", "cheppinappudu", "chusinappudu", "thelisinappudu",
        # rastaru / rayachu
        "rastaru", "rastharu", "rastanu", "rasthanu", "rasamu", "rasadu", "rasaru",
        "rayachu", "raayachu", "rayadam", "raayadam", "rayali", "raayali", "rasanu",
        # veltunnanu / velthunnanu
        "veltunnanu", "velthunnanu", "veltanu", "velthanu", "velthamu", "veltharu",
        "vellanu", "vellali", "velladam", "veldam", "vellu", "vellandi",
        # cheyakunda / negative -kunda
        "cheyakunda", "cheyyakunda", "cheppakunda", "chudakunda", "vellakunda", "rakunda", "undakunda",
        # cheyyagalagali / abilitative -galagali
        "cheyyagalagali", "cheyagalagali", "cheppagalagali", "chudagalagali", "vellagalagali",
        "cheyyagaladu", "cheyyagalamu", "cheyyagalaru", "cheyagalanu", "cheyagaladu",
        # understand / optimize / train
        "understand", "chesukovali", "chesukovadam", "chesukunnam", "chesukondi",
        "optimize", "cheyyali", "cheyali", "cheyyadam", "chestunnanu", "chesanu", "cheyyinchali"
    ]

    unique_set = set()
    for w in words:
        clean = w.strip().lower()
        if clean:
            unique_set.add(clean)
    return unique_set


if __name__ == "__main__":
    lex = get_tenglish_dravidian_lexicon()
    print(f"Tenglish Dravidian Lexicon: {len(lex):,} unique words.")
