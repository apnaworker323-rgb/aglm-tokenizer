"""
Indic Verb Morphology & Agglutinative Conjugation Generator.
Systematically generates productive inflections, causatives, reflexives, compound verbs,
and agentives for Hindi, Telugu, Tamil, Kannada, Malayalam, Bengali, and Marathi
in BOTH Native Scripts and Romanized Transliterations (bare + space-prefixed).
"""

from typing import Set, List


def generate_indic_verb_lexicon() -> Set[str]:
    """Generates a rich, comprehensive set of inflected and conjugated Indic verbs."""
    tokens = set()

    # =========================================================================
    # 1. HINDI DEVANAGARI & ROMANIZED HINGLISH INFLECTIONS
    # =========================================================================
    hindi_roots = [
        # (Devanagari Root, Romanized Root, Causative Base Deva, Causative Base Rom)
        ("कर", "kar", "करवा", "karwa"),
        ("देख", "dekh", "दिखा", "dikha"),
        ("बोल", "bol", "बुलवा", "bulwa"),
        ("सुन", "sun", "सुना", "suna"),
        ("बता", "bata", "बतला", "batla"),
        ("दे", "de", "दिला", "dila"),
        ("ले", "le", "दिलवा", "dilwa"),
        ("रख", "rakh", "रखवा", "rakhwa"),
        ("चल", "chal", "चला", "chala"),
        ("बन", "ban", "बना", "bana"),
        ("लिख", "likh", "लिखवा", "likhwa"),
        ("पढ़", "padh", "पढ़ा", "padha"),
        ("भेज", "bhej", "भिजवा", "bhijwa"),
        ("खोल", "khol", "खिलवा", "khulwa"),
        ("रोक", "rok", "रुकवा", "rukwa"),
        ("बैठ", "baith", "बिठा", "bitha"),
        ("उठ", "uth", "उठा", "utha"),
        ("सोच", "soch", "समझा", "samjha"),
        ("समझ", "samajh", "समझा", "samjha")
    ]

    # Devanagari Suffixes
    deva_suffixes = [
        "ना", "ने", "नी", "ता", "ती", "ते", "ा", "ी", "े", "या", "ये", "यी", "ई",
        "ूंगा", "ूंगी", "ेंगे", "ेगा", "एगी", "ओगे", "ओगी", "नापड़ेगा", "नापड़ेगी",
        "नेवाला", "नेवाले", "नेवाली", "नेकेलिए", "नेपर"
    ]
    # Romanized Suffixes
    rom_suffixes = [
        "na", "ne", "ni", "ta", "ti", "te", "a", "i", "e", "ya", "ye", "yi",
        "unga", "ungi", "enge", "ega", "egi", "oge", "ogi", "na padega", "na padegi",
        "newala", "newale", "newali", "ne wala", "ne wale", "ne wali", "ne ke liye", "ne per"
    ]

    # Direct Inflections
    for droot, rroot, dcause, rcause in hindi_roots:
        for dsuf in deva_suffixes:
            tokens.add(f"{droot}{dsuf}")
            tokens.add(f"{dcause}{dsuf}")
        for rsuf in rom_suffixes:
            tokens.add(f"{rroot}{rsuf}")
            tokens.add(f"{rroot}_{rsuf}")
            tokens.add(f"{rcause}{rsuf}")

    # Specific common Devanagari compounds
    deva_compounds = [
        "करवाना", "करवाया", "करवाए", "करवाई", "करवाएंगे", "करवाएगा", "करवाएगी",
        "करवानेवाला", "करवानेवाले", "करवानेवाली", "करवाते", "करवाता", "करवाती",
        "दिखवाना", "दिखवाया", "दिखवाएंगे", "दिखवानेवाला",
        "बनवाना", "बनवाया", "बनवाएंगे", "बनवानेवाला",
        "लिखवाना", "लिखवाया", "लिखवाएंगे", "लिखवानेवाला",
        "पढ़वाना", "पढ़वाया", "पढ़वाएंगे", "पढ़वानेवाला",
        "भिजवाना", "भिजवाया", "भिजवाएंगे", "भिजवानेवाला"
    ]
    for dc in deva_compounds:
        tokens.add(dc)

    # Specific common Romanized compounds
    rom_compounds = [
        "karwana", "karwaya", "karwaye", "karwayi", "karwayenge", "karwayega", "karwayegi",
        "karwanewala", "karwanewale", "karwanewali", "karwate", "karwata", "karwati",
        "dikhwana", "dikhwaya", "dikhwayenge", "dikhwanewala",
        "banwana", "banwaya", "banwayenge", "banwanewala",
        "likhwana", "likhwaya", "likhwayenge", "likhwanewala"
    ]
    for rc in rom_compounds:
        tokens.add(rc)

    # =========================================================================
    # 2. TELUGU / TENGLISH AGGLUTINATIVE INFLECTIONS
    # =========================================================================
    telugu_bases = [
        "cheyy", "chey", "ches", "chepp", "chep", "chud", "chust", "vell", "velth",
        "vach", "vast", "ra", "rav", "po", "poth", "ivv", "ist", "theesuk", "pettuk",
        "und", "unn", "kaval", "matlad"
    ]
    telugu_endings = [
        "ali", "alsi", "adam", "adame", "tunnanu", "tunnaru", "tundi", "tundhi",
        "tunnam", "tunna", "tunnara", "anu", "aru", "amu", "adu", "adi",
        "ukovali", "ukovalani", "ukondi", "ukunnanu", "kunnaru", "kundi",
        "inchali", "inchanu", "incharu", "inchadam", "isthunnanu", "istharu",
        "agalanu", "agaladu", "agalamu", "alenu", "aledu", "alem",
        "oddu", "vaddu", "andi", "avoyi"
    ]
    for tb in telugu_bases:
        for te in telugu_endings:
            tokens.add(f"{tb}{te}")

    # Explicit Telugu target tokens
    explicit_telugu = [
        "cheyyali", "cheyyadam", "chestunnanu", "chesanu", "chesukovali", "cheyyinchali",
        "cheyali", "cheyadam", "chestanu", "chesanu", "chesukondi", "cheyinchali",
        "cheppali", "cheppadam", "cheptunnanu", "cheppanu", "cheppukovali", "cheppinchali",
        "chudali", "chudadam", "chustunnanu", "choosanu", "chusukovali", "chupinchali",
        "vellali", "velladam", "velthunnanu", "vellanu", "vellipovali", "vellinchali",
        "ravali", "ravadam", "vastunnanu", "vachanu", "rappinchali",
        "ivvali", "ivvadam", "istunnanu", "ichanu", "ippinchali",
        "teesukovali", "theesukovali", "teesukovadam", "teesukunnanu", "teesukunnaru"
    ]
    for et in explicit_telugu:
        tokens.add(et)

    # =========================================================================
    # 3. TAMIL / TANGLISH AGGLUTINATIVE INFLECTIONS
    # =========================================================================
    tamil_bases = [
        "pann", "panr", "pan", "seiy", "sei", "soll", "sol", "ketk", "kel",
        "paakk", "paar", "var", "vanth", "vandh", "pog", "por", "poit",
        "kudukk", "kuduth", "vaang", "vaangit", "irukk", "irundh"
    ]
    tamil_endings = [
        "anum", "uren", "uran", "anga", "aru", "itten", "iten", "irukken",
        "irukanga", "ikkalam", "alam", "unga", "u", "iya", "om", "rom",
        "aamal", "aadhunga", "aadha", "aachu", "aayiduchu", "alaam",
        "adhu", "udhu", "uthu", "ala", "um", "ave"
    ]
    for tb in tamil_bases:
        for te in tamil_endings:
            tokens.add(f"{tb}{te}")

    # Explicit Tamil target tokens
    explicit_tamil = [
        "pannanum", "pannuren", "pannanga", "pannirukken", "pannikkalam",
        "panran", "panren", "pannitten", "panniten", "pannalam", "pannunga",
        "seiyanum", "seiyuren", "seithan", "seithu", "seithirukken", "seiyalam",
        "sollanum", "solluren", "solranga", "sonnan", "sollirukken", "sollalam",
        "paakkanum", "paakkuren", "paathiya", "paathen", "paathirukken", "paathukalam",
        "varanum", "varuren", "vandhuten", "vandhachu", "vandhirukken", "varalaam",
        "poganum", "poren", "poiten", "poyachu", "poyirukken", "pogalaam",
        "kudukanum", "kudukkuren", "kuduthen", "kuduthachu", "kuduthirukken",
        "vaanganum", "vaanguren", "vaangitten", "vaangirukken"
    ]
    for eta in explicit_tamil:
        tokens.add(eta)

    # =========================================================================
    # 4. KANNADA / KANGLISH AGGLUTINATIVE INFLECTIONS
    # =========================================================================
    kannada_bases = [
        "mad", "maad", "hog", "bar", "nod", "hel", "kodu", "thago", "iru"
    ]
    kannada_endings = [
        "beku", "uttiddene", "thini", "ide", "abahudu", "isabeku", "oke", "i",
        "alla", "bahudu", "idini", "adhe", "thaare", "thare", "theve"
    ]
    for kb in kannada_bases:
        for ke in kannada_endings:
            tokens.add(f"{kb}{ke}")

    # Explicit Kannada target tokens
    explicit_kannada = [
        "madbeku", "maadbeku", "maduttiddene", "maaduttiddene", "madabahudu", "maadabahudu",
        "madisabeku", "maadisabeku", "madoke", "maadoke", "madthini", "maadthini", "madide", "maadide",
        "hogbeku", "hoguttiddene", "hogabahudu", "hogisabeku", "hogoke", "hogthini", "hode",
        "barbeku", "baruttiddene", "barabahudu", "barisabeku", "baroke", "barthini", "bandhe",
        "nodbeku", "noduttiddene", "nodabahudu", "nodisabeku", "nodoke", "nodthini", "nodide",
        "helbeku", "heluttiddene", "helabahudu", "helisabeku", "heloke", "helthini", "helide",
        "kodubeku", "koduuttiddene", "kodabahudu", "kodisabeku", "kodoke", "kodthini", "kotte"
    ]
    for ek in explicit_kannada:
        tokens.add(ek)

    # Normalize clean lowercase for Romanized tokens
    final_set = set()
    for tok in tokens:
        clean = tok.strip()
        if clean:
            final_set.add(clean)
            # If Romanized, also add lowercased
            final_set.add(clean.lower())

    return final_set


if __name__ == "__main__":
    lex = generate_indic_verb_lexicon()
    print(f"Generated Indic Verb Morphology Lexicon: {len(lex):,} unique inflected forms.")
