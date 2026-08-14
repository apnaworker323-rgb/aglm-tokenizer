"""
Comprehensive Colloquial Romanized Indic & Code-Mixed Core Lexicon.
Covers high-frequency conversational words across Hindi/Hinglish, Telugu/Tenglish,
Tamil/Tanglish, Kannada/Kanglish, Malayalam/Manglish, Bengali/Bonglish, Marathi, Gujarati, Punjabi, Urdu.
Ensures BOTH bare ("word") and space-prefixed (" word") representations in AGLM vocabulary.
"""

from typing import Set


def get_conversational_indic_lexicon() -> Set[str]:
    """Returns a comprehensive set of high-frequency colloquial Romanized Indic words."""
    words = [
        # =========================================================================
        # 1. TAMIL / TANGLISH (Conversational Core)
        # =========================================================================
        # Demonstratives, Pronouns, Interrogatives
        "indha", "andha", "endha", "idha", "adha", "edha", "idhai", "adhai", "edhai",
        "idhu", "adhu", "edhu", "indhadhu", "andhadhu", "endhadhu", "idhellam", "adhellam",
        "naan", "enakku", "ennoda", "enne", "ennai", "ennaiye", "namma", "namakku", "nammoda",
        "neenga", "ungalukku", "ungala", "ungappan", "ungamma", "ungalku", "unga", "unaku",
        "nee", "un", "unna", "unnoda", "unnai", "unnakku",
        "avan", "avanku", "avanoda", "avana", "aval", "avalku", "avaloda", "avala",
        "avanga", "avangalukku", "avangaloda", "avangala", "ivanga", "ivangalukku",
        "yaar", "yaarukku", "yaaroda", "yaara", "evano", "evalo",
        "enna", "ethu", "ethuku", "edhuku", "edhukaga", "yen", "yean", "yenda", "yendi",
        "epdi", "eppadi", "eppadiyo", "enga", "enge", "engirundhu", "eppo", "eppodhu", "eppavum",
        # Common Nouns & Adjectives
        "prachanai", "pirachanai", "pirachinai", "prachana", "vishayam", "sangathi", "neram",
        "velai", "kaalam", "seekiram", "seekirama", "seekram", "seekrama", "vegam", "vegama",
        "sekiram", "sekirama", "nalla", "nalladhu", "nallavan", "nallavala", "romba", "rombaave",
        "konjam", "konjama", "chinna", "periya", "pudhu", "palaya", "azhaga", "azhagu",
        "sari", "seri", "sariya", "seriya", "sariyanadhu", "thappu", "thappa", "thevai", "theva",
        "mukkiyam", "mukkiyama", "avasiyam", "sandhosham", "kavala", "bayama", "kobam",
        "kaasu", "panam", "veedu", "veetuku", "ooru", "oorukku", "kadai", "thambi", "anna", "akka",
        "thala", "machi", "machan", "nanba", "nanbane", "paappa", "kannu", "kanna",
        # Common Verbs & Conjugations
        "varudhu", "varuthu", "varathu", "varudha", "vandhadhu", "vandhathu", "varum", "vandha",
        "varen", "varala", "varaadhu", "vandhuten", "vandhachu", "vandhachi", "vaanga", "vaa",
        "puriyala", "purila", "puriyudhu", "puriyuthu", "puriyum", "puriyadhu", "purinjidhu", "purinjitha",
        "theriyala", "therila", "theriyudhu", "theriyum", "theriyadhu", "therinjidhu", "therinjitha",
        "pannanum", "panran", "panren", "pannitten", "pannunga", "pannalam", "panniten", "pannrom",
        "panrom", "pannu", "panniya", "panna", "pannalaam", "pannirukken", "seiyanum", "seiya", "seithu",
        "poganum", "poran", "poren", "poiten", "poyachu", "poyachi", "poitu", "ponga", "po", "pogalaam",
        "irukku", "irukka", "irukken", "irukkeengala", "irundhuchu", "irundhadhu", "irukkum", "irukadhu",
        "irundha", "irukalam", "irundhen", "irukkrom",
        "solla", "sollu", "sollunga", "sonnan", "sonnen", "solranga", "solran", "solren",
        "ketka", "kelu", "kettan", "ketten", "kelunga", "kekudhu", "kekala",
        "kuduka", "kudu", "kudunga", "kuduthan", "kuduthen", "kuduthachu",
        "paakka", "paaru", "paathiya", "paathen", "paathukalam", "paaringa",
        "mudiyum", "mudiyadhu", "mudiyala", "mudinjidhu", "mudichachu",
        "aachu", "aayiduchu", "aagudhu", "aagala", "aagum",
        # Suffixes, Particles, Connectives
        "nu", "ennu", "endru", "aga", "aaga", "la", "laa", "kooda", "mattum", "mattume",
        "illama", "illadha", "illa", "illai", "illaiya", "dhane", "dhaane", "thaane",
        "aana", "aanal", "aanaalum", "appo", "ippo", "eppo", "adhunaala", "idhunaala",

        # =========================================================================
        # 2. TELUGU / TENGLISH (Conversational Core)
        # =========================================================================
        # Pronouns, Demonstratives, Interrogatives
        "nenu", "naaku", "naadi", "naatho", "nannu", "nuvvu", "neeku", "needi", "neetho", "ninnu",
        "meeru", "meeku", "meedi", "meetho", "mimmulni", "manamu", "manaki", "manadi",
        "athanu", "athani", "aame", "aamenu", "vaallu", "vaallaki", "vaallani", "veellu", "veellaki",
        "idi", "idhi", "adi", "adhi", "edhi", "edi", "ivi", "avi", "evi", "annitini",
        "evaru", "evariki", "evadiki", "evadu", "enti", "enduku", "ela", "elaga", "eppudu", "ekkada",
        "deniki", "deeniki", "daaniki", "denitho", "deenitho",
        # Common Nouns & Adjectives
        "prashna", "samasya", "samasyalu", "vishayam", "sangathulu", "panulu", "pani",
        "samayam", "kaalam", "roju", "ivvala", "ee roju", "repu", "ninna", "ellundi",
        "twaraga", "ventane", "manchi", "manchiga", "chala", "konchem", "chinna", "pedda",
        "sari", "sariga", "sarainadi", "thappu", "avasaram", "mukhyam", "mukhyanga",
        "anandam", "badha", "kopam", "bhayam", "dabbu", "dabbulu", "illu", "intiki", "ooru",
        "annayya", "thammudu", "akka", "chelli", "mithrama", "macha", "bro",
        # Common Verbs & Conjugations
        "vastundi", "vastundhi", "vachindi", "vachindhi", "vastanu", "vastunna", "vastunnanu",
        "ravali", "ravalsi", "ravadam", "vachanu", "vachamu", "vacharu", "randi", "ra",
        "velthundi", "velthunna", "velthunnanu", "vellali", "vellanu", "veldam", "vellu", "vellandi",
        "ardhamkaledu", "arthamkaledu", "arthamavvaledhu", "ardhamayindi", "theliyadhu", "teliyadu",
        "thelidhu", "telusu", "telusukovali",
        "cheyyali", "cheyali", "chey", "cheyyi", "cheyyandi", "chestunna", "chestunnanu", "chesanu",
        "chesam", "chesaru", "cheyadam", "cheyagalanu", "cheyalenu",
        "chudali", "chudu", "chudandi", "choosanu", "chustunna", "chustunnanu", "choodu",
        "cheppali", "cheppu", "cheppandi", "cheppanu", "cheptunna", "cheptunnanu",
        "undali", "unnaavu", "unnanu", "unnaru", "unnara", "undi", "undhi", "unnayi", "undedi",
        "kavali", "vaddu", "oddu", "ivvali", "ivvandi", "ichanu", "istunna", "theesukovali",
        "matladali", "matladu", "matladandi", "matladanu", "matladuthunna",
        # Particles & Connectives
        "kani", "kaani", "kabatti", "anduke", "ayithe", "kuda", "kooda", "matrame", "leka",
        "ippudu", "appudu", "eppudu", "mundu", "tharuvatha", "tarvatha", "lo", "tho", "ki", "ku",

        # =========================================================================
        # 3. KANNADA / KANGLISH (Conversational Core)
        # =========================================================================
        "naanu", "nanage", "nanna", "nannanu", "neevu", "nimge", "nimma", "avaru", "avara", "avarge",
        "idu", "adu", "yenu", "yaake", "hege", "hegide", "elli", "yaavaga", "yaaru", "yaarge",
        "samasyegalu", "samasye", "vishaya", "kelasa", "samaya", "thumba", "tumba", "swalpa",
        "bega", "veegavagi", "chikka", "dodda", "ollaya", "chennagi", "thappu", "beku", "beda",
        "gothu", "gothilla", "arike", "arivilla", "maadbeku", "maadthini", "maadide", "maadoke", "maadi",
        "hogbeku", "hogthini", "hode", "hogi", "barbeku", "barthini", "bandhe", "banni", "baa",
        "nodbeku", "nodi", "helbeku", "heli", "idhe", "illa", "aadre", "naale", "ivattu", "ninne",

        # =========================================================================
        # 4. MALAYALAM / MANGLISH (Conversational Core)
        # =========================================================================
        "njan", "enikku", "ente", "enne", "ningal", "ningalkku", "ningalude", "avar", "avarkku",
        "ithu", "athu", "enthu", "enthukondu", "engane", "evide", "eppol", "aaru", "aarkku",
        "prashnam", "kaaryam", "pani", "samayam", "vegam", "nerathe", "valare", "kurachu",
        "nallathu", "thettu", "venam", "venda", "ariyaam", "ariyilla", "manasilayi", "manasilayilla",
        "cheyyanam", "cheyyuka", "cheythu", "cheyyam", "pokanam", "pokum", "poyi", "varanam", "varum",
        "vannu", "nokkanam", "nokku", "parayanam", "parayu", "und", "illa", "pakshe", "naale", "innu",

        # =========================================================================
        # 5. HINDI / URDU / HINGLISH (Conversational Core)
        # =========================================================================
        # Pronouns, Demonstratives, Interrogatives
        "mujhe", "tujhe", "hume", "hame", "humko", "tumko", "aapko", "usko", "isko", "unhe", "inhe",
        "mera", "meri", "mere", "tera", "teri", "tere", "uska", "uski", "uske", "iska", "iski", "iske",
        "unka", "unki", "unke", "inka", "inki", "inke", "humara", "humari", "humare", "hamara", "hamari", "hamare",
        "tumhara", "tumhari", "tumhare", "aapka", "aapki", "aapke", "apna", "apni", "apne",
        "kisi", "kisiko", "kisko", "kiska", "kiski", "kiske", "sabko", "sabka", "sabki", "sabke",
        "kya", "kyu", "kyun", "kyon", "kyuki", "kyunki", "kyonki", "kyoki", "kab", "kaha", "kahan",
        "kidhar", "kaise", "kaisa", "kaisi", "kitna", "kitni", "kitne", "kaun",
        # Common Nouns & Adjectives
        "samajh", "problem", "prashn", "itna", "itni", "itne", "chota", "choti", "chote", "bada", "badi", "bade",
        "time", "waqt", "samay", "kaam", "baat", "cheez", "koshish", "zaroorat", "zarurat", "zaroori", "zaruri",
        "important", "tension", "aasan", "mushkil", "thoda", "thodi", "thode", "zyada", "jyada", "bahut", "bohot",
        "bilkul", "sahi", "galat", "accha", "acha", "acchi", "achi", "acche", "ache", "bura", "buri", "bure",
        "sundar", "theek", "thik", "kharab", "fresh", "jaldi", "late", "aaram", "aram", "subah", "shaam", "raat",
        "din", "aaj", "kal", "parso", "ghar", "office", "station", "college", "school", "market", "rasta", "jagah",
        "khana", "paani", "chai", "coffee", "dost", "yaar", "bhai", "behen", "sir", "madam", "boss", "team",
        "project", "meeting", "call", "msg", "message", "reply", "update", "details", "info", "system", "file",
        # Verbs & Conjugations
        "karna", "karo", "kare", "karenge", "karega", "karegi", "karte", "karta", "karti", "kiya", "kiye", "kiyi",
        "karunga", "karungi", "karoge", "raha", "rahe", "rahi", "rahega", "rahegi", "rahenge", "hai", "hain",
        "tha", "thi", "the", "hoga", "hogi", "hoge", "honge", "hona", "hota", "hoti", "hote", "hua", "hue", "hui",
        "jana", "jao", "jaana", "jaayenge", "jaaunga", "jaaungi", "jaoge", "jaata", "jaati", "jaate", "gaya", "gaye", "gayi",
        "aana", "aao", "aata", "aati", "aate", "aaya", "aaye", "aayi", "aayega", "aayegi", "aayenge", "aaunga", "aaungi",
        "dekhna", "dekho", "dekha", "dekhe", "dekhi", "dekh", "dekhunga", "dekhungi", "dekhenge",
        "bolna", "bolo", "bola", "bole", "boli", "bol", "bolunga", "bolungi", "bolenge",
        "samajhna", "samjho", "samjha", "samjhe", "samjhi",
        "lagna", "laga", "lagi", "lage", "lagta", "lagti", "lagte", "lag", "lagega", "lagegi",
        "chalna", "chalo", "chala", "chale", "chali", "chal", "chalte", "chalta", "chalti", "chalega", "chalegi",
        "dena", "do", "diya", "diye", "dijiye", "de", "denge", "dega", "degi",
        "lena", "lo", "liya", "liye", "lijiye", "le", "lenge", "lega", "legi",
        "rakhna", "rakho", "rakha", "rakhe", "rakhi", "rakh",
        "paana", "paya", "paye", "payi", "paa", "paunga", "paungi", "paenge",
        "padna", "pada", "pade", "padi", "pad", "padega", "padegi", "padenge",
        "sunna", "suno", "suna", "sune", "suni", "sun",
        "batana", "batao", "bataya", "bataye", "batayi", "bata",
        "milna", "milo", "mila", "mile", "mili", "mil", "milenge",
        # Connectives & Particles
        "lekin", "magar", "par", "aur", "ya", "isliye", "taki", "warna", "agar", "yadi",
        "jab", "tab", "tabhi", "abhi", "phir", "fir", "waise", "jaise", "kuch", "koi",
        "sab", "sabhi", "toh", "to", "bhi", "hi", "si", "se", "sa", "tak", "mein", "par",
        "ko", "ne", "ke", "ki", "ka",

        # =========================================================================
        # 6. BENGALI / BONGLISH (Conversational Core)
        # =========================================================================
        "aami", "aamake", "aamar", "tumi", "tomake", "tomar", "aamra", "aamader", "she", "taake", "taar",
        "eita", "oita", "ki", "kyano", "kobe", "kothay", "kemon", "shomosya", "kaaj", "shomoy",
        "khub", "ektu", "bhalo", "kharap", "korte", "hobe", "korbo", "korchi", "jaabo", "aschhi",

        # =========================================================================
        # 7. MARATHI (Conversational Core)
        # =========================================================================
        "mee", "mala", "maajha", "maajhi", "maajhe", "tumhi", "tumhala", "tumcha", "tumchi", "tumche", "aapan",
        "he", "te", "kaay", "kasa", "kashi", "kadhi", "kuthe", "ka", "adhik", "kami", "chaan",
        "karnyachi", "ahe", "hotey", "kartoy", "jaato", "yeto", "karaycha", "pahije"
    ]

    unique_set = set()
    for w in words:
        clean = w.strip().lower()
        if clean:
            unique_set.add(clean)
    return unique_set


if __name__ == "__main__":
    lex = get_conversational_indic_lexicon()
    print(f"Colloquial Indic Lexicon: {len(lex):,} unique words.")
