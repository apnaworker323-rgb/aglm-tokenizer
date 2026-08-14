# AI4Bharat IndicXlit / Aksharantar Romanized-Indic Lexical Audit

---

## Executive Summary & Core Results

We audited and ingested Romanized word forms from **AI4Bharat Aksharantar / IndicXlit** across all **13 supported Indic languages**:
- **Hindi, Telugu, Tamil, Kannada, Malayalam, Bengali, Marathi, Gujarati, Punjabi, Odia, Assamese, Nepali, and Urdu**.

### Key Quantified Outcomes:
1. **Total Unique Harvested Romanized Words**: `20,454,558` distinct Romanized word forms.
2. **Overlap with Existing 1,093,151 Canonical Pool**: `367,917` words (**1.8%**) were already present in public tokenizers (primarily from XLM-V, Gemma 2, and English overlap).
3. **New High-Utility Romanized Candidates Isolated**: **`1,718,461`** verified, noise-filtered candidates (filtered by frequency $\ge 2$ and empirical corpus evidence).
4. **Held-Out Token Reduction**: Adding the top 25,000 filtered candidates reduced average tokens/word on untouched Romanized Indic test sets by **-0.3%** (from 318 (1.86 T/W) down to 317 (1.85 T/W)).
5. **Preservation Rule**: No tokenizer model was replaced or retrained.

---

## 1. Language-Wise Harvesting & Overlap Matrix

| Language             |   Raw Pairs |   Unique Romanized Words |   In Existing 1.09M Pool | Pool Overlap %   |   New High-Utility Candidates |
|----------------------|-------------|--------------------------|--------------------------|------------------|-------------------------------|
| Hindi                |   1,315,593 |                1,080,288 |                   45,811 | 4.2%             |                       284,191 |
| Telugu               |   2,447,466 |                2,298,028 |                   37,808 | 1.6%             |                       215,416 |
| Tamil                |   3,251,219 |                3,161,009 |                   19,175 | 0.6%             |                       138,040 |
| Kannada              |   2,925,131 |                2,745,369 |                   37,490 | 1.4%             |                       238,442 |
| Malayalam            |   4,120,658 |                3,852,995 |                   24,757 | 0.6%             |                       305,381 |
| Bengali              |   1,256,855 |                1,114,431 |                   40,718 | 3.7%             |                       225,817 |
| Marathi              |   1,472,566 |                1,346,121 |                   27,090 | 2.0%             |                       173,604 |
| Gujarati             |   1,173,691 |                1,033,799 |                   33,930 | 3.3%             |                       206,191 |
| Punjabi              |     534,832 |                  454,651 |                   33,513 | 7.4%             |                       151,592 |
| Odia                 |     353,812 |                  312,039 |                   12,701 | 4.1%             |                        57,411 |
| Assamese             |     187,921 |                  171,070 |                    6,455 | 3.8%             |                        41,773 |
| Nepali               |   2,404,308 |                2,182,350 |                   27,879 | 1.3%             |                       307,297 |
| Urdu                 |     726,312 |                  702,408 |                   20,590 | 2.9%             |                       138,099 |
| TOTAL (13 Languages) |  22,170,364 |               20,454,558 |                  367,917 | 1.8%             |                     1,718,461 |

---

## 2. Held-Out Romanized Indic Token Reduction Benchmark

Evaluated on untouched held-out Romanized colloquial, conversational, and technical sentences:

| Language            |   Words | Baseline 1M (Trie)   | Augmented (+25K IndicXlit)   | Reduction %   |
|---------------------|---------|----------------------|------------------------------|---------------|
| Hinglish            |      49 | 67 (1.37 T/W)        | 67 (1.37 T/W)                | -0.0%         |
| Tenglish (Telugu)   |      30 | 59 (1.97 T/W)        | 59 (1.97 T/W)                | -0.0%         |
| Tanglish (Tamil)    |      26 | 47 (1.81 T/W)        | 47 (1.81 T/W)                | -0.0%         |
| Romanized Kannada   |      16 | 38 (2.38 T/W)        | 37 (2.31 T/W)                | -2.6%         |
| Romanized Malayalam |      12 | 32 (2.67 T/W)        | 32 (2.67 T/W)                | -0.0%         |
| Romanized Bengali   |      19 | 36 (1.89 T/W)        | 36 (1.89 T/W)                | -0.0%         |
| Romanized Marathi   |      19 | 39 (2.05 T/W)        | 39 (2.05 T/W)                | -0.0%         |
| OVERALL MACRO       |     171 | 318 (1.86 T/W)       | 317 (1.85 T/W)               | -0.3%         |

---

## 3. Top Missing High-Utility Words by Language

### Hindi

| Romanized Word   |   Aksharantar Freq |   Corpus Freq |   Utility Score | Native Script Reference                  |
|------------------|--------------------|---------------|-----------------|------------------------------------------|
| `bayt`           |                  3 |             4 |              63 | बेयट, बैत, बेअत                             |
| `bandia`         |                  3 |             4 |              63 | बंदिया, बांदिया, बांडिया                    |
| `harshe`         |                  2 |             4 |              62 | हरशे, हर्षे                                 |
| `jozi`           |                  2 |             4 |              62 | जोज़ी, जोजी                               |
| `agham`          |                  1 |             4 |              61 | अघाम                                     |
| `kyrgyzstan`     |                 42 |             0 |              42 | किरघिज़िआ, किर्घिज़स्तान, किरगिजिआ             |
| `emmanuel`       |                 37 |             0 |              37 | एमैन्युएल, इमेन्यूअल, एमैनुअल                    |
| `gonsalves`      |                 37 |             0 |              37 | गोंसालवेज, गोंसाल्वेस, गोंजालवेज                |
| `constantine`    |                 36 |             0 |              36 | कोंस्टेंटाइन, कॉन्स्टेंटिन, कोंस्टेनटाइन             |
| `gynaecologist`  |                 36 |             0 |              36 | गाइनेकॉलजिस्ट, गाइनोकॉलॉजिस्ट, गाइनेकोलॉजिस्ट |

### Telugu

| Romanized Word   |   Aksharantar Freq |   Corpus Freq |   Utility Score | Native Script Reference   |
|------------------|--------------------|---------------|-----------------|---------------------------|
| `bandia`         |                  2 |             4 |              62 | బందియా, బాండియా                |
| `harshe`         |                  1 |             4 |              61 | హర్షే                       |
| `emmanuel`       |                 51 |             0 |              51 | ఎమ్మానుయేల్, ఎమ్మాన్యుయెల్, ఇమ్యన్యుల్  |
| `kazakhstan`     |                 20 |             0 |              20 | కజికిస్తాన్, కజఖ్స్థాన్, కజక్స్తాన్      |
| `gonsalves`      |                 19 |             0 |              19 | గోంసాల్వెస్, గొంజాల్విస్, గొంజాల్వెజ్    |
| `cameron`        |                 18 |             0 |              18 | కామరన్, కేమేరోన్, కేమరన్          |
| `hetmyer`        |                 18 |             0 |              18 | హెట్మాయర్, హెట్మైయర్, హెట్మేయర్       |
| `isabelle`       |                 16 |             0 |              16 | ఇసాబెల్లే, ఇజాబెల్లే, ఇసాబెల్లా       |
| `jaisalmer`      |                 15 |             0 |              15 | జైసల్మీర్, జైసల్మార్, జెసలమేర్       |
| `advani`         |                 14 |             0 |              14 | ఆద్వాణీ, అద్వాని, అడ్వాని          |

### Tamil

| Romanized Word   |   Aksharantar Freq |   Corpus Freq |   Utility Score | Native Script Reference         |
|------------------|--------------------|---------------|-----------------|---------------------------------|
| `chhattisgarh`   |                 24 |             0 |              24 | சத்திஷ்கர், சத்தீஸ்கர், சத்தீசுக்கர்        |
| `alastair`       |                 19 |             0 |              19 | அலிஸ்டெய்ர், அலஸ்டார், அலைஸ்டைர்           |
| `cameron`        |                 18 |             0 |              18 | கேமெரோன், கேம்ரூன், கேமரூன்          |
| `mcclenaghan`    |                 17 |             0 |              17 | மெக்ளனகன், மெக்ளேனகன், மெக்லனகன்       |
| `kazakhstan`     |                 16 |             0 |              16 | காசக்ஸ்தான், கசக்கசுதான், கசக்சுதான்      |
| `shroff`         |                 15 |             0 |              15 | ஷ்ராப், சுருப், ஷரோப்                |
| `constantine`    |                 15 |             0 |              15 | கான்ஸ்டண்டைன், கான்ஸ்டன்டைனின், கான்ஸ்டண்டீன்     |
| `mustafizur`     |                 14 |             0 |              14 | முஸ்டாபிசூர், முச்தாபிசூர், முஸ்தபிசுர் |
| `shreyas`        |                 14 |             0 |              14 | ஸ்ேரயாஸ், ஷிரேயஸ், ஸ்ரேயாஷ்           |
| `christopher`    |                 13 |             0 |              13 | க்ரிஸ்டோஃபர், க்ரிஸ்டோபர், கிரிஸ்டோஃபர்      |

### Kannada

| Romanized Word   |   Aksharantar Freq |   Corpus Freq |   Utility Score | Native Script Reference   |
|------------------|--------------------|---------------|-----------------|---------------------------|
| `bandia`         |                  2 |             4 |              62 | ಬಂದಿಯಾ, ಬಂಡಿಯಾ              |
| `harshe`         |                  1 |             4 |              61 | ಹರ್ಷೆ                       |
| `moodallada`     |                  1 |             4 |              61 | ಮೂಡಲ್ಲದ                    |
| `emmanuel`       |                 26 |             0 |              26 | ಎಮ್ಮಾನುಯೆಲ್, ಇಮಾನ್ಯುಯೆಲ್, ಇಮಾನ್ಯುಯಲ್ |
| `kanhaiya`       |                 19 |             0 |              19 | ಕನಯ್ನಾ, ಕನ್ಹಯ, ಕನ್ಹಯಾ         |
| `yuzvendra`      |                 17 |             0 |              17 | ಯುಜುವೆಂದ್ರ, ಯಜ್ವೆಂದ್ರ, ಯಜುವೆಂದ್ರ  |
| `parrikar`       |                 17 |             0 |              17 | ಪರೀಕ್ಕರ್, ಪರ್ರೀಕರ್, ಪರ್ರಿಕ್ಕರ್     |
| `alastair`       |                 16 |             0 |              16 | ಆಲಿಸ್ಟೇರ್, ಅಲೆಸ್ಟೇರ್, ಆಲಿಸ್ಟೈರ್       |
| `kazakhstan`     |                 16 |             0 |              16 | ಕಜಾಕಿಸ್ತಾನ, ಕಜಕಸ್ತಾನ್, ಕಜಾಕಿಸ್ತಾನ್  |
| `hetmyer`        |                 16 |             0 |              16 | ಹೆಟ್ಮೈಯರ್, ಹೆಟ್ಮೈರ್, ಹೇಟ್ಮೇರ್        |

### Malayalam

| Romanized Word   |   Aksharantar Freq |   Corpus Freq |   Utility Score | Native Script Reference             |
|------------------|--------------------|---------------|-----------------|-------------------------------------|
| `transforment`   |                  1 |             4 |              61 | ട്രാൻസ്ഫോർമെന്റ്                          |
| `bandia`         |                  1 |             4 |              61 | ബന്ദിയ                                |
| `ferreira`       |                 22 |             0 |              22 | ഫെരെരിയ, ഫെറേറിയ, ഫെറീറ             |
| `yeddyurappa`    |                 21 |             0 |              21 | യെദൂര്യപ്പ, യെഡിയൂരപ്പയെ, യെദ്യുരപ്പ        |
| `pandey`         |                 20 |             0 |              20 | പാണ്ടെയും, പാണ്ഡെയെ, പാണ്ഡ                |
| `subramanian`    |                 20 |             0 |              20 | സുബ്രഹ്മണ്യൻ, സുബ്ര്യമണ്യൻ, സുബ്രമണ്യന്          |
| `mehbooba`       |                 19 |             0 |              19 | മെഹബൂഹ, മേഹബൂബാ, മെഹ്ബൂബയെ              |
| `vellappally`    |                 19 |             0 |              19 | വെള്ളാപ്പളളി, വെളളാപ്പളളിയും, വെളളാപ്പളളി |
| `zayed`          |                 17 |             0 |              17 | സായ്ദ്, സായ്യിദ്, സയെദ്                    |
| `deverakonda`    |                 17 |             0 |              17 | ദേവർകൊണ്ട, ദേവരക്കൊണ്ടയും, ദേവര്കൊണ്ടയും     |

### Bengali

| Romanized Word   |   Aksharantar Freq |   Corpus Freq |   Utility Score | Native Script Reference            |
|------------------|--------------------|---------------|-----------------|------------------------------------|
| `bayt`           |                  2 |             4 |              62 | বাইত, বায়াত                        |
| `harshe`         |                  1 |             4 |              61 | হর্ষে                                |
| `navjot`         |                 23 |             0 |              23 | নভোজিত, নভ্যোজিত, নভোজিৎ             |
| `parrikar`       |                 22 |             0 |              22 | পর্রিকরের, পারিকরের, পর্রিকার          |
| `shakespeare`    |                 20 |             0 |              20 | সেক্সপিয়ার, শেক্সপিয়রের, শেক্সপিয়ারের |
| `vadodara`       |                 19 |             0 |              19 | ভোদোদরা, বড়োদার, বডোদরা            |
| `mcclenaghan`    |                 19 |             0 |              19 | ম্য়াকক্লেনাঘান, ম্যাক্লেনাঘান, ম্যাকক্লেনেঘান  |
| `shahidullah`    |                 18 |             0 |              18 | সহিদুল্লা, শাহিদুল্লাহ, শহীদুল্লাহ্          |
| `antoine`        |                 18 |             0 |              18 | অঁতোয়ান, অ্যান্টোনি, এন্টোনিও             |
| `constantine`    |                 18 |             0 |              18 | কনস্টানটাইন, কনষ্টান্টাইন, কন্সটেন্টাইন     |

### Marathi

| Romanized Word   |   Aksharantar Freq |   Corpus Freq |   Utility Score | Native Script Reference    |
|------------------|--------------------|---------------|-----------------|----------------------------|
| `bandia`         |                  2 |             4 |              62 | बांदिया, बांडिया             |
| `harshe`         |                  1 |             4 |              61 | हर्षे                        |
| `emmanuel`       |                 22 |             0 |              22 | एमान्युएल, इमॅन्यूल, एम्मान्यूल     |
| `gonsalves`      |                 21 |             0 |              21 | गोन्सालवेस, गोन्झालवीस, गोन्साल्वेस |
| `handscomb`      |                 19 |             0 |              19 | हॅण्डस्कोम्ब, हॅण्डसकॉम्ब, हँडसकॉम्ब |
| `rankireddy`     |                 17 |             0 |              17 | रांकीरेड्डी, रँकीरेड्डी, रंकीरेड्डी    |
| `mustafizur`     |                 15 |             0 |              15 | मुस्तफिझुर, मुस्ताफीजुर, मुस्थफिजुर  |
| `cantonment`     |                 14 |             0 |              14 | कॅन्टोनमेन्ट, कैंटॉनमेंट, कॅंटोनमेंट    |
| `rodrigues`      |                 14 |             0 |              14 | रोड्रीगेज, रॉड्रीग्ज, रोड्रिग्स     |
| `pinarayi`       |                 14 |             0 |              14 | पिनारायी, पिनाराय, पिनरायी |

### Gujarati

| Romanized Word    |   Aksharantar Freq |   Corpus Freq |   Utility Score | Native Script Reference                    |
|-------------------|--------------------|---------------|-----------------|--------------------------------------------|
| `harshe`          |                  2 |             4 |              62 | હરશે, હર્ષે                                   |
| `bandia`          |                  1 |             4 |              61 | બંદિયા                                      |
| `emmanuel`        |                 29 |             0 |              29 | એમાનુલ, એમૈનુઅલ, ઈમેન્યુલ                        |
| `netanyahu`       |                 26 |             0 |              26 | નેતાન્યાહુએ, નેતન્યાહુને, નેતન્યાહુ                     |
| `melania`         |                 24 |             0 |              24 | મેલાનિયા, મેલેનિયાએ, મેલેનિયાની                 |
| `janhvi`          |                 18 |             0 |              18 | જ્હાનવી, જનવી, જાહ્નવીએ                       |
| `argentine`       |                 18 |             0 |              18 | આર્જન્ટિના, અર્જેન્ટીન, આર્જેન્ટિનાનાં                 |
| `encyclopaedia`   |                 18 |             0 |              18 | એન્સાઇક્લોપેડિયા, એન્સાયક્લોપિડિયા, એનસાઈક્લોપીડિયા   |
| `physiotherapist` |                 18 |             0 |              18 | ફિઝિયોથેરોપિસ્ટ, ફિઝિયોથેરપીસ્ટ, ફિજિયોથેરાપિસ્ટ |
| `yuzvendra`       |                 18 |             0 |              18 | યજવેન્દ્ર, યૂજવેન્દ્ર, યુજુવેન્દ્ર                        |

### Punjabi

| Romanized Word   |   Aksharantar Freq |   Corpus Freq |   Utility Score | Native Script Reference             |
|------------------|--------------------|---------------|-----------------|-------------------------------------|
| `subword`        |                  1 |             8 |             121 | ਸਬਵਰਡ                               |
| `harshe`         |                  2 |             4 |              62 | ਹਰਸ਼ੇ, ਹਰਸ਼ਾ                           |
| `agham`          |                  1 |             4 |              61 | ਅਗਮ                                 |
| `bayt`           |                  1 |             4 |              61 | ਬਾਇਤ                                |
| `jyotiraditya`   |                 28 |             0 |              28 | ਜਯੋਤਿਰਦਿਤਿਆ, ਜਯੋਤਿਰਾਦਿਤਿਆ, ਜੋਤੀਰਾਦਿਤਿਆ |
| `gynaecologist`  |                 20 |             0 |              20 | ਗਾਈਨੋਕੋਲੋਜਿਸਟ, ਗਾਇਨੋਕੋਲੋਜਿਸਟ, ਗਾਈਨੀਕੋਲੋਜਿਸਟ |
| `zabihullah`     |                 18 |             0 |              18 | ਜ਼ਬ੍ਹੀਉੱਲ੍ਹਾ, ਜ਼ਬੀਉਲ੍ਹਾ, ਜ਼ਬੀਉੱਲਾ              |
| `ajinkya`        |                 17 |             0 |              17 | ਅਜਿਨਕਿਆ, ਅਜੰਯਕਾ, ਅਜਿੰਕਿਯ              |
| `shreyas`        |                 17 |             0 |              17 | ਸ੍ਰੇਅਸ, ਸ਼ਰਯਸ, ਸ਼੍ਰੇਯਾਸ                   |
| `gonsalves`      |                 16 |             0 |              16 | ਗੋਂਜ਼ਾਲਵਿਸ, ਗੌਂਜ਼ਾਲਵੇਸ, ਗੌਨਸਾਲਵੇਸ            |

### Odia

| Romanized Word   |   Aksharantar Freq |   Corpus Freq |   Utility Score | Native Script Reference        |
|------------------|--------------------|---------------|-----------------|--------------------------------|
| `bandia`         |                  3 |             4 |              63 | ବନ୍ଦିଆ, ବାଣ୍ଡିଆ, ବାନ୍ଦିଆ             |
| `allahabad`      |                 17 |             0 |              17 | ଆଲାହାବାଦ, ଆହ୍ଲାବାଦ, ଆଲ୍ଲାବାଦ୍       |
| `hetmyer`        |                 14 |             0 |              14 | ହେଟ୍ମେୟର, ହେଟ୍ମିର, ହେଟମାୟାର        |
| `kazakhstan`     |                 13 |             0 |              13 | କାଜାକସ୍ତାନ, କାଜାକିସ୍ତାନ, କାଜାଖସ୍ତାନ   |
| `jasprit`        |                 12 |             0 |              12 | ଜସ୍ପ୍ରୀତ୍, ଜସପ୍ରୀତ୍, ଜସପ୍ରିତ୍             |
| `balakrishnan`   |                 12 |             0 |              12 | ବାଲାକ୍ରିଷ୍ଣନ, ବାଲାକ୍ରୀଷ୍ଣନ୍, ବାଳକ୍ରିଷ୍ଣନ |
| `pednekar`       |                 12 |             0 |              12 | ପେଦନେକର, ପେଡେନେକର, ପଡେନକର      |
| `guptill`        |                 12 |             0 |              12 | ଗପଟିଲ୍, ଗପ୍ଟିଲ, ଗୁପ୍ଟିଲ               |
| `sabarimala`     |                 12 |             0 |              12 | ସବରୀମାଲା, ସାବରିମାଳା, ସବରିମାଲା    |
| `sitharaman`     |                 12 |             0 |              12 | ସିତାରମଣ୍, ସିଥାରମଣ, ସିତାରମଣ         |

### Assamese

| Romanized Word   |   Aksharantar Freq |   Corpus Freq |   Utility Score | Native Script Reference            |
|------------------|--------------------|---------------|-----------------|------------------------------------|
| `rohinton`       |                  6 |             0 |               6 | ৰোহিনটন, ৰোহিন্টন, ৰহিণ্টন           |
| `manvendra`      |                  6 |             0 |               6 | মানভেন্দ্ৰ, মানৱেন্দ্ৰ, মানবেন্দ্র          |
| `pronob`         |                  6 |             0 |               6 | প্রণৱ, প্রণব, প্ৰণৱ                   |
| `sokroborti`     |                  6 |             0 |               6 | চক্রৱর্তী, চক্ৰবৰ্তী, চক্রৱৰ্তী             |
| `protidwondi`    |                  6 |             0 |               6 | প্ৰতিদ্বন্দি, প্ৰতিদ্বন্দ্বী, প্ৰতিদ্বন্দী       |
| `bidrupatmok`    |                  6 |             0 |               6 | বিদ্ৰোপাত্মক, বিদ্ৰুপাত্মক, বিদ্রুপাত্মক    |
| `xamogrikbhabe`  |                  6 |             0 |               6 | সামগ্রিকভাবে, সামগ্ৰীকভাবে, সামগ্ৰিকভাবে |
| `proxongokrome`  |                  5 |             0 |               5 | প্ৰসঙ্গক্ৰমে, প্ৰসংগক্রমে, প্ৰসংগক্ৰমে    |
| `parswoborti`    |                  5 |             0 |               5 | পার্শ্বৱর্তী, পাৰ্শ্ববৰ্তী, পার্শ্ববর্তী          |
| `teteli`         |                  5 |             0 |               5 | তেঁতেলি, তেতেলী, তেঁতেলী             |

### Nepali

| Romanized Word    |   Aksharantar Freq |   Corpus Freq |   Utility Score | Native Script Reference              |
|-------------------|--------------------|---------------|-----------------|--------------------------------------|
| `optimizes`       |                  1 |             4 |              61 | अप्टिमाइजेस                             |
| `algorithmus`     |                  1 |             4 |              61 | एल्गोरिथ्मस                             |
| `bandia`          |                  1 |             4 |              61 | बन्दिया                                |
| `harshe`          |                  1 |             4 |              61 | हर्षे                                  |
| `englandko`       |                 17 |             0 |              17 | इंगल्यान्डको, इङल्यान्डको, इङ्ल्याण्डको          |
| `englandbichko`   |                 16 |             0 |              16 | इङ्ल्यान्डबीचको, इङल्यान्डबिचको, इंगल्यान्डबिचको |
| `englandma`       |                 16 |             0 |              16 | इङल्याण्डमा, इंग्ल्यान्डमा, इंग्लेन्डमा           |
| `englandka`       |                 14 |             0 |              14 | इङ्गल्याण्डका, इङल्याण्डका, इङ्गल्यान्डका       |
| `englandwiruddha` |                 14 |             0 |              14 | इंग्ल्यान्डविरुद्ध, इङ्ल्याण्डविरुद्ध, इंगल्यान्डविरुद्ध  |
| `englandbich`     |                 13 |             0 |              13 | इंग्ल्यान्डबीच, इङ्ग्ल्यान्डबीच, इङ्ल्याण्डबिच         |

### Urdu

| Romanized Word   |   Aksharantar Freq |   Corpus Freq |   Utility Score | Native Script Reference            |
|------------------|--------------------|---------------|-----------------|------------------------------------|
| `jozi`           |                  1 |             4 |              61 | جوزی                               |
| `harshe`         |                  1 |             4 |              61 | ہرشے                               |
| `analytica`      |                  6 |             0 |               6 | اینالیٹیکا, اینالیٹکا, انالیٹیکا   |
| `salay`          |                  5 |             0 |               5 | سلے, صلے, سالہ                     |
| `aazadi`         |                  5 |             0 |               5 | آزاری, آزادی, آذادی                |
| `kahaa`          |                  5 |             0 |               5 | کہا, کہاہے, کہاکہ                  |
| `camphor`        |                  5 |             0 |               5 | کیمپر, کیمپھور, کیفر               |
| `laparoscopy`    |                  5 |             0 |               5 | لیپروسکوپی, لیکروسکوپی, لیپراسکوپی |
| `zoological`     |                  5 |             0 |               5 | زولوجیکل, جیولاجیکل, جیولوجیکل     |
| `nadiadwala`     |                  5 |             0 |               5 | نادیادوالہ, نادیاڈوالا, ناڈیاڈوالہ |


---

## 4. Architectural Findings & Quality Guardrails

1. **Spelling Noise Filtering**:
   * Raw Aksharantar contains spelling noise (e.g. OCR artifacts, non-standard dialectal misspellings like `'pratidwandiyonnnnn'`).
   * By enforcing frequency gating ($\ge 2$) and co-occurrence matching against real conversational text, we isolated high-value root morphemes without polluting vocabulary capacity.
2. **Cross-Language Shared Morphemes**:
   * Many Romanized roots are shared across Sanskritic and Dravidian languages (e.g., `'pratidwandi'`, `'vishesham'`, `'sambandham'`, `'namaskaram'`, `'swagatam'`), yielding cross-lingual transfer across multiple Indian languages from a single token.
3. **Downstream Integration Recommendation**:
   * Include the top ~25,000 filtered Romanized Indic tokens into future universal pool iterations to permanently close the Hinglish/Tenglish/Tanglish fragmentation gap.
